"""GWAS workflow stage report: inspect a ``gwas`` load manifest and its tables.

Opens the primary output of the end-to-end **export -> regenie -> load** GWAS
workflow -- the **load manifest** written by ``GwasLoadPipeline.get_manifest``
(``{checkpoint}/intermediate/regenie/manifest.json.gz``) -- which indexes every
loaded ``(table, region)`` dataset with its ``gwas``/``burden`` hail-table paths
and its provenance (``regenie_version``, ``responses``, the filter globals and
the effect-mask info). The report screens what the workflow produced:

- **which tests ran** -- the ``(table x region)`` fan-out, one manhattan per
  GWAS ``(dataset, phenotype, test)`` and per burden ``(dataset, phenotype,
  mask, aaf)``;
- **which filters ran** -- each filter global's variant/sample counts and
  cutoffs, and how much each kept;
- **the callset metrics** -- the effect-mask class counts and the per-filter
  counts, plotted as scatter/heatmaps when they grow with filters/region sets.

Manhattan rows are **downsampled** (top-N by ``log10_p`` plus a random
background, capped at ``GMEB_GWAS_MANHATTAN_MAX``, default 100k) because a
production ``gwas`` table holds millions of rows; every genome-wide-significant
variant is kept. Burden region coordinates are inferred from each exported
dataset's ``variant_sets`` TSV.

Run it where the Hail/Spark stack works (the integration container, per
``AGENTS.md`` and ADR-0002). Path is ``GMEB_GWAS_MANIFEST`` else the manifest
the run's settings resolve to.
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App()

with app.setup:
    import os
    import sys
    from pathlib import Path

    import marimo as mo
    import pandas as pd

    def _find_repo_root() -> Path:
        here = Path.cwd().resolve()
        for candidate in (here, *here.parents):
            if (candidate / "src" / "gnomad_meb").is_dir():
                return candidate
        raise RuntimeError(
            "could not locate the repository root (no src/gnomad_meb)"
        )

    REPO = _find_repo_root()
    sys.path.insert(0, str(REPO / "src"))

    from gnomad_meb import reporting
    from gnomad_meb.config import get_config
    from gnomad_meb.hail import hail as hl
    from gnomad_meb.hail import hail_init
    from gnomad_meb.lib.gwas.manifest import read_gwas_manifest

    # Reports read artifacts on a Hail-capable machine; always initialize at
    # module level so every cell below runs against the same Spark session.
    hail_init()

    # --- what to report -----------------------------------------------------
    # The load manifest is the one index of every loaded table; override with
    # ``GMEB_GWAS_MANIFEST``, else resolve through the run's settings.
    def _default_manifest() -> Path:
        from gnomad_meb.lib.gwas.manifest import gwas_manifest_path

        try:
            resolved = gwas_manifest_path(get_config())
        except Exception:
            resolved = None
        return (
            Path(resolved)
            if resolved
            else (
                REPO
                / "data"
                / "results"
                / "grch38"
                / "intermediate"
                / "regenie"
                / "manifest.json.gz"
            )
        )

    # _manifest_env = os.environ.get("GMEB_GWAS_MANIFEST")
    # GWAS_MANIFEST = Path(_manifest_env) if _manifest_env else _default_manifest()
    GWAS_MANIFEST = os.environ.get(
        "GMEB_GWAS_MANIFEST",
        Path(
            "/data/scratch/gnomad/checkpoints/gnomad-meb-test/intermediate/regenie/manifest.json.gz"
        ),
    )
    #: Manhattan downsampling cap, kept per table (top-N plus a random
    #: background). Every genome-wide-significant variant is always retained.
    MANHATTAN_MAX = int(os.environ.get("GMEB_GWAS_MANHATTAN_MAX", "100000"))

    MANIFEST = read_gwas_manifest(GWAS_MANIFEST)

    RESULTS: dict[str, object] = {}

    def render(frame, title, note=""):
        """A titled table, the standard output shape for the report cells."""
        blocks = [mo.md(f"### {title}")]
        if note:
            blocks.append(mo.md(note))
        blocks.append(mo.ui.table(frame))
        return mo.vstack(blocks)


@app.cell
def _():
    mo.md(f"""
    # GWAS workflow — stage report

    Primary output: **`{GWAS_MANIFEST}`** (the load manifest).

    This notebook reads the manifest — the index of every loaded ``(table,
    region)`` dataset, its filter globals and effect-mask provenance — and the
    terminal ``gwas`` / ``burden`` hail tables it names. It reports which
    tests and filters ran, plots their metadata, and draws a manhattan for every
    GWAS and burden test. It aggregates with Hail and downsamples the manhattan
    rows; it never unions a callset (ADR-0015). Path is `GMEB_GWAS_MANIFEST`,
    else the manifest the run's settings resolve to.
    """)
    return


@app.cell
def _():
    _provenance = reporting.run_provenance(MANIFEST)
    RESULTS["provenance"] = _provenance
    _datasets = reporting.manifest_datasets(MANIFEST)
    RESULTS["datasets"] = _datasets
    mo.vstack(
        [
            mo.md(
                f"**Project**: `{_provenance['project']}` "
                f"({_provenance['reference_genome']}).\n\n"
                f"**Datasets**: {_provenance['n_datasets']} loaded "
                f"across region sets {_provenance['regions']}.\n\n"
                f"**Responses**: {_provenance['responses']}.\n\n"
                f"**Filters / tables**: {_provenance['sources']}.\n\n"
                f"**regenie**: {_provenance['regenie_versions']}."
            ),
            render(
                _datasets,
                "Loaded datasets",
                "One row per `(table, region)` dataset: its source, region set, "
                "responses, regenie version and the `gwas` / `burden` table "
                "paths the workflow produced.",
            ),
        ]
    )
    return


@app.cell
def _():
    _matrix = reporting.dataset_matrix(RESULTS["datasets"])
    RESULTS["dataset_matrix"] = _matrix
    reporting.dataset_region_heatmap(_matrix)
    return


@app.cell
def _():
    _variant_counts = reporting.filter_variant_counts(MANIFEST)
    _sample_counts = reporting.filter_sample_counts(MANIFEST)
    _cutoffs = reporting.filter_cutoffs(MANIFEST)
    RESULTS["filter_variant_counts"] = _variant_counts
    RESULTS["filter_sample_counts"] = _sample_counts
    RESULTS["filter_cutoffs"] = _cutoffs
    mo.vstack(
        [
            mo.md(
                "**Filters executed.** Each dataset's provenance records the "
                "filter globals it inherited (the high-quality basis plus any "
                "variant/region filter), each with its step-wise variant and "
                "sample counts. The counts are plotted, not tabulated wide, so "
                "an arbitrary set of filters x datasets stays legible."
            ),
            render(
                _variant_counts,
                "Variants per filter step",
                "`step` is the filter global's own counter key "
                "(`default`/`high_quality`/`inbreeding`/`vqsr`/`final`).",
            ),
            reporting.filter_variant_scatter(_variant_counts),
            reporting.filter_sample_scatter(_sample_counts),
            render(
                _cutoffs,
                "Filter cutoffs",
                "The thresholds each filter global applied.",
            ),
        ]
    )
    return


@app.cell
def _():
    _effect = reporting.effect_mask_counts(MANIFEST)
    RESULTS["effect_mask_counts"] = _effect
    mo.vstack(
        [
            mo.md(
                "**Effect masks.** The `effect_mask_info` each dataset recorded: "
                "the per-class variant count for every regenie mask (`lof`, "
                "`missense`, `cadd`, `clinvar`, `rare`) and the variant-class "
                "breakdown. Rendered as a bar per dataset and a heatmap over "
                "datasets, so the mask x dataset grid never grows into a table."
            ),
            render(_effect, "Effect-mask class counts"),
            reporting.effect_mask_bar(_effect),
            reporting.effect_mask_heatmap(_effect),
        ]
    )
    return


@app.cell
def _():
    # One overview + manhattan per loaded GWAS table. The tables are read lazily
    # by the manifest paths; the manhattan rows are downsampled inside the
    # collector, so the driver pull is bounded whatever the table's size.
    _blocks = []
    for _dataset in MANIFEST.datasets:
        _ht = hl.read_table(_dataset.gwas)
        _overview = reporting.gwas_overview(_ht)
        RESULTS[f"gwas_overview::{_dataset.label}"] = _overview
        _manhattan = reporting.gwas_manhattan(_ht, max_rows=MANHATTAN_MAX)
        RESULTS[f"gwas_manhattan::{_dataset.label}"] = _manhattan
        _lambda = reporting.gwas_lambda(_ht)
        RESULTS[f"gwas_lambda::{_dataset.label}"] = _lambda
        _blocks.append(
            mo.vstack(
                [
                    mo.md(
                        f"### GWAS — `{_dataset.label}`\n\n"
                        f"{_overview['n_rows']:,} variant-tests over "
                        f"phenotypes {_overview['phenotypes']}, tests "
                        f"{_overview['tests']}; "
                        f"{_overview['n_sig']:,} genome-wide significant "
                        f"(p < 5e-8); best "
                        f"`-log10(p)` {_overview['max_log10_p']:.1f}."
                    ),
                    reporting.manhattan_plot(
                        _manhattan,
                        title=f"GWAS manhattan — {_dataset.label}",
                        facet="phenotype",
                    ),
                    reporting.lambda_bar(_lambda),
                ]
            )
        )
    mo.vstack(_blocks) if _blocks else mo.md("_No GWAS datasets loaded._")
    return


@app.cell
def _():
    # QQ per GWAS table (bounded pull), drawn as small multiples by phenotype.
    _blocks = []
    for _dataset in MANIFEST.datasets:
        _qq = reporting.gwas_qq(hl.read_table(_dataset.gwas))
        RESULTS[f"gwas_qq::{_dataset.label}"] = _qq
        _blocks.append(
            mo.vstack(
                [
                    mo.md(f"### QQ — `{_dataset.label}`"),
                    reporting.qq_plot(_qq, facet="phenotype"),
                ]
            )
        )
    mo.vstack(_blocks) if _blocks else mo.md("_No GWAS datasets loaded._")
    return


@app.cell
def _():
    # One overview + burden manhattan per loaded burden table, one manhattan
    # point per region test. Region coordinates come from the dataset's
    # exported ``variant_sets`` TSV; gene ids carry a TODO (see reporting.gwas).
    _blocks = []
    for _dataset in MANIFEST.datasets:
        _ht = hl.read_table(_dataset.burden)
        _overview = reporting.burden_overview(_ht)
        RESULTS[f"burden_overview::{_dataset.label}"] = _overview
        _coordinates = reporting.region_coordinates(_dataset.dataset)
        _manhattan = reporting.burden_manhattan(_ht, _coordinates)
        RESULTS[f"burden_manhattan::{_dataset.label}"] = _manhattan
        _summary = reporting.burden_summary(_ht)
        RESULTS[f"burden_summary::{_dataset.label}"] = _summary
        _blocks.append(
            mo.vstack(
                [
                    mo.md(
                        f"### Burden — `{_dataset.label}`\n\n"
                        f"{_overview['n_rows']:,} region tests over "
                        f"{_overview['n_regions']:,} regions, masks "
                        f"{_overview['masks']}, AAF strata {_overview['aafs']}, "
                        f"phenotypes {_overview['phenotypes']}; best "
                        f"`-log10(p)` {_overview['max_log10_p']:.1f}."
                    ),
                    reporting.manhattan_plot(
                        _manhattan,
                        title=f"Burden manhattan — {_dataset.label}",
                        facet="phenotype",
                    ),
                    reporting.burden_mask_scatter(_summary),
                ]
            )
        )
    mo.vstack(_blocks) if _blocks else mo.md("_No burden datasets loaded._")
    return


@app.cell
def _():
    _lines = ["## Reading the results"]
    _provenance = RESULTS.get("provenance") or {}
    if _provenance:
        _lines.append(
            f"\n- **Run**: `{_provenance.get('project')}` "
            f"({_provenance.get('reference_genome')}), "
            f"{_provenance.get('n_datasets', 0)} datasets, regenie "
            f"{_provenance.get('regenie_versions')}."
        )
    _variant_counts = RESULTS.get("filter_variant_counts")
    if isinstance(_variant_counts, pd.DataFrame) and not _variant_counts.empty:
        _n_filters = _variant_counts["filter"].nunique()
        _lines.append(
            f"\n- **Filters**: {_n_filters} distinct filter globals; "
            f"{len(_variant_counts)} dataset x filter x step rows plotted."
        )
    _effect = RESULTS.get("effect_mask_counts")
    if isinstance(_effect, pd.DataFrame) and not _effect.empty:
        _lines.append(
            f"\n- **Effect masks**: {_effect['mask'].nunique()} masks over "
            f"{_effect['label'].nunique()} datasets."
        )
    mo.md("\n".join(_lines))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
