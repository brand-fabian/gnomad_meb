"""Variant QC stage report: inspect a ``variant_qc`` hail table.

Opens the primary output of the ``variant-qc`` stage — the ``variant_qc`` hail
table written by ``VariantQcImportPipeline.merge`` — and reports what it holds:
how many variants VQSR removed, the applied cutoffs, the VQSLOD score
distribution, the variant classes, the culprits and the per-contig filter rate.

The VQSR provenance (target sensitivities, applied cutoffs, recalibration params,
GATK version) travels as the table's **globals** (ticket 08), so the report reads
it with ``index_globals()`` and needs no side-loaded tranches.

Run it where the Hail/Spark stack works (the integration container, per
``AGENTS.md`` and ADR-0002), or point it at another table with
``GMEB_VARIANT_QC_HT``.
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
        raise RuntimeError("could not locate the repository root (no src/gnomad_meb)")

    REPO = _find_repo_root()
    sys.path.insert(0, str(REPO / "src"))

    from gnomad_meb import reporting
    from gnomad_meb.hail import hail as hl
    from gnomad_meb.hail import hail_init

    # Reports read artifacts on a Hail-capable machine; always initialize at
    # module level so every cell below runs against the same Spark session.
    hail_init()

    # --- what to report -----------------------------------------------------
    # The staged table is known; override with ``GMEB_VARIANT_QC_HT``.
    VARIANT_QC_HT = os.environ.get(
        "GMEB_VARIANT_QC_HT",
        REPO / "data" / "results" / "grch38" / "variant_qc.ht",
    )
    VARIANT_QC = hl.read_table(str(VARIANT_QC_HT))

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
    # Variant QC — stage report

    Primary output: **`{VARIANT_QC_HT}`**.

    This notebook reads the merged ``variant_qc`` table (keyed by locus/alleles)
    and reports how VQSR filtered it. The VQSR provenance — target
    sensitivities, applied VQSLOD cutoffs, recalibration params and the GATK
    version — is read from the table's globals. It aggregates with Hail; it never
    unions the callset (ADR-0015). Path is `GMEB_VARIANT_QC_HT`, else the staged
    table.
    """)
    return


@app.cell
def _():
    _provenance = reporting.vqsr_provenance(VARIANT_QC)
    RESULTS["provenance"] = _provenance
    _overview = reporting.variant_overview(VARIANT_QC)
    RESULTS["overview"] = _overview
    mo.vstack(
        [
            mo.md(
                f"**Variants**: {_overview.get('n_variants', 0):,} — "
                f"**{_overview.get('n_pass', 0):,}** pass, "
                f"**{_overview.get('n_filtered', 0):,}** filtered "
                f"({_overview.get('n_lowqual', 0):,} low-qual).\n\n"
                f"Class: {_overview.get('n_snv', 0):,} SNP / "
                f"{_overview.get('n_indel', 0):,} INDEL; "
                f"Ti/Tv {_overview.get('ti_tv', float('nan')):.3f}.\n\n"
                f"Split rows: {_overview.get('n_split_rows', 0):,}; "
                f"training sites {_overview.get('n_positive_train', 0):,} positive / "
                f"{_overview.get('n_negative_train', 0):,} negative."
            ),
            render(
                pd.DataFrame({"value": _provenance["vqsr_params"]}).reset_index(
                    names="parameter"
                ),
                "VQSR parameters",
                "The recalibration settings the stage recorded as globals.",
            ),
            render(
                pd.DataFrame({"value": _provenance["gatk_version"]}).reset_index(
                    names="key"
                ),
                "GATK provenance",
                "The GATK version that produced the VQSR scores.",
            ),
        ]
    )
    return


@app.cell
def _():
    _cutoffs = reporting.vqsr_cutoffs(VARIANT_QC)
    RESULTS["vqsr_cutoffs"] = _cutoffs
    render(
        _cutoffs,
        "VQSR cutoffs",
        "The applied truth-sensitivity target and the resulting `minVQSLod` "
        "per mode. The histogram below draws the cutoffs as reference lines.",
    )
    return


@app.cell
def _():
    _counts = reporting.filter_counts(VARIANT_QC)
    RESULTS["filter_counts"] = _counts
    _note = (
        f"{int(_counts['n_variants'].sum()):,} filter assignments across "
        f"{len(_counts)} FILTER values."
        if not _counts.empty
        else "No variant carries a non-empty FILTER."
    )
    mo.vstack(
        [
            render(_counts, "Filtered variants", _note),
            reporting.filter_counts_bar(_counts),
        ]
    )
    return


@app.cell
def _():
    _summary = reporting.vqslod_summary(VARIANT_QC)
    RESULTS["vqslod_summary"] = _summary
    _histogram = reporting.vqslod_bins(VARIANT_QC)
    RESULTS["vqslod_histogram"] = _histogram
    _cutoffs = RESULTS["vqsr_cutoffs"]
    _thresholds = dict(zip(_cutoffs["mode"], _cutoffs["vqslod_cutoff"], strict=True))
    mo.vstack(
        [
            render(
                _summary,
                "VQSLOD summary",
                "The `AS_VQSLOD` distribution per mode (SNP = `allele_type`"
                " `snv`, INDEL = everything else, matching GATK's two models).",
            ),
            reporting.vqslod_histogram(_histogram, _thresholds),
        ]
    )
    return


@app.cell
def _():
    _classes = reporting.variant_class_counts(VARIANT_QC)
    RESULTS["variant_class_counts"] = _classes
    mo.vstack(
        [
            render(
                _classes,
                "Variant class",
                "Pass and filtered counts per (`allele_type`, `variant_type`) "
                "with the filter rate.",
            ),
            reporting.variant_class_bar(_classes),
        ]
    )
    return


@app.cell
def _():
    _culprits = reporting.culprit_counts(VARIANT_QC)
    RESULTS["culprit_counts"] = _culprits
    _note = (
        f"{len(_culprits)} distinct culprits among the filtered variants."
        if not _culprits.empty
        else "No filtered variant carries an `AS_culprit`."
    )
    mo.vstack(
        [
            render(_culprits, "VQSR culprit", _note),
            reporting.culprit_bar(_culprits),
        ]
    )
    return


@app.cell
def _():
    _contigs = reporting.filter_by_contig(VARIANT_QC)
    RESULTS["filter_by_contig"] = _contigs
    mo.vstack(
        [
            render(
                _contigs,
                "By contig",
                "Variant count, filtered count and filter rate per contig.",
            ),
            reporting.filter_rate_by_contig(_contigs),
        ]
    )
    return


@app.cell
def _():
    _top = reporting.top_filtered_sites(VARIANT_QC)
    RESULTS["top_filtered_sites"] = _top
    render(
        _top,
        "Top filtered sites",
        "The highest-`AS_VQSLOD` rows that still failed — bounded examples, not "
        "the full filtered set.",
    )
    return


@app.cell
def _():
    _lines = ["## Reading the results"]
    _overview = RESULTS.get("overview") or {}
    if _overview:
        _lines.append(
            f"\n- **Variants**: {_overview.get('n_variants', 0):,}; "
            f"{_overview.get('n_pass', 0):,} pass, "
            f"{_overview.get('n_filtered', 0):,} filtered, "
            f"{_overview.get('n_lowqual', 0):,} low-qual."
        )
        _lines.append(
            f"\n- **Class**: {_overview.get('n_snv', 0):,} SNP / "
            f"{_overview.get('n_indel', 0):,} INDEL; "
            f"Ti/Tv {_overview.get('ti_tv', float('nan')):.3f}."
        )
    _cutoffs = RESULTS.get("vqsr_cutoffs")
    if isinstance(_cutoffs, pd.DataFrame) and not _cutoffs.empty:
        _applied = ", ".join(
            f"`{row.mode}` target {row.target_truth_sensitivity}"
            f" @ VQSLOD {row.vqslod_cutoff}"
            for row in _cutoffs.itertuples()
        )
        _lines.append(f"\n- **Cutoffs**: {_applied}.")
    _culprits = RESULTS.get("culprit_counts")
    if isinstance(_culprits, pd.DataFrame) and not _culprits.empty:
        _top = ", ".join(
            f"`{row.culprit}`={int(row.n_variants)}"
            for row in _culprits.head(3).itertuples()
        )
        _lines.append(f"\n- **Top culprits**: {_top}.")
    mo.md("\n".join(_lines))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
