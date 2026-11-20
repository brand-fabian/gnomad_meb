"""VDS-create stage report: inspect a built ``VariantDataset``.

Opens the primary output of the ``vds create`` stage — the dataset's
``.metadata.json`` plus its split VDSes — and reports what it built: the input
GVCFs, the split plan, the per-task status, and the per-split variant and
reference content.

Run it where the Hail/Spark stack works (the integration container, per
``AGENTS.md`` and ADR-0002). Point it at another dataset with
``GMEB_VDS_DIR`` / ``GMEB_VDS_NAME``.
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
    from gnomad_meb.hail import hail_init
    from gnomad_meb.lib.vds import VariantDataset

    # Reports read artifacts on a Hail-capable machine; always initialize at
    # module level so every cell below runs against the same Spark session.
    hail_init()

    # --- what to report -----------------------------------------------------
    VDS_DIR = Path(
        os.environ.get("GMEB_VDS_DIR", REPO / "data" / "results" / "grch38.vds")
    )
    VDS_NAME = os.environ.get("GMEB_VDS_NAME", "vds")
    # Root at the metadata file's directory, not ``params.base``: a staged
    # dataset's ``base`` still names the host that produced it.
    VDS = VariantDataset.open(VDS_DIR / f"{VDS_NAME}.metadata.json")

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
    # VDS create — stage report

    Primary output: **`{VDS_DIR / VDS_NAME}.metadata.json`** plus its
    `{VDS_NAME}.<split>.vds` split datasets.

    This notebook reads the dataset handle and reports what the build produced.
    It aggregates **per split** with Hail aggregators — it never unions the
    callset (ADR-0015). Path is `GMEB_VDS_DIR` / `GMEB_VDS_NAME`; root is the
    metadata file's directory.
    """)
    return


@app.cell
def _():
    _params = reporting.build_parameters(VDS)
    RESULTS["build_parameters"] = _params
    _frame = pd.DataFrame({"value": _params}).reset_index(names="parameter")
    render(
        _frame,
        "Build parameters",
        "The configuration the dataset was built with, plus the input GVCF and "
        "task counts derived from it.",
    )
    return


@app.cell
def _():
    _coverage = reporting.interval_coverage(VDS)
    RESULTS["interval_coverage"] = _coverage
    mo.vstack(
        [
            render(
                _coverage,
                "Split plan",
                "One row per split: how many intervals it spans, the bases it "
                "covers, and its first/last locus.",
            ),
            reporting.intervals_per_split(_coverage),
        ]
    )
    return


@app.cell
def _():
    _status = reporting.task_status(VDS)
    RESULTS["task_status"] = _status
    _blocks = [
        render(
            _status,
            "Task status",
            "Per-split resume state from the metadata. A finished build reads "
            "all `finished`; `error` rows carry the failure message below.",
        )
    ]
    _errors = _status[_status["status"] == "error"]
    if not _errors.empty:
        _lines = "\n".join(
            f"- `{row.split}`: {row.message}" for row in _errors.itertuples()
        )
        _blocks.append(mo.md(f"**Errors**\n\n{_lines}"))
    mo.vstack(_blocks)
    return


@app.cell
def _():
    _variants = reporting.split_variant_stats(VDS)
    RESULTS["split_variant_stats"] = _variants
    if not _variants.empty:
        _total = int(_variants["n_variant_rows"].sum())
        _note = f"{_total:,} variant rows across {len(_variants)} splits."
    else:
        _note = "No finished splits to read."
    mo.vstack(
        [
            render(_variants, "Variant content per split", _note),
            reporting.variant_class_by_split(_variants),
            reporting.split_sizes(_variants),
        ]
    )
    return


@app.cell
def _():
    _reference = reporting.split_reference_stats(VDS)
    RESULTS["split_reference_stats"] = _reference
    mo.vstack(
        [
            render(
                _reference,
                "Reference content per split",
                "Reference-block rows and their total / mean length: where the "
                "sparse representation is dense.",
            )
        ]
    )
    return


@app.cell
def _():
    _samples = reporting.samples_summary(VDS)
    RESULTS["samples"] = _samples
    mo.md(
        f"### Samples\n\n"
        f"**{_samples['n_samples']}** samples; "
        f"consistent across splits = **{_samples['consistent']}**."
    )
    return


@app.cell
def _():
    _lines = ["## Reading the results"]
    _params = RESULTS.get("build_parameters") or {}
    if _params:
        _lines.append(
            f"\n- **Build**: `{_params['reference_genome']}`, "
            f"`interval_size={_params['interval_size']:,}`, "
            f"`n_tasks={_params['n_tasks']}`, `n_gvcfs={_params['n_gvcfs']}`."
        )
    _status = RESULTS.get("task_status")
    if isinstance(_status, pd.DataFrame) and not _status.empty:
        _by_status = _status["status"].value_counts().to_dict()
        _lines.append(f"\n- **Status**: {_by_status}.")
    _variants = RESULTS.get("split_variant_stats")
    if isinstance(_variants, pd.DataFrame) and not _variants.empty:
        _total = int(_variants["n_variant_rows"].sum())
        _empty = _variants[_variants["n_variant_rows"] == 0]["split"].tolist()
        _lines.append(
            f"\n- **Variants**: {_total:,} rows across "
            f"{len(_variants)} splits"
            + (f"; empty splits = {_empty}" if _empty else "")
            + "."
        )
    _samples = RESULTS.get("samples")
    if isinstance(_samples, dict):
        _lines.append(
            f"\n- **Samples**: {_samples['n_samples']} "
            f"(consistent = {_samples['consistent']})."
        )
    mo.md("\n".join(_lines))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
