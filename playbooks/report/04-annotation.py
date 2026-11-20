"""Annotation stage report: inspect an ``annotation`` matrix table.

Opens the primary output of the ``annotation`` stage — the ``annotation`` matrix
table written by ``AnnotationPipeline.merge`` — and reports what it holds: the
VEP, in-silico-predictor and ClinVar annotations, the gnomAD and cohort allele
frequencies, and the callset statistics, over both the produced samples (columns)
and the variants (rows).

Two validations get their own sections:

- **SOCS1** — the sites overlapping ``data/bed/socs1_sites.bed`` are pulled out
  and their VEP consequence and CADD score are shown, because this region is a
  known place where the reference lookup missed variants.
- **potentially protein-changing variants** — every row whose most severe VEP
  consequence can change the protein must be annotated, including the rows that
  are absent from gnomAD or have a gnomAD allele frequency of zero.

Run it where the Hail/Spark stack works (the integration container, per
``AGENTS.md`` and ADR-0002), or point it at another matrix with
``GMEB_ANNOTATION_MT`` / ``GMEB_SOCS1_BED``.
"""

import marimo

__generated_with = "0.25.2"
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
    # The staged matrix is known; override with ``GMEB_ANNOTATION_MT``.
    ANNOTATION_MT = os.environ.get(
        "GMEB_ANNOTATION_MT",
        REPO / "data" / "results" / "grch38" / "annotation.mt",
    )
    SOCS1_BED = os.environ.get(
        "GMEB_SOCS1_BED",
        REPO / "data" / "bed" / "socs1_sites.bed",
    )
    ANNOTATION = hl.read_matrix_table(str(ANNOTATION_MT))

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
    # Annotation — stage report

    Primary output: **`{ANNOTATION_MT}`**.

    SOCS1 validation interval: **`{SOCS1_BED}`**.

    This notebook reads the final annotated callset (VEP, in-silico scores,
    ClinVar, gnomAD and cohort frequencies, and the Sample QC columns) and
    reports what it holds. It aggregates with Hail; it never unions the callset
    (ADR-0015). Paths are `GMEB_ANNOTATION_MT` / `GMEB_SOCS1_BED`, else the
    staged artifact.
    """)
    return


@app.cell
def _():
    _provenance = reporting.annotation_provenance(ANNOTATION)
    RESULTS["provenance"] = _provenance
    _overview = reporting.annotation_overview(ANNOTATION)
    RESULTS["overview"] = _overview
    _presence = reporting.annotation_presence(ANNOTATION)
    RESULTS["presence"] = _presence
    _meta = _provenance["project_meta"]
    _vep = _provenance["vep_globals"]
    mo.vstack(
        [
            mo.md(
                f"**Project**: {_meta.get('project')} ({_meta.get('date')}, "
                f"caller {_meta.get('caller')} {_meta.get('caller_version')})\n\n"
                f"**Callset**: {_overview.get('n_variants', 0):,} variants x "
                f"{_provenance.get('n_samples', 0):,} samples — "
                f"**{_overview.get('n_pass', 0):,}** pass, "
                f"**{_overview.get('n_filtered', 0):,}** filtered "
                f"({_overview.get('n_uninformative', 0):,} `vqc_uninformative`).\n\n"
                f"**Class**: {_overview.get('n_snv', 0):,} SNP / "
                f"{_overview.get('n_indel', 0):,} INDEL; "
                f"Ti/Tv {_overview.get('ti_tv', float('nan')):.3f}.\n\n"
                f"**VEP**: local run `{_vep.get('vep_version')}` — "
                f"{_vep.get('defined', 0):,} rows from the reference lookup, "
                f"{_vep.get('missing', 0):,} VEP'd locally; "
                f"reference gnomAD VEP "
                f"`{_provenance['gnomad_vep_globals'].get('vep_version')}`."
            ),
            render(
                _presence,
                "Annotation presence",
                "The validation table: the fraction of rows carrying each "
                "annotation source. VEP and the consequence must be 100%.",
            ),
            reporting.annotation_presence_bar(_presence),
        ]
    )
    return


@app.cell
def _():
    _consequences = reporting.vep_consequence_counts(ANNOTATION)
    RESULTS["consequences"] = _consequences
    _note = (
        f"{len(_consequences)} distinct most-severe consequences; "
        f"{int(_consequences['n_variants'].head(6).sum()):,} rows in the six "
        "most common."
        if not _consequences.empty
        else "No VEP consequence is populated."
    )
    mo.vstack(
        [
            render(_consequences, "VEP most severe consequence", _note),
            reporting.vep_consequence_bar(_consequences),
        ]
    )
    return


@app.cell
def _():
    _insilico = reporting.in_silico_presence(ANNOTATION)
    RESULTS["insilico"] = _insilico
    mo.vstack(
        [
            render(
                _insilico,
                "In-silico predictor presence",
                "Scores are consequence-dependent (CADD for coding, REVEL / "
                "SIFT / PolyPhen / AlphaMissense for missense, SpliceAI / "
                "Pangolin for splice-adjacent), so sub-100% is expected.",
            ),
            reporting.in_silico_presence_bar(_insilico),
        ]
    )
    return


@app.cell
def _():
    _clinvar = reporting.clinvar_counts(ANNOTATION)
    RESULTS["clinvar"] = _clinvar
    _note = (
        f"{int(_clinvar['n_variants'].sum()):,} classified rows across "
        f"{len(_clinvar)} significance values."
        if not _clinvar.empty
        else "No row carries a ClinVar classification (none of the variants "
        "are in ClinVar)."
    )
    mo.vstack(
        [
            render(_clinvar, "ClinVar clinical significance", _note),
            reporting.clinvar_significance_bar(_clinvar),
        ]
    )
    return


@app.cell
def _():
    _freq = reporting.freq_by_stratum(ANNOTATION)
    RESULTS["freq_by_stratum"] = _freq
    mo.vstack(
        [
            render(
                _freq,
                "Cohort allele frequency by stratum",
                "The produced callset's own frequencies (`freq`), one row per "
                "`freq_meta` stratum: overall, by genetic ancestry, by sex and "
                "by cohort. `n_af_gt0` is the polymorphic fraction.",
            ),
            reporting.freq_stratum_bar(_freq),
        ]
    )
    return


@app.cell
def _():
    _gnomad = reporting.gnomad_freq_by_population(ANNOTATION)
    RESULTS["gnomad_freq_by_population"] = _gnomad
    mo.vstack(
        [
            render(
                _gnomad,
                "gnomAD reference frequency by population",
                "The reference `gnomad_freq` the join supplies, per genetic "
                "ancestry (`*_adj`). `n_af_gt0` is the fraction polymorphic in "
                "that reference population.",
            ),
            reporting.gnomad_freq_population_bar(_gnomad),
        ]
    )
    return


@app.cell
def _():
    if (
        "coverage_stats" not in ANNOTATION.row.dtype.fields
        or "coverage_stats_meta" not in ANNOTATION.globals.dtype.fields
    ):
        _output = mo.md(
            "### Coverage\n\n"
            "The matrix carries no coverage annotations (`coverage_stats` / "
            "`coverage_stats_meta`), so there is nothing to summarise. This "
            "happens when the artifact predates the coverage join "
            "(`annotate_coverage_stats`); regenerate it to populate coverage."
        )
    else:
        _by_group = reporting.coverage_by_group(ANNOTATION)
        RESULTS["coverage_by_group"] = _by_group
        _by_contig = reporting.coverage_by_contig(ANNOTATION)
        RESULTS["coverage_by_contig"] = _by_contig

        _pivot = _by_contig.pivot_table(
            index="contig", columns="stratum", values="mean_coverage"
        ).reset_index()
        _order = reporting.ordered_splits(_pivot["contig"]).argsort()
        _pivot = _pivot.loc[_order].reset_index(drop=True)

        _output = mo.vstack(
            [
                render(
                    _by_group,
                    "Coverage by group",
                    "Per-locus depth aggregated over the whole callset, one row "
                    "per `coverage_stats_meta` group: the sample-QC "
                    "pass/filtered split, plus the cohort strata when "
                    "`group_by_cohort` is configured. `mean_coverage` is the "
                    "mean per-site mean depth, `median_coverage` the median of "
                    "those site means, and `fraction_over_10` the mean fraction "
                    "of samples above 10x.",
                ),
                render(
                    _pivot,
                    "Mean coverage by contig",
                    f"`mean_coverage` per contig, one column per group. Sites: "
                    f"**{int(_by_contig['n_sites'].sum()):,}** across "
                    f"**{len(_pivot)}** contigs and **{_by_group.shape[0]}** "
                    "groups.",
                ),
                reporting.coverage_contig_bar(_by_contig),
            ]
        )
    _output
    return


@app.cell
def _():
    _intervals = reporting.bed_intervals(SOCS1_BED)
    _region = reporting.region_variants(ANNOTATION, _intervals)
    RESULTS["socs1_region"] = _region
    if _region.empty:
        _note = (
            "**No variant in the SOCS1 interval.** The region is not covered by "
            "the callset, so there is nothing for VEP to annotate here — this "
            "is a coverage gap, not a missing annotation."
        )
    else:
        _unannotated = _region["most_severe_consequence"].isna().sum()
        _note = (
            f"{len(_region)} variant(s) in the SOCS1 interval; "
            f"{_unannotated} without a consequence."
        )
    render(_region, "SOCS1 region", _note)
    return


@app.cell
def _():
    _check = reporting.protein_changing_check(ANNOTATION)
    RESULTS["protein_changing"] = _check
    _unannotated = reporting.unannotated_protein_changing(ANNOTATION)
    RESULTS["unannotated_protein_changing"] = _unannotated
    _ok = (
        _check.get("n_unannotated", 0) == 0
        and _check.get("n_af0_unannotated", 0) == 0
        and _check.get("n_absent_unannotated", 0) == 0
    )
    _status = "PASS" if _ok else "**FAIL**"
    mo.vstack(
        [
            mo.md(
                f"### Protein-changing validation — {_status}\n\n"
                f"- Potentially protein-changing variants: "
                f"**{_check.get('n_protein_changing', 0):,}**\n"
                f"- VEP-annotated: **{_check.get('n_annotated', 0):,}** "
                f"(unannotated: {_check.get('n_unannotated', 0):,})\n"
                f"- Absent from gnomAD: {_check.get('n_absent_gnomad', 0):,} "
                f"(unannotated: {_check.get('n_absent_unannotated', 0):,})\n"
                f"- gnomAD AF = 0: {_check.get('n_af0', 0):,} "
                f"(unannotated: {_check.get('n_af0_unannotated', 0):,})"
            ),
            render(
                _unannotated,
                "Unannotated protein-changing variants",
                "Empty is the expected result: every protein-changing row must "
                "carry a VEP consequence, even with gnomAD AF 0 or absent.",
            ),
        ]
    )
    return


@app.cell
def _():
    _lines = ["## Reading the results"]
    _overview = RESULTS.get("overview") or {}
    if _overview:
        _lines.append(
            f"\n- **Callset**: {_overview.get('n_variants', 0):,} variants; "
            f"{_overview.get('n_pass', 0):,} pass, "
            f"{_overview.get('n_filtered', 0):,} filtered."
        )
    _presence = RESULTS.get("presence")
    if isinstance(_presence, pd.DataFrame) and not _presence.empty:
        _missing = _presence.loc[_presence["fraction"] < 1.0, "annotation"]
        _lines.append(
            "\n- **Annotation presence**: "
            + (
                "every source present on every row."
                if _missing.empty
                else "incomplete for " + ", ".join(f"`{m}`" for m in _missing) + "."
            )
        )
    _check = RESULTS.get("protein_changing") or {}
    if _check:
        _lines.append(
            f"\n- **Protein-changing**: {_check.get('n_protein_changing', 0):,} "
            f"annotated; {_check.get('n_af0', 0):,} with gnomAD AF 0, "
            f"{_check.get('n_absent_gnomad', 0):,} absent from gnomAD; "
            f"unannotated {_check.get('n_unannotated', 0):,} / "
            f"{_check.get('n_af0_unannotated', 0):,} / "
            f"{_check.get('n_absent_unannotated', 0):,}."
        )
    _region = RESULTS.get("socs1_region")
    if isinstance(_region, pd.DataFrame):
        _lines.append(
            f"\n- **SOCS1**: {len(_region)} variant(s) in the interval."
        )
    _coverage = RESULTS.get("coverage_by_group")
    if isinstance(_coverage, pd.DataFrame) and not _coverage.empty:
        _overall = _coverage.loc[_coverage["stratum"] == "all", "mean_coverage"]
        if not _overall.empty and pd.notna(_overall.iloc[0]):
            _lines.append(
                f"\n- **Coverage**: overall mean depth "
                f"{float(_overall.iloc[0]):.1f}x across "
                f"{_coverage.shape[0]} group(s)."
            )
    mo.md("\n".join(_lines))
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
