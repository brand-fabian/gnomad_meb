"""Sample QC stage report: inspect a ``sample_qc`` hail table.

Opens the primary output of the ``sample-qc`` stage — the ``sample_qc`` hail
table written by ``compute_hard_filters`` — together with the three relatedness
artifacts (the ``king`` matrix and the ``relatedness`` / ``relatedness2``
pc-relate tables), and reports what it holds: the hard filters, coverage, sex
inference, contamination, ancestry, and a comparison of the relatedness
methods.

Run it where the Hail/Spark stack works (the integration container, per
``AGENTS.md`` and ADR-0002). Run it with the same ``GMEB_CHECKPOINT`` /
``GMEB_PROJECT`` as the producing run, or point it at the tables directly with
``GMEB_SAMPLE_QC_HT`` / ``GMEB_RELATEDNESS_HT`` / ``GMEB_RELATEDNESS2_HT`` /
``GMEB_KING_MT``.
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
    from gnomad_meb.config import get_config
    from gnomad_meb.hail import hail as hl
    from gnomad_meb.hail import hail_init
    from gnomad_meb.lib.checkpoint import CHECKPOINT_TYPE, Artifact

    # Reports read artifacts on a Hail-capable machine; always initialize at
    # module level so every cell below runs against the same Spark session.
    hail_init()

    # --- what to report -----------------------------------------------------
    # A hail table is not self-describing, so resolve through the run's
    # settings: env-var override first, else the stage's declared artifacts.
    CONFIG = get_config()
    SAMPLE_QC_HT = os.environ.get(
        "GMEB_SAMPLE_QC_HT",
        REPO / "data" / "results" / "grch38" / "sample_qc.ht",
    )
    RELATEDNESS_HT = os.environ.get(
        "GMEB_RELATEDNESS_HT",
        REPO / "data" / "results" / "grch38" / "intermediate" / "sample_qc" / "relatedness.ht",
    )
    RELATEDNESS2_HT = os.environ.get(
        "GMEB_RELATEDNESS2_HT",
        REPO
        / "data"
        / "results"
        / "grch38"
        / "intermediate"
        / "sample_qc"
        / "relatedness2.ht",
    )
    KING_MT = os.environ.get(
        "GMEB_KING_MT",
        REPO / "data" / "results" / "grch38" / "intermediate" / "sample_qc" / "king.mt",
    )
    SAMPLE_QC = hl.read_table(str(SAMPLE_QC_HT))
    RELATEDNESS = hl.read_table(str(RELATEDNESS_HT))
    RELATEDNESS2 = hl.read_table(str(RELATEDNESS2_HT))
    KING = hl.read_matrix_table(str(KING_MT))
    SAMPLES = reporting.sample_ids(SAMPLE_QC)

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
    # Sample QC — stage report

    Primary output: **`{SAMPLE_QC_HT}`**.

    Relatedness artifacts: **`{KING_MT}`** (KING ``phi``), **`{RELATEDNESS_HT}`**
    (pc-relate projected onto the gnomAD loadings), **`{RELATEDNESS2_HT}`**
    (pc-relate on the KING-pruned, self-fit PCA).

    This notebook reads the ``sample_qc`` table (coverage, sex, contamination,
    ancestry and the hard filters) plus the three relatedness outputs, and
    compares them. It aggregates with Hail; it never unions the callset
    (ADR-0015). Paths are `GMEB_SAMPLE_QC_HT` / `GMEB_RELATEDNESS_HT` /
    `GMEB_RELATEDNESS2_HT` / `GMEB_KING_MT`, else resolved from the run's
    settings.
    """)
    return


@app.cell
def _():
    _config = reporting.config_summary(SAMPLE_QC)
    RESULTS["config"] = _config
    _overview = reporting.sample_overview(SAMPLE_QC)
    RESULTS["overview"] = _overview
    _cutoffs = pd.DataFrame(
        {"value": _config["sample_qc_filter_cutoffs"]}
    ).reset_index(names="cutoff")
    mo.vstack(
        [
            render(
                _cutoffs,
                "Hard-filter cutoffs",
                "The thresholds the stage applied; the plot builders below draw "
                "them as reference lines.",
            ),
            mo.md(
                f"**Cohort**: {_overview.get('n_samples', 0)} samples — "
                f"**{_overview.get('n_pass', 0)}** pass, "
                f"**{_overview.get('n_fail', 0)}** fail "
                f"({_overview.get('n_related_dropped', 0)} for relatedness).\n\n"
                f"Sex karyotypes: {_overview.get('sex', {})}\n\n"
                f"Ancestry: {_overview.get('ancestry', {})}"
            ),
        ]
    )
    return


@app.cell
def _():
    _counts = reporting.hard_filter_counts(SAMPLE_QC)
    RESULTS["hard_filter_counts"] = _counts
    if not _counts.empty:
        _total = int(_counts["n_samples"].sum())
        _note = f"{_total:,} filter failures across {len(_counts)} filters."
    else:
        _note = "No sample carries a hard filter."
    mo.vstack(
        [
            render(_counts, "Hard filters", _note),
            reporting.hard_filter_counts_bar(_counts),
        ]
    )
    return


@app.cell
def _():
    _membership = reporting.hard_filter_membership(SAMPLE_QC)
    RESULTS["hard_filter_membership"] = _membership
    reporting.hard_filter_upset(_membership)
    return


@app.cell
def _():
    _metrics = reporting.coverage_metrics(SAMPLE_QC)
    RESULTS["coverage_metrics"] = _metrics
    _histograms = reporting.coverage_histograms(SAMPLE_QC)
    RESULTS["coverage_histograms"] = _histograms
    _cutoffs = RESULTS["config"]["sample_qc_filter_cutoffs"]
    mo.vstack(
        [
            render(
                _metrics,
                "Coverage metrics",
                "One row per sample: autosomal depth, called bases over DP 1 /"
                " DP 20, call rates and the singleton / het / hom-var counts.",
            ),
            render(
                _histograms,
                "Coverage distributions",
                "Depth and adjusted call rate aggregated in Hail, so a large "
                "cohort still returns a handful of rows.",
            ),
            reporting.coverage_dp_histogram(
                _metrics, _cutoffs.get("autosomal_min_cov")
            ),
            reporting.bases_over_dp_scatter(_metrics, _cutoffs),
            reporting.singleton_het_hom_scatter(_metrics, _cutoffs),
            reporting.callrate_histogram(
                _metrics, _cutoffs.get("min_qc_mt_adj_callrate")
            ),
        ]
    )
    return


@app.cell
def _():
    _sex = reporting.sex_inference(SAMPLE_QC)
    RESULTS["sex_inference"] = _sex
    _ploidy_params = RESULTS["config"]["ploidy_params"]
    mo.vstack(
        [
            render(
                _sex,
                "Sex inference",
                "The ploidy struct flattened: chrX / chrY ploidy and depth, "
                "the inbreeding f_stat and the inferred karyotypes.",
            ),
            reporting.sex_ploidy_scatter(_sex),
            reporting.f_stat_histogram(_sex, _ploidy_params.get("f_stat_cutoff")),
        ]
    )
    return


@app.cell
def _():
    _contamination = reporting.contamination(SAMPLE_QC)
    RESULTS["contamination"] = _contamination
    _cutoffs = RESULTS["config"]["sample_qc_filter_cutoffs"]
    mo.vstack(
        [
            render(
                _contamination,
                "Contamination",
                "The per-sample charr estimate (and the assigned ancestry).",
            ),
            reporting.contamination_histogram(
                _contamination, _cutoffs.get("max_contamination_estimate")
            ),
        ]
    )
    return


@app.cell
def _():
    # The production tables are written with no ``min_kinship`` pruning
    # (``None``), so read the complete pair set — no display cap — for the
    # matrices below.
    _rel = reporting.relatedness_pairs(RELATEDNESS, SAMPLE_QC, cap=None)
    _rel2 = reporting.relatedness_pairs(RELATEDNESS2, SAMPLE_QC, cap=None)
    _king = reporting.king_pairs(KING, SAMPLE_QC, cap=None)
    RESULTS["relatedness_pairs"] = _rel
    RESULTS["relatedness2_pairs"] = _rel2
    RESULTS["king_pairs"] = _king
    _min_kinship = (
        CONFIG.sample_qc.min_kinship if CONFIG.sample_qc is not None else None
    )
    _prune = (
        "no `min_kinship` pruning"
        if _min_kinship is None
        else f"`min_kinship = {_min_kinship}`"
    )
    _summary = (
        f"{len(_rel):,} pairs reported ({_prune}); median kinship "
        f"{_rel['kinship'].median():.3f}."
        if not _rel.empty
        else f"No pair survived ({_prune})."
    )
    mo.vstack(
        [
            render(
                _rel,
                "Related pairs — relatedness (projected pc-relate)",
                _summary,
            ),
            reporting.relatedness_ibd_scatter(_rel),
        ]
    )
    return


@app.cell
def _():
    _comparison = reporting.relatedness_comparison(
        {
            "king": RESULTS["king_pairs"],
            "relatedness": RESULTS["relatedness_pairs"],
            "relatedness2": RESULTS["relatedness2_pairs"],
        }
    )
    RESULTS["relatedness_comparison"] = _comparison
    _concordance = reporting.relatedness_concordance(_comparison)
    RESULTS["relatedness_concordance"] = _concordance
    _counts = " · ".join(
        f"{label}: {len(frame):,}"
        for label, frame in {
            "KING": RESULTS["king_pairs"],
            "relatedness": RESULTS["relatedness_pairs"],
            "relatedness2": RESULTS["relatedness2_pairs"],
        }.items()
    )
    mo.vstack(
        [
            render(
                _comparison,
                "Relatedness methods — comparison",
                "One row per sample pair any method reports (outer join on "
                f"`i` / `j`); pairs per method — {_counts}.",
            ),
            render(
                _concordance,
                "Inter-method concordance",
                "Pearson / Spearman kinship correlation over the pairs both "
                "methods report (complete cases).",
            ),
        ]
    )
    return


@app.cell
def _():
    _threshold = reporting.RELATEDNESS_DISPLAY_THRESHOLD
    _methods = {
        "KING (phi)": RESULTS["king_pairs"],
        "relatedness (projected pc-relate)": RESULTS["relatedness_pairs"],
        "relatedness2 (KING-pruned pc-relate)": RESULTS["relatedness2_pairs"],
    }
    mo.vstack(
        [
            mo.md(
                "**Per-method kinship heatmaps** — the complete sample x sample "
                "matrix; the colour ramp is anchored at the display threshold "
                f"`{_threshold}` (no pair is dropped)."
            ),
            *[
                reporting.relatedness_heatmap(
                    frame, _threshold, samples=SAMPLES, title=label
                )
                for label, frame in _methods.items()
            ],
        ]
    )
    return


@app.cell
def _():
    _comparison = RESULTS["relatedness_comparison"]
    mo.vstack(
        [
            mo.md(
                "**Method agreement** — one method's kinship against another's; "
                "the dashed line is `y = x`."
            ),
            reporting.relatedness_method_scatter(
                _comparison,
                "kinship_relatedness",
                "kinship_king",
                "Projected pc-relate vs KING",
            ),
            reporting.relatedness_method_scatter(
                _comparison,
                "kinship_relatedness2",
                "kinship_king",
                "KING-pruned pc-relate vs KING",
            ),
            reporting.relatedness_method_scatter(
                _comparison,
                "kinship_relatedness",
                "kinship_relatedness2",
                "Projected vs KING-pruned pc-relate",
            ),
        ]
    )
    return


@app.cell
def _():
    _ancestry = reporting.ancestry_scores(SAMPLE_QC)
    RESULTS["ancestry_scores"] = _ancestry
    _pca = reporting.ancestry_pca_scatter(_ancestry)
    mo.vstack(
        [
            render(
                _ancestry,
                "Ancestry",
                "Per-sample population, inferred probabilities and PCA scores.",
            ),
            _pca,
            reporting.ancestry_probability_bar(_ancestry),
        ]
    )
    return


@app.cell
def _():
    _lines = ["## Reading the results"]
    _overview = RESULTS.get("overview") or {}
    if _overview:
        _lines.append(
            f"\n- **Cohort**: {_overview.get('n_samples', 0)} samples; "
            f"{_overview.get('n_pass', 0)} pass, "
            f"{_overview.get('n_fail', 0)} fail "
            f"({_overview.get('n_related_dropped', 0)} relatedness drops)."
        )
    _counts = RESULTS.get("hard_filter_counts")
    if isinstance(_counts, pd.DataFrame) and not _counts.empty:
        _top = ", ".join(
            f"`{row.filter}`={int(row.n_samples)}"
            for row in _counts.head(3).itertuples()
        )
        _lines.append(f"\n- **Hard filters**: top failures {_top}.")
    _comparison = RESULTS.get("relatedness_comparison")
    if isinstance(_comparison, pd.DataFrame) and not _comparison.empty:
        _lines.append(
            f"\n- **Relatedness**: {len(_comparison):,} pairs compared across "
            "KING, projected pc-relate and KING-pruned pc-relate."
        )
    _ancestry = RESULTS.get("ancestry_scores")
    if isinstance(_ancestry, pd.DataFrame) and "gen_anc" in _ancestry:
        _lines.append(
            f"\n- **Ancestry**: {_ancestry['gen_anc'].value_counts().to_dict()}."
        )
    mo.md("\n".join(_lines))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
