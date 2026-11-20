"""The Sample QC stage: collectors over the ``sample_qc`` hail table (and the
relatedness sibling) and the ``plotnine``/``UpSet`` builders over their frames.

The stage's declared primary output is the ``sample_qc`` table written by
``compute_hard_filters``; it carries ancestry, sex, contamination, coverage and
the hard filters, so one ``hl.read_table`` covers everything except relatedness,
which lives in three sibling artifacts under ``intermediate/sample_qc``: the
``king`` matrix (``hl.king`` between-family ``phi``), and the two pc-relate
tables ``relatedness`` (projected onto the gnomAD loadings) and ``relatedness2``
(KING-pruned, PCA fit on the unrelated remainder). The report compares all
three.

Each collector is stateless, returns a tidy ``pandas.DataFrame`` (or a plain
dict for scalars) with stable columns, and does not write to disk. No collector
unions the callset: Sample QC's table is already split-aggregated by the
pipeline (ADR-0015), so a report is a read.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as pn
from upsetplot import UpSet, from_memberships

from gnomad_meb.hail import hail as hl
from gnomad_meb.reporting.theme import THEME

__all__ = [
    "RELATEDNESS_DISPLAY_THRESHOLD",
    "ancestry_pca_scatter",
    "ancestry_probability_bar",
    "ancestry_scores",
    "bases_over_dp_scatter",
    "callrate_histogram",
    "config_summary",
    "contamination",
    "contamination_histogram",
    "coverage_dp_histogram",
    "coverage_histograms",
    "coverage_metrics",
    "f_stat_histogram",
    "hard_filter_counts",
    "hard_filter_counts_bar",
    "hard_filter_membership",
    "hard_filter_upset",
    "king_pairs",
    "relatedness_comparison",
    "relatedness_concordance",
    "relatedness_heatmap",
    "relatedness_ibd_scatter",
    "relatedness_method_scatter",
    "relatedness_pairs",
    "sample_ids",
    "sample_overview",
    "sex_inference",
    "sex_ploidy_scatter",
    "singleton_het_hom_scatter",
]

#: The ancestry populations the classification model assigns.
ANCESTRY_POPS = ("afr", "ami", "amr", "eas", "mid", "nfe", "sas", "other")

#: The ploidy/sex struct fields, in a stable order for the flattened frame.
PLOIDY_FIELDS = (
    "chrX_ploidy",
    "chrY_ploidy",
    "chrX_mean_dp",
    "chrY_mean_dp",
    "autosomal_mean_dp",
    "f_stat",
    "n_called",
    "expected_homs",
    "observed_homs",
    "X_karyotype",
    "Y_karyotype",
    "sex_karyotype",
)

#: Hard filters ``compute_hard_filters`` can attach to a sample.
HARD_FILTERS = (
    "high_n_singleton",
    "high_r_het_hom_var",
    "low_bases_dp_over_1",
    "low_bases_dp_over_20",
    "high_contamination",
    "low_adj_callrate",
    "low_coverage",
    "relatedness",
)

#: Index into ``bases_over_dp_threshold`` for the ``dp > 1`` / ``dp > 20`` bins
#: (the default ``dp_bins`` are ``[0, 1, 10, 20, 30]``); the same bins the hard
#: filters use.
_DP_OVER_1_INDEX = 1
_DP_OVER_20_INDEX = 3

#: Columns of the flattened coverage frame, in order.
_COVERAGE_COLUMNS = (
    "autosomal_mean_dp",
    "bases_over_dp_1",
    "bases_over_dp_20",
    "callrate",
    "callrate_adj",
    "n_singleton",
    "r_het_hom_var",
    "n_het",
    "n_hom_var",
)

#: Columns of the tidy relatedness-pair frame, in order.
_RELATEDNESS_COLUMNS = (
    "i",
    "j",
    "kinship",
    "ibd0",
    "ibd1",
    "ibd2",
    "gen_anc_i",
    "gen_anc_j",
)

#: Lower bound for the relatedness heatmap colour ramp. The production tables
#: are written with no ``min_kinship`` pruning (``None``), so the matrix is
#: dominated by undifferentiated background pairs and only a handful carry the
#: signal; anchoring the fill scale at the conventional KING third-degree
#: kinship cutoff (``0.0884``) keeps related pairs legible against that
#: background. This is a display-only constant — no pair is dropped.
RELATEDNESS_DISPLAY_THRESHOLD = 0.0884


# --- pure reshape helpers ---------------------------------------------------
# Each collector is a thin hail read; the shaping below is ordinary Python so it
# is unit-testable without a Spark session (the ``vds.py`` precedent).


def _as_plain(value: Any) -> Any:
    """Return a plain-Python copy of a hail struct, dict or SimpleNamespace."""
    if value is None:
        return None
    if isinstance(value, dict):
        return {key: _as_plain(item) for key, item in value.items()}
    fields = getattr(value, "_fields", None)
    if isinstance(fields, dict):  # ``hl.utils.Struct``
        return {key: _as_plain(item) for key, item in fields.items()}
    if isinstance(value, list | tuple):
        return [_as_plain(item) for item in value]
    if hasattr(value, "__dict__"):
        return {key: _as_plain(item) for key, item in vars(value).items()}
    return value


def _array_get(index: int):
    """A mapper returning ``value[index]`` or ``nan`` when out of range."""

    def get(value: Any) -> Any:
        if value is None:
            return np.nan
        try:
            return value[index]
        except (IndexError, TypeError):
            return np.nan

    return get


def _sample_frame(ht: hl.Table, **fields: Any) -> pd.DataFrame:
    """A per-sample frame indexed by the sample id ``s``.

    ``Table.to_pandas`` leaves the key as a column, so it is promoted to the
    index here (and left alone if a future Hail promotes it already).
    """
    frame = ht.select(**fields).to_pandas()
    if "s" in frame.columns:
        frame = frame.set_index("s")
    return frame


def _filter_counts_frame(
    counts: dict[str, int], n_samples: int
) -> pd.DataFrame:
    """A ``filter``/``n_samples``/``fraction`` frame from a filter counter."""
    if not counts:
        return pd.DataFrame(columns=["filter", "n_samples", "fraction"])
    frame = pd.DataFrame(
        {"filter": list(counts.keys()), "n_samples": list(counts.values())}
    )
    frame = frame.sort_values("n_samples", ascending=False, ignore_index=True)
    frame["fraction"] = (
        frame["n_samples"] / n_samples if n_samples else 0.0
    )
    return frame[["filter", "n_samples", "fraction"]]


def _as_filter_set(value: Any) -> set[str]:
    """A hail ``set<str>`` cell (or NaN) as a plain Python set."""
    if value is None:
        return set()
    if isinstance(value, float) and math.isnan(value):
        return set()
    return set(value)


def _filter_member(name: str):
    def has(value: Any) -> bool:
        return name in _as_filter_set(value)

    return has


def _membership_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """One bool column per filter, one row per sample; all-False columns dropped.

    ``rows`` is indexed by sample and carries a ``hard_filters`` set column (the
    shape ``Table.to_pandas`` returns for a ``set<str>``).
    """
    values = rows["hard_filters"] if "hard_filters" in rows.columns else rows.iloc[:, 0]
    present = sorted({f for value in values for f in _as_filter_set(value)})
    frame = pd.DataFrame(
        {name: values.map(_filter_member(name)) for name in present},
        index=rows.index,
    )
    if frame.shape[0]:
        frame = frame.loc[:, frame.any(axis=0)]
    return frame


def _coverage_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """Flatten ``bases_over_dp_threshold`` into named, over-DP columns."""
    frame = rows.copy()
    if "bases_over_dp_threshold" in frame.columns:
        values = frame.pop("bases_over_dp_threshold")
        frame["bases_over_dp_1"] = values.map(_array_get(_DP_OVER_1_INDEX))
        frame["bases_over_dp_20"] = values.map(_array_get(_DP_OVER_20_INDEX))
    return frame.reindex(
        columns=[column for column in _COVERAGE_COLUMNS if column in frame]
    )


def _histogram_frame(hist: Any, metric: str) -> pd.DataFrame:
    """A ``metric``/``bin_center``/``count`` frame from a ``hl.agg.hist`` result."""
    edges = list(hist["bin_edges"] if isinstance(hist, dict) else hist.bin_edges)
    freqs = list(hist["bin_freq"] if isinstance(hist, dict) else hist.bin_freq)
    centers = [
        (edges[index] + edges[index + 1]) / 2 for index in range(len(freqs))
    ]
    return pd.DataFrame({"metric": metric, "bin_center": centers, "count": freqs})


def _ancestry_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """Expand the ``ancestry.scores`` array into ``pc1``…``pcN`` columns."""
    frame = pd.DataFrame(index=rows.index)
    if "gen_anc" in rows.columns:
        frame["gen_anc"] = rows["gen_anc"]
    for column in rows.columns:
        if column.startswith("prob_"):
            frame[column] = rows[column]
    if "pca_scores" in rows.columns:
        scores = rows["pca_scores"]
        widths = [len(value) for value in scores if value is not None]
        n_pcs = max(widths) if widths else 0
        for index in range(n_pcs):
            frame[f"pc{index + 1}"] = scores.map(_array_get(index))
    return frame


def _relatedness_frame(
    pairs: pd.DataFrame, pop: pd.Series, cap: int = 50_000
) -> pd.DataFrame:
    """Join each ``(i, j)`` pair to its two ancestry populations, then cap rows.

    The pairwise table survives ``min_kinship`` pruning but a dense cohort can
    still be large, so the scatter is bounded to the ``cap`` most-related pairs.
    """
    frame = pairs.copy()
    if "kin" in frame.columns:
        frame = frame.rename(columns={"kin": "kinship"})
    if "i" not in frame.columns:
        frame = frame.reset_index()
    frame["gen_anc_i"] = frame["i"].map(pop)
    frame["gen_anc_j"] = frame["j"].map(pop)
    if cap and len(frame) > cap:
        frame = (
            frame.nlargest(cap, "kinship")
            if "kinship" in frame.columns
            else frame.head(cap)
        )
    return frame.reset_index(drop=True).reindex(
        columns=[column for column in _RELATEDNESS_COLUMNS if column in frame]
    )


def _kinship_matrix(
    pairs: pd.DataFrame, samples: pd.Index | list[str] | None = None
) -> pd.DataFrame:
    """A pairwise frame as a symmetric ``i``/``j``/``kinship`` long matrix.

    ``pc_relate`` stores each unordered pair once, so every value is mirrored
    to both orientations. When ``samples`` is given the matrix is reindexed to
    that full axis, so the result is the complete ``len(samples)``-square grid
    with a blank tile for any pair a method did not report; otherwise the axis
    is the sorted union of the samples seen. The diagonal is undefined (a
    self-pair is not a relatedness estimate): with no ``samples`` it is dropped,
    while a complete matrix keeps it blank so the grid stays square. Empty input
    yields an empty frame.
    """
    if pairs.empty or not {"i", "j", "kinship"} <= set(pairs.columns):
        return pd.DataFrame(columns=["i", "j", "kinship"])
    matrix = pairs.pivot_table(
        index="i", columns="j", values="kinship", aggfunc="first"
    )
    matrix = matrix.combine_first(matrix.T)
    if samples is None:
        axis = sorted(matrix.index)
        drop_undefined = True  # the diagonal carries no estimate
    else:
        axis = list(samples)
        drop_undefined = False  # complete grid keeps blank tiles explicitly
    matrix = matrix.reindex(index=axis, columns=axis)
    long = matrix.stack(dropna=drop_undefined).rename("kinship").reset_index()
    long.columns = ["i", "j", "kinship"]
    return long


def _config_summary(globals_plain: dict[str, Any], n_samples: int) -> dict[str, Any]:
    """The plain-globals view of a Sample QC table's configuration."""
    return {
        "n_samples": n_samples,
        "sample_qc_filter_cutoffs": globals_plain.get("sample_qc_filter_cutoffs")
        or {},
        "charr_params": globals_plain.get("charr_params") or {},
        "ploidy_params": globals_plain.get("ploidy_params") or {},
        "ancestry_params": globals_plain.get("ancestry_params") or {},
    }


# --- collectors -------------------------------------------------------------


def config_summary(ht: hl.Table) -> dict[str, Any]:
    """The table's globals structs plus ``n_samples``.

    The cutoffs feed the plot builders' threshold lines; the params describe the
    contamination, ploidy and ancestry computations.
    """
    globals_plain = _as_plain(hl.eval(ht.index_globals()))
    return _config_summary(globals_plain, ht.count())


def sample_overview(ht: hl.Table) -> dict[str, Any]:
    """Scalar pass/fail tallies and sex/ancestry breakdowns."""
    stats = ht.aggregate(
        hl.struct(
            n_samples=hl.agg.count(),
            n_pass=hl.agg.count_where(hl.len(ht.hard_filters) == 0),
            n_fail=hl.agg.count_where(hl.len(ht.hard_filters) > 0),
            n_related_dropped=hl.agg.count_where(
                ht.hard_filters.contains("relatedness")
            ),
            sex=hl.agg.counter(ht.ploidy.sex_karyotype),
            ancestry=hl.agg.counter(ht.ancestry.gen_anc),
        )
    )
    return _as_plain(stats)


def hard_filter_counts(ht: hl.Table) -> pd.DataFrame:
    """One row per filter: how many samples fail it and the cohort fraction."""
    counts = ht.aggregate(
        hl.agg.explode(lambda name: hl.agg.counter(name), ht.hard_filters)
    )
    return _filter_counts_frame(dict(counts), ht.count())


def hard_filter_membership(ht: hl.Table) -> pd.DataFrame:
    """One row per sample, one bool column per filter (the UpSet input).

    All-False filter columns are dropped; the index is the sample id.
    """
    membership = _sample_frame(ht, hard_filters=ht.hard_filters)
    return from_memberships(membership["hard_filters"], membership.index)


def coverage_metrics(ht: hl.Table) -> pd.DataFrame:
    """One row per sample: coverage and call-rate metrics plus the hard-filter
    inputs, with ``bases_over_dp_threshold`` flattened into named columns."""
    rows = _sample_frame(
        ht,
        autosomal_mean_dp=ht.ploidy.autosomal_mean_dp,
        bases_over_dp_threshold=ht.bases_over_dp_threshold,
        callrate=ht.callrate,
        callrate_adj=ht.callrate_adj,
        n_singleton=ht.n_singleton,
        r_het_hom_var=ht.r_het_hom_var,
        n_het=ht.n_het,
        n_hom_var=ht.n_hom_var,
    )
    return _coverage_frame(rows)


def coverage_histograms(ht: hl.Table, nbins: int = 40) -> pd.DataFrame:
    """Hail-aggregated depth/call-rate histograms: ``metric``, ``bin_center``,
    ``count``. A large cohort still returns a handful of rows."""
    bounds = ht.aggregate(
        hl.struct(
            min_dp=hl.agg.min(ht.ploidy.autosomal_mean_dp),
            max_dp=hl.agg.max(ht.ploidy.autosomal_mean_dp),
        )
    )
    lower = float(bounds.min_dp)
    upper = float(bounds.max_dp)
    if not upper > lower:
        upper = lower + 1.0
    hists = ht.aggregate(
        hl.struct(
            autosomal_mean_dp=hl.agg.hist(
                ht.ploidy.autosomal_mean_dp, lower, upper, nbins
            ),
            callrate_adj=hl.agg.hist(ht.callrate_adj, 0.0, 1.0, nbins),
        )
    )
    return pd.concat(
        [
            _histogram_frame(hists.autosomal_mean_dp, "autosomal_mean_dp"),
            _histogram_frame(hists.callrate_adj, "callrate_adj"),
        ],
        ignore_index=True,
    )


def sex_inference(ht: hl.Table) -> pd.DataFrame:
    """One row per sample: the ``ploidy`` struct flattened into its fields."""
    available = ht.ploidy.dtype.fields
    fields = {name: ht.ploidy[name] for name in PLOIDY_FIELDS if name in available}
    return _sample_frame(ht, **fields)


def ancestry_scores(ht: hl.Table) -> pd.DataFrame:
    """One row per sample: ``gen_anc``, ``prob_*`` and the PCA pca_scores as ``pc<i>``."""
    ancestry = ht.ancestry
    available = ancestry.dtype.fields
    fields: dict[str, Any] = {}
    if "gen_anc" in available:
        fields["gen_anc"] = ancestry.gen_anc
    for population in ANCESTRY_POPS:
        name = f"prob_{population}"
        if name in available:
            fields[name] = ancestry[name]
    if "pca_scores" in available:
        fields["pca_scores"] = ancestry.pca_scores
    rows = _sample_frame(ht, **fields)
    return _ancestry_frame(rows)


def contamination(ht: hl.Table) -> pd.DataFrame:
    """One row per sample: ``charr`` and the assigned ancestry ``gen_anc``."""
    return _sample_frame(ht, charr=ht.charr, gen_anc=ht.ancestry.gen_anc)


def sample_ids(ht: hl.Table) -> list[str]:
    """The cohort's sample ids, sorted, as the complete heatmap axis."""
    ht = ht.key_by()
    rows = _sample_frame(ht, s=ht.s)
    return sorted(rows.index)


def relatedness_pairs(
    rel_ht: hl.Table, ht: hl.Table, cap: int | None = 50_000
) -> pd.DataFrame:
    """Pairwise ``i``/``j`` relatedness estimates joined to both ancestries.

    The production tables are written with no ``min_kinship`` pruning
    (``None``), so passing ``cap=None`` returns the complete set of pairs — the
    report does this for its heatmaps. A dense cohort can still OOM the driver,
    so the default caps the frame to the ``cap`` most-related pairs.
    """
    fields = [
        name
        for name in ("kin", "ibd0", "ibd1", "ibd2")
        if name in rel_ht.row.dtype.fields
    ]
    selected = {
        ("kinship" if name == "kin" else name): rel_ht[name] for name in fields
    }
    pairs = rel_ht.select(**selected).to_pandas()
    if "i" not in pairs.columns:
        pairs = pairs.reset_index()
    pop = _sample_frame(ht, gen_anc=ht.ancestry.gen_anc)["gen_anc"]
    return _relatedness_frame(pairs, pop, cap)


def king_pairs(
    king_mt: hl.MatrixTable, ht: hl.Table, cap: int | None = 50_000
) -> pd.DataFrame:
    """Pairwise KING kinship from a ``hl.king`` matrix, joined to ancestries.

    ``hl.king`` returns a matrix keyed by sample on both axes — row key ``s_1``,
    column key ``s`` — with a single entry, ``phi``. The pairs are read out and
    renamed to the ``relatedness_pairs`` shape (``kinship``) so the same frame
    builders apply; KING reports no IBD probabilities, so those columns are
    absent. ``cap=None`` returns every pair for a complete heatmap.
    """
    row_key = next(iter(king_mt.row_key))
    col_key = next(iter(king_mt.col_key))
    entries = king_mt.entries()
    pairs = entries.select(
        i=entries[col_key], j=entries[row_key], kin=entries.phi
    ).to_pandas()
    if "i" not in pairs.columns:
        pairs = pairs.reset_index()
    pop = _sample_frame(ht, gen_anc=ht.ancestry.gen_anc)["gen_anc"]
    return _relatedness_frame(pairs, pop, cap)


def relatedness_comparison(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join per-method relatedness frames into one pair-per-row comparison.

    ``frames`` maps a method label to its pair frame (as returned by
    ``relatedness_pairs`` or ``king_pairs``); each frame's ``kinship`` column is
    suffixed ``kinship_<label>``. The join is an outer join on ``(i, j)``, so
    every pair any method reports is kept and a method that omits a pair shows a
    missing value rather than dropping the row. Rows are ordered by kinship,
    most related first.
    """
    comparison: pd.DataFrame | None = None
    ancestry: pd.DataFrame | None = None
    for label, frame in frames.items():
        if frame.empty:
            continue
        if ancestry is None and {"gen_anc_i", "gen_anc_j"} <= set(frame.columns):
            ancestry = frame[["i", "j", "gen_anc_i", "gen_anc_j"]]
        renamed = frame[["i", "j", "kinship"]].rename(
            columns={"kinship": f"kinship_{label}"}
        )
        comparison = (
            renamed
            if comparison is None
            else comparison.merge(renamed, on=["i", "j"], how="outer")
        )
    if comparison is None:
        return pd.DataFrame(columns=["i", "j", "gen_anc_i", "gen_anc_j"])
    if ancestry is not None:
        comparison = comparison.merge(ancestry, on=["i", "j"], how="left")
    kinship_columns = [
        column for column in comparison.columns if column.startswith("kinship_")
    ]
    comparison = comparison.reindex(
        columns=["i", "j", "gen_anc_i", "gen_anc_j", *kinship_columns]
    )
    # Order by the pair's strongest kinship across methods, so the ordering does
    # not depend on the order the frames were supplied in.
    order = comparison[kinship_columns].max(axis=1)
    return (
        comparison.assign(pair_max_kinship=order)
        .sort_values("pair_max_kinship", ascending=False, na_position="last")
        .drop(columns="pair_max_kinship")
        .reset_index(drop=True)
    )


def relatedness_concordance(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Pearson/Spearman correlation between the methods' kinships.

    Correlations are computed over the pairs *both* methods report
    (complete-case), so each row states its own ``n_pairs``.
    """
    methods = [column for column in df.columns if column.startswith("kinship_")]
    rows = []
    for left, right in itertools.combinations(methods, 2):
        shared = df[[left, right]].dropna()
        enough = len(shared) > 1
        rows.append(
            {
                "method_a": left.removeprefix("kinship_"),
                "method_b": right.removeprefix("kinship_"),
                "n_pairs": len(shared),
                "pearson": shared[left].corr(shared[right]) if enough else np.nan,
                "spearman": (
                    shared[left].corr(shared[right], method="spearman")
                    if enough
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(
        rows, columns=["method_a", "method_b", "n_pairs", "pearson", "spearman"]
    )


# --- plot builders ----------------------------------------------------------
# Each takes the matching collector frame; thresholds arrive as arguments (from
# ``config_summary``) so a builder never reads config or globals.


def ancestry_pca_scatter(df: pd.DataFrame) -> pn.ggplot:
    """PC1 x PC2, coloured by assigned ancestry."""
    return (
        pn.ggplot(df, pn.aes(x="pc1", y="pc2", color="gen_anc"))
        + pn.geom_point(alpha=0.6)
        + pn.labs(x="PC1", y="PC2", color="Ancestry", title="Ancestry PCA")
        + THEME
    )


def ancestry_probability_bar(df: pd.DataFrame) -> pn.ggplot:
    """Mean inferred ``prob_*`` per assigned ancestry."""
    prob_columns = [c for c in df.columns if c.startswith("prob_")]
    long = df.melt(
        id_vars="gen_anc",
        value_vars=prob_columns,
        var_name="population",
        value_name="probability",
    )
    long["population"] = long["population"].str.removeprefix("prob_")
    means = long.groupby(["gen_anc", "population"], as_index=False)["probability"].mean()
    return (
        pn.ggplot(means, pn.aes(x="population", y="probability", fill="gen_anc"))
        + pn.geom_col(position="dodge")
        + pn.labs(
            x="Population",
            y="Mean probability",
            fill="Ancestry",
            title="Ancestry probabilities",
        )
        + THEME
    )


def sex_ploidy_scatter(df: pd.DataFrame) -> pn.ggplot:
    """chrX ploidy x chrY ploidy, coloured by inferred sex karyotype."""
    return (
        pn.ggplot(df, pn.aes(x="chrX_ploidy", y="chrY_ploidy", color="sex_karyotype"))
        + pn.geom_point(alpha=0.6)
        + pn.labs(
            x="chrX ploidy",
            y="chrY ploidy",
            color="Karyotype",
            title="Sex inference",
        )
        + THEME
    )


def f_stat_histogram(df: pd.DataFrame, cutoff: float | None = None) -> pn.ggplot:
    """The inbreeding ``f_stat`` distribution with the cutoff line."""
    plot = (
        pn.ggplot(df, pn.aes(x="f_stat"))
        + pn.geom_histogram(bins=50, fill="#4c72b0", color="white")
        + pn.labs(x="f_stat", y="Samples", title="Inbreeding coefficient")
        + THEME
    )
    if cutoff is not None:
        plot = plot + pn.geom_vline(
            xintercept=cutoff, linetype="dashed", color="red"
        )
    return plot


def relatedness_heatmap(
    df: pd.DataFrame,
    min_kinship: float | None = None,
    samples: pd.Index | list[str] | None = None,
    title: str = "Pairwise relatedness",
) -> pn.ggplot:
    """Pairwise kinship as a symmetric sample x sample heatmap.

    ``df`` is a ``relatedness_pairs``/``king_pairs`` frame; it is mirrored into a
    symmetric matrix (:func:`_kinship_matrix`) and each pair drawn as a tile with
    ``kinship`` as the fill. Passing ``samples`` reindexes the matrix to the full
    cohort so the heatmap is the complete sample x sample grid (blank tiles where
    a method reported no pair); otherwise the axes span only the samples seen.

    ``min_kinship`` anchors the fill scale's lower limit — pass
    :data:`RELATEDNESS_DISPLAY_THRESHOLD` for production tables written with
    ``min_kinship = None`` so the ramp reads from "related enough" upward instead
    of compressing the signal against the undifferentiated background.
    """
    matrix = _kinship_matrix(df, samples)
    limits = None
    if min_kinship is not None and not matrix.empty:
        upper = float(matrix["kinship"].max())
        if upper > min_kinship:
            limits = (float(min_kinship), upper)
    return (
        pn.ggplot(matrix, pn.aes(x="i", y="j", fill="kinship"))
        + pn.geom_tile()
        + pn.scale_fill_gradient(low="#f7f7f7", high="#c44e52", limits=limits)
        + pn.labs(x="", y="", fill="Kinship", title=title)
        + THEME
    )


def relatedness_ibd_scatter(df: pd.DataFrame) -> pn.ggplot:
    """IBD0 x IBD1 for each related pair, coloured by kinship."""
    return (
        pn.ggplot(df, pn.aes(x="ibd0", y="ibd1", color="kinship"))
        + pn.geom_point(alpha=0.6)
        + pn.labs(
            x="IBD0", y="IBD1", color="Kinship", title="Relatedness IBD"
        )
        + THEME
    )


def relatedness_method_scatter(
    df: pd.DataFrame, x: str, y: str, title: str = "Relatedness methods"
) -> pn.ggplot:
    """One relatedness method's kinship against another's, with ``y = x``.

    ``df`` is the ``relatedness_comparison`` frame; ``x``/``y`` name its
    ``kinship_<method>`` columns. The dashed diagonal is the line the two
    methods would trace if they agreed exactly.
    """
    return (
        pn.ggplot(df, pn.aes(x=x, y=y))
        + pn.geom_point(alpha=0.6)
        + pn.labs(
            x=x.removeprefix("kinship_"),
            y=y.removeprefix("kinship_"),
            title=title,
        )
        + pn.geom_abline(intercept=0, slope=1, linetype="dashed", color="grey")
        + THEME
    )


def hard_filter_counts_bar(df: pd.DataFrame) -> pn.ggplot:
    """Samples failing each hard filter."""
    return (
        pn.ggplot(df, pn.aes(x="reorder(filter, n_samples)", y="n_samples"))
        + pn.geom_col(fill="#c44e52")
        + pn.coord_flip()
        + pn.labs(x="", y="Samples", title="Hard-filter failures")
        + THEME
    )


def hard_filter_upset(membership: pd.DataFrame) -> plt.Figure:
    """The co-failure view of the hard filters as a matplotlib UpSet figure.

    ``membership`` is the bool frame ``hard_filter_membership`` returns; it is
    converted to upsetplot's per-sample membership lists.
    """
    figure = plt.figure(figsize=(8, 5))
    if membership is None or membership.shape[1] == 0:
        return figure
    UpSet(
        membership,
        sort_by="cardinality",
        show_counts=False,
    ).plot(fig=figure)
    return figure


def coverage_dp_histogram(df: pd.DataFrame, min_cov: float | None = None) -> pn.ggplot:
    """The autosomal mean-depth distribution with the minimum-coverage line."""
    plot = (
        pn.ggplot(df, pn.aes(x="autosomal_mean_dp"))
        + pn.geom_histogram(bins=50, fill="#4c72b0", color="white")
        + pn.labs(x="Autosomal mean depth", y="Samples", title="Autosomal depth")
        + THEME
    )
    if min_cov is not None:
        plot = plot + pn.geom_vline(
            xintercept=min_cov, linetype="dashed", color="red"
        )
    return plot


def bases_over_dp_scatter(
    df: pd.DataFrame, cutoffs: dict[str, Any] | None = None
) -> pn.ggplot:
    """Bases over DP 1 x bases over DP 20, log-log, with the threshold lines."""
    plot = (
        pn.ggplot(df, pn.aes(x="bases_over_dp_1", y="bases_over_dp_20"))
        + pn.geom_point(alpha=0.5)
        + pn.scale_x_log10()
        + pn.scale_y_log10()
        + pn.labs(
            x="Bases over DP 1 (log10)",
            y="Bases over DP 20 (log10)",
            title="Depth coverage",
        )
        + THEME
    )
    if cutoffs:
        low = cutoffs.get("min_bases_dp_over_1")
        high = cutoffs.get("min_bases_dp_over_20")
        if low is not None:
            plot = plot + pn.geom_vline(
                xintercept=low, linetype="dashed", color="red"
            )
        if high is not None:
            plot = plot + pn.geom_hline(
                yintercept=high, linetype="dashed", color="red"
            )
    return plot


def singleton_het_hom_scatter(
    df: pd.DataFrame, cutoffs: dict[str, Any] | None = None
) -> pn.ggplot:
    """Singleton count x het/hom-var ratio with the two hard-filter cutoffs."""
    mapping = {"x": "n_singleton", "y": "r_het_hom_var"}
    if "hard_filters" in df.columns:
        mapping["color"] = "hard_filters"
    plot = (
        pn.ggplot(df, pn.aes(**mapping))
        + pn.geom_point(alpha=0.6)
        + pn.labs(
            x="Singletons",
            y="het/hom-var ratio",
            title="Singleton and het/hom-var outliers",
        )
        + THEME
    )
    if cutoffs:
        max_singletons = cutoffs.get("max_n_singleton")
        max_ratio = cutoffs.get("max_r_het_hom_var")
        if max_singletons is not None:
            plot = plot + pn.geom_vline(
                xintercept=max_singletons, linetype="dashed", color="red"
            )
        if max_ratio is not None:
            plot = plot + pn.geom_hline(
                yintercept=max_ratio, linetype="dashed", color="red"
            )
    return plot


def callrate_histogram(df: pd.DataFrame, cutoff: float | None = None) -> pn.ggplot:
    """The adjusted call-rate distribution with the minimum line."""
    plot = (
        pn.ggplot(df, pn.aes(x="callrate_adj"))
        + pn.geom_histogram(bins=50, fill="#4c72b0", color="white")
        + pn.labs(x="Adjusted call rate", y="Samples", title="Adjusted call rate")
        + THEME
    )
    if cutoff is not None:
        plot = plot + pn.geom_vline(
            xintercept=cutoff, linetype="dashed", color="red"
        )
    return plot


def contamination_histogram(
    df: pd.DataFrame, cutoff: float | None = None
) -> pn.ggplot:
    """The ``charr`` contamination distribution with the maximum line."""
    plot = (
        pn.ggplot(df, pn.aes(x="charr"))
        + pn.geom_histogram(bins=50, fill="#4c72b0", color="white")
        + pn.labs(x="charr", y="Samples", title="Contamination estimate")
        + THEME
    )
    if cutoff is not None:
        plot = plot + pn.geom_vline(
            xintercept=cutoff, linetype="dashed", color="red"
        )
    return plot
