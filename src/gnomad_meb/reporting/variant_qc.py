"""The Variant QC stage: collectors over the ``variant_qc`` hail table and the
``plotnine`` builders over their frames.

The stage's declared primary output is the ``variant_qc`` table written by
``VariantQcImportPipeline.merge``. It is a **merged global** table (no split
column), so the natural breakdowns are variant class, contig and the VQSR
culprit/score. The VQSR provenance — the target sensitivities, the applied
``minVQSLod`` cutoffs, the recalibration params and the GATK version — travels as
the table's **globals** (ticket ``08``), so the report reads them with a single
``ht.index_globals()`` rather than side-loading the tranches files.

Each collector is stateless, returns a tidy ``pandas.DataFrame`` (or a plain
dict for scalars) with stable columns, and does not write to disk. Following
:mod:`gnomad_meb.reporting.sample_qc`, each Hail collector is a thin ``hl.Table``
read plus a **pure reshape helper**, so the shaping is unit-testable without a
Spark session. No collector unions the callset: ``variant_qc`` is already the
terminal merge (ADR-0015), so a report is a read.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import plotnine as pn

from gnomad_meb.hail import hail as hl
from gnomad_meb.reporting.sample_qc import (
    _as_plain,
    _filter_counts_frame,
    _histogram_frame,
)
from gnomad_meb.reporting.theme import THEME, ordered_splits

__all__ = [
    "culprit_bar",
    "culprit_counts",
    "filter_by_contig",
    "filter_counts",
    "filter_counts_bar",
    "filter_rate_by_contig",
    "top_filtered_sites",
    "variant_class_bar",
    "variant_class_counts",
    "variant_overview",
    "vqslod_bins",
    "vqslod_histogram",
    "vqslod_summary",
    "vqsr_cutoffs",
    "vqsr_provenance",
]

#: The two VQSR modes; every per-mode view is keyed by these.
VQSR_MODES = ("snp", "indel")

#: Columns of the ``vqslod_summary`` frame, in order.
_VQSLOD_SUMMARY_COLUMNS = (
    "variant_class",
    "n",
    "min",
    "p25",
    "median",
    "p75",
    "max",
    "mean",
)

#: Columns of the ``variant_class_counts`` frame, in order.
_VARIANT_CLASS_COLUMNS = (
    "allele_type",
    "variant_type",
    "n_variants",
    "n_pass",
    "n_filtered",
    "filter_rate",
)

#: Columns of the ``filter_by_contig`` frame, in order.
_CONTIG_COLUMNS = ("contig", "n_variants", "n_filtered", "filter_rate")

#: Columns of the ``top_filtered_sites`` frame, in order.
_TOP_FILTERED_COLUMNS = (
    "locus",
    "alleles",
    "allele_type",
    "filters",
    "AS_VQSLOD",
    "AS_culprit",
    "QD",
    "FS",
    "MQ",
    "SOR",
)


# --- pure reshape helpers ---------------------------------------------------
# Each collector is a thin hail read; the shaping below is ordinary Python so it
# is unit-testable without a Spark session (the ``sample_qc.py`` precedent).


def _is_filtered(ht: hl.Table) -> hl.expr.BooleanExpression:
    """Whether a row failed a filter: VQSR writes a non-empty ``FILTER`` set."""
    return hl.len(ht.filters) > 0


def _is_snv(ht: hl.Table) -> hl.expr.BooleanExpression:
    """Whether a row is a SNP (the split-biallelic ``allele_type`` view)."""
    return ht.allele_info.allele_type == "snv"


def _overview(stats: Any) -> dict[str, Any]:
    """The overview struct with the derived transition/transversion ratio."""
    plain = _as_plain(stats)
    n_ti = plain.get("n_ti") or 0
    n_tv = plain.get("n_tv") or 0
    plain["ti_tv"] = (n_ti / n_tv) if n_tv else float("nan")
    return plain


def _vqsr_provenance(globals_plain: dict[str, Any], n_variants: int) -> dict[str, Any]:
    """The plain-globals view of a Variant QC table's VQSR provenance."""
    return {
        "n_variants": n_variants,
        "variant_qc_cutoffs": globals_plain.get("variant_qc_cutoffs") or {},
        "vqsr_params": globals_plain.get("vqsr_params") or {},
        "gatk_version": globals_plain.get("gatk_version") or {},
    }


def _vqsr_cutoffs_frame(cutoffs: dict[str, Any]) -> pd.DataFrame:
    """One row per VQSR mode: its target sensitivity and applied VQSLOD cutoff."""
    rows = []
    for mode in VQSR_MODES:
        entry = cutoffs.get(mode) or {}
        rows.append(
            {
                "mode": mode,
                "target_truth_sensitivity": entry.get("target_truth_sensitivity"),
                "vqslod_cutoff": entry.get("vqslod_cutoff"),
            }
        )
    return pd.DataFrame(
        rows, columns=["mode", "target_truth_sensitivity", "vqslod_cutoff"]
    )


def _vqslod_stats_frame(stats: Any) -> pd.DataFrame:
    """One row per VQSR mode from a per-class VQSLOD stats struct."""
    plain = _as_plain(stats)
    rows = []
    for mode in VQSR_MODES:
        entry = plain.get(mode) or {}
        rows.append({"variant_class": mode, **entry})
    return pd.DataFrame(rows).reindex(columns=_VQSLOD_SUMMARY_COLUMNS)


def _vqslod_histogram_frame(hists: Any) -> pd.DataFrame:
    """One ``bin_center``/``count`` series per VQSR mode from per-class hists."""
    plain = _as_plain(hists)
    frames = []
    for mode in VQSR_MODES:
        hist = plain.get(mode)
        if hist is None:
            continue
        frame = _histogram_frame(hist, mode)
        frames.append(frame.rename(columns={"metric": "variant_class"}))
    if not frames:
        return pd.DataFrame(columns=["variant_class", "bin_center", "count"])
    return pd.concat(frames, ignore_index=True)


def _histogram_bounds(lower: Any, upper: Any) -> tuple[float, float]:
    """Finite, positive-width bounds for an ``hl.agg.hist``.

    ``AS_VQSLOD`` is imported from a VCF whose header declares it a *string*
    and GATK writes non-finite scores for alleles the mixture model cannot
    score (``Infinity``/``-Infinity``/``NaN``). ``hl.agg.hist`` rejects a
    non-finite ``end`` ("requires positive bin size"), so the raw min/max are
    sanitised here: a non-finite or missing bound degrades to a unit-width
    range around the finite lower bound.
    """
    low = float(lower) if lower is not None else 0.0
    high = float(upper) if upper is not None else low + 1.0
    if not math.isfinite(low):
        low = 0.0
    if not math.isfinite(high) or not high > low:
        high = low + 1.0
    return low, high


def _variant_class_frame(counts: Any, filtered: Any) -> pd.DataFrame:
    """One row per ``(allele_type, variant_type)`` from the two tuple counters."""
    counts = _as_plain(counts) or {}
    filtered = _as_plain(filtered) or {}
    rows = []
    for key, n_variants in counts.items():
        allele_type, variant_type = key
        n_filtered = filtered.get(key, 0)
        rows.append(
            {
                "allele_type": allele_type,
                "variant_type": variant_type,
                "n_variants": n_variants,
                "n_pass": n_variants - n_filtered,
                "n_filtered": n_filtered,
                "filter_rate": (n_filtered / n_variants) if n_variants else 0.0,
            }
        )
    return pd.DataFrame(rows, columns=_VARIANT_CLASS_COLUMNS).sort_values(
        "n_variants", ascending=False, ignore_index=True
    )


def _filter_by_contig_frame(counts: Any, filtered: Any) -> pd.DataFrame:
    """One row per contig from the per-contig variant/filtered counters."""
    counts = _as_plain(counts) or {}
    filtered = _as_plain(filtered) or {}
    rows = []
    for contig, n_variants in counts.items():
        n_filtered = filtered.get(contig, 0)
        rows.append(
            {
                "contig": contig,
                "n_variants": n_variants,
                "n_filtered": n_filtered,
                "filter_rate": (n_filtered / n_variants) if n_variants else 0.0,
            }
        )
    return pd.DataFrame(rows, columns=_CONTIG_COLUMNS)


def _culprit_frame(counts: Any, n_filtered: int) -> pd.DataFrame:
    """One row per ``AS_culprit`` among filtered rows, sorted by count."""
    counts = _as_plain(counts) or {}
    if not counts:
        return pd.DataFrame(columns=["culprit", "n_variants", "fraction"])
    frame = pd.DataFrame(
        {"culprit": list(counts.keys()), "n_variants": list(counts.values())}
    )
    frame = frame.sort_values("n_variants", ascending=False, ignore_index=True)
    frame["fraction"] = frame["n_variants"] / n_filtered if n_filtered else 0.0
    return frame[["culprit", "n_variants", "fraction"]]


def _top_filtered_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """Format a bounded top-filtered frame: the ``filters`` set as a string."""
    frame = rows.copy()
    if "filters" in frame.columns:
        frame["filters"] = frame["filters"].map(
            lambda value: ",".join(sorted(value)) if value else ""
        )
    return frame.reindex(columns=[c for c in _TOP_FILTERED_COLUMNS if c in frame])


# --- collectors -------------------------------------------------------------


def vqsr_provenance(ht: hl.Table) -> dict[str, Any]:
    """The table's globals (VQSR cutoffs/params/GATK version) plus ``n_variants``.

    The cutoffs feed the plot builders' reference lines.
    """
    globals_plain = _as_plain(hl.eval(ht.index_globals()))
    return _vqsr_provenance(globals_plain, ht.count())


def variant_overview(ht: hl.Table) -> dict[str, Any]:
    """Scalar tallies: the pass/filtered split, class, class and train sites."""
    stats = ht.aggregate(
        hl.struct(
            n_variants=hl.agg.count(),
            n_pass=hl.agg.count_where(~_is_filtered(ht)),
            n_filtered=hl.agg.count_where(_is_filtered(ht)),
            n_lowqual=hl.agg.count_where(ht.AS_lowqual),
            n_snv=hl.agg.count_where(_is_snv(ht)),
            n_indel=hl.agg.count_where(~_is_snv(ht)),
            n_ti=hl.agg.count_where(
                hl.is_transition(ht.alleles[0], ht.alleles[1])
            ),
            n_tv=hl.agg.count_where(
                hl.is_transversion(ht.alleles[0], ht.alleles[1])
            ),
            n_split_rows=hl.agg.count_where(ht.was_split),
            n_positive_train=hl.agg.count_where(ht.info.POSITIVE_TRAIN_SITE),
            n_negative_train=hl.agg.count_where(ht.info.NEGATIVE_TRAIN_SITE),
        )
    )
    return _overview(stats)


def vqsr_cutoffs(ht: hl.Table) -> pd.DataFrame:
    """One row per VQSR mode: the applied target sensitivity and VQSLOD cutoff.

    Read from the table's globals (ticket ``08``); a legacy table with no
    globals degrades to missing cutoffs rather than failing.
    """
    globals_plain = _as_plain(hl.eval(ht.index_globals()))
    return _vqsr_cutoffs_frame(globals_plain.get("variant_qc_cutoffs") or {})


def filter_counts(ht: hl.Table) -> pd.DataFrame:
    """One row per FILTER value: how many variants fail it and the fraction.

    This is the "how many variants VQSR filtered" table.
    """
    counts = ht.aggregate(
        hl.agg.explode(lambda name: hl.agg.counter(name), ht.filters)
    )
    frame = _filter_counts_frame(dict(counts), ht.count())
    return frame.rename(columns={"n_samples": "n_variants"})


def _class_stats(expr: hl.expr.NumericExpression) -> hl.expr.StructExpression:
    """The ``AS_VQSLOD`` summary struct for one VQSR mode (missing dropped)."""
    return hl.struct(
        n=hl.agg.count_where(hl.is_defined(expr)),
        min=hl.agg.min(expr),
        p25=hl.agg.approx_quantiles(expr, 0.25),
        median=hl.agg.approx_quantiles(expr, 0.5),
        p75=hl.agg.approx_quantiles(expr, 0.75),
        max=hl.agg.max(expr),
        mean=hl.agg.mean(expr),
    )


def vqslod_summary(ht: hl.Table) -> pd.DataFrame:
    """Per VQSR mode: the ``AS_VQSLOD`` distribution summary (min..max)."""
    stats = ht.aggregate(
        hl.struct(
            snp=_class_stats(hl.or_missing(_is_snv(ht), ht.info.AS_VQSLOD)),
            indel=_class_stats(hl.or_missing(~_is_snv(ht), ht.info.AS_VQSLOD)),
        )
    )
    return _vqslod_stats_frame(stats)


def vqslod_bins(ht: hl.Table, nbins: int = 60) -> pd.DataFrame:
    """The ``AS_VQSLOD`` distribution per VQSR mode: a handful of binned rows.

    Non-finite ``AS_VQSLOD`` values are excluded from the bounds so
    ``hl.agg.hist`` always receives a finite, positive-width range.
    """
    finite = hl.or_missing(hl.is_finite(ht.info.AS_VQSLOD), ht.info.AS_VQSLOD)
    bounds = ht.aggregate(
        hl.struct(
            lower=hl.agg.min(finite),
            upper=hl.agg.max(finite),
        )
    )
    lower, upper = _histogram_bounds(bounds.lower, bounds.upper)
    hists = ht.aggregate(
        hl.struct(
            snp=hl.agg.filter(
                _is_snv(ht), hl.agg.hist(finite, lower, upper, nbins)
            ),
            indel=hl.agg.filter(
                ~_is_snv(ht), hl.agg.hist(finite, lower, upper, nbins)
            ),
        )
    )
    return _vqslod_histogram_frame(hists)


def variant_class_counts(ht: hl.Table) -> pd.DataFrame:
    """One row per ``(allele_type, variant_type)``: pass/filtered counts and rate."""
    key = hl.tuple([ht.allele_info.allele_type, ht.allele_info.variant_type])
    result = ht.aggregate(
        hl.struct(
            counts=hl.agg.counter(key),
            filtered=hl.agg.filter(_is_filtered(ht), hl.agg.counter(key)),
        )
    )
    return _variant_class_frame(result.counts, result.filtered)


def filter_by_contig(ht: hl.Table) -> pd.DataFrame:
    """One row per contig: variant count, filtered count and filter rate."""
    result = ht.aggregate(
        hl.struct(
            counts=hl.agg.counter(ht.locus.contig),
            filtered=hl.agg.filter(
                _is_filtered(ht), hl.agg.counter(ht.locus.contig)
            ),
        )
    )
    return _filter_by_contig_frame(result.counts, result.filtered)


def culprit_counts(ht: hl.Table) -> pd.DataFrame:
    """One row per ``AS_culprit`` among filtered rows: why VQSR dropped them."""
    result = ht.aggregate(
        hl.struct(
            n_filtered=hl.agg.count_where(_is_filtered(ht)),
            culprits=hl.agg.filter(
                _is_filtered(ht), hl.agg.counter(ht.info.AS_culprit)
            ),
        )
    )
    return _culprit_frame(result.culprits, int(result.n_filtered or 0))


def top_filtered_sites(ht: hl.Table, n: int = 100) -> pd.DataFrame:
    """The ``n`` highest-``AS_VQSLOD`` rows that still failed, as examples.

    A bounded driver pull, capped like the Sample QC relatedness frame so a
    callset-scale table never lands on the driver.
    """
    filtered = ht.filter(_is_filtered(ht))
    filtered = filtered.order_by(hl.desc(filtered.info.AS_VQSLOD))
    rows = filtered.select(
        locus=hl.str(filtered.locus),
        alleles=hl.delimit(filtered.alleles, ","),
        allele_type=filtered.allele_info.allele_type,
        filters=filtered.filters,
        AS_VQSLOD=filtered.info.AS_VQSLOD,
        AS_culprit=filtered.info.AS_culprit,
        QD=filtered.info.QD,
        FS=filtered.info.FS,
        MQ=filtered.info.MQ,
        SOR=filtered.info.SOR,
    )
    return _top_filtered_frame(rows.head(n).to_pandas())


# --- plot builders ----------------------------------------------------------
# Each takes the matching collector frame; thresholds arrive as arguments (from
# ``config_summary`` / ``vqsr_cutoffs``) so a builder never reads config or files.


def filter_counts_bar(df: pd.DataFrame) -> pn.ggplot:
    """Variants failing each FILTER value."""
    return (
        pn.ggplot(df, pn.aes(x="reorder(filter, n_variants)", y="n_variants"))
        + pn.geom_col(fill="#c44e52")
        + pn.coord_flip()
        + pn.labs(x="", y="Variants", title="Filtered variants")
        + THEME
    )


def filter_rate_by_contig(df: pd.DataFrame) -> pn.ggplot:
    """The filter rate per contig, in semantic contig order."""
    frame = df.copy()
    frame["contig"] = ordered_splits(frame["contig"])
    return (
        pn.ggplot(frame, pn.aes(x="contig", y="filter_rate"))
        + pn.geom_col(fill="#c44e52")
        + pn.labs(x="", y="Filter rate", title="Filter rate by contig")
        + THEME
    )


def vqslod_histogram(
    df: pd.DataFrame, cutoffs: dict[str, float | None] | None = None
) -> pn.ggplot:
    """The ``AS_VQSLOD`` distribution by VQSR mode (log-count y), with the
    applied VQSLOD cutoff drawn per mode."""
    plot = (
        pn.ggplot(df, pn.aes(x="bin_center", y="count", fill="variant_class"))
        + pn.geom_col()
        + pn.scale_y_log10()
        + pn.labs(
            x="AS_VQSLOD",
            y="Variants (log10)",
            fill="Class",
            title="VQSLOD distribution",
        )
        + THEME
    )
    if cutoffs:
        for mode, value in cutoffs.items():
            if value is not None and mode in set(df["variant_class"]):
                plot = plot + pn.geom_vline(
                    xintercept=value, linetype="dashed", color="red"
                )
    return plot


def variant_class_bar(df: pd.DataFrame) -> pn.ggplot:
    """Stacked bar of pass/filtered counts per ``allele_type``."""
    long = df.melt(
        id_vars=["allele_type", "variant_type"],
        value_vars=["n_pass", "n_filtered"],
        var_name="outcome",
        value_name="n",
    )
    return (
        pn.ggplot(long, pn.aes(x="allele_type", y="n", fill="outcome"))
        + pn.geom_col()
        + pn.labs(
            x="Allele type", y="Variants", fill="", title="Variant class"
        )
        + THEME
    )


def culprit_bar(df: pd.DataFrame) -> pn.ggplot:
    """Filtered variants by ``AS_culprit``."""
    return (
        pn.ggplot(df, pn.aes(x="reorder(culprit, n_variants)", y="n_variants"))
        + pn.geom_col(fill="#c44e52")
        + pn.coord_flip()
        + pn.labs(x="", y="Variants", title="VQSR culprits")
        + THEME
    )
