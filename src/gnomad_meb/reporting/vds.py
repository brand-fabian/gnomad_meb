"""The VDS-build stage: collectors over a :class:`~gnomad_meb.lib.vds.VariantDataset`
and the ``plotnine`` builders over their frames.

Every collector is stateless and returns a tidy ``pandas.DataFrame`` (or a
plain dict for scalar summaries) with stable columns. The Hail-bearing
collectors aggregate **per split** with Hail aggregators and combine, rather
than unioning the callset (ADR-0015), so a report over many splits stays cheap.

One module per stage: the builders that visualise the collectors live here too,
so VDS tables and VDS plots sit together (see
:mod:`gnomad_meb.reporting.sample_qc` for the same convention).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotnine as pn

from gnomad_meb.hail import hail as hl
from gnomad_meb.lib.vds import VariantDataset
from gnomad_meb.reporting.theme import THEME, ordered_splits

__all__ = [
    "build_parameters",
    "interval_coverage",
    "intervals_per_split",
    "samples_summary",
    "split_reference_stats",
    "split_sizes",
    "split_variant_stats",
    "task_status",
    "variant_class_by_split",
]


def _frame(
    rows: dict[str, dict[str, Any]], columns: list[str]
) -> pd.DataFrame:
    """Turn a ``{split: {field: value}}`` mapping into a frame with a split column."""
    if not rows:
        return pd.DataFrame(columns=["split", *columns])
    frame = pd.DataFrame.from_dict(rows, orient="index").reset_index(names="split")
    return frame.reindex(columns=["split", *columns])


def build_parameters(vds: VariantDataset) -> dict[str, Any]:
    """The dataset's build parameters plus derived counts (hail-free)."""
    params = vds.params
    return {
        "name": params.name,
        "base": str(params.base),
        "reference_genome": params.reference_genome,
        "interval_size": params.interval_size,
        "n_splits": params.n_splits,
        "branch_factor": params.branch_factor,
        "target_records": params.target_records,
        "overwrite": params.overwrite,
        "n_gvcfs": len(params.gvcfs),
        "n_tasks": len(params.tasks),
    }


def interval_coverage(vds: VariantDataset) -> pd.DataFrame:
    """One row per split: its interval count, span and covered bases (hail-free)."""
    rows = []
    for task in vds.params.tasks:
        intervals = list(task.intervals)
        if intervals:
            first, last = intervals[0], intervals[-1]
            total_bp = sum(
                intv.end.position - intv.start.position + 1 for intv in intervals
            )
            first_contig, first_pos = first.start.contig, first.start.position
            last_contig, last_pos = last.end.contig, last.end.position
        else:
            total_bp = 0
            first_contig = first_pos = last_contig = last_pos = None
        rows.append(
            {
                "split": task.name,
                "n_intervals": len(intervals),
                "total_bp": total_bp,
                "first_contig": first_contig,
                "first_pos": first_pos,
                "last_contig": last_contig,
                "last_pos": last_pos,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "split",
            "n_intervals",
            "total_bp",
            "first_contig",
            "first_pos",
            "last_contig",
            "last_pos",
        ],
    )


def task_status(vds: VariantDataset) -> pd.DataFrame:
    """One row per task: its status and message (hail-free)."""
    statuses = vds.params.task_status
    rows = []
    for task in vds.params.tasks:
        status, message = statuses.get(task.name, ("init", ""))
        rows.append({"split": task.name, "status": status, "message": message})
    return pd.DataFrame(rows, columns=["split", "status", "message"])


def _variant_class(vmt: hl.MatrixTable) -> hl.expr.StringExpression:
    """Classify each variant row as a SNP, an INDEL or other."""
    ref, alt = vmt.alleles[0], vmt.alleles[1]
    return (hl.case()
        .when(hl.is_snp(ref, alt), "n_snps")
        .when(hl.is_indel(ref, alt), "n_indels")
        .default("n_other")
    )


def split_variant_stats(vds: VariantDataset) -> pd.DataFrame:
    """Per finished split: variant row count and SNP/INDEL/other breakdown."""

    def _stats(split_vds: hl.vds.VariantDataset) -> dict[str, int]:
        vmt = split_vds.variant_data
        classes = vmt.aggregate_rows(hl.agg.counter(_variant_class(vmt)))
        return {
            "n_variant_rows": vmt.count_rows(),
            "n_snps": classes.get("n_snps", 0),
            "n_indels": classes.get("n_indels", 0),
            "n_other": classes.get("n_other", 0),
        }

    frame = _frame(
        vds.map_datasets(_stats),
        ["n_variant_rows", "n_snps", "n_indels", "n_other"],
    )
    return frame


def split_reference_stats(vds: VariantDataset) -> pd.DataFrame:
    """Per finished split: reference-block count and total/mean block length.

    Reference data is dense across samples, so the total reference length is the
    summed block ``LEN`` over entries divided by the column count.
    """

    def _stats(split_vds: hl.vds.VariantDataset) -> dict[str, Any]:
        rmt = split_vds.reference_data
        n_samples = max(rmt.count_cols(), 1)
        total_len = rmt.aggregate_entries(hl.agg.sum(rmt.LEN))
        n_ref_rows = rmt.count_rows()
        total_bp = int(total_len or 0) // n_samples
        mean_bp = (total_bp / n_ref_rows) if n_ref_rows else float("nan")
        return {
            "n_ref_rows": n_ref_rows,
            "total_ref_bp": total_bp,
            "mean_ref_block_bp": mean_bp,
        }

    frame = _frame(
        vds.map_datasets(_stats),
        ["n_ref_rows", "total_ref_bp", "mean_ref_block_bp"],
    )
    return frame


def samples_summary(vds: VariantDataset) -> dict[str, Any]:
    """The sample set and whether it is identical across finished splits."""

    def _samples(split_vds: hl.vds.VariantDataset) -> set[str]:
        vmt = split_vds.variant_data
        return set(vmt.aggregate_cols(hl.agg.collect_as_set(vmt.s)))

    per_split = vds.map_datasets(_samples)
    if per_split:
        union: set[str] = set().union(*per_split.values())
        consistent = len({frozenset(s) for s in per_split.values()}) <= 1
    else:
        union, consistent = set(), True
    return {
        "n_samples": len(union),
        "samples": sorted(union),
        "consistent": consistent,
    }


def intervals_per_split(df: pd.DataFrame) -> pn.ggplot:
    """Bar chart of the interval count per split, in semantic split order."""
    frame = df.copy()
    frame["split"] = ordered_splits(frame["split"])
    return (
        pn.ggplot(frame, pn.aes(x="split", y="n_intervals"))
        + pn.geom_col(fill="#4c72b0")
        + pn.coord_flip()
        + pn.labs(x="", y="Intervals", title="Intervals per split")
        + THEME
    )


def variant_class_by_split(df: pd.DataFrame) -> pn.ggplot:
    """Stacked bar of SNP/INDEL/other counts per split, in split order."""
    value_vars = [c for c in ("n_snps", "n_indels", "n_other") if c in df.columns]
    long = df.melt(
        id_vars="split",
        value_vars=value_vars,
        var_name="variant_class",
        value_name="n",
    )
    long["split"] = ordered_splits(long["split"])
    return (
        pn.ggplot(long, pn.aes(x="split", y="n", fill="variant_class"))
        + pn.geom_col()
        + pn.coord_flip()
        + pn.labs(x="", y="Variants", fill="", title="Variant class per split")
        + THEME
    )


def split_sizes(df: pd.DataFrame) -> pn.ggplot:
    """Bar chart of variant rows per split (log-scaled y), in split order."""
    frame = df.copy()
    frame["split"] = ordered_splits(frame["split"])
    return (
        pn.ggplot(frame, pn.aes(x="split", y="n_variant_rows"))
        + pn.geom_col(fill="#4c72b0")
        + pn.coord_flip()
        + pn.scale_y_log10()
        + pn.labs(x="", y="Variant rows (log10)", title="Split sizes")
        + THEME
    )
