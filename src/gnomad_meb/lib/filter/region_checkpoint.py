from __future__ import annotations

from typing import TYPE_CHECKING

from gnomad_meb.lib.filter.config import RegionCheckpoint

if TYPE_CHECKING:
    from gnomad_meb.hail import hail as hl


def apply_region_checkpoint(
    mt: hl.MatrixTable,
    *,
    region: RegionCheckpoint,
):
    """Filter the given MatrixTable based on an interval set."""
    from gnomad_meb.hail import hail as hl

    if region.intervals is None:
        return mt

    intervals_ht = hl.import_bed(
        region.intervals, reference_genome=region.reference_genome
    )

    n_intervals = intervals_ht.count()
    interval_size = intervals_ht.aggregate(
        hl.agg.sum(
            intervals_ht.interval.end.position
            - intervals_ht.interval.start.position
        )
    )

    mt = hl.filter_intervals(
        mt,
        intervals_ht.interval,
        keep=region.keep,
    )
    mt = mt.annotate_rows(region=intervals_ht[mt.locus].target)
    mt = mt.annotate_globals(
        filter_region_info=hl.struct(
            region=region.intervals,
            n_intervals=n_intervals,
            n_variants=mt.count_rows(),
            total_size_bp=interval_size,
            keep=region.keep,
        )
    )
    return mt
