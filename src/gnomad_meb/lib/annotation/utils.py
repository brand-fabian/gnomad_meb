import gzip
import re
from contextlib import contextmanager
from pathlib import Path

from gnomad_meb.config import get_config
from gnomad_meb.lib.checkpoint import is_s3_path
from gnomad_meb.logging import get_logger


@contextmanager
def __read_s3(vcf):
    cfg = get_config()
    logger = get_logger()

    try:
        with cfg.s3fs.open(vcf, "rb") as raw_f:
            with gzip.open(raw_f, "rb") as gzip_f:
                yield gzip_f
    except Exception as e:
        logger.error(f"encountered error trying to s3 read vcf {vcf}: {e}")
        raise e
    finally:
        logger.debug(f"finished reading {vcf} from s3 bucket")


@contextmanager
def __read_fs(vcf):
    logger = get_logger()
    try:
        with gzip.open(vcf, "rb") as in_f:
            yield in_f
    finally:
        logger.debug(f"finished reading {vcf}")


def get_clinvar_filedate(vcf: str, s3_uri_schemes: list[str] = ["s3", "s3a"]):
    """Infert the clinvar version from the filepath or vcf file contents.

    This method tries to infer the clinvar version either from the filename,
    matching the standard clinvar names `clinvar_YYYYMMDD` or by using the
    VCF header `##fileDate`.

    Returns
    -------
    str
        A datestring of the format YYYY-MM-DD.
    """
    path = Path(vcf)
    clinvar_date_match = re.match(
        r"clinvar_(?P<year>[0-9]{4})(?P<month>[0-9]{2})(?P<day>[0-9]{2})\.vcf\.gz",
        path.name,
    )
    if clinvar_date_match is not None:
        return "-".join(
            [
                clinvar_date_match.group("year"),
                clinvar_date_match.group("month"),
                clinvar_date_match.group("day"),
            ]
        )
    else:
        if is_s3_path(vcf, s3_uri_schemes=s3_uri_schemes):
            open = __read_s3
        else:
            open = __read_fs
        with open(vcf) as in_vcf:
            for line in in_vcf:
                line = line.decode().strip()
                if not line.startswith("##"):
                    # Stop reading on the first non-header line
                    break
                line = line[2:]
                if line.startswith("fileDate"):
                    _, datestr = line.split("=", maxsplit=1)
                    return datestr
    return None


def compute_coverage_stats(
    mt,
    *,
    strata_expr: list[dict[str, object]],
    coverage_over_x_bins: list[int] = [1, 5, 10, 15, 20, 25, 30, 50, 100],
    dp_field: str = "DP",
):
    """Per-locus coverage over the rows of a dense MatrixTable.

    Aggregates mean, median, total DP and samples-over-X bins at every
    representative locus in ``mt``, stratified by ``strata_expr``. The
    resulting Table is keyed by ``locus`` with the same ``coverage_stats``
    array and ``coverage_stats_meta`` / ``coverage_stats_meta_sample_count``
    globals produced by ``gnomad.utils.sparse_mt.compute_coverage_stats``.

    Note
    ----
    Derived from:
        https://github.com/broadinstitute/gnomad_qc/blob/main/gnomad_qc/v4/annotations/compute_coverage.py#L470
        https://github.com/populationgenomics/production-pipelines/blob/main/cpg_workflows/large_cohort/compute_stats_for_all_sites.py#L740

    The stratification, coverage-bin finalization, and global-setting blocks
    are kept as close as possible to those implementations. The only intended
    difference is the input: those references densify a VDS over a
    ``reference_ht`` before aggregating, whereas this function aggregates the
    already-dense matrix produced by the preceding annotation steps.
    """
    from gnomad.utils.annotations import (
        agg_by_strata,
        generate_freq_group_membership_array,
    )
    from gnomad.utils.sparse_mt import get_coverage_agg_func

    from gnomad_meb.hail import hail as hl

    if dp_field not in mt.entry:
        raise ValueError(
            f"Entry field {dp_field!r} is missing from the input matrix. Coverage "
            "aggregation requires 'DP' on both the VDS reference_data and "
            "variant_data entries; an older VDS does not carry it. Rebuild the VDS "
            "with '--overwrite' so its entries include DP."
        )

    # Materialize the strata on columns and rebuild strata_expr against them. This
    # runs on the input matrix because the caller's strata expressions are bound to
    # it; re-keying rows below yields a new Hail source that invalidates them.
    cols = mt.annotate_cols(
        **{k: v for d in strata_expr for k, v in d.items()}
    ).cols()
    strata_expr = [{k: cols[k] for k in d} for d in strata_expr]

    group_membership_ht = generate_freq_group_membership_array(
        cols, strata_expr, no_raw_group=True
    )
    # generate_freq_group_membership_array labels every group "adj"; coverage does
    # no adj filtering, so relabel every group "raw".
    group_membership_ht = group_membership_ht.annotate_globals(
        freq_meta=group_membership_ht.freq_meta.map(
            lambda x: hl.dict(
                x.items().map(
                    lambda m: hl.if_else(m[0] == "group", ("group", "raw"), m)
                )
            )
        )
    )

    cov_bins = sorted(coverage_over_x_bins)
    rev_cov_bins = list(reversed(cov_bins))
    max_cov_bin = cov_bins[-1]
    cov_bins = hl.array(cov_bins)
    entry_agg_funcs = {
        "coverage_stats": get_coverage_agg_func(
            dp_field=dp_field, max_cov_bin=max_cov_bin
        )
    }

    # One row per locus, replacing the references' reference-table join; the dense
    # rows already are the coverage site set.
    mt = mt.key_rows_by("locus").distinct_by_row()

    ht = agg_by_strata(mt, entry_agg_funcs, group_membership_ht=group_membership_ht)
    group_globals = group_membership_ht.index_globals()
    ht = ht.annotate_globals(
        strata_meta=group_globals.freq_meta,
        strata_sample_count=group_globals.freq_meta_sample_count,
    )

    # This expression aggregates the DP counter in reverse order of the cov_bins and
    # computes the cumulative sum over them. It needs to be in reverse order because
    # we want the sum over samples covered by > X.
    def _cov_stats(
        cov_stat: hl.expr.StructExpression, n: hl.expr.Int32Expression
    ) -> hl.expr.StructExpression:
        # The coverage was already floored to the max_coverage_bin, so no more
        # aggregation is needed for the max bin.
        count_expr = cov_stat.coverage_counter
        max_bin_expr = hl.int32(count_expr.get(max_cov_bin, 0))

        # For each of the other bins, coverage is summed between the boundaries.
        bin_expr = hl.range(hl.len(cov_bins) - 1, 0, step=-1)
        bin_expr = bin_expr.map(
            lambda i: hl.sum(
                hl.range(cov_bins[i - 1], cov_bins[i]).map(
                    lambda j: hl.int32(count_expr.get(j, 0))
                )
            )
        )
        bin_expr = hl.cumulative_sum(hl.array([max_bin_expr]).extend(bin_expr))

        bin_expr = {f"over_{x}": bin_expr[i] / n for i, x in enumerate(rev_cov_bins)}

        return cov_stat.annotate(**bin_expr).drop("coverage_counter")

    # The per-stratum sample counts are read directly off ``ht`` rather than through
    # ``ht.index_globals()``: on this un-checkpointed path an index-globals reference
    # merged into the row expression trips Hail 0.2's optimizer.
    if isinstance(ht.coverage_stats, hl.expr.ArrayExpression):
        cov_stats_expr = {
            "coverage_stats": hl.map(
                lambda c, n: _cov_stats(c, n),
                ht.coverage_stats,
                ht.strata_sample_count,
            )
        }
    else:
        cov_stats_expr = _cov_stats(ht.coverage_stats, ht.strata_sample_count)

    # Finalize the rows first, then re-derive the coverage globals from the
    # transmuted table's own globals. Binding the metastratum map to the same Hail
    # source as the finalizer keeps the lowering on one source; merging a
    # foreign-source global array into the row expression trips Hail 0.2's
    # optimizer. This is the spec's finalize-then-set-globals order.
    ht = ht.transmute(**cov_stats_expr)
    ht = ht.select_globals(
        coverage_stats_meta=ht.strata_meta.map(
            lambda x: hl.dict(x.items().filter(lambda m: m[0] != "group"))
        ),
        coverage_stats_meta_sample_count=ht.strata_sample_count,
    )

    return ht


def cohort_expr(
    mt,
    table: str | None = None,
    cohort: str | None = None,
    key_col: str = "s",
    cohort_col: str = "cohort",
):
    """Returns an expression to add cohort data to a given MatrixTable.

    Arguments
    ---------
    table : str | None
            A table containing a sample to cohort matching. Takes precedence
            over `cohort` if specified.
    cohort : str | None
             Static cohort annotation.
    key_col : str
              Name of the key column in the table.
    cohort_col : str
                 Name of the cohort column in the table.

    Returns
    -------
    hl.expr.StringExpression
        An expression containing the cohort annotation for each sample in the
        MatrixTable `mt`.
    """
    from gnomad_meb.hail import hail as hl

    logger = get_logger()

    if table is not None:
        logger.debug(f"annotating cohorts based on table {table}")
        ht = hl.import_table(str(Path(table).resolve()), key=key_col)
        assert cohort_col in ht.row, (
            f"missing cohort column {cohort} in table rows, got {ht.row!s}"
        )
        return ht[mt.col_key][cohort_col]
    elif cohort is not None:
        logger.debug(f"adding static cohort annotation cohort = {cohort}")
        return cohort
    else:
        logger.warning("no source for cohort annotation specified")
        return "<unknown>"
