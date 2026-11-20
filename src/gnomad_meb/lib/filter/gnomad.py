from collections.abc import Iterable

from gnomad_meb.hail import hail as hl


def filter_variants_gnomad(
    mt: hl.MatrixTable,
    min_inbreeding_coef: float = -0.3,
    min_high_quality_calls: int = 1,
    apply_sample_filters: bool = True,
    sample_filters_ignore: Iterable[str] | None = None,
):
    """Filter the matrix table to high-quality variants according to gnomAD.

    Applies the variant-QC and sample-QC results and returns only non-filtered
    rows. Variants carrying a filter are removed, as are variants with fewer
    than ``min_high_quality_calls`` alleles in the ``adj`` freq field and those
    failing ``min_inbreeding_coef``. Samples are removed when ``hard_filters``
    is defined and non-empty, unless ``apply_sample_filters`` is false (keep
    every sample) or the only filters present are in ``sample_filters_ignore``.
    """
    adj_index = hl.eval(mt.freq_meta).index({"group": "adj"})
    vqsr_expr = (~hl.is_defined(mt.filters) | (hl.len(mt.filters) == 0))
    high_qual_expr = (mt.freq[adj_index].AC >= min_high_quality_calls)
    inbreeding_expr = (mt.inbreeding_coef > min_inbreeding_coef)

    # Compute and report filter statistics before application of any variant
    # filter.
    variant_info=hl.struct(
        cutoffs=hl.struct(
            min_inbreeding_coef=min_inbreeding_coef,
            min_high_quality_calls=min_high_quality_calls,
        ),
        n_variants=hl.struct(
            default=mt.count_rows(),
            **mt.aggregate_rows(hl.struct(
                vqsr=hl.agg.count_where(vqsr_expr),
                high_quality=hl.agg.count_where(high_qual_expr),
                inbreeding=hl.agg.count_where(inbreeding_expr),
                final=hl.agg.count_where(
                    vqsr_expr
                    & high_qual_expr
                    & inbreeding_expr
                ),
            )),
        ),
    )
    
    mt = mt.filter_rows(
        vqsr_expr
        & high_qual_expr
        & inbreeding_expr
    )

    ignored_filters = set(sample_filters_ignore or ())
    if ignored_filters:
        sample_filters_ignore_expr = hl.literal(ignored_filters)
        hard_filters_expr = hl.is_defined(mt.hard_filters) & (
            hl.len(mt.hard_filters.difference(sample_filters_ignore_expr)) == 0
        )
    else:
        sample_filters_ignore_expr = hl.missing(hl.tset(hl.tstr))
        hard_filters_expr = hl.is_defined(mt.hard_filters) & (
            hl.len(mt.hard_filters) == 0
        )

    sample_qc_expr = (
        hl.literal(not apply_sample_filters)
        | (~hl.is_defined(mt.hard_filters) | hard_filters_expr)
    )
    sample_info=hl.struct(
        sample_info=hl.struct(
            apply_sample_filters=hl.literal(apply_sample_filters),
            sample_filters_ignore=sample_filters_ignore_expr,
        ),
        n_samples=hl.struct(
            default=mt.count_cols(),
            **mt.aggregate_cols(hl.struct(
                sample_qc=hl.agg.count_where(sample_qc_expr),
                final=hl.agg.count_where(sample_qc_expr),
            )),
        ),
    )
    
    mt = mt.filter_cols(sample_qc_expr)
    mt = mt.annotate_globals(
        filter_high_quality_gnomad=hl.struct(
            variant_info=variant_info,
            sample_info=sample_info,
        ),
    )

    return mt, dict(**hl.eval(variant_info)), dict(**hl.eval(sample_info))


def annotate_lof_constraint(
    mt: hl.MatrixTable,
    constraints_ht: hl.Table,
):
    """Index the gnomad constraints ht for the given matrix table mt.
    
    Returns
    -------
    mt : hl.MatrixTable
        MatrixTable with an added key `constraints` that contains the scores from
        the constraints hl.Table object.
    """
    mt = mt.annotate_rows(
        constraints=hl.struct(**constraints_ht[(
            mt.vep.worst_csq_for_variant_canonical.gene_symbol,
            mt.vep.worst_csq_for_variant_canonical.gene_id,
            mt.vep.worst_csq_for_variant_canonical.transcript_id,
            hl.is_defined(mt.vep.worst_csq_for_variant_canonical.canonical),
            hl.is_defined(mt.vep.worst_csq_for_variant_canonical.mane_select),
        )]),
    )
    return mt