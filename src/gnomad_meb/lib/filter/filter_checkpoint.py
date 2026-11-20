from typing import Any

from gnomad_meb.hail import get_default_partitions
from gnomad_meb.hail import hail as hl
from gnomad_meb.lib.filter.config import FilterCheckpoint


def hail_struct_to_dict(val):
    """Recursively convert Hail Structs (or evaluated results) into standard Python dicts."""
    # If it's a Hail Struct or has .items() / fields, convert it
    def _coerce_key_dtype(key: Any):
        if key is None:
            return "None"
        if isinstance(key, (bool,)):
            return str(key)
        return key
    if hasattr(val, "items"):
        return {_coerce_key_dtype(k): hail_struct_to_dict(v) for k, v in val.items()}
    elif isinstance(val, list):
        return [hail_struct_to_dict(x) for x in val]
    else:
        return val


def gnomad_freq_meta_key(gnomad_group: str | None) -> dict[str, str]:
    """Build the ``gnomad_freq_meta`` index key for ``gnomad_group``.

    The requested ancestry group is used verbatim; ``None`` falls back to the
    reference group ``nfe``. This is the only place the lookup key is built, so
    the group passed to :func:`filter_variants_by_af` actually takes effect.
    """
    return {"gen_anc": "nfe" if gnomad_group is None else gnomad_group, "group": "adj"}


def filter_variants_by_af(
    mt: hl.MatrixTable,
    min_gnomad_af: float | None = None,
    max_gnomad_af: float | None = None,
    filter_snps: bool = False,
    gnomad_group: str | None = None,
) -> tuple[hl.MatrixTable, hl.expr.StructExpression]:

    gnomad_af_idx = hl.eval(mt.gnomad_freq_meta).index(
        gnomad_freq_meta_key(gnomad_group)
    )
    gnomad_af = mt.gnomad_freq[gnomad_af_idx].AF

    gnomad_af_expr = hl.literal(True)
    if min_gnomad_af is not None:
        gnomad_af_expr = gnomad_af_expr & (gnomad_af >= min_gnomad_af)
    if max_gnomad_af is not None:
        gnomad_af_expr = gnomad_af_expr & (gnomad_af < max_gnomad_af)

    if filter_snps:
        filter_snps_expr = (mt.allele_info.variant_type == "snv")
    else:
        filter_snps_expr = hl.literal(True)

    variant_info = hl.struct(
        cutoffs=hl.struct(
            min_gnomad_af=(
                min_gnomad_af if min_gnomad_af is not None else hl.missing("float64")
            ),
            max_gnomad_af=(
                max_gnomad_af if max_gnomad_af is not None else hl.missing("float64")
            ),
        ),
        n_variants=hl.struct(
            default=mt.count_rows(),
            **mt.aggregate_rows(hl.struct(
                gnomad=hl.agg.count_where(gnomad_af_expr),
                snps=hl.agg.count_where(filter_snps_expr),
                final=hl.agg.count_where(gnomad_af_expr & filter_snps_expr),
            ))
        ),
    )

    mt = mt.filter_rows(gnomad_af_expr & filter_snps_expr)
    mt = mt.repartition(get_default_partitions())

    return mt, variant_info


def filter_lof_variants(
    mt: hl.MatrixTable,
    min_z_score: float | None = 5.0,
    min_pLI: float | None = 0.9,
    min_cadd: float | None = 20.0,
    impact: set | None = None,
) -> tuple[hl.MatrixTable, hl.expr.StructExpression]:
    """Filter for potentially disease causing rare LoF variants.

    This filter applies the gnomAD constraint metrics pLI for LoF
    variants and the Z-score for missense variants to select variants
    that are likely to be damaging. By default, we use a cutoff of
    5 for missense z-scores and 0.9 for LoF pLI Scores.

    A variant passes the constraint test when it is *either* LoF (its pLI meets
    ``min_pLI``) *or* missense (its z-score meets ``min_z_score``) -- the
    ``lof_expr | mis_expr`` union, not the intersection -- and additionally
    meets the CADD cutoff and falls in the ``impact`` set.

    Additionally, we filter these variants further to those with
    high cadd scores to reduce the filtered variant subset.

    ``impact`` defaults to the high-impact transcript consequences
    ``{"HIGH", "MODERATE"}`` when unset; pass an explicit set to override.
    """
    if impact is None:
        impact = {"HIGH", "MODERATE"}

    if min_z_score is not None:
        mis_expr = (mt.constraints.mis.z_score >= min_z_score)
    else:
        mis_expr = hl.literal(True)

    if min_pLI is not None:
        lof_expr = (mt.constraints.lof.pLI >= min_pLI)
    else:
        lof_expr = hl.literal(True)

    if min_cadd is not None:
        cadd_expr = (mt.in_silico_predictors.cadd.phred >= min_cadd)
    else:
        cadd_expr = hl.literal(True)

    impact_expr = (
        hl.literal(impact).contains(mt.vep.worst_csq_for_variant_canonical.impact)
    )

    # Union of the LoF and missense constraint tests (a variant need only meet
    # one), then narrowed by CADD and impact.
    constraint_expr = ((lof_expr | mis_expr) & cadd_expr & impact_expr)

    variant_info = hl.struct(
        cutoffs=hl.struct(
            min_pli=(
                min_pLI if min_pLI is not None else hl.missing("float64")
            ),
            min_z_score=(
                min_z_score if min_z_score is not None else hl.missing("float64")
            ),
            min_cadd=(
                min_cadd if min_cadd is not None else hl.missing("float64")
            ),
            impact=impact,
        ),
        n_variants=hl.struct(
            default=mt.count_rows(),
            **mt.aggregate_rows(hl.struct(
                lof=hl.agg.count_where(lof_expr),
                mis=hl.agg.count_where(mis_expr),
                cadd=hl.agg.count_where(cadd_expr),
                impact=hl.agg.count_where(impact_expr),
                final=hl.agg.count_where(constraint_expr),
            )),
        ),
    )

    mt = mt.filter_rows(constraint_expr)
    mt = mt.repartition(get_default_partitions())

    return mt, variant_info


def filter_clinvar_pathogenic(
    mt: hl.MatrixTable,
    pathogenic_expr: hl.expr.SetExpression | hl.tset = hl.literal({  # noqa: B008
        "Likely_pathogenic",
        "Likely_pathogenic|protective",
        # 'Likely_risk_allele' includes intronic variants with low CADD etc.,
        # so very low penetrance.
        "Pathogenic",
        "Pathogenic/Likely_pathogenic",
        "Pathogenic/Pathogenic",
        "Pathogenic|risk_factor",
    }),
) -> tuple[hl.MatrixTable, hl.expr.StructExpression]:
    """Filter for clinvar pathogenic and likely pathogenic variants.

    Parameters
    ----------
    mt : hl.MatrixTable
         Input matrix table.
    pathogenic_expr: hl.expr.SetExpression | hl.set
                     SetExpression containing all possible pathogenic variant
                     significance annotations to filter for.

    Returns
    -------
    mt : hl.MatrixTable
    """
    clinvar_expr = (
        hl.is_defined(mt.clinvar.clinical_significance)
        & (
            hl.len(
                hl.set(mt.clinvar.clinical_significance).intersection(
                    pathogenic_expr
                )
            )
            > 0
        )
    )

    variant_info = hl.struct(
        cutoffs=hl.struct(
            clinvar=hl.eval(pathogenic_expr),
        ),
        n_variants=hl.struct(
            default=mt.count_rows(),
            **mt.aggregate_rows(hl.struct(
                clinvar=hl.agg.count_where(clinvar_expr),
                final=hl.agg.count_where(clinvar_expr),
            )),
        ),
    )

    mt = mt.filter_rows(clinvar_expr)
    mt = mt.repartition(get_default_partitions())

    return mt, variant_info


def filter_splice_variants(
    mt: hl.MatrixTable,
    csq: set = {"splice_acceptor_variant", "splice_donor_variant"},
) -> tuple[hl.MatrixTable, hl.expr.StructExpression]:
    """Filter variants for HIGH impact splice variants.
    
    Parameters
    ----------
    mt : hl.MatrixTable
        A matrix table containing the variants to filter.
    csq : set
        A set of strings to consider as splice variants. These strings are
        expected to be splice site VEP consequence annotations, which will be
        searched for in `mt`.

    Returns
    -------
    mt : hl.MatrixTable
    """
    splice_hl = hl.literal(csq)
    splice_expr = (
        hl.is_defined(
            mt.vep.worst_csq_for_variant_canonical.most_severe_consequence
        )
        & splice_hl.contains(
            mt.vep.worst_csq_for_variant_canonical.most_severe_consequence
        )
    )

    variant_info = hl.struct(
        cutoffs=hl.struct(
            csq=splice_hl,
        ),
        n_variants=hl.struct(
            default=mt.count_rows(),
            **mt.aggregate_rows(hl.struct(
                splice=hl.agg.count_where(splice_expr),
                final=hl.agg.count_where(splice_expr),
            )),
        ),
    )

    mt = mt.filter_rows(splice_expr)
    mt = mt.repartition(get_default_partitions())

    return mt, variant_info


def apply_filter_checkpoint(
    mt: hl.MatrixTable,
    *,
    flt: FilterCheckpoint,
) -> tuple[hl.MatrixTable, dict[str, Any]]:
    """Apply one configured filter's sub-filters in order."""
    variant_info = hl.struct(filters=hl.empty_array(hl.tstr))
    # Gate on the fields that actually drive the AF sub-filter -- its bounds and
    # the SNP toggle. `ancestry_group` only selects *which* frequency to read,
    # and is None by default, so it must not gate the step.
    if flt.gnomad_af_conditions():
        mt, gnomad_filter_s = filter_variants_by_af(
            mt,
            min_gnomad_af=flt.min_gnomad_af,
            max_gnomad_af=flt.max_gnomad_af,
            filter_snps=flt.filter_snps,
            gnomad_group=flt.ancestry_group,
        )
        variant_info = variant_info.annotate(
            filters=variant_info.filters.append("gnomad_freq"),
            gnomad_freq=gnomad_filter_s,
        )

    if any(
        x is not None
        for x in [flt.min_z_score, flt.min_pLI, flt.min_cadd, flt.impact]
    ):
        mt, lof_filter_s = filter_lof_variants(
            mt,
            min_z_score=flt.min_z_score,
            min_pLI=flt.min_pLI,
            min_cadd=flt.min_cadd,
            impact=flt.impact,
        )
        variant_info = variant_info.annotate(
            filters=variant_info.filters.append("lof"),
            lof=lof_filter_s,
        )

    if flt.filter_clinvar_pathogenic:
        mt, clinvar_filter_s = filter_clinvar_pathogenic(
            mt
        )
        variant_info = variant_info.annotate(
            filters=variant_info.filters.append("clinvar"),
            clinvar=clinvar_filter_s,
        )

    if flt.filter_splice_variants:
        mt, splice_filter_s = filter_splice_variants(mt)
        variant_info = variant_info.annotate(
            filters=variant_info.filters.append("splice_variants"),
            splice_variants=splice_filter_s,
        )

    mt = mt.annotate_globals(filter_variant_info=variant_info)

    variant_info_d = hl.eval(variant_info)
    variant_info_d = hail_struct_to_dict(variant_info_d)
    return mt, variant_info_d
