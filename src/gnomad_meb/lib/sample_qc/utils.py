from typing import TYPE_CHECKING

from gnomad_meb.logging import get_logger

if TYPE_CHECKING:
    from gnomad_meb.hail import hail as hl

logger = get_logger()


def filter_ambiguous_samples(ploidy_ht, f_stat_cutoff: float):
    """Restrict the table used to derive sex-karyotype cutoffs.

    The ploidy and hom-alt cutoffs must not be set by a single intermediate
    (aneuploid or contaminated) sample, so the derivation aggregate is computed
    over a screened subset while every sample still receives a karyotype.

    Samples are split into roughly 'XX' (``f_stat < f_stat_cutoff``) and 'XY' (above),
    the median ``chrx_frac_hom_alt_adj`` of each group is aggregated into Python
    floats, and a sample is kept only when its own fraction is *not* strictly
    between the two medians. A sample with an undefined fraction cannot be judged
    concordant and is left out of the derivation set.

    Parameters
    ----------
    ploidy_ht : hl.Table
        Stage-1 table of per-sample sex-chromosome ploidies, f-stat and chrX
        hom-alt fraction.
    f_stat_cutoff : float
        f-stat used to split samples into roughly 'XX' and 'XY'.

    Returns
    -------
    tuple
        ``(derivation_ht, band)``: the subset of ``ploidy_ht`` used to derive
        cutoffs, and the ``(lower, upper)`` median pair that defines the
        ambiguous band (exported as the ``screening_band`` global).
    """
    from gnomad_meb.hail import hail as hl

    group = hl.if_else(ploidy_ht.f_stat < f_stat_cutoff, "XX", "XY")
    # Hail is lazy: materialise the two medians as Python floats before building
    # the concordance filter, so the band is a plain scalar comparison.
    medians = ploidy_ht.aggregate(
        hl.agg.group_by(group, hl.agg.approx_median(ploidy_ht.chrx_frac_hom_alt_adj))
    )
    if "XX" not in medians or "XY" not in medians:
        raise ValueError(
            "cannot screen samples without both groups; f-stat groups present: "
            f"{sorted(medians)} (f_stat_cutoff={f_stat_cutoff})"
        )

    xx_median = float(medians["XX"])
    xy_median = float(medians["XY"])
    lower, upper = sorted((xx_median, xy_median))

    logger.info(
        f"removing ambiguous samples from sex karyotype cutoff derivation: chrx_frac_hom_alt_adj = [{lower}, {upper}], XX = {xx_median}, XY = {xy_median}"
    )

    concordant = (ploidy_ht.chrx_frac_hom_alt_adj <= lower) | (
        ploidy_ht.chrx_frac_hom_alt_adj >= upper
    )
    return ploidy_ht.filter(concordant), (lower, upper)


def infer_sex_karyotype(
    ploidy_ht,
    *,
    f_stat_cutoff: float,
    normal_ploidy_cutoff: int = 5,
    aneuploidy_cutoff: int = 6,
    x_ploidy_cutoffs: tuple[float, tuple[float, float], float] | None = None,
    y_ploidy_cutoffs: tuple[tuple[float, float], float] | None = None,
    chr_x_frac_hom_alt_cutoffs: tuple[tuple[float, float], float] | None = None,
):
    """Infer sex karyotype given the sample ploidy hail table.

    Ports gnomad.sample_qc.pipeline.infer_sex_karyotype with additional
    settings and refinements to handle aneuploid samples in smaller
    cohorts as the default gnomad cohorts.

    The X/Y ploidy cutoffs and the chrX hom-alt cutoffs are derived from a
    screened subset (see :func:`filter_ambiguous_samples`) while the karyotype
    is assigned to every sample. All cutoffs, the screening rule and the
    ambiguous band are exported as table globals.

    When ``x_ploidy_cutoffs``, ``y_ploidy_cutoffs`` and
    ``chr_x_frac_hom_alt_cutoffs`` are all supplied (pinned reference cutoffs),
    they are used as-is and screening is skipped; the globals report the pinned
    values, and ``screening_band`` is missing because no band was derived. They
    must be supplied together or not at all. The f-stat grouping and reporting
    are unchanged either way.

    Parameters
    ----------
    ploidy_ht : hl.Table
        Stage-1 table of per-sample ``chrX_ploidy``, ``chrY_ploidy``, ``f_stat``
        and ``chrx_frac_hom_alt_adj``.
    f_stat_cutoff : float
        f-stat to roughly divide 'XX' from 'XY' samples for cutoff derivation.
    normal_ploidy_cutoff : int
        Number of standard deviations for XX/XY ploidy cutoffs.
    aneuploidy_cutoff : int
        Number of standard deviations for sex aneuploidy ploidy cutoffs.
    x_ploidy_cutoffs : tuple[float, tuple[float, float], float]
        Pinned ``(upper_cutoff_X, (lower_cutoff_XX, upper_cutoff_XX),
        lower_cutoff_XXX)``, or ``None`` to derive.
    y_ploidy_cutoffs : tuple[tuple[float, float], float]
        Pinned ``((lower_cutoff_Y, upper_cutoff_Y), lower_cutoff_YY)``, or
        ``None`` to derive.
    chr_x_frac_hom_alt_cutoffs : tuple[tuple[float, float], float]
        Pinned ``((lower_cutoff_more_than_one_X, upper_cutoff_more_than_one_X),
        lower_cutoff_single_X)``, or ``None`` to derive.

    Returns
    -------
    A table of samples' imputed sex karyotype (or "ambiguous" if this cant be
    decided), annotated with the derived cutoffs as globals.
    """
    from gnomad.sample_qc.sex import (
        get_chr_x_hom_alt_cutoffs,
        get_ploidy_cutoffs,
        get_sex_expr,
    )

    from gnomad_meb.hail import hail as hl

    logger.info("inferring sex karyotype")

    pinned = (x_ploidy_cutoffs, y_ploidy_cutoffs, chr_x_frac_hom_alt_cutoffs)
    if any(cutoff is not None for cutoff in pinned) and not all(
        cutoff is not None for cutoff in pinned
    ):
        raise ValueError(
            "pinned cutoffs must be supplied together: x_ploidy_cutoffs, "
            "y_ploidy_cutoffs and chr_x_frac_hom_alt_cutoffs (all or none)"
        )

    if x_ploidy_cutoffs is not None:
        logger.info("using pinned sex karyotype cutoffs; skipping screening")
        # Screening never ran, so the band is unknown; keep a stable global
        # schema by emitting missing floats rather than omitting the field.
        screening_band = (hl.missing(hl.tfloat64), hl.missing(hl.tfloat64))
    else:
        derivation_ht, screening_band = filter_ambiguous_samples(
            ploidy_ht, f_stat_cutoff
        )

        x_ploidy_cutoffs, y_ploidy_cutoffs = get_ploidy_cutoffs(
            derivation_ht,
            f_stat_cutoff=f_stat_cutoff,
            normal_ploidy_cutoff=normal_ploidy_cutoff,
            aneuploidy_cutoff=aneuploidy_cutoff,
        )
        chr_x_frac_hom_alt_cutoffs = get_chr_x_hom_alt_cutoffs(
            derivation_ht,
            derivation_ht.chrx_frac_hom_alt_adj,
            f_stat_cutoff=f_stat_cutoff,
        )

    # Annotate rather than select: every Stage-1 column (chrX_ploidy,
    # chrY_ploidy, f_stat, chrx_frac_hom_alt*) must survive for
    # get_combined_sample_qc_table, which nests all non-key ploidy fields.
    karyotype_ht = ploidy_ht.annotate(
        **get_sex_expr(
            ploidy_ht.chrX_ploidy,
            ploidy_ht.chrY_ploidy,
            x_ploidy_cutoffs,
            y_ploidy_cutoffs,
            chr_x_frac_hom_alt_expr=ploidy_ht.chrx_frac_hom_alt_adj,
            chr_x_frac_hom_alt_cutoffs=chr_x_frac_hom_alt_cutoffs,
        )
    )

    return karyotype_ht.annotate_globals(
        normal_ploidy_cutoff=normal_ploidy_cutoff,
        aneuploidy_cutoff=aneuploidy_cutoff,
        f_stat_cutoff=f_stat_cutoff,
        screening_band=hl.struct(
            lower=screening_band[0],
            upper=screening_band[1],
        ),
        x_ploidy_cutoffs=hl.struct(
            upper_cutoff_X=x_ploidy_cutoffs[0],
            lower_cutoff_XX=x_ploidy_cutoffs[1][0],
            upper_cutoff_XX=x_ploidy_cutoffs[1][1],
            lower_cutoff_XXX=x_ploidy_cutoffs[2],
        ),
        y_ploidy_cutoffs=hl.struct(
            lower_cutoff_Y=y_ploidy_cutoffs[0][0],
            upper_cutoff_Y=y_ploidy_cutoffs[0][1],
            lower_cutoff_YY=y_ploidy_cutoffs[1],
        ),
        x_frac_hom_alt_cutoffs=hl.struct(
            lower_cutoff_more_than_one_X=chr_x_frac_hom_alt_cutoffs[0][0],
            upper_cutoff_more_than_one_X=chr_x_frac_hom_alt_cutoffs[0][1],
            lower_cutoff_single_X=chr_x_frac_hom_alt_cutoffs[1],
        ),
    )


def get_related_samples(
    king_mt: "hl.MatrixTable", kin_threshold: float = 0.1
) -> "hl.Table":
    from gnomad_meb.hail import hail as hl

    s1 = next(iter(king_mt.row_key))
    s2 = next(iter(king_mt.col_key))

    related_pairs_ht = king_mt.entries()
    related_pairs_ht = related_pairs_ht.filter(
        (related_pairs_ht.phi > kin_threshold)
        & (related_pairs_ht[s1] != related_pairs_ht[s2])
    )
    related_ht = hl.maximal_independent_set(
        related_pairs_ht[s1], related_pairs_ht[s2], keep=False
    )
    related_ht = related_ht.key_by()
    related_ht = related_ht.select(s=related_ht.node)
    # ``run_pca_with_relateds`` looks samples up as ``related[mt.col_key]``, so
    # the result must be keyed by ``s``, not left unkeyed.
    return related_ht.key_by("s")


def get_king_pairs(king_mt: "hl.MatrixTable") -> "hl.Table":
    from gnomad_meb.hail import get_default_partitions

    row_key = next(iter(king_mt.row_key))
    col_key = next(iter(king_mt.col_key))
    entries = king_mt.entries()
    king_pairs_ht = entries.select(
        i=entries[col_key], j=entries[row_key], kin=entries.phi
    )
    king_pairs_ht = king_pairs_ht.filter(king_pairs_ht.i < king_pairs_ht.j)
    king_pairs_ht = king_pairs_ht.key_by("i", "j").repartition(get_default_partitions())
    return king_pairs_ht
