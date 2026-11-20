from typing import NamedTuple

from gnomad_meb.hail import hail as hl
from gnomad_meb.logging import get_logger
from gnomad_meb.lib.variant_qc.constants import (
    REQUIRED_ALLELE_SPECIFIC_ANNOTATIONS,
    REQUIRED_ANNOTATIONS,
)


class AlleleDataFieldConfig(NamedTuple):
    GT: str = "LGT"
    ALLELES: str = "LA"
    AD: str = "LAD"
    PL: str = "LPL"
    DP: str = "DP"


def correct_variant_annotations(
    mt: hl.MatrixTable,
    infer_as_annotations: bool = True,
    fields: AlleleDataFieldConfig = AlleleDataFieldConfig(),
) -> hl.MatrixTable:
    """Infer AS annotations from known values and correct all other fields to adhere to VariantQC requirements.
    
    Returns
    -------
    hl.MatrixTable
        A matrix table containing all fields required to perform VariantQC on the dataset.
    """
    logger = get_logger()
    import gnomad_qc.v4.annotations.generate_variant_qc_annotations as v4ag

    mt = mt.annotate_rows(
        alt_alleles_range_array=hl.range(1, hl.len(mt.alleles))
    )
    mt = mt.filter_rows(
        (hl.len(mt.alleles) > 1) & (hl.agg.any(mt[fields.GT].is_non_ref()))
    )
    mt = mt.annotate_rows(
        site_dp=hl.agg.sum(mt[fields.DP]),
        ANS=hl.agg.count_where(hl.is_defined(mt[fields.GT])) * 2,
    )
    
    if all(
        x in mt.gvcf_info
        for x in [*REQUIRED_ANNOTATIONS, *REQUIRED_ALLELE_SPECIFIC_ANNOTATIONS]
    ):
        pass
    elif infer_as_annotations:
        logger.warning(
            "missing some gvcf annotations, overriding values with inferred "
            "annotations"
        )
        target_allele = (
            mt[fields.ALLELES][1]
            if fields.ALLELES == "LAD"
            else hl.if_else(mt[fields.GT].ploidy == 1, mt[fields.GT][0], mt[fields.GT][1])
        )
        mt = mt.annotate_entries(
            gvcf_info=mt.gvcf_info.annotate(
                QUALapprox=mt[fields.PL][0],
                VarDP=mt[fields.DP],
                AS_QUALapprox=hl.range(hl.len(mt[fields.ALLELES])).map(
                    lambda x: hl.if_else(x == 0, mt[fields.PL][x], 0)
                ),
                AS_SB_TABLE=hl.range(hl.len(mt[fields.ALLELES])).map(
                    lambda x: hl.if_else(
                        2 * x + 1 < hl.len(mt.SB),
                        [mt.SB[2 * x], mt.SB[2 * x + 1]],
                        [0, 0],
                    )
                ),
                AS_RAW_MQ=hl.range(hl.len(mt[fields.ALLELES])).map(
                    lambda x: hl.if_else(
                        x == 1,
                        hl.if_else(
                            mt.gvcf_info.RAW_MQandDP[1] > 0,
                            hl.sqrt(
                                mt.gvcf_info.RAW_MQandDP[0]
                                / mt.gvcf_info.RAW_MQandDP[1]
                            ),
                            0,
                        ),
                        0,
                    )
                ),
                AS_RAW_ReadPosRankSum=hl.range(hl.len(mt[fields.ALLELES])).map(
                    lambda x: hl.if_else(
                        x == target_allele, [mt.gvcf_info.ReadPosRankSum], [0]
                    )
                ),
                AS_RAW_MQRankSum=hl.range(hl.len(mt[fields.ALLELES])).map(
                    lambda x: hl.if_else(
                        x == target_allele, [mt.gvcf_info.MQRankSum], [0]
                    )
                ),
                AS_VarDP=mt[fields.AD],
            ),
        )
    else:
        missing_annotations = {
            *REQUIRED_ANNOTATIONS,
            *REQUIRED_ALLELE_SPECIFIC_ANNOTATIONS,
        } - set(mt.gvcf_info.keys())
        logger.error(
            f"missing gvcf annotations {missing_annotations}, and "
            "infer_as_annotations = False"
        )
        raise ValueError(f"missing annotations: missing = {missing_annotations}")

    correct_mt = mt.annotate_entries(
        gvcf_info=mt.gvcf_info.annotate(
            AS_QUALapprox=v4ag.recompute_as_qualapprox_from_lpl(mt),
            **v4ag.correct_as_annotations(mt),
        )
    )
    return correct_mt

def compute_info_table(
    mt: hl.MatrixTable,
    retain_cdfs: bool = False,
    cdf_k: int = 200,
):
    """Compute the info annotations for each row in the table.
    
    This function uses `default_compute_info` to annotate allele and site
    specific info values for each row in the input matrix table `mt` and
    applies some basic postprocessing.

    Returns
    -------
    hl.Table
        Hail info table containing site and allele specific info fields.
    """
    from gnomad.utils.sparse_mt import (
        default_compute_info,
    )

    quasi_info_ht = default_compute_info(
        mt,
        as_annotations=True,
        site_annotations=True,
        n_partitions=None,
        retain_cdfs=retain_cdfs,
        cdf_k=cdf_k,
    )
    quasi_info_ht = quasi_info_ht.annotate(
        info=quasi_info_ht.info.annotate(
            DB=mt.rows()[quasi_info_ht.key].site_dp
        )
    )
    as_info = quasi_info_ht.info
    as_keys = hl.eval(
        hl.array(list(as_info.keys())).group_by(
            lambda x: (
                hl.switch(x[:2])
                .when("AS", "AS_info")
                .when("AC", "AC_info")
                .default("site_info")
            )
        )
    )
    quasi_info = {k: as_info.select(*v) for k, v in as_keys.items()}
    info_ht = quasi_info_ht.select(
        AS_info=quasi_info["AS_info"],
        AC_info=quasi_info["AC_info"],
        site_info=quasi_info["site_info"],
        quasi_info=quasi_info_ht["quasi_info"],
        lowqual=quasi_info_ht.lowqual,
        AS_lowqual=quasi_info_ht.AS_lowqual,
    )
    return info_ht


def build_info_table(
    mt: hl.MatrixTable,
    infer_as_annotations: bool = True,
    retain_cdfs: bool = False,
    cdf_k: int = 200,
    fields: AlleleDataFieldConfig = AlleleDataFieldConfig(),
) -> hl.Table:
    """Convert the variant data from the matrix table to a info ht.

    Performs similar filtering steps to generate a site-only VCF as proposed by
    the two refererence implementations, skipping the three different methods
    separation from the gnomAD pipeline:

        https://github.com/broadinstitute/gnomad_methods/blob/main/gnomad/utils/sparse_mt.py
        https://github.com/populationgenomics/production-pipelines/blob/94e4d06cf51425141e7955213e1ef807957a232d/cpg_workflows/large_cohort/site_only_vcf.py

    Arguments
    ---------
    vds : hl.vds.VariantDataset
        Input variant dataset.
    infer_as_annotations : bool
                        If set and the Allele-Specific (AS) annotations are
                        missing from the original dataset, we infer them as
                        best as possible from the defined values.
    retain_cds : bool
                Whether the CDF's will be included in the final info hail table.
    cdf_k : int
            ...

    Returns
    -------
    hl.Table
        Hail table containing the annotations `site_info`, `quasi_info`,
        `AS_info`, `AC_info`, `lowqual` and `AS_lowqual`.
    """
    correct_mt = correct_variant_annotations(mt, infer_as_annotations=infer_as_annotations, fields=fields)
    return compute_info_table(correct_mt, retain_cdfs=retain_cdfs, cdf_k=cdf_k)


def select_info_table(
    ht: hl.Table,
    extra_field: str,
) -> hl.Table:
    """Select a specific info field from the meta info table.
    
    By default, there are allele specific (AS), site specific and general (quasi) info fields,
    collected into the meta info table (cf. `build_info_table`). This function selects and 
    normalizes the info field to one of these fields under `extra_field`.

    Note
    ----
    Lifted from: cpg_workflows.large_cohort.site_only_vcf.build_info_ht(ht: hl.Table, extra_field: str) -> hl.Table:  # noqa: E501

    Arguments
    ---------
    ht : hl.Table
        Meta info table containing allele specific, site specific and quasi info fields.
    extra_field : str
        Name of the quasi info field to select. Should be one of `AS_info` or `quasi_info`.

    Returns
    -------
    hl.Table
        Hail Table containing a normalized and selected info annotation.
    """
    from gnomad.utils.vcf import adjust_vcf_incompatible_types

    ht = ht.select("AC_info", extra_field, "site_info", "lowqual", "AS_lowqual")
    ht = ht.annotate(
        info=hl.struct(**ht.AC_info, **ht[extra_field], **ht.site_info),
    )
    ht = ht.drop(
        "AC_info",
        extra_field,
        "site_info",
    )

    ht = adjust_vcf_incompatible_types(
        ht,
        # With default INFO_VCF_AS_PIPE_DELIMITED_FIELDS, AS_VarDP will be
        # converted into a pipe-delimited value e.g.: VarDP=|132.1|140.2
        # which breaks VQSR parser (it doesn't recognise the delimiter and
        # treats it as an array with a single string value "|132.1|140.2",
        # leading to an IndexOutOfBound exception when trying to access
        # value for second allele)
        pipe_delimited_annotations=[],
    )
    return ht


def split_info(info_ht: hl.Table) -> hl.Table:
    """
    Splits multi-allelic sites in the provided info Table.

    Note
    ----
    This function was lifted from https://github.com/populationgenomics/production-pipelines/blob/main/cpg_workflows/large_cohort/load_vqsr.py#L21,  # noqa: E501
    who adapted it from the `gnomad_qc` function `gnomad_qc.v4.annotations.generate_variant_qc_annotations.py` # noqa
    multi-allelic sites in the info Table.

    The `annotate_allele_info` function from `gnomad_methods` is used to split
    multi-allelic sites before splitting the `info` annotation. This ensures
    that all sites in the returned Table are properly annotated with
    allele-specific information.

    Parameters
    ----------
    info_ht : hl.Table
        Info Table containing unsplit multi-allelic sites.

    Returns
    -------
    hl.Table
        Info Table with multi-allelic sites split into bi-allelic sites.
    """
    from gnomad.utils.annotations import annotate_allele_info
    from gnomad.utils.sparse_mt import (
        split_info_annotation,
        split_lowqual_annotation,
    )

    info_ht = annotate_allele_info(info_ht)
    info_annotations_to_split = ["info"]
    info_ht = info_ht.annotate(
        **{
            a: info_ht[a].annotate(
                **split_info_annotation(info_ht[a], info_ht.a_index),
            )
            for a in info_annotations_to_split
        },
        AS_lowqual=split_lowqual_annotation(
            info_ht.AS_lowqual, info_ht.a_index
        ),
    )

    return info_ht