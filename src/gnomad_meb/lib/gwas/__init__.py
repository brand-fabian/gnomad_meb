"""The GWAS pipeline package: export, run and load stages.

Importing this package must not initialize hail or Snakemake; the stage
pipelines register with the factory, and hail is reached only through lazy
method bodies.
"""

from gnomad_meb.lib.gwas.constants import (
    AAF,
    ANNOTATION,
    COVARIATES,
    EFFECT_MASKS,
    INTERMEDIATE_PREFIX,
    PHENOTYPE,
    PLINK_REFERENCE_NAME,
    REGENIE_VERSION_FILENAME,
    TABLE_NAMES,
    VARIANT_CLASS_ORDER,
    VARIANT_SETS,
)
from gnomad_meb.lib.gwas.export import GwasExportPipeline
from gnomad_meb.lib.gwas.inventory import GwasInventory, GwasTable
from gnomad_meb.lib.gwas.load import GwasLoadPipeline
from gnomad_meb.lib.gwas.metadata import (
    GwasDatasetMetadata,
    GwasMetadata,
    GwasMetadataException,
)
from gnomad_meb.lib.gwas.names import DatasetKey
from gnomad_meb.lib.gwas.run import GwasRunPipeline
from gnomad_meb.lib.gwas.utils import (
    RegenieDataset,
    build_covariate_exprs,
    define_plink_reference,
    get_aaf_ht,
    get_annotation_ht,
    get_covariate_ht,
    get_effects_ht,
    get_freq_expr,
    get_gene_region_expr,
    get_impact_expr,
    get_phenotype_ht,
    get_regenie_input,
    get_region_name,
    get_sex_cat_expr,
    get_variant_classification,
    get_variant_sets_ht,
    get_weight_expr,
    region_label,
    resolve_iid_fid,
    snake_case,
    str_to_expr,
    to_hail_literal_value,
    variant_id_expr,
)
from gnomad_meb.lib.gwas.workflow import GwasWorkflowPipeline

__all__ = [
    "AAF",
    "ANNOTATION",
    "COVARIATES",
    "EFFECT_MASKS",
    "INTERMEDIATE_PREFIX",
    "PHENOTYPE",
    "PLINK_REFERENCE_NAME",
    "REGENIE_VERSION_FILENAME",
    "TABLE_NAMES",
    "VARIANT_CLASS_ORDER",
    "VARIANT_SETS",
    "DatasetKey",
    "GwasDatasetMetadata",
    "GwasExportPipeline",
    "GwasInventory",
    "GwasLoadPipeline",
    "GwasMetadata",
    "GwasMetadataException",
    "GwasRunPipeline",
    "GwasTable",
    "GwasWorkflowPipeline",
    "RegenieDataset",
    "build_covariate_exprs",
    "define_plink_reference",
    "get_aaf_ht",
    "get_annotation_ht",
    "get_covariate_ht",
    "get_effects_ht",
    "get_freq_expr",
    "get_gene_region_expr",
    "get_impact_expr",
    "get_phenotype_ht",
    "get_regenie_input",
    "get_region_name",
    "get_sex_cat_expr",
    "get_variant_classification",
    "get_variant_sets_ht",
    "get_weight_expr",
    "region_label",
    "resolve_iid_fid",
    "snake_case",
    "str_to_expr",
    "to_hail_literal_value",
    "variant_id_expr",
]
