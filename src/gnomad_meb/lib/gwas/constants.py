"""Domain constants for the GWAS stages.

Public names only: the intermediate path template, the six regenie table
names, the PLINK reference name, the run provenance filename, and the fixed
variant-class ordering that is the regenie ``effect`` / ``effect_masks``
contract (data, not configuration).
"""
import re

#: Intermediate artifact prefix for every GWAS stage step.
INTERMEDIATE_PREFIX = "{project}/intermediate/gwas"

#: The terminal regenie result prefix the load stage produces.
REGENIE_PREFIX = "{project}/intermediate/regenie"

#: The GWAS load manifest: the one index of the loaded tables and their
#: provenance, written next to the terminal regenie tables.
REGENIE_MANIFEST_PREFIX = f"{REGENIE_PREFIX}/manifest"

#: The GWAS load manifest schema version. Bump when the shape changes
#: incompatibly.
GWAS_MANIFEST_SCHEMA_VERSION = 1

#: The dataset GWAS metadata file key holding the table's variant count, recorded
#: once per table by the export and read by the run stage to derive its splits.
N_VARIANTS_KEY = "n_variants"

#: The GWAS metadata file keys recording a table's provenance, so the load stage
#: can index each loaded table by what it was without re-parsing its label.
TABLE_SOURCE_KEY = "source"
TABLE_NAME_KEY = "filter_name"
REGION_NAME_KEY = "region_name"

#: The GWAS metadata file key holding the effect-masks provenance: the
#: ``variant_info`` the export aggregated for the region's ``Mask-<class>`` rows
#: (the per-class variant counter, plus the classification gate counters when the
#: default classification produced it). The load stage attaches it as a global on
#: the loaded tables.
EFFECT_MASK_INFO_KEY = "effect_mask_info"

#: The six regenie input tables, in the order the export writes them.
PHENOTYPE = "phenotype"
COVARIATES = "covariates"
AAF = "aaf"
ANNOTATION = "annotation"
VARIANT_SETS = "variant_sets"
EFFECT_MASKS = "effect_masks"
TABLE_NAMES: tuple[str, ...] = (
    PHENOTYPE,
    COVARIATES,
    AAF,
    ANNOTATION,
    VARIANT_SETS,
    EFFECT_MASKS,
)

#: The region-independent base-chain steps: step kind -> checkpoint path suffix
#: under INTERMEDIATE_PREFIX. Shared by the export's BASE_CHAIN and DatasetKey.prefix.
BASE_CHAIN_PREFIXES: dict[str, str] = {
    "gwas_base": "gwas_base",
    "normalize_regenie": "normalized_regenie",
    "filter_regenie_chromosomes": "regenie_base",
    "phenotype": f"dataset/{PHENOTYPE}",
    "covariates": f"dataset/{COVARIATES}",
    "aaf": f"dataset/{AAF}",
}

#: The un-prefixed GRCh38 reference regenie expects (``1..22,X,Y,MT``).
PLINK_REFERENCE_NAME = "plink"

#: The regenie provenance file the run writes next to its output, holding the
#: detected regenie version; the load stage reads it back for a global.
REGENIE_VERSION_FILENAME = "regenie.version.json"

#: The single GWAS metadata file the export writes next to the dataset. It holds
#: the ``response -> display name`` map and the ``response -> kind`` map
#: (``binary``/``quantitative``) the run reads to apply the right phenotype gate,
#: plus whichever filter metadata globals the exported MT carried (see
#: :data:`FILTER_GLOBAL_NAMES`). One file, so the run and load stages read the
#: dataset's metadata from a single place.
#:
#: The suffix is deliberately distinct from the ``.metadata.json`` extension the
#: native-``VariantDataset`` checkpoint kind uses, so ``Artifact.from_path``
#: cannot mistake this JSON (which carries response and filter metadata, not VDS
#: params) for a dataset handle.
METADATA_SUFFIX = ".gwas.md.json"

#: The keys the GWAS metadata file's response maps live under.
RESPONSE_NAME_MAP_KEY = "response_name_map"
RESPONSE_KIND_MAP_KEY = "response_kind_map"

#: The filter metadata globals the export copies into the GWAS metadata file. The
#: high-quality global is the basis the configured filters layer on, so an MT
#: may carry it alone, with a filter's ``filter_variant_info``, with a region
#: filter's ``filter_region_info``, or with both. Any of them may be absent, so
#: the export records only those present and non-empty and the load attaches
#: only those the file actually has.
FILTER_GLOBAL_NAMES: tuple[str, ...] = (
    "filter_high_quality_gnomad",
    "filter_variant_info",
    "filter_region_info",
)

#: The fixed variant-class labels and mask ordering (the regenie contract).
VARIANT_CLASS_ORDER: tuple[str, ...] = (
    "clinvar",
    "lof",
    "deleterious_missense",
    "rare",
    "common",
)

BURDEN_ID_PATTERN = re.compile(r"^(?P<region>.+?)\.(?P<mask>Mask-[^.]+)\.(?P<aaf>.+)$")