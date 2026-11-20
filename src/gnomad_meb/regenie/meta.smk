from snakemake.logging import logger
import pandas

annotations_gzipped = any(x.endswith("gz") for x in [
    config['covariates'],
    config['phenotype'],
    config['annotation'],
    config['variant_sets'],
    config['aaf'],
    config['effect_masks'],
])

pheno_df = pandas.read_csv(
    config['phenotype'],
    compression='gzip' if config['phenotype'].endswith('gz') else None,
    sep="\t"
)
phenotypes_base = set(pheno_df.columns) - { "FID", "IID" }
logger.info(f"extracted base phenotypes: n_phenotypes = {phenotypes_base}")

# Extract the phenotypes deemed valid by regenie and make sure they are in
# the same order as the columns of the phenotype file. A quantitative response
# is gated on its non-missing sample count; every other response on its case
# count (the column sum).
response_kinds = config.get("response_kinds", {})

def _passes_min_case_count(column: str) -> bool:
    if response_kinds.get(column) == "quantitative":
        count = pheno_df[column].notna().sum()
    else:
        count = pheno_df[column].sum()
    return count >= config["min_case_count"]

phenotypes = [k for k in phenotypes_base if _passes_min_case_count(k)]
phenotypes = list(sorted(phenotypes, key=list(pheno_df.columns).index))
logger.info(f"extracted valid phenotypes: n_phenotypes = {len(phenotypes)}, phenotypes = {phenotypes}")

def get_regenie_inputs_base(wildcards) -> dict[str, str]:
    if not annotations_gzipped:
        return {
            "covariates": config["covariates"],
            "phenotype": config["phenotype"],
            "aaf": config["aaf"],
            "snplist": f"{config['output_dir']}/pass.snplist",
        }
    else:
        return {
            "covariates": f"{config['output_dir']}/annotations/covariates.tsv",
            "phenotype": f"{config['output_dir']}/annotations/phenotype.tsv",
            "aaf": f"{config['output_dir']}/annotations/aaf.tsv",
            "snplist": f"{config['output_dir']}/pass.snplist",
        }

def get_regenie_burden_annotations_input(wildcards) -> dict[str, str]:
    base = get_regenie_inputs_base(wildcards)
    if not annotations_gzipped:
        return {
            **base,
            "annotation": config["annotation"],
            "variant_sets": config["variant_sets"],
            "effect_masks": config["effect_masks"],
        }
    else:
        return {
            **base,
            "annotation": f"{config['output_dir']}/annotations/annotation.tsv",
            "variant_sets": f"{config['output_dir']}/annotations/variant_sets.reformatted.tsv",
            "effect_masks": f"{config['output_dir']}/annotations/effect_masks.tsv",
        }

def get_regenie_chunk_inputs(wildcards):
    return {
        "bed": config["bed"],
        **get_regenie_inputs_base(wildcards),
    }

def get_regenie_step1_inputs(wildcards):
    return {
        "bed": config["bed"],
        "chunk_master": f"{config['output_dir']}/l0/{config['project']}.master",
        "chunk": f"{config['output_dir']}/l0/{config['project']}_job{wildcards.chunk_id}.snplist",
        **get_regenie_inputs_base(wildcards),
    }

def get_regenie_merge_inputs(wildcards):
    return {
        "bed": config["bed"],
        "chunk_master": f"{config['output_dir']}/l0/{config['project']}.master",
        "chunk_files": [
            f"{config['output_dir']}/l0/{config['project']}_job{chunk_id + 1}_l0_Y{wildcards.phenotype_id}"
            for chunk_id in range(config['n_chunks'])
        ],
        **get_regenie_inputs_base(wildcards),
    }

def get_regenie_gwas_inputs(wildcards):
    return {
        "bgen": f"{config['output_dir']}/{config['project']}.bgen",
        "predictions": f"{config['output_dir']}/l1/{wildcards.phenotype}/{config['project']}.Y{wildcards.phenotype_id}.l1_pred.list",
        **get_regenie_inputs_base(wildcards),
    }

def get_regenie_burden_inputs(wildcards):
    return {
        "bgen": f"{config['output_dir']}/{config['project']}.bgen",
        "predictions": f"{config['output_dir']}/l1/{wildcards.phenotype}/{config['project']}.Y{wildcards.phenotype_id}.l1_pred.list",
        **get_regenie_burden_annotations_input(wildcards),
    }

def get_snp_filter_str() -> str:
    """Assemble the plink2 filter criteria to assemble the final snplist for regenie."""
    min_mac = config.get("min_mac", 3)
    min_geno_frac = config.get("min_geno_frac", 0.01)
    min_maf = config.get("min_maf", 0.01)
    hwe = config.get("hwe", 1e-15)
    snps_only = config.get("snps_only", True)
    snps_only_str = "--snps-only" if snps_only else ""

    return f"--mac {min_mac} --maf {min_maf} --geno {min_geno_frac} --hwe {hwe} {snps_only_str}"

wildcard_constraints:
    chunk_id="|".join(map(lambda x: str(x + 1), range(config['n_chunks']))),
    phenotype="|".join(phenotypes),
    phenotype_id="|".join(map(lambda x: str(x + 1), range(len(phenotypes)))),