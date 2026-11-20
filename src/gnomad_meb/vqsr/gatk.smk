tasks = list(config["tasks"])

rule concat_as_info:
    """Concatenate the per-task sites-only as_info VCFs into one global VCF.

    Hail writes the per-task VCF headers, so their ##contig/INFO/FORMAT lines
    match and no ``bcftools reheader`` normalization is needed. The input order
    is the task order, so the concatenation is deterministic.
    """
    input:
        vcf=[config["as_info"][task] for task in tasks],
    output:
        vcf=f"{config['output_dir']}/site_only.vcf.bgz",
    params:
        extra="",
    log:
        f"{config['log_dir']}/concat/as_info.log"
    threads: 8
    resources:
        runtime=720,
        mem_mb=120000,
    wrapper: f"file:{config['wrapper_dir']}/bcftools/concat"

rule variant_recalibrator_create_model:
    """Train the global SNP recalibration model on the concatenated site-only VCF."""
    input:
        vcf=f"{config['output_dir']}/site_only.vcf.bgz",
    output:
        vcf=f"{config['output_dir']}/step1/snp.vcf.bgz",
        tbi=f"{config['output_dir']}/step1/snp.vcf.bgz.tbi",
        tranches=f"{config['output_dir']}/step1/snp.tranches.txt",
        model=f"{config['output_dir']}/step1/snp.model.txt",
    params:
        step="CREATE_MODEL",
        mode="SNP",
        allele_specific=config["vqsr_params"]["allele_specific"],
        sample_nth_variant=config["vqsr_params"]["variants_to_sample"],
        max_gaussians=config["vqsr_params"]["max_gaussians"],
        tranche_values=config["vqsr_params"]["snp_tranche_values"],
        hapmap=config["resources"]["hapmap"],
        omni=config["resources"]["omni"],
        thousand_genome=config["resources"]["thousand_genome"],
        dbsnp=config["resources"]["dbsnp"],
    log:
        f"{config['log_dir']}/step1/snp.log"
    threads: 32
    resources:
        runtime=720,
        mem_mb=120000,
    wrapper: f"file:{config['wrapper_dir']}/gatk/VariantRecalibrator"

rule variant_recalibrator_create_model_indel:
    """Train the global INDEL recalibration model on the concatenated site-only VCF."""
    input:
        vcf=f"{config['output_dir']}/site_only.vcf.bgz",
    output:
        vcf=f"{config['output_dir']}/step1/indel.vcf.bgz",
        tbi=f"{config['output_dir']}/step1/indel.vcf.bgz.tbi",
        tranches=f"{config['output_dir']}/step1/indel.tranches.txt",
        model=f"{config['output_dir']}/step1/indel.model.txt",
    params:
        step="CREATE_MODEL",
        mode="INDEL",
        allele_specific=config["vqsr_params"]["allele_specific"],
        sample_nth_variant=config["vqsr_params"]["variants_to_sample"],
        max_gaussians=config["vqsr_params"]["max_gaussians"],
        tranche_values=config["vqsr_params"]["indel_tranche_values"],
        mills=config["resources"]["mills"],
        axiom_poly=config["resources"]["axiom_poly"],
        dbsnp=config["resources"]["dbsnp"],
    log:
        f"{config['log_dir']}/step1/indel.log"
    threads: 32
    resources:
        runtime=720,
        mem_mb=120000,
    wrapper: f"file:{config['wrapper_dir']}/gatk/VariantRecalibrator"

rule variant_recalibrator_apply:
    """Apply the allele-specific variant recalibrator for all snps."""
    input:
        vcf=f"{config['output_dir']}/site_only.vcf.bgz",
        model=rules.variant_recalibrator_create_model.output['model'],
        interval=lambda wildcards: config["calling_intervals"][wildcards.task],
    output:
        vcf=f"{config['output_dir']}/task-{{task}}/step2/snp.vcf.bgz",
        tbi=f"{config['output_dir']}/task-{{task}}/step2/snp.vcf.bgz.tbi",
        tranches=f"{config['output_dir']}/task-{{task}}/step2/snp.tranches.txt",
    params:
        step="APPLY_RECALIBRATION",
        mode="SNP",
        allele_specific=config["vqsr_params"]["allele_specific"],
        sample_nth_variant=config["vqsr_params"]["variants_to_sample"],
        max_gaussians=config["vqsr_params"]["max_gaussians"],
        tranche_values=config["vqsr_params"]["snp_tranche_values"],
        hapmap=config["resources"]["hapmap"],
        omni=config["resources"]["omni"],
        thousand_genome=config["resources"]["thousand_genome"],
        dbsnp=config["resources"]["dbsnp"],
    log:
        f"{config['log_dir']}/variant_recalibrator_apply/snp.{{task}}.log"
    threads: 32
    resources:
        runtime=720,
        mem_mb=120000,
    wrapper: f"file:{config['wrapper_dir']}/gatk/VariantRecalibrator"

rule variant_recalibrator_apply_indel:
    """Apply allele-specific variant recalibrator to indels."""
    input:
        vcf=f"{config['output_dir']}/site_only.vcf.bgz",
        model=rules.variant_recalibrator_create_model_indel.output['model'],
        interval=lambda wildcards: config["calling_intervals"][wildcards.task],
    output:
        vcf=f"{config['output_dir']}/task-{{task}}/step2/indel.vcf.bgz",
        tbi=f"{config['output_dir']}/task-{{task}}/step2/indel.vcf.bgz.tbi",
        tranches=f"{config['output_dir']}/task-{{task}}/step2/indel.tranches.txt",
    params:
        step="APPLY_RECALIBRATION",
        mode="INDEL",
        allele_specific=config["vqsr_params"]["allele_specific"],
        sample_nth_variant=config["vqsr_params"]["variants_to_sample"],
        max_gaussians=config["vqsr_params"]["max_gaussians"],
        tranche_values=config["vqsr_params"]["indel_tranche_values"],
        mills=config["resources"]["mills"],
        axiom_poly=config["resources"]["axiom_poly"],
        dbsnp=config["resources"]["dbsnp"],
    log:
        f"{config['log_dir']}/variant_recalibrator_apply_indel/indel.{{task}}.log"
    threads: 32
    resources:
        runtime=720,
        mem_mb=120000,
    wrapper: f"file:{config['wrapper_dir']}/gatk/VariantRecalibrator"

rule gather_tranches:
    """Gather all tranches files"""
    input:
        tranches=[
            f"{config['output_dir']}/task-{task}/step2/snp.tranches.txt"
            for task in tasks
        ]
    output:
        tranches=f"{config['output_dir']}/vqsr.snp.tranches.txt"
    params:
        mode="SNP"
    log:
        f"{config['log_dir']}/gather_tranches/snp.log"
    threads: 4
    resources:
        runtime=60,
        mem_mb=64000,
    wrapper: f"file:{config['wrapper_dir']}/gatk/GatherTranches"

rule gather_tranches_indel:
    """Gather all indel tranches files"""
    input:
        tranches=[
            f"{config['output_dir']}/task-{task}/step2/indel.tranches.txt"
            for task in tasks
        ]
    output:
        tranches=f"{config['output_dir']}/vqsr.indel.tranches.txt"
    params:
        mode="INDEL"
    log:
        f"{config['log_dir']}/gather_tranches/indel.log"
    threads: 4
    resources:
        runtime=60,
        mem_mb=64000,
    wrapper: f"file:{config['wrapper_dir']}/gatk/GatherTranches"

rule apply_snp:
    """Apply the global SNP model to one task's callable sites."""
    input:
        vcf=f"{config['output_dir']}/site_only.vcf.bgz",
        recal=rules.variant_recalibrator_apply.output["vcf"],
        tranches=rules.gather_tranches.output["tranches"],
        interval=lambda wildcards: config["calling_intervals"][wildcards.task],
    output:
        vcf=f"{config['output_dir']}/task-{{task}}/snp.vcf.bgz",
    params:
        mode="SNP",
        allele_specific=config["vqsr_params"]["allele_specific"],
        filter_value=config["target_truth_sensitivity"]["snp"],
    log:
        f"{config['log_dir']}/apply_vqsr/snp.{{task}}.log"
    threads: 4
    resources:
        runtime=720,
        mem_mb=120000,
    wrapper: f"file:{config['wrapper_dir']}/gatk/ApplyVQSR"

rule apply_vqsr:
    """Apply the global INDEL model to one task's SNP-applied VCF."""
    input:
        vcf=f"{config['output_dir']}/task-{{task}}/snp.vcf.bgz",
        recal=rules.variant_recalibrator_apply_indel.output["vcf"],
        tranches=rules.gather_tranches_indel.output["tranches"],
        interval=lambda wildcards: config["calling_intervals"][wildcards.task],
    output:
        vcf=f"{config['output_dir']}/task-{{task}}/vqsr.vcf.bgz",
    params:
        mode="INDEL",
        allele_specific=config["vqsr_params"]["allele_specific"],
        filter_value=config["target_truth_sensitivity"]["indel"],
    log:
        f"{config['log_dir']}/apply_vqsr/{{task}}.log"
    threads: 4
    resources:
        runtime=720,
        mem_mb=120000,
    wrapper: f"file:{config['wrapper_dir']}/gatk/ApplyVQSR"
