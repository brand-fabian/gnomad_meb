def extract_file(args):
    in_gz, out_f = args
    if in_gz.endswith("gz"):
        subprocess.run(f"gzip -dc {in_gz} > {out_f}", shell=True, check=True)


rule extract_annotations:
    input:
        covariates=config["covariates"],
        phenotype=config["phenotype"],
        annotation=config["annotation"],
        variant_sets=config["variant_sets"],
        aaf=config["aaf"],
        effect_masks=config["effect_masks"],
    output:
        covariates=f"{config['output_dir']}/annotations/covariates.tsv",
        phenotype=f"{config['output_dir']}/annotations/phenotype.tsv",
        annotation=f"{config['output_dir']}/annotations/annotation.tsv",
        variant_sets=f"{config['output_dir']}/annotations/variant_sets.tsv",
        aaf=f"{config['output_dir']}/annotations/aaf.tsv",
        effect_masks=f"{config['output_dir']}/annotations/effect_masks.tsv",
    log:
        f"{config['log_dir']}/extract_annotations/{config['project']}.log",
    threads: 6
    resources:
        runtime=60,
        mem_mb=64_000,
    run:
        import subprocess
        import multiprocessing as mp


        with mp.Pool(processes=threads) as pool:
            pool.map(extract_file, zip(input, output))


rule fix_variant_sets:
    input:
        variant_sets=rules.extract_annotations.output.variant_sets,
    output:
        variant_sets=f"{config['output_dir']}/annotations/variant_sets.reformatted.tsv",
    log:
        f"{config['log_dir']}/fix_variant_sets/{config['project']}.log",
    threads: 6
    resources:
        runtime=60,
        mem_mb=32_000,
    localrule: True
    run:
        import pandas

        df = pandas.read_csv(input[0], sep="\t")
        assert set(df.columns) == { "contig", "region", "position", "variants" }, f"invalid pandas dataframe columns: {set(df.columns)}"
        # ensure that the columns are in the precise ordering that regenie
        # expects
        df[["region", "contig", "position", "variants"]].to_csv(output[0], sep="\t", index=False, header=False)


rule write_snplist:
    input:
        bed=config["bed"],
    output:
        snplist=f"{config['output_dir']}/pass.snplist",
    params:
        preset="write-snplist",
        extra=get_snp_filter_str(),
    log:
        f"{config['log_dir']}/write_snplist/{config['project']}.log",
    threads: 8
    resources:
        runtime=60,
        mem_mb=64_000,
    wrapper:
        f"file:{config['wrapper_dir']}/plink2"


rule export_bgen:
    input:
        bed=config["bed"],
    output:
        bgen=f"{config['output_dir']}/{config['project']}.bgen",
        sample=f"{config['output_dir']}/{config['project']}.sample",
    params:
        preset="export-bgen",
    log:
        f"{config['log_dir']}/export_bgen/{config['project']}.log",
    threads: 8
    resources:
        runtime=60,
        mem_mb=64_000,
    wrapper:
        f"file:{config['wrapper_dir']}/plink2"


rule regenie_chunk:
    input:
        unpack(get_regenie_chunk_inputs),
    output:
        master=f"{config['output_dir']}/l0/{config['project']}.master",
        chunk_files=[
            f"{config['output_dir']}/l0/{config['project']}_job{i + 1}.snplist"
            for i in range(config["n_chunks"])
        ],
    params:
        preset="chunk",
    log:
        f"{config['log_dir']}/regenie/chunk/{config['project']}.log",
    threads: 20
    resources:
        runtime=120,
        mem_mb=80_000,
    wrapper:
        f"file:{config['wrapper_dir']}/regenie"


rule regenie_step1:
    input:
        unpack(get_regenie_step1_inputs),
    output:
        chunk_files=[
            f"{config['output_dir']}/l0/{config['project']}_job{{chunk_id}}_l0_Y{i + 1}"
            for i in range(len(phenotypes))
        ],
    params:
        preset="step1",
        extra=f"--minCaseCount {config["min_case_count"]}"
    log:
        f"{config['log_dir']}/regenie/l0/{config['project']}_job{{chunk_id}}.log",
    threads: 20
    resources:
        runtime=120,
        mem_mb=80_000,
    wrapper:
        f"file:{config['wrapper_dir']}/regenie"


rule regenie_merge:
    input:
        unpack(get_regenie_merge_inputs),
    output:
        predictions=f"{config['output_dir']}/l1/{{phenotype}}/{config['project']}.Y{{phenotype_id}}.l1_pred.list",
        loco=f"{config['output_dir']}/l1/{{phenotype}}/{config['project']}.Y{{phenotype_id}}.l1_{{phenotype_id}}.loco.gz",
    params:
        preset="merge",
        phenotype=lambda w: phenotypes[int(w.phenotype_id) - 1],
        extra=f"--minCaseCount {config["min_case_count"]}",
    log:
        f"{config['log_dir']}/regenie/merge/{config['project']}.Y{{phenotype_id}}_{{phenotype}}.log",
    threads: 20
    resources:
        runtime=720,
        mem_mb=100_000,
    wrapper:
        f"file:{config['wrapper_dir']}/regenie"


rule regenie_gwas:
    input:
        unpack(get_regenie_gwas_inputs),
    output:
        regenie=f"{config['output_dir']}/gwas/{{phenotype}}/{config['project']}_Y{{phenotype_id}}_{{phenotype}}.regenie.gz",
    params:
        preset="step2",
        phenotype=lambda w: phenotypes[int(w.phenotype_id) - 1],
        extra=f"--minCaseCount {config["min_case_count"]}",
    log:
        f"{config['log_dir']}/regenie/gwas/{config['project']}.Y{{phenotype_id}}_{{phenotype}}.log",
    threads: 20
    resources:
        runtime=240,
        mem_mb=80_000,
    wrapper:
        f"file:{config['wrapper_dir']}/regenie"


rule regenie_burden:
    input:
        unpack(get_regenie_burden_inputs),
    output:
        regenie=f"{config['output_dir']}/burden/{{phenotype}}/{config['project']}_Y{{phenotype_id}}_{{phenotype}}.regenie.gz",
    params:
        preset="burden",
        phenotype=lambda w: phenotypes[int(w.phenotype_id) - 1],
        extra=f"--minCaseCount {config["min_case_count"]}",
    log:
        f"{config['log_dir']}/regenie/burden/{config['project']}.Y{{phenotype_id}}_{{phenotype}}.log",
    threads: 20
    resources:
        runtime=720,
        mem_mb=100_000,
    wrapper:
        f"file:{config['wrapper_dir']}/regenie"
