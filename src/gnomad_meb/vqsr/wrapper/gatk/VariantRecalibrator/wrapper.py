##
# Snakemake wrapper script for gatk VariantRecalibrator
__author__ = 'Fabian Brand'
__email__ = 'brand@imbie.uni-bonn.de'

import os
from tempfile import TemporaryDirectory, gettempdir
from textwrap import dedent
from uuid import uuid4

from snakemake.logging import logger
from snakemake.shell import shell

VARIANT_RECALIBRATOR_STEPS = ["CREATE_MODEL", "APPLY_RECALIBRATION", "COMBINED"]
VARIANT_RECALIBRATOR_MODES = ["SNP", "INDEL"]

VARIANTS_TO_SAMPLE = 10
MAX_GAUSSIANS = 4

##
# Tool configurations lifted from the gnomad / populationgenomics pipelines:
#     https://github.com/broadinstitute/gnomad_qc/blob/main/gnomad_qc/v4/variant_qc/vqsr.py
#     https://github.com/broadinstitute/gnomad_qc/blob/main/gnomad_qc/v4/resources/variant_qc.py#L25
#     https://github.com/populationgenomics/production-pipelines/blob/main/cpg_workflows/jobs/vqsr.py
#
SNP_RECALIBRATION_TRANCHE_VALUES = [
    100.0,
    99.95,
    99.9,
    99.8,
    99.6,
    99.5,
    99.4,
    99.3,
    99.0,
    98.0,
    97.0,
    90.0,
]
INDEL_RECALIBRATION_TRANCHE_VALUES = [
    100.0,
    99.95,
    99.9,
    99.5,
    99.0,
    97.0,
    96.0,
    95.0,
    94.0,
    93.5,
    93.0,
    92.0,
    91.0,
    90.0,
]
STANDARD_FEATURES = [
    'ReadPosRankSum',
    'MQRankSum',
    'QD',
    'FS',
    'SOR',
]
SNP_STANDARD_FEATURES = STANDARD_FEATURES + ['MQ']
INDEL_STANDARD_FEATURES = STANDARD_FEATURES

ALLELE_SPECIFIC_FEATURES = [
    'AS_ReadPosRankSum',
    'AS_MQRankSum',
    'AS_QD',
    'AS_FS',
    'AS_SOR',
    # Not using depth for the following reasons:
    # 1. The Broad pipelines don't use it;
    # 2. -G AS_StandardAnnotation flag to GenotypeGVCFs doesn't include it;
    # 3. For exomes, depth is an irrelevant feature and should be skipped:
    # 'AS_VarDP'
    # Note that for consistency, we also skip it for WGS.
]
SNP_ALLELE_SPECIFIC_FEATURES = ALLELE_SPECIFIC_FEATURES + ['AS_MQ']
INDEL_ALLELE_SPECIFIC_FEATURES = ALLELE_SPECIFIC_FEATURES

def get_default_tranch_values_for_mode(mode: str):
    if mode == "SNP":
        return SNP_RECALIBRATION_TRANCHE_VALUES
    elif mode == "INDEL":
        return INDEL_RECALIBRATION_TRANCHE_VALUES
    else:
        raise ValueError(f"invalid mode: mode = {mode}")

shell.executable("/bin/bash")

vcf_in = snakemake.input.get("vcf", None)
assert vcf_in is not None, "missing input vcf file"

step = snakemake.params.get("step", None)
assert step in ["CREATE_MODEL", "APPLY_RECALIBRATION", "COMBINED"], f"invalid variant recalibrator step: {step} != {VARIANT_RECALIBRATOR_STEPS}"

mode = snakemake.params.get("mode", "SNP")
assert mode in ["SNP", "INDEL"], f"invalid variant recalibrator mode: {mode} != {VARIANT_RECALIBRATOR_MODES}"

allele_specific = snakemake.params.get("allele_specific", True)
assert type(allele_specific) == bool, f"invalid type for allele_specific: {type(allele_specific)}"

# General params for the different steps / modes. These are settings parameters
# (``vqsr_params``), sourced from the rules via the Snakemake config.
variants_to_sample = snakemake.params.get("sample_nth_variant", VARIANTS_TO_SAMPLE)
max_gaussians = snakemake.params.get("max_gaussians", MAX_GAUSSIANS)
tranche_values = snakemake.params.get("tranche_values", get_default_tranch_values_for_mode(mode))

# Variant Databases for recalibration
hapmap = snakemake.params.get("hapmap", None)
omni = snakemake.params.get("omni", None)
thousand_genome = snakemake.params.get("thousand_genome", None)
dbsnp = snakemake.params.get("dbsnp", None)
axiom_poly = snakemake.params.get("axiom_poly", None)
mills = snakemake.params.get("mills", None)
if mode == "SNP":
    assert all(x is not None for x in [hapmap, omni, thousand_genome, dbsnp]), "missing some required vqsr reference databases"
elif mode == "INDEL":
    assert all(x is not None for x in [dbsnp, axiom_poly, mills]), "missing some required vqsr reference databases"

# Output Files depend on the mode / step selection:
output_files = {
    "vcf": "{prefix}.vcf.bgz",
    "tbi": "{prefix}.vcf.bgz.tbi",
    "tranches": "{prefix}.tranches.txt",
}
if step == "CREATE_MODEL":
    output_files["model"] = "{prefix}.vqsr.txt"
    output_files["rscript"] = "{prefix}.R"

extra = snakemake.params.get("extra", "")
jvm_args = snakemake.params.get("jvm_args", "")
memory = snakemake.resources.get("mem_mb", None)
log = snakemake.log_fmt_shell(stdout=True, stderr=True)
threads = snakemake.threads

# Massage the jvm args to fit the desired defaults, without overwriting user input
if 'Xms' not in jvm_args and memory is not None:
    jvm_args += f" -Xms{memory}m "

if '-XX:+UseParallelGC' not in jvm_args:
    jvm_args += " -XX:+UseParallelGC "

if '-XX:ParallelGCThreads=' not in jvm_args:
    jvm_args += f" -XX:ParallelGCThreads={threads - 2} "

as_cmd = " --use-allele-specific-annotations " if allele_specific else ""

tranche_cmd = " ".join(f"-tranche {t}" for t in tranche_values)
an_cmd = ""
db_cmd = ""
if mode == "SNP":
    if allele_specific:
        an_cmd = " ".join(map(lambda f: f"-an {f}", SNP_ALLELE_SPECIFIC_FEATURES))
    else:
        an_cmd = " ".join(map(lambda f: f"-an {f}", SNP_STANDARD_FEATURES))

    db_cmd = dedent(f"""\
        -resource:hapmap,known=false,training=true,truth=true,prior=15 {hapmap} \\
        -resource:omni,known=false,training=true,truth=true,prior=12 {omni} \\
        -resource:1000G,known=false,training=true,truth=false,prior=10 {thousand_genome} \\
        -resource:dbsnp,known=true,training=false,truth=false,prior=7 {dbsnp} \\
    """)
elif mode == "INDEL":
    if allele_specific:
        an_cmd = " ".join(map(lambda f: f"-an {f}", INDEL_ALLELE_SPECIFIC_FEATURES))
    else:
        an_cmd = " ".join(map(lambda f: f"-an {f}", INDEL_STANDARD_FEATURES))

    db_cmd = dedent(f"""\
        -resource:mills,known=false,training=true,truth=true,prior=12 {mills} \\
        -resource:axiomPoly,known=false,training=true,truth=false,prior=10 {axiom_poly} \\
        -resource:dbsnp,known=true,training=false,truth=false,prior=2 {dbsnp} \\
    """)
else:
    raise Exception(f"invalid variant recalibrator mode: {mode}")

# Use the $SCRATCH_DIR as default temp directory, since it is available on the local HPC
# system, use gettempdir() as fallback
with TemporaryDirectory(dir=os.getenv("SCRATCH_DIR", gettempdir())) as tempd:
    prefix = os.path.join(tempd, str(uuid4()))

    step_cmd = ""
    if step == "CREATE_MODEL":
        step_cmd = dedent(f"""\
            --sample-every-Nth-variant {variants_to_sample} \\
            --output-model {prefix}.vqsr.txt \\
            --rscript-file {prefix}.R \\
        """)
    elif step == "APPLY_RECALIBRATION":
        model_in = snakemake.input.get("model", None)
        assert model_in is not None, "model input is required for the APPLY_RECALIBRATION step"
        
        interval_in = snakemake.input.get("interval", None)
        assert interval_in is not None, "interval input is required for the APPLY_RECALIBRATION step"

        step_cmd = dedent(f"""\
            --input-model {model_in} \\
            --output-tranches-for-scatter \\
            -L {interval_in} \\
        """)
    elif step == "COMBINED":
        # For smaller cohorts or INDEL callsets, we do avoid the two step workflow
        pass

    out_cmd = ""
    for key, value in snakemake.output.items():
        if key not in output_files:
            logger.warning(f"no known output file for type {key} exists, ignoring")
        else:
            out_cmd += f"mv -v {output_files[key].format(prefix=prefix)} {value}\n"

    shell(dedent("""\
        export TMPDIR={tempd}

        set -ex    
        
        {{
            gatk --java-options "{jvm_args}" VariantRecalibrator \\
                {extra} \\
                -V {vcf_in} \\
                -O {prefix}.vcf.bgz \\
                --tranches-file {prefix}.tranches.txt \\
                --max-gaussians {max_gaussians} \\
                --trust-all-polymorphic \\
                {tranche_cmd} \\
                {an_cmd} \\
                -mode {mode} \\
                {as_cmd} \\
                {step_cmd} \\
                {db_cmd}

            {out_cmd}
        }} {log}
    """))