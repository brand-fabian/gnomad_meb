##
# Snakemake wrapper script for gatk ApplyVQSR
__author__ = 'Fabian Brand'
__email__ = 'brand@imbie.uni-bonn.de'

import os
from tempfile import TemporaryDirectory, gettempdir
from textwrap import dedent

from snakemake.logging import logger
from snakemake.shell import shell

# SNP and Indel default sensitivity filter values from populationgenomics pipeline:
#     https://github.com/populationgenomics/production-pipelines/blob/1132f9c133b8f734b482dbdaf5c6206c4f7d3af0/configs/defaults/rd_combiner.toml#L77
SNP_SENSITIVITY_FILTER_VALUE = 99.7
INDEL_SENSITIVITY_FILTER_VALUE = 99.0


def get_default_sensitivity_filter_for_mode(mode: str):
    if mode == "SNP":
        return SNP_SENSITIVITY_FILTER_VALUE
    elif mode == "INDEL":
        return INDEL_SENSITIVITY_FILTER_VALUE
    else:
        raise ValueError(f"invalid mode: mode = {mode}")


vcf_in = snakemake.input.get("vcf", None)
assert vcf_in is not None, "missing input vcf"

recalibrated_in = snakemake.input.get("recal", None)
assert recalibrated_in is not None, "missing variant recal file"

tranches_in = snakemake.input.get("tranches", None)
assert tranches_in is not None, "missing tranches file"

interval_in = snakemake.input.get("interval", None)
interval_cmd = ""
if interval_in is None:
    logger.warning("no intervals specified, processing the whole input vcf")
else:
    interval_cmd = f" -L {interval_in} "

vcf_out = snakemake.output.get("vcf", None)
assert vcf_out is not None, "missing output vcf path"

# There is also mode = "BOTH" in gatk ApplyVQSR, but since this is
# not for use in production as per the gatk Documentation, we intentionally
# exclude it here.
mode = snakemake.params.get("mode", None)
assert mode in ["SNP", "INDEL"], "mode must be in ['SNP', 'INDEL']"

allele_specific = snakemake.params.get("allele_specific", True)
assert type(allele_specific) == bool, f"allele_specific must be a boolean value, got {type(allele_specific)}"
as_cmd = "--use-allele-specific-annotations" if allele_specific else ""

# The truth-sensitivity filter level is a settings parameter, sourced from the
# rules (``target_truth_sensitivity`` in the Snakemake config), not hardcoded
# here.
filter_value = snakemake.params.get("filter_value", get_default_sensitivity_filter_for_mode(mode))
filter_value = float(filter_value)

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

with TemporaryDirectory(dir=os.getenv("SCRATCH_DIR", gettempdir())) as tempd:
    shell(dedent("""\
        TMPDIR="{tempd}"
        set -ex

        {{
            gatk --java-options "{jvm_args}" \\
                ApplyVQSR \\
                -O {vcf_out} \\
                -V {vcf_in} \\
                --recal-file {recalibrated_in} \\
                {interval_cmd} \\
                --tranches-file {tranches_in} \\
                --truth-sensitivity-filter-level {filter_value} \\
                {as_cmd} \\
                -mode {mode}
        }} {log}
    """))
