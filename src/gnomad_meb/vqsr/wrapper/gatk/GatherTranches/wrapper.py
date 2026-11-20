##
# Snakemake wrapper script for gatk GatherTranches
__author__ = 'Fabian Brand'
__email__ = 'brand@imbie.uni-bonn.de'

from textwrap import dedent

from snakemake.shell import shell

tranches_in = snakemake.input.get("tranches", None)
assert tranches_in is not None and len(tranches_in) > 0, "missing input tranches"

tranches_out = snakemake.output.get("tranches", None)
assert tranches_out is not None, "missing output tranches path"

mode = snakemake.params.get("mode", None)
assert mode in ["SNP", "INDEL"], "mode must be in ['SNP', 'INDEL']"

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

input_cmd = " ".join(map(lambda s: f"--input {s}", tranches_in))

shell(dedent("""\
    set -ex

    {{
        gatk --java-options "{jvm_args}" \\
            GatherTranches \\
            --mode {mode} \\
            {input_cmd} \\
            --output {tranches_out}
    }} {log}
"""))
