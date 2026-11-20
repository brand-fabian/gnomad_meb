__author__ = 'Fabian Brand'
__email__ = 'brand@imbie.uni-bonn.de'

import os
from tempfile import TemporaryDirectory, gettempdir
from textwrap import dedent

from snakemake.shell import shell

shell.executable('/bin/bash')

in_vcf = snakemake.input
if isinstance(in_vcf, str):
    in_vcf = [in_vcf]
if len(in_vcf) == 0:
    raise Exception("Error: please provide at least one input vcf.")

output_vcf = snakemake.output.get("vcf", None)
if output_vcf is None:
    raise Exception("Error: Please provide an output vcf file")

extra = snakemake.params.get("extra", "")

vcf_type_out = ""
if output_vcf.split(".")[-1] == "gz" or output_vcf.split(".")[-1] == "bgz":
    vcf_type_out += " -O z"
elif output_vcf.split(".")[-1] == "bcf":
    vcf_type_out += " -O b"

mem = snakemake.resources.get("mem_mb", None)
if mem is not None:
    mem_str = f"-m {mem!s}M"
else:
    mem_str = ""

log = snakemake.log_fmt_shell(stdout=True, stderr=True)
with TemporaryDirectory(dir=os.getenv("SCRATCH_DIR", gettempdir())) as tempdir:
    if len(in_vcf) > 1:
        in_vcf_str = " ".join(in_vcf)
        shell(dedent("""
            set -x
            {{
                bcftools concat \
                    {extra} \
                    {in_vcf_str} \
                | bcftools sort \
                    {vcf_type_out} -o {output_vcf} \
                    -T {tempdir}/bcftools.XXXXXX {mem_str}
                tabix {output_vcf}
            }} {log}
        """))
    elif len(in_vcf) == 1:
        in_vcf_str = in_vcf[0]
        shell(dedent("""
            set -x
            {{
                bcftools view {vcf_type_out} -o {output_vcf} {in_vcf_str}
                tabix {output_vcf}
            }} {log}
        """))
