# Snakemake wrapper for regenie
__author__ = 'Fabian Brand'
__email__ = 'brand@imbie.uni-bonn.de'

import os
from tempfile import TemporaryDirectory, gettempdir
from textwrap import dedent
from typing import Any
from uuid import uuid4
from pathlib import Path
import re

from snakemake.shell import shell

PRESETS = {
    "write-snplist",
    "export-bgen",
}
"""Valid presets to determine regenie analysis options and defaults."""

#
# Required Genomic Input Data
#
bed_in = snakemake.input.get("bed", None)

if bed_in is not None:
    bed_p = Path(bed_in).resolve()
    bed_p = bed_p.with_name(bed_p.stem)
    in_str = f" --bfile {str(bed_p)}"
else:
    raise ValueError(f"missing input bed/bim/fam files")

preset = snakemake.params.get("preset", None)
extra = snakemake.params.get("extra", "")
threads = snakemake.threads or 1
log = snakemake.log_fmt_shell(stdout=True, stderr=True)

with TemporaryDirectory(dir=os.getenv("SCRATCH_DIR", gettempdir())) as tempd:
    if preset == "write-snplist":
        output = snakemake.output.get("snplist", None)
        assert output is not None and output.endswith("snplist"), "invalid or no output file specified"

        snplist_p = Path(output)
        snplist_p = str(snplist_p.with_name(snplist_p.stem))
        preset_str = dedent(f"""\
            --write-snplist \\
            --out {snplist_p}
        """)
    elif preset == "export-bgen":
        bgen_out = snakemake.output.get("bgen", None)
        sample_out = snakemake.output.get("sample", None)
        assert bgen_out is not None and sample_out is not None, "missing output files for bgen conversion"

        bgen_prefix = os.path.commonprefix([ bgen_out, sample_out ])
        if bgen_prefix[-1] in {".", "_", "-"}:
            bgen_prefix = bgen_prefix[:-1]
        preset_str = dedent(f"""\
            --export bgen-1.2 'bits=8' \\
            --out {bgen_prefix}
        """)
    else:
        raise ValueError(f"invalid preset, preset = {preset}")
    
    shell(dedent("""\
        export TMPDIR={tempd}

        set -ex
    
        {{
            plink2 {extra} \\
                {in_str} \\
                {preset_str}
        }} {log}
    """))
