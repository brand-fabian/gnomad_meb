# Snakemake wrapper for regenie
__author__ = 'Fabian Brand'
__email__ = 'brand@imbie.uni-bonn.de'

import os
from tempfile import TemporaryDirectory, gettempdir
from textwrap import dedent
from typing import Any
from uuid import uuid4
from pathlib import Path
from itertools import chain
import re
import shlex

from snakemake.shell import shell

PRESETS = {
    "chunk",
    "step1",
    "merge",
    "step2",
    "burden",
}
"""Valid presets to determine regenie analysis options and defaults."""

def get_input_str(name: str, param: str, default: Any | None = None) -> str:
    v = snakemake.input.get(name, default)
    if v is None:
        return ""

    if param == "":
        return str(v)
    
    return f" {param} {v} "

#
# Required Genomic Input Data
#
bgen_in = snakemake.input.get("bgen", None)
bed_in = snakemake.input.get("bed", None)

if bgen_in is not None:
    bgen_p = Path(bgen_in).resolve()
    sample_p = bgen_p.with_suffix(".sample")

    in_root_p = bgen_p.with_name(bgen_p.stem)
    in_str = f" --bgen {str(bgen_p)} --sample {str(sample_p)}"
elif bed_in is not None:
    bed_p = Path(bed_in).resolve()
    bed_p = bed_p.with_name(bed_p.stem)

    in_root_p = bed_p
    in_str = f" --bed {str(bed_p)}"
else:
    raise ValueError(f"missing input bgen and bed/bim/fam files")

#
# Optional Covariates and Phenotype inputs
#
covar_file = get_input_str("covariates", "--covarFile")
pheno_file = get_input_str("phenotype", "--phenoFile")
anno_file = get_input_str("annotation", "--anno-file")
set_list_file = get_input_str("variant_sets", "--set-list")
aaf_file = get_input_str("aaf", "--aaf-file")
mask_file = get_input_str("effect_masks", "--mask-def")

snplist = get_input_str("snplist", "--extract")
exclude_sets = get_input_str("exclude_sets", "--exclude-sets")

master_file = get_input_str("chunk_master", "")
chunk_file = get_input_str("chunk", "")

pred_file = get_input_str("predictions", "--pred")

#
# General parameters
#
preset = snakemake.params.get("preset", None)
if preset in { "chunk", "step1", "merge" }:
    step = "1"
elif preset in { "step2", "burden" }:
    step = "2"
else:
    raise ValueError(f"invalid preset, preset = {preset}")

if step == "1":
    step_str = f" --step {str(step)} --force-step1 "
else:
    step_str = f" --step {str(step)} "

extra = snakemake.params.get("extra", "")
bsize = snakemake.params.get("bsize", 1_000)

low_memory = snakemake.params.get("low_memory", True)

binary_trait = snakemake.params.get("binary_trait", True)
if binary_trait:
    binary_trait_str = "--bt"
else:
    binary_trait_str = ""


# Preset and step specific parameters
phenotype = snakemake.params.get("phenotype", None)
gwas_test = snakemake.params.get("gwas_test", "additive")
burden_test = snakemake.params.get("burden_test", "skato,acato-full")

threads = snakemake.threads or 1
log = snakemake.log_fmt_shell(stdout=True, stderr=True)

with TemporaryDirectory(dir=os.getenv("SCRATCH_DIR", gettempdir())) as tempd:
    analysis_id = str(uuid4())
    analysis_dir = Path(tempd) / analysis_id
    analysis_dir.mkdir(exist_ok=True, parents=True)

    temp_path = analysis_dir / "regenie_temp"
    out_path = analysis_dir / "regenie_out"

    if low_memory:
        low_memory_str = f" --lowmem --lowmem-prefix {str(temp_path)} "
    else:
        low_memory_str = ""

    prepare_cmd = ""
    if preset == "chunk":
        master_file = snakemake.output.get("master", None)
        assert master_file is not None, "no master file output specified"

        chunk_files = snakemake.output.get("chunk_files", None)
        assert isinstance(chunk_files, list), "chunk files must be specified as list"

        master_prefix = Path(master_file)
        master_prefix = master_prefix.with_name(master_prefix.stem)

        chunk_files_prefix = os.path.commonprefix(chunk_files)
        
        # This returns n snplist files at {chunk_files_prefix}_job{i}.snplist
        # Hence, in snakemake these files should be named accordingly.
        preset_cmd = dedent(f"""\
            --split-l0 {master_prefix},{len(chunk_files)} \\
            --out {chunk_files_prefix}
        """)

    elif preset == "step1":
        assert master_file != "", "master file is required for running l0 analyses"
        assert os.path.exists(chunk_file), f"missing chunk file {chunk_file}"
        chunk_re = re.search(r"_job(?P<jid>\d+)$", Path(chunk_file).stem)
        assert chunk_re is not None, f"could not find chunk id in path, chunk_file = {chunk_file}"
        chunk_id = chunk_re.group("jid")

        chunk_files = snakemake.output.get("chunk_files", None)
        assert isinstance(chunk_files, list), "chunk files must be specified as list with len(chunk_files) == n_phenotypes"

        chunk_prefix, _ = re.split(r"_l0", os.path.commonprefix(chunk_files), maxsplit=1)

        # l0 model predictions are always written alongside the input chunk at
        # {chunk_file}_l0_Y{i} for i in n_phenotypes. --out writes a log file at
        # the specified location.
        preset_cmd = dedent(f"""\
            --gz \\
            --use-relative-path \\
            --run-l0 {master_file},{chunk_id} \\
            --out {chunk_prefix}
        """)

    elif preset == "merge":
        assert master_file != "", "master file is required when merging l0 splits."
        if phenotype is not None:
            phenotype_str = f" --l1-phenoList {phenotype} "
        else:
            phenotype_str = ""

        prediction_list = snakemake.output.get("predictions", None)
        loco_out = snakemake.output.get("loco", None)
        assert (prediction_list is not None) and (loco_out is not None), "missing output parameter"
        output_prefix = os.path.commonprefix([prediction_list, loco_out])

        if output_prefix[-1] in {".", "_", "-"}:
            output_prefix = output_prefix[:-1]

        # Copy input files to the local scratch in order to optimize file read
        # performance.
        input_dir = analysis_dir / "regenie_input"
        input_dir.mkdir(exist_ok=True, parents=True)

        rsync_tokens = ["rsync", "-av"]
        master_p = Path(master_file)
        local_master_p = input_dir / master_p.name
        with open(master_p, "r") as in_f, open(local_master_p, "w") as out_f:
            for i, line in enumerate(in_f):
                if i == 0:
                    out_f.write(line)
                else:
                    path, n_chunks, n_vars = line.strip().split()
                    chunk_p = Path(path)

                    rsync_tokens.extend(map(lambda p: str(p.resolve()), chunk_p.parent.glob(f"{chunk_p.name}*")))
                    out_f.write("\t".join([
                        str((input_dir / chunk_p.name).resolve()),
                        str(n_chunks),
                        str(n_vars),
                    ]) + "\n")

        rsync_tokens.append(str(input_dir.resolve()))
        prepare_cmd = dedent(f"""\
            rsync -av {in_root_p}* {input_dir}
            {shlex.join(rsync_tokens)}

            echo "===== master file content ====="
            cat {local_master_p}
            echo "===== master file end     ====="
        """)

        if "--bgen" in in_str:
            in_str = f" --bgen {str(input_dir / in_root_p.name)}.bgen --sample {str(input_dir / in_root_p.name)}.sample"
        elif "--bed" in in_str:
            in_str = f" --bed {str(input_dir / in_root_p.name)}"
        else:
            raise ValueError(f"unrecognized input format: {in_str}")

        # l1 analysis generates two output files: {prefix}_pred.list containg
        # a description file with all prediction files and {prefix}_{i}.loco.gz where
        # {i} is the index of the phenotype specified.
        preset_cmd = dedent(f"""\
            --gz \\
            {phenotype_str} \\
            --use-relative-path \\
            --run-l1 {str(local_master_p)} \\
            --keep-l0 \\
            --out {output_prefix}
        """)

    elif preset == "step2":
        regenie_out = snakemake.output.get("regenie", None)
        assert regenie_out is not None, "missing output file path"
        assert pred_file != "", "missing predictions list for step 2 testing"
        
        pred_prefix, _ = re.split(rf"_{phenotype}.regenie.gz", regenie_out, maxsplit=1)

        # Output files are written to {prefix}_{phenotype}.regenie.gz:
        preset_cmd = dedent(f"""\
            --firth \\
            --approx \\
            --test {gwas_test} \\
            --gz \\
            {pred_file} \\
            --out {pred_prefix}
        """)

    elif preset == "burden":
        assert (
            anno_file != ""
            and set_list_file != ""
            and mask_file != ""
        ), "missing set list specifications for gene or region based testing"
        assert pred_file != "", "missing predictions list for step 2 testing"

        regenie_out = snakemake.output.get("regenie", None)
        assert regenie_out is not None, "missing output file path"
        assert pred_file != "", "missing predictions list for step 2 testing"
        
        pred_prefix, _ = re.split(rf"_{phenotype}.regenie.gz", regenie_out, maxsplit=1)

        preset_cmd = dedent(f"""\
            {anno_file} \\
            --weights-col 4 \\
            {set_list_file} \\
            {mask_file} \\
            {exclude_sets} \\
            {pred_file} \\
            --gz \\
            --firth \\
            --approx \\
            --rgc-gene-p \\
            --write-mask \\
            --out {pred_prefix}
        """)
    else:
        raise ValueError(f"invalid preset, preset = {preset}")

    shell(dedent("""\
        export TMPDIR={tempd}

        set -ex
    
        {{
            {prepare_cmd}

            regenie {extra} \\
                {step_str} \\
                {in_str} \\
                {covar_file} \\
                {pheno_file} \\
                {aaf_file} \\
                {snplist} \\
                {binary_trait_str} \\
                {low_memory_str} \\
                --bsize {bsize} \\
                --threads {threads} \\
                {preset_cmd}
        }} {log}
    """))