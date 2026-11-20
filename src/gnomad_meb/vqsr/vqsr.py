"""Snakemake pipeline for VQSR jobs.

This module prepares and executes a snakemake based VQSR pipeline for recalibration
of site-only SNP and INDEL VCF's for variant QC. The pipeline is model after two
reference implementations from gnomAD and populationgenomics:
    https://github.com/populationgenomics/production-pipelines/blob/main/cpg_workflows/jobs/vqsr.py
    https://github.com/broadinstitute/gnomad_qc/blob/main/gnomad_qc/v4/variant_qc/vqsr.py

The scatter unit is the VDS task list: the caller passes one input bundle per
task. The workflow concatenates the per-task site-only VCFs once, trains the SNP
and INDEL models globally on that concatenated VCF, then applies each model back
across the tasks as a scatter-gather pass — ``VariantRecalibrator`` is rerun per
task with ``--input-model`` / ``--output-tranches-for-scatter`` to emit that
task's recalibration table and per-task tranches, ``GatherTranches`` collates the
per-task tranches into one file per mode, and ``ApplyVQSR`` filters each task
with its own recalibration table plus the gathered tranches. This mirrors the
two-stage workflow of the cited references. The per-task calling-interval lists
are computed by the stage (see :mod:`gnomad_meb.lib.variant_qc.vqsr`) and travel
in the config.
"""

import json
import string
from collections.abc import Mapping
from pathlib import Path
from tempfile import gettempdir
from typing import TypedDict

import yaml
from snakemake.cli import args_to_api, parse_args
from snakemake.exceptions import print_exception

from gnomad_meb.config import VariantQcConfig, get_config
from gnomad_meb.lib.variant_qc.constants import GATK_VERSION_FILENAME
from gnomad_meb.logging import get_logger

SNAKEFILE_TEMPLATE = (Path(__file__).parent / "Snakefile.template").resolve()
SNAKEFILE_BASE = Path(__file__).parent.resolve()


class TaskInput(TypedDict):
    """One VDS task's per-split VQSR inputs (the keys ``_make_config`` reads)."""

    as_info: str
    calling_intervals: str


def _make_config(
    task_inputs: Mapping[str, TaskInput],
    output_dir: str | Path,
    target: str | Path | None = None,
) -> Path:
    """Write the Snakemake config for the per-task VQSR workflow.

    ``task_inputs`` maps each VDS task name to its ``as_info`` VCF and its
    per-task ``calling_intervals.interval_list``. The task order is preserved in
    the dumped YAML (``sort_keys=False``), so the ``concat_as_info`` rule reads
    the per-task VCFs in the same order the dataset enumerates them and the
    concatenated ``site_only.vcf.bgz`` is deterministic.
    """
    cfg = get_config()
    logger = get_logger()

    if target is None:
        target = Path(gettempdir()) / "config.yaml"
    elif isinstance(target, str):
        target = Path(target)

    assert cfg.vqsr is not None, "missing vqsr configuration section"
    assert cfg.vqsr.resources is not None, "missing vqsr reference resources"

    # The VQSR tunables live on the Variant QC settings (they are annotated as
    # the ``vqsr_params``/``variant_qc_cutoffs`` globals downstream); the VQSR
    # stage only projects them into the Snakemake config. Defaults apply when no
    # ``[variant_qc]`` section is configured.
    variant_qc = cfg.variant_qc or VariantQcConfig()

    tasks = list(task_inputs)

    logger.debug(f"creating snakemake pipeline configuration at {target}")
    with open(target, "w") as out_f:
        yaml.dump(
            {
                "output_dir": str(output_dir),
                "wrapper_dir": str(cfg.vqsr.wrapper_dir),
                "log_dir": str(cfg.vqsr.log_dir),
                "resources": cfg.vqsr.resources.to_dict(),
                "target_truth_sensitivity": (
                    variant_qc.target_truth_sensitivity.model_dump()
                ),
                "vqsr_params": variant_qc.vqsr_params.model_dump(),
                "tasks": tasks,
                "as_info": {
                    task: task_inputs[task]["as_info"] for task in tasks
                },
                "calling_intervals": {
                    task: task_inputs[task]["calling_intervals"] for task in tasks
                },
            },
            out_f,
            sort_keys=False,
        )
    return target


def _make_snakefile(
    config: str | Path,
    target: str | Path | None = None,
    source: Path = SNAKEFILE_TEMPLATE,
) -> Path:
    logger = get_logger()

    if target is None:
        target = Path(gettempdir()) / "Snakefile"
    elif isinstance(target, str):
        target = Path(target)

    with open(source) as in_f:
        source_t = string.Template(in_f.read())

    logger.debug(f"creating snakefile at {target}")
    with open(target, "w") as out_f:
        out_f.write(
            source_t.safe_substitute(
                {
                    "configfile": str(config),
                    "vqsr_base": SNAKEFILE_BASE,
                }
            )
        )
    return target


def _detect_gatk_version(working_directory: Path) -> dict[str, str] | None:
    """Detect the GATK version Snakemake deployed the pipeline with.

    Snakemake writes each conda rule's environment spec to
    ``{working_directory}/.snakemake/conda/<hash>.yaml`` and deploys it to the
    sibling ``<hash>/`` prefix. Find the spec that requires ``gatk4`` and read
    the installed package's ``conda-meta/gatk4-<version>-<build>.json`` record.

    Returns the record's ``version``/``build``/``url``, or ``None`` when no GATK
    environment was deployed.
    """
    logger = get_logger()
    conda_root = working_directory / ".snakemake" / "conda"

    for spec_path in sorted(conda_root.glob("*.yaml")):
        spec = yaml.safe_load(spec_path.read_text()) or {}
        dependencies = spec.get("dependencies", [])
        if not any(str(dependency).startswith("gatk4") for dependency in dependencies):
            continue

        records = sorted(
            (conda_root / spec_path.stem / "conda-meta").glob("gatk4-*.json")
        )
        if not records:
            logger.warning(
                f"gatk4 conda environment without a package record: {spec_path}"
            )
            continue

        record = json.loads(records[-1].read_text())
        return {
            "version": record["version"],
            "build": record["build"],
            "url": record.get("url", ""),
        }
    return None


def vqsr(
    task_inputs: Mapping[str, TaskInput],
    output_dir: str | Path,
    snakemake_args: tuple[str, ...],
):
    """Run the VQSR workflow once for the whole callset.

    The VDS task list is the scatter: each entry in ``task_inputs`` is one
    per-task apply. The config and Snakefile are written under
    ``{scratch_dir}/vqsr``.
    """
    cfg = get_config()
    logger = get_logger()

    working_directory = Path(cfg.scratch_dir) / "vqsr"
    working_directory.mkdir(parents=True, exist_ok=True)

    cfg_p = _make_config(
        task_inputs, output_dir, target=working_directory / "config.yaml"
    )
    smk_p = _make_snakefile(cfg_p, target=working_directory / "Snakefile")

    if "--snakefile" in snakemake_args:
        logger.warning(
            "found --snakefile arg in snakemake arguments for this workflow"
            ", this wil be overriden"
        )
        idx = snakemake_args.index("--snakefile")
        snakemake_args = (
            *snakemake_args[:idx],
            *snakemake_args[(idx + 2) :],
        )

    if "--directory" in snakemake_args:
        logger.warning(
            "found --directory arg in snakemake arguments for this workflow"
            ", this wil be overriden"
        )
        idx = snakemake_args.index("--directory")
        snakemake_args = (
            *snakemake_args[:idx],
            *snakemake_args[(idx + 2) :],
        )

    snakemake_args = (
        *snakemake_args,
        "--snakefile",
        str(smk_p),
        # Run Snakemake from the workflow's scratch working directory so its
        # relative paths, locks and logs stay out of the caller's cwd.
        "--directory",
        str(working_directory),
    )
    logger.debug(f"got snakemake argument list: {snakemake_args}")

    # Lifted from the snakemake main function with some additions
    # for the integration between the different pipelines:
    # https://github.com/snakemake/snakemake/blob/main/src/snakemake/cli.py#L2246
    try:
        logger.info(
            f"starting vqsr workflow for {len(task_inputs)} task(s) into {output_dir}"
        )
        parser, args = parse_args(snakemake_args)
        success = args_to_api(args, parser)
    except Exception as e:
        logger.error("error executing snakemake workflow")
        print_exception(e)
        return False

    if success:
        # Write the GATK provenance next to the per-split VCFs, so the import
        # can read it back when it imports them (the VCF output marks success).
        gatk = _detect_gatk_version(working_directory)
        if gatk is None:
            logger.warning(
                "could not detect the gatk version from the deployed conda "
                "environments; no gatk provenance will be written"
            )
        else:
            provenance = Path(output_dir) / GATK_VERSION_FILENAME
            provenance.parent.mkdir(parents=True, exist_ok=True)
            provenance.write_text(json.dumps(gatk, indent=2))
            logger.debug(f"wrote gatk version provenance to {provenance}")
    return success
