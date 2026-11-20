"""The ``gnomad_workflow`` command: run the composed whole-pipeline workflow.

It parses the VDS-build options into a :class:`GnomadWorkflowRun` task, composes
every stage (VDS build, QC, annotation and filter) over the VDS that build
produces, and executes the whole graph
through :class:`~gnomad_meb.pipeline.PipelineRunner` via
:class:`~gnomad_meb.lib.gnomad_workflow.GnomadWorkflowPipeline`. The global
``--describe`` flag prints the task description and exits before hail starts.
The composed stages' own options are settings (not flags); see the command help.
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import get_args

import click

from gnomad_meb.cmd.task import GnomadWorkflowRun, Task, task_command
from gnomad_meb.config import ENV_PREFIX
from gnomad_meb.constants import REFERENCE_GENOMES
from gnomad_meb.logging import get_logger


def run_gnomad_workflow(
    task: Task, dry_run: bool = False, output_format: str = "json"
) -> bool:
    """Build and run the composed workflow for ``task``.

    :class:`GnomadWorkflowPipeline` composes every ported stage over the VDS the
    task's build options describe and registers their steps in the central
    registry; the runner executes the whole graph in dependency order. With
    ``dry_run`` the pipeline is built and its manifest printed, but no step
    executes.
    """
    from gnomad_meb.lib.gnomad_workflow import GnomadWorkflowPipeline
    from gnomad_meb.pipeline import PipelineRunner

    command: GnomadWorkflowRun = task.command
    logger = get_logger()
    logger.info("starting gnomad workflow")

    pipeline = GnomadWorkflowPipeline(task)
    runner = PipelineRunner(task, [pipeline], overwrite=command.overwrite)
    if dry_run:
        from gnomad_meb.cmd.output import emit_dry_run

        emit_dry_run(runner, output_format)
        return True
    runner.run()
    logger.info(
        "finished gnomad workflow: finished = "
        f"{runner.has_finished()}, error = {runner.has_error()}"
    )
    return runner.has_finished() and not runner.has_error()


@click.command()
@click.option(
    "-g",
    "--gvcf",
    multiple=True,
    default=[],
    type=click.Path(exists=True, readable=True),
    help="GVCF file path or S3 URL.",
)
@click.option(
    "-b",
    "--bucket",
    multiple=True,
    default=[],
    help="Buckets to search gvcf files in.",
    type=str,
)
@click.option(
    "-s",
    "--search",
    multiple=True,
    default=[],
    help="Search directory for gvcfs on the local filesystem.",
    type=click.Path(exists=True),
)
@click.option(
    "-n",
    "--vds-name",
    help="Name of the VDS files.",
    default=os.getenv(f"{ENV_PREFIX}VDS_NAME", "vds"),
    type=str,
)
@click.option(
    "-o",
    "--output-base",
    help="Local output base directory for the dataset.",
    default=os.getenv(f"{ENV_PREFIX}OUTPUT_BASE", "./output"),
    type=str,
)
@click.option(
    "--reference-genome",
    help="Reference genome build to use.",
    default="GRCh38",
    type=click.Choice(get_args(REFERENCE_GENOMES), case_sensitive=True),
)
@click.option(
    "--s3-search-pattern",
    help="Glob patterns to use in searches in the S3 buckets.",
    multiple=True,
    default=["**/*gvcf.gz"],
    type=str,
)
@click.option(
    "--local-search-pattern",
    help="Search patterns to use on the local filesystem.",
    multiple=True,
    default=["*gvcf.gz"],
    type=str,
)
@click.option(
    "--n-splits",
    help=(
        "Number of splits to build the dataset in. Zero or fewer than the "
        "contig count gives one split per contig."
    ),
    default=0,
    type=int,
)
@click.option(
    "--interval-size",
    help="Size of each interval inside a split.",
    default=1_200_000,
    type=int,
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Re-run every step, even finished ones.",
)
@task_command(GnomadWorkflowRun, pipeline="gnomad-workflow")
def gnomad_workflow(task: Task, dry_run: bool = False, output_format: str = "json"):
    """Build and run the whole gvcf -> VDS -> QC -> annotation -> filter workflow.

    This command carries the VDS-build options; the composed stages' own knobs
    are settings, so a run is configured through the TOML file or ``GMEB_``
    environment rather than a growing set of flags. In particular, cluster
    arguments for the Snakemake-backed VQSR stage go in the config, e.g.

    \b
    [vqsr]
    snakemake = ["--profile", "slurm", "--executor", "slurm"]

    Calling intervals come from ``resources.calling_intervals``.
    """
    try:
        succeeded = run_gnomad_workflow(task, dry_run, output_format)
    except Exception as e:
        logger = get_logger()
        logger.error(f"error processing gnomad workflow: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

    if not succeeded:
        logger = get_logger()
        logger.error("gnomad workflow finished with errored steps")
        sys.exit(1)
