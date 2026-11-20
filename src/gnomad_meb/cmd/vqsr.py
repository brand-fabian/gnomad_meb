"""VQSR CLI application."""

import sys

import click

from gnomad_meb.cmd.task import Task, VqsrRun, task_command
from gnomad_meb.logging import get_logger


@click.group()
def vqsr():
    """Execute the VQSR pipeline for a given site only vcf."""


def run_vqsr(task: Task, dry_run: bool = False, output_format: str = "json") -> bool:
    """Build and run the VQSR stage for ``task``; return whether it succeeded.

    This is the shared body of the ``vqsr pipeline`` CLI command and the flow's
    serialized ``vqsr.pipeline`` executor, so both replay the identical
    invocation. The stage's single Snakemake-backed step is the whole callset
    workflow; the VDS task list is its scatter. With ``dry_run`` the pipeline is
    built and its manifest printed, but no step executes.
    """
    from gnomad_meb.lib.variant_qc import VqsrPipeline
    from gnomad_meb.pipeline import PipelineRunner

    logger = get_logger()
    logger.info("starting vqsr pipeline")
    pipeline = VqsrPipeline(task)
    runner = PipelineRunner(task, [pipeline])
    if dry_run:
        from gnomad_meb.cmd.output import emit_dry_run

        emit_dry_run(runner, output_format)
        return True
    runner.run()

    return runner.has_finished() and not runner.has_error()


@vqsr.command(context_settings={"ignore_unknown_options": True})
@click.option(
    "-V",
    "--vds",
    default=None,
    required=True,
    type=str,
    help="Path to the VariantDataset metadata file whose task list is the scatter.",
)
@click.option(
    "-a",
    "--as-info",
    default=None,
    required=True,
    type=str,
    help="Directory of per-task sites-only as_info VCFs the workflow concatenates.",
)
@click.option(
    "-o",
    "--output-dir",
    default=None,
    required=True,
    type=str,
    help="Directory the per-task VQSR VCFs are written under.",
)
@click.option(
    "-c",
    "--calling-intervals",
    default=None,
    required=True,
    type=str,
    help="Global picard interval_list the per-task callable intervals "
    "are derived from.",
)
@click.argument(
    "snakemake",
    nargs=-1,
)
@task_command(VqsrRun, pipeline="vqsr")
def pipeline(task: Task, dry_run: bool = False, output_format: str = "json"):
    """Run the VQSR Pipeline.

    Execute the snakemake pipeline from the VQSR module for a given VDS.

    Cluster arguments (``--profile`` / ``--executor``) belong in the
    ``[vqsr].snakemake`` setting, e.g.

    \b
    [vqsr]
    snakemake = ["--profile", "slurm", "--executor", "slurm"]

    or ``GMEB_VQSR__SNAKEMAKE='["--profile", "slurm"]'``. The composed
    ``gnomad-workflow`` command reads that setting; any SNAKEMAKE argument
    passed here overrides it for this invocation.
    """
    if not run_vqsr(task, dry_run, output_format):
        # error messages are printed in the vqsr function
        sys.exit(1)
