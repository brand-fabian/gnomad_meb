"""Sample QC CLI application."""

import sys
import traceback

import click

from gnomad_meb.cmd.task import SampleQcRun, Task, task_command
from gnomad_meb.logging import get_logger


@click.group()
def sample_qc():
    """Run the Sample QC pipeline for a given VariantDataset."""


def run_sample_qc(
    task: Task, dry_run: bool = False, output_format: str = "json"
) -> bool:
    """Build and run the Sample QC pipeline for ``task``.

    This is the shared body of the ``sample-qc run`` CLI command and the
    serialized ``sample_qc.run`` executor, so the CLI and a worker replay the
    identical invocation. Hail is reached only through the pipeline's methods.
    With ``dry_run`` the pipeline is built and its manifest printed, but no step
    executes.
    """
    from gnomad_meb.lib.sample_qc import SampleQcPipeline
    from gnomad_meb.pipeline import PipelineRunner

    logger = get_logger()
    logger.info("starting sample QC pipeline")

    pipeline = SampleQcPipeline(task)
    runner = PipelineRunner(task, [pipeline])
    if dry_run:
        from gnomad_meb.cmd.output import emit_dry_run

        emit_dry_run(runner, output_format)
        return True
    runner.run()
    return runner.has_finished() and not runner.has_error()


@sample_qc.command()
@click.option(
    "-V",
    "--vds",
    default=None,
    required=True,
    type=str,
    help="Path to a VariantDataset metadata file to compute sample stats for.",
)
@task_command(SampleQcRun, pipeline="sample_qc")
def run(task: Task, dry_run: bool = False, output_format: str = "json"):
    try:
        succeeded = run_sample_qc(task, dry_run, output_format)
    except Exception as e:
        logger = get_logger()
        logger.error(f"error processing sample QC pipeline: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

    if not succeeded:
        logger = get_logger()
        logger.error("sample QC pipeline finished with errored steps")
        sys.exit(1)
