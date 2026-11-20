"""Filter CLI application."""

import sys
import traceback

import click

from gnomad_meb.cmd.task import FilterRun, Task, task_command
from gnomad_meb.logging import get_logger


@click.group()
def filter():
    """Apply the configured filters to an annotated matrix table."""


def run_filter(task: Task, dry_run: bool = False, output_format: str = "json") -> bool:
    """Build and run the Filter pipeline for ``task``.

    This is the shared body of the ``filter run`` CLI command and the serialized
    ``filter.run`` executor, so the CLI and a worker replay the identical
    invocation. Hail is reached only through the pipeline's methods. With
    ``dry_run`` the pipeline is built and its manifest printed, but no step
    executes.
    """
    from gnomad_meb.lib.filter import FilterPipeline
    from gnomad_meb.pipeline import PipelineRunner

    logger = get_logger()
    logger.info("starting filter pipeline")

    pipeline = FilterPipeline(task)
    runner = PipelineRunner(task, [pipeline])
    if dry_run:
        from gnomad_meb.cmd.output import emit_dry_run

        emit_dry_run(runner, output_format)
        return True
    runner.run()
    return runner.has_finished() and not runner.has_error()


@filter.command()
@click.option(
    "-M",
    "--mt",
    default=None,
    required=True,
    type=str,
    help="Path to an annotated matrix table to filter.",
)
@task_command(FilterRun, pipeline="filter")
def run(task: Task, dry_run: bool = False, output_format: str = "json"):
    """Filter the given matrix table with the configured filters and regions."""
    try:
        succeeded = run_filter(task, dry_run, output_format)
    except Exception as e:
        logger = get_logger()
        logger.error(f"error processing filter pipeline: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

    if not succeeded:
        logger = get_logger()
        logger.error("filter pipeline finished with errored steps")
        sys.exit(1)
