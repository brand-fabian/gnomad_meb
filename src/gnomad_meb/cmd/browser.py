"""Browser export CLI application."""

import sys
import traceback

import click

from gnomad_meb.cmd.task import BrowserExportRun, Task, task_command
from gnomad_meb.logging import get_logger


@click.group()
def browser():
    """Export a browser-ready sites table and per-contig parquet files."""


def run_browser(task: Task, dry_run: bool = False, output_format: str = "json") -> bool:
    """Run the browser-export pipeline for ``task``.

    Shared by the ``browser run`` CLI command and the serialized ``browser.run``
    executor. With ``dry_run`` the pipeline is built and its manifest printed,
    but no step executes.
    """
    from gnomad_meb.lib.browser import BrowserExportPipeline
    from gnomad_meb.pipeline import PipelineRunner

    logger = get_logger()
    logger.info("starting browser export pipeline")
    pipeline = BrowserExportPipeline(task)
    runner = PipelineRunner(task, [pipeline])
    if dry_run:
        from gnomad_meb.cmd.output import emit_dry_run

        emit_dry_run(runner, output_format)
        return True
    runner.run()
    logger.info(
        "finished browser export pipeline: finished = "
        f"{runner.has_finished()}, error = {runner.has_error()}"
    )
    return runner.has_finished() and not runner.has_error()


@browser.command()
@click.option(
    "--mt",
    default=None,
    required=False,
    type=str,
    help=(
        "Path to the MatrixTable to export. Defaults to the configured "
        "annotation checkpoint."
    ),
)
@task_command(BrowserExportRun, pipeline="browser_export")
def run(task: Task, dry_run: bool = False, output_format: str = "json"):
    """Reshape a MatrixTable into the browser site table and per-contig parquet.

    The input is the annotation checkpoint by default; ``--mt`` overrides it
    with any MatrixTable.
    """
    try:
        succeeded = run_browser(task, dry_run, output_format)
    except Exception as e:
        logger = get_logger()
        logger.error(f"error processing browser export pipeline: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

    if not succeeded:
        logger = get_logger()
        logger.error("browser export pipeline finished with errored steps")
        sys.exit(1)
