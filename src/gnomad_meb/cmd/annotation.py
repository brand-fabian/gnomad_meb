"""Annotation CLI application."""

import sys
import traceback

import click

from gnomad_meb.cmd.task import AnnotationRun, Task, task_command
from gnomad_meb.logging import get_logger


@click.group()
def annotation():
    """Run the annotation pipeline on a VariantDataset."""


def run_annotation(
    task: Task, dry_run: bool = False, output_format: str = "json"
) -> bool:
    """Run the Annotation pipeline for ``task``.

    Shared by the ``annotation run`` CLI command and the serialized
    ``annotation.run`` executor. ``AnnotationPipeline.run`` binds the VDS and
    executes through the runner. With ``dry_run`` the pipeline is built and its
    manifest printed, but no step executes.
    """
    from gnomad_meb.lib.annotation import AnnotationPipeline
    from gnomad_meb.pipeline import PipelineRunner

    logger = get_logger()
    logger.info("starting annotation pipeline")
    pipeline = AnnotationPipeline(task)
    runner = PipelineRunner(task, [pipeline])
    if dry_run:
        from gnomad_meb.cmd.output import emit_dry_run

        emit_dry_run(runner, output_format)
        return True
    runner.run()
    logger.info(
        "finished annotation pipeline: finished = "
        f"{runner.has_finished()}, error = {runner.has_error()}"
    )
    return runner.has_finished() and not runner.has_error()


@annotation.command()
@click.option(
    "-V",
    "--vds",
    default=None,
    required=True,
    type=str,
    help="Path to a VariantDataset metadata file.",
)
@task_command(AnnotationRun, pipeline="annotation")
def run(task: Task, dry_run: bool = False, output_format: str = "json"):
    """Annotate the given VariantDataset with information from external sources and QC.

    This pipeline uses information from the Variant QC and Sample QC pipelines, as well
    as gnomAD and clinvar data to annotate the current VariantDataset and return a
    gnomAD-like formatted output.
    """
    try:
        succeeded = run_annotation(task, dry_run, output_format)
    except Exception as e:
        logger = get_logger()
        logger.error(f"error processing annotation pipeline: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

    if not succeeded:
        logger = get_logger()
        logger.error("annotation pipeline finished with errored steps")
        sys.exit(1)
