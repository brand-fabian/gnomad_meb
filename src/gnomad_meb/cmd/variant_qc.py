"""Variant QC CLI application."""

import sys
import traceback

import click

from gnomad_meb.cmd.task import (
    Task,
    VariantQcExport,
    VariantQcLoadVqsr,
    task_command,
)
from gnomad_meb.logging import get_logger


@click.group()
def variant_qc():
    """Execute the Variant QC pipeline for a given VariantDataset."""


def export_variant_qc(
    task: Task, dry_run: bool = False, output_format: str = "json"
) -> bool:
    """Build and run the Variant QC export pipeline for ``task``.

    Shared by the ``variant-qc export`` CLI command and the serialized
    ``variant_qc.export`` executor. At least one output VCF must be requested.
    With ``dry_run`` the pipeline is built and its manifest printed, but no step
    executes.
    """
    logger = get_logger()
    cmd = task.command
    if cmd.as_info is None and cmd.quasi_info is None:
        raise ValueError(
            "no output files specified, at least one of -a|--as-info and "
            "-q|--quasi-info must be set"
        )

    from gnomad_meb.lib.variant_qc import VariantQcExportPipeline
    from gnomad_meb.pipeline import PipelineRunner

    logger.info("starting variant qc export pipeline")
    pipeline = VariantQcExportPipeline(task)
    runner = PipelineRunner(task, [pipeline])
    if dry_run:
        from gnomad_meb.cmd.output import emit_dry_run

        emit_dry_run(runner, output_format)
        return True
    runner.run()
    return runner.has_finished() and not runner.has_error()


def load_vqsr_variant_qc(
    task: Task, dry_run: bool = False, output_format: str = "json"
) -> bool:
    """Build and run the Variant QC VQSR-import pipeline for ``task``.

    Shared by the ``variant-qc load-vqsr`` CLI command and the serialized
    ``variant_qc.load_vqsr`` executor. With ``dry_run`` the pipeline is built and
    its manifest printed, but no step executes.
    """
    from gnomad_meb.lib.variant_qc import VariantQcImportPipeline
    from gnomad_meb.pipeline import PipelineRunner

    logger = get_logger()
    logger.info("starting variant qc import pipeline")
    pipeline = VariantQcImportPipeline(task)
    runner = PipelineRunner(task, [pipeline])
    if dry_run:
        from gnomad_meb.cmd.output import emit_dry_run

        emit_dry_run(runner, output_format)
        return True
    runner.run()
    return runner.has_finished() and not runner.has_error()


@variant_qc.command()
@click.option(
    "-V",
    "--vds",
    default=None,
    required=True,
    type=str,
    help="Path to a VariantDataset metadata file to export to the vcf files.",
)
@click.option(
    "-a",
    "--as-info",
    default=None,
    type=str,
    help=(
        "Output directory for the per-task as_info sites-only vcf files; one "
        "is written to <dir>/task-<task>/as_info.vcf.bgz per VDS split."
    ),
)
@click.option(
    "-q",
    "--quasi-info",
    default=None,
    type=str,
    help=(
        "Output directory for the per-task quasi_info sites-only vcf files; one "
        "is written to <dir>/task-<task>/quasi_info.vcf.bgz per VDS split."
    ),
)
@click.option(
    "--infer-as-annotations/--no-infer-as-annotations",
    default=True,
    help=(
        "If False is set, the function will throw an error instead of inferring "
        "missing gvcf annotations"
    ),
)
@click.option("--cdf-k", default=200, type=int, help="...")
@click.option(
    "--retain-cdfs",
    default=False,
    is_flag=True,
    help="If True, retain the CDFs generated when creating the info ht (cf. https://broadinstitute.github.io/gnomad_methods/api_reference/utils/sparse_mt.html#gnomad.utils.sparse_mt.default_compute_info)",
)
@task_command(VariantQcExport, pipeline="variant_qc_export")
def export(task: Task, dry_run: bool = False, output_format: str = "json"):
    """Run the export pipeline to generate a VQSR sites-only VCF."""
    try:
        succeeded = export_variant_qc(task, dry_run, output_format)
    except Exception as e:
        logger = get_logger()
        logger.error(f"error processing variant qc export pipeline: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

    if not succeeded:
        logger = get_logger()
        logger.error("variant qc export pipeline finished with errored steps")
        sys.exit(1)


@variant_qc.command()
@click.option(
    "-V",
    "--vds",
    default=None,
    required=True,
    type=str,
    help=(
        "Path to a VariantDataset metadata file whose split tasks are loaded "
        "one at a time."
    ),
)
@click.option(
    "--vcf",
    default=None,
    required=True,
    type=str,
    help=(
        "Directory of per-split VQSR output vcf files; one is read from "
        "<dir>/task-<task>/vqsr.vcf.bgz per VDS split."
    ),
)
@click.option(
    "--as-info",
    default=None,
    required=True,
    type=str,
    help=(
        "Directory of per-split as_info hail tables; one is read from "
        "<dir>/task-<task>/as_info.ht per VDS split for AS_lowqual."
    ),
)
@task_command(VariantQcLoadVqsr, pipeline="variant_qc")
def load_vqsr(task: Task, dry_run: bool = False, output_format: str = "json"):
    """Import the result of the VQSR pipeline as hail table to the current project."""
    try:
        succeeded = load_vqsr_variant_qc(task, dry_run, output_format)
    except Exception as e:
        logger = get_logger()
        logger.error(f"error processing variant qc import pipeline: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

    if not succeeded:
        logger = get_logger()
        logger.error("variant qc import pipeline finished with errored steps")
        sys.exit(1)
