"""GWAS CLI application."""

import sys

import click

from gnomad_meb.cmd.task import (
    GwasExport,
    GwasLoad,
    GwasPipeline,
    GwasWorkflowRun,
    Task,
    task_command,
)
from gnomad_meb.logging import get_logger


@click.group()
def gwas():
    """Execute the GWAS pipeline for a given variant dataset."""


def export_gwas(task: Task, dry_run: bool = False, output_format: str = "json") -> bool:
    """Build and run the GWAS export pipeline for ``task``.

    Shared by the ``gwas export`` CLI command and the serialized
    ``gwas.export`` executor. With ``dry_run`` the pipeline is built and its
    manifest printed, but no step executes.
    """
    from gnomad_meb.lib.gwas import GwasExportPipeline
    from gnomad_meb.pipeline import PipelineRunner

    logger = get_logger()
    logger.info("starting gwas export pipeline")
    pipeline = GwasExportPipeline(task)
    runner = PipelineRunner(task, [pipeline])
    if dry_run:
        from gnomad_meb.cmd.output import emit_dry_run

        emit_dry_run(runner, output_format)
        return True
    runner.run()
    return runner.has_finished() and not runner.has_error()


@gwas.command()
@click.option(
    "--mt",
    default=None,
    required=False,
    type=str,
    help="Path to a single high_quality_gnomad MatrixTable to export.",
)
@click.option(
    "--manifest",
    default=None,
    type=str,
    help="Path to the Filter stage's manifest.json.gz (one dataset per table).",
)
@click.option(
    "--include",
    multiple=True,
    default=[],
    type=str,
    help="Only export tables matching a token (label, filter name or region).",
)
@click.option(
    "--exclude",
    multiple=True,
    default=[],
    type=str,
    help="Drop tables matching a token (label, filter name or region).",
)
@click.option(
    "--metadata-csv",
    default=None,
    type=str,
    help="Override the [gwas].metadata_csv study metadata path.",
)
@click.option(
    "--base",
    default=None,
    type=str,
    help=(
        "Override the dataset output base directory "
        "(default [gwas].output_dir/dataset)."
    ),
)
@task_command(GwasExport, pipeline="gwas_export")
def export(task: Task, dry_run: bool = False, output_format: str = "json"):
    """Export regenie datasets from an MT or the Filter manifest.

    Provide exactly one of ``--mt`` or ``--manifest``. With ``--manifest``, one
    dataset is written per selected table (``--include``/``--exclude`` narrow the
    set), each namespaced under ``{base}/{table}/{region}``.
    """
    logger = get_logger()
    try:
        success = export_gwas(task, dry_run, output_format)
    except ValueError as exc:
        logger.error(str(exc))
        sys.exit(1)

    if success:
        logger.info("gwas export finished successfully")
    else:
        logger.warning("encountered error during gwas export")


def run_gwas(task: Task, dry_run: bool = False, output_format: str = "json") -> bool:
    """Build and run the GWAS run pipeline for ``task``; return success.

    Shared by the ``gwas pipeline`` CLI command and the flow's serialized
    ``gwas.pipeline`` executor. The snakemake/regenie stack is imported lazily.
    With ``dry_run`` the pipeline is built and its manifest printed, but no step
    executes.
    """
    from gnomad_meb.lib.gwas import GwasRunPipeline
    from gnomad_meb.pipeline import PipelineRunner

    logger = get_logger()
    logger.info("starting gwas run pipeline")
    pipeline = GwasRunPipeline(task)
    runner = PipelineRunner(task, [pipeline])
    if dry_run:
        from gnomad_meb.cmd.output import emit_dry_run

        emit_dry_run(runner, output_format)
        return True
    runner.run()
    return runner.has_finished() and not runner.has_error()


@gwas.command(context_settings={"ignore_unknown_options": True})
@click.option(
    "--dataset",
    default=None,
    required=True,
    type=str,
    help=(
        "Exported region directory (PLINK prefix + six TSVs), as written by "
        "``gwas export``."
    ),
)
@click.option(
    "--output-dir",
    default=None,
    required=True,
    type=str,
    help="Directory the regenie ``gwas``/``burden`` outputs are written under.",
)
@click.argument(
    "snakemake",
    nargs=-1,
)
@task_command(GwasPipeline, pipeline="gwas")
def pipeline(task: Task, dry_run: bool = False, output_format: str = "json"):
    """Run the regenie GWAS pipeline over an exported dataset.

    Cluster arguments (``--profile`` / ``--executor``) belong in the
    ``[gwas].snakemake`` setting, e.g.

    \b
    [gwas]
    snakemake = ["--profile", "slurm", "--executor", "slurm"]

    or ``GMEB_GWAS__SNAKEMAKE='["--profile", "slurm"]'``. Any SNAKEMAKE argument
    passed here overrides that setting for this invocation.
    """
    logger = get_logger()
    try:
        success = run_gwas(task, dry_run, output_format)
    except (ValueError, RuntimeError) as exc:
        logger.error(str(exc))
        sys.exit(1)

    if success:
        logger.info("gwas pipeline finished successfully")
    else:
        logger.warning("encountered error during gwas pipeline")


def load_gwas(task: Task, dry_run: bool = False, output_format: str = "json") -> bool:
    """Build and run the GWAS load pipeline for ``task``.

    Shared by the ``gwas load`` CLI command and the serialized ``gwas.load``
    executor. With ``dry_run`` the pipeline is built and its manifest printed,
    but no step executes.
    """
    from gnomad_meb.lib.gwas import GwasLoadPipeline
    from gnomad_meb.pipeline import PipelineRunner

    logger = get_logger()
    logger.info("starting gwas load pipeline")
    pipeline = GwasLoadPipeline(task)
    runner = PipelineRunner(task, [pipeline])
    if dry_run:
        from gnomad_meb.cmd.output import emit_dry_run

        emit_dry_run(runner, output_format)
        return True
    runner.run()
    return runner.has_finished() and not runner.has_error()


@gwas.command()
@click.option(
    "--dataset",
    default=None,
    required=True,
    type=str,
    help="Exported region directory (holds the phenotype table).",
)
@click.option(
    "--gwas",
    default=None,
    required=True,
    type=str,
    help="regenie GWAS output directory (``<output-dir>/gwas``).",
)
@click.option(
    "--burden",
    default=None,
    required=True,
    type=str,
    help="regenie burden output directory (``<output-dir>/burden``).",
)
@task_command(GwasLoad, pipeline="gwas_load")
def load(task: Task, dry_run: bool = False, output_format: str = "json"):
    """Load regenie GWAS results back into keyed hail tables."""
    logger = get_logger()
    try:
        success = load_gwas(task, dry_run, output_format)
    except ValueError as exc:
        logger.error(str(exc))
        sys.exit(1)

    if success:
        logger.info("gwas load finished successfully")
    else:
        logger.warning("encountered error during gwas load")


def run_gwas_workflow(
    task: Task, dry_run: bool = False, output_format: str = "json"
) -> bool:
    """Build and run the composed GWAS workflow for ``task``; return success.

    :class:`GwasWorkflowPipeline` composes export, run and load over the run's
    dataset inventory and registers their steps in the central registry; the
    runner executes the whole graph in dependency order in one invocation. With
    ``dry_run`` the pipeline is built and its manifest printed, but no step
    executes.
    """
    from gnomad_meb.lib.gwas import GwasWorkflowPipeline
    from gnomad_meb.pipeline import PipelineRunner

    logger = get_logger()
    logger.info("starting gwas workflow")
    pipeline = GwasWorkflowPipeline(task)
    runner = PipelineRunner(task, [pipeline])
    if dry_run:
        from gnomad_meb.cmd.output import emit_dry_run

        emit_dry_run(runner, output_format)
        return True
    runner.run()
    logger.info(
        "finished gwas workflow: finished = "
        f"{runner.has_finished()}, error = {runner.has_error()}"
    )
    return runner.has_finished() and not runner.has_error()


@gwas.command(context_settings={"ignore_unknown_options": True})
@click.option(
    "--mt",
    default=None,
    type=str,
    help="Path to a single high_quality_gnomad MatrixTable (xor --manifest).",
)
@click.option(
    "--manifest",
    default=None,
    type=str,
    help="Path to the Filter stage's manifest.json.gz (xor --mt).",
)
@click.option(
    "--include",
    multiple=True,
    default=[],
    type=str,
    help="Only run tables matching a token (label, filter name or region).",
)
@click.option(
    "--exclude",
    multiple=True,
    default=[],
    type=str,
    help="Drop tables matching a token (label, filter name or region).",
)
@click.option(
    "--output-base",
    default=None,
    type=str,
    help="Override the exported dataset base directory.",
)
@click.option(
    "--output-dir",
    default=None,
    type=str,
    help="Override the regenie output root ([gwas].output_dir).",
)
@click.argument(
    "snakemake",
    nargs=-1,
)
@task_command(GwasWorkflowRun, pipeline="gwas_workflow")
def workflow(task: Task, dry_run: bool = False, output_format: str = "json"):
    """Run export -> regenie -> load as one deterministic DAG.

    Provide exactly one of ``--mt`` or ``--manifest``. With ``--manifest``, the
    workflow runs once per selected table and region (``--include``/``--exclude``
    narrow the tables), producing one keyed ``gwas`` and ``burden`` hail table per
    dataset plus the GWAS load manifest.

    Cluster arguments (``--profile`` / ``--executor``) belong in the
    ``[gwas].snakemake`` setting; any argument passed here overrides it.
    """
    logger = get_logger()
    try:
        success = run_gwas_workflow(task, dry_run, output_format)
    except (ValueError, RuntimeError) as exc:
        logger.error(str(exc))
        sys.exit(1)

    if success:
        logger.info("gwas workflow finished successfully")
    else:
        logger.warning("encountered error during gwas workflow")
