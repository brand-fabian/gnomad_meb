"""VDS CLI application."""

import json
import os
import sys
from typing import get_args

import click
import yaml

from gnomad_meb.cmd.task import (
    Task,
    VdsCreate,
    VdsFind,
    VdsInspect,
    task_command,
)
from gnomad_meb.config import ENV_PREFIX
from gnomad_meb.constants import REFERENCE_GENOMES
from gnomad_meb.logging import get_logger
from gnomad_meb.s3 import S3ConnectionOptions, S3Provider

logger = get_logger()


@click.group()
def vds():
    """Commands to create and manipulate vds objects."""


@vds.command()
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
    help="Name of the VDS files",
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
    help="Reference genome build to use",
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
    help="Rebuild every split, discarding prior results.",
)
@task_command(VdsCreate, pipeline="vds")
def create(task: Task, dry_run: bool = False, output_format: str = "json"):
    """Create a new VDS from a set of gvcf files or search directories."""
    from gnomad_meb.lib.vds import VdsCreatePipeline
    from gnomad_meb.pipeline import PipelineRunner

    logger = get_logger()

    cmd: VdsCreate = task.command
    logger.info(f"creating new VDS at {cmd.output_base}")

    try:
        pipeline = VdsCreatePipeline(task)
        runner = PipelineRunner(task, [pipeline], overwrite=cmd.overwrite)
        if dry_run:
            from gnomad_meb.cmd.output import emit_dry_run

            emit_dry_run(runner, output_format)
            return

        runner.run()

        if runner.has_error() or not runner.has_finished():
            sys.exit(1)
    except Exception as e:
        import traceback

        logger.error(f"error processing variant qc export pipeline: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


@vds.command()
@click.option(
    "-V",
    "--vds",
    default=None,
    required=True,
    help="VDS instance to inspect.",
    type=str,
)
@click.option(
    "--format",
    help="Output format.",
    default="json",
    type=click.Choice(["yaml", "json"], case_sensitive=True),
)
@task_command(VdsInspect)
def inspect(task: Task, dry_run: bool = False, output_format: str = "json"):
    """Inspect a given VDS dataset, printing its parameters."""
    if dry_run:
        from gnomad_meb.cmd.output import emit, manifest_payload

        emit(manifest_payload(None, task), output_format)
        return
    inspect_dataset(task)


def inspect_dataset(task: Task) -> None:
    """Print a VDS dataset's parameters, importing the VDS lib lazily."""
    from gnomad_meb.lib.vds import VariantDataset

    cmd: VdsInspect = task.command
    logger.info(f"inspecting vds dataset at {cmd.vds}")
    dataset = VariantDataset.open(cmd.vds)
    summary = dataset.inspect()
    if cmd.format == "yaml":
        print(yaml.dump(summary))
    elif cmd.format == "json":
        print(json.dumps(summary, indent=4))
    else:
        logger.error(f"invalid format specifier: {cmd.format}")
        sys.exit(1)


@vds.command()
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
    "--format",
    help=(
        "Output format for the command. Yaml and Json will output a "
        "semi-structured format, split by source directory or bucket."
    ),
    default="json",
    type=click.Choice(["text", "yaml", "json"], case_sensitive=True),
)
@task_command(VdsFind)
def find(task: Task, dry_run: bool = False, output_format: str = "json"):
    """Search local files and S3 buckets for files.

    Performs the same search routines that will be used to find files for VDS
    creation in the specified search directories and S3 buckets.
    """
    if dry_run:
        from gnomad_meb.cmd.output import emit, manifest_payload

        emit(manifest_payload(None, task), output_format)
        return
    find_files(task)


def find_files(task: Task) -> None:
    """Discover files for ``vds.find``; pure discovery, no hail."""
    from gnomad_meb.lib.utils import glob

    cfg = task.settings
    cmd: VdsFind = task.command

    base_files = list(*cmd.gvcf)
    s3_files = []
    local_files = []

    if len(cmd.bucket) > 0:
        logger.info(f"searching for files in {len(cmd.bucket)} buckets")
        s3 = S3Provider(
            S3ConnectionOptions(
                cfg.s3.access_key,
                cfg.s3.secret_key,
                cfg.s3.endpoint_url,
                cfg.s3.verify,
            )
        )
        s3_files.extend(
            list(s3.glob(cmd.bucket, patterns=cmd.s3_search_pattern, return_url=True))
        )
    if len(cmd.search) > 0:
        logger.info(f"searching for files in {len(cmd.search)} local directories")
        local_files.extend(glob(cmd.search, patterns=cmd.local_search_pattern))

    logger.info(
        f"found {len(base_files) + len(s3_files) + len(local_files)} files "
        "matching the glob patterns."
    )
    if cmd.format == "text":
        print(
            "\n".join(
                [
                    *base_files,
                    *s3_files,
                    *local_files,
                ]
            )
        )
    elif cmd.format == "json":
        print(
            json.dumps(
                {
                    "base": base_files,
                    "local": local_files,
                    "s3": s3_files,
                },
                indent=4,
            )
        )
    elif cmd.format == "yaml":
        print(
            yaml.dump(
                {
                    "base": base_files,
                    "local": local_files,
                    "s3": s3_files,
                }
            )
        )
    else:
        logger.error(f"invalid format specifier: {cmd.format}")
        sys.exit(1)
