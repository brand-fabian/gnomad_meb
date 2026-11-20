"""Main CLI application."""

from pathlib import Path
from typing import Any

import click
from click.core import ParameterSource
from click.decorators import pass_context

from gnomad_meb.config import resolve, set_config
from gnomad_meb.logging import configure_logging, get_log_level, set_level


@click.group()
@click.option(
    "-v",
    "--verbose",
    default="info",
    type=click.Choice(get_log_level().keys(), case_sensitive=False),
    help="Set output verbosity.",
)
@click.option(
    "-c",
    "--config",
    required=False,
    default=None,
    type=click.Path(exists=True, readable=True, dir_okay=False),
    help="Configuration file for this utility, detailing filter and annotation settings.",
)
@click.option(
    "-p",
    "--project-name",
    default=None,
    type=str,
    help="Overwrite or set the project name in the configuration",
)
@click.option(
    "--describe",
    is_flag=True,
    default=False,
    help="Print the resolved task description and exit without running.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help=(
        "Print the run plan and a manifest of every step, then exit without "
        "executing or initializing hail."
    ),
)
@click.option(
    "--output-format",
    default="json",
    type=click.Choice(["json", "yaml"], case_sensitive=False),
    help="Serialization format for --describe and --dry-run output.",
)
@pass_context
def main(ctx, verbose, config, project_name, describe, dry_run, output_format):
    """Gnomad analysis pipeline CLI."""
    overrides: dict[str, Any] = {}
    if ctx.get_parameter_source("verbose") == ParameterSource.COMMANDLINE:
        overrides["verbose"] = verbose
    if project_name is not None:
        overrides["project"] = project_name

    if describe and dry_run:
        raise click.UsageError(
            "--describe and --dry-run are mutually exclusive.", ctx=ctx
        )

    set_level(verbose)
    settings = None
    if config is not None:
        settings = resolve(Path(config), overrides=overrides)
        configure_logging(
            settings.verbose,
            hail_level=settings.hail.log_level,
            scratch_dir=settings.scratch_dir,
        )
        set_config(settings, config=Path(config))
    elif ctx.invoked_subcommand != "run":
        # ``run`` is the one command that carries its settings in the manifest
        # (spec D7); every other leaf command still requires ``--config``.
        raise click.UsageError("Missing option '-c' / '--config'.", ctx=ctx)

    ctx.obj = {
        "settings": settings,
        "describe": describe,
        "dry_run": dry_run,
        "output_format": output_format,
    }
