"""Replay a ``--describe`` or ``--dry-run`` manifest as an executed run.

``gmeb run PATH`` consumes the run envelope emitted by the two inspection flags
(JSON or YAML), reconstructs the graph through
:class:`~gnomad_meb.pipeline.factory.PipelineFactory`, and executes it. The load
order is load-bearing (spec "Manifest load order"): the root ``task`` is
validated and its settings re-injected into every step, an optional ``--config``
replaces those settings *before* any pipeline is built (so pipeline construction
and the runner's own ``set_config`` read the override, D7), the graph is rebuilt,
its recovery is verified against the manifest (step ids and per-step resource
edges), and only then is the runner constructed and hail started.

A manifest that names no pipeline (``runnable: false``) describes a command that
does its work directly (``vds inspect``, ``vds find``) and cannot be replayed, so
``run`` rejects it with a clear error before building anything (D9).
"""

from __future__ import annotations

from pathlib import Path

import click

from gnomad_meb.cmd.output import load_payload, parse_manifest
from gnomad_meb.config import resolve, set_config
from gnomad_meb.pipeline.execution import PipelineMismatchError, PipelineRunner
from gnomad_meb.pipeline.factory import PipelineFactory
from gnomad_meb.pipeline.runtime import LocalRuntime
from gnomad_meb.pipeline.spec import RunManifest
from gnomad_meb.pipeline.step import StepRegistry


def ensure_runnable(manifest: RunManifest) -> None:
    """Raise a :class:`click.ClickException` unless ``manifest`` is runnable.

    ``runnable`` is ``False`` exactly when ``pipeline`` is ``None`` (the model
    validator enforces it), meaning the description came from a command that
    executes no pipeline — it has nothing to reconstruct and would otherwise run
    nothing at all. ``run`` therefore refuses it loudly.
    """
    if not manifest.runnable:
        raise click.ClickException(
            "manifest is not runnable: it names no pipeline, so there is no "
            "run to reconstruct (commands like 'vds inspect' and 'vds find' "
            "do their work directly and cannot be replayed)."
        )


def verify_recovery(manifest: RunManifest, registry: StepRegistry) -> None:
    """Assert the rebuilt graph recovers every step and resource edge named.

    Every id in ``manifest.steps`` must exist in ``registry``, and each
    manifest step's declared ``resources`` edge mapping must be a subset of the
    rebuilt step's (so a pipeline that dropped a resource edge is detected).
    ``depends_on`` edges and ordering are not compared — the dependency closure
    is recalculated fresh. A miss raises :class:`PipelineMismatchError`, reusing
    the durable-state recovery vocabulary, before hail starts.
    """
    if manifest.steps is None:
        return

    serialized_ids = {step.id for step in manifest.steps}
    registered_ids = set(registry.steps())
    missing_ids = serialized_ids - registered_ids
    if missing_ids:
        raise PipelineMismatchError(
            "manifest steps are missing from the rebuilt graph: "
            f"missing = {sorted(missing_ids)}"
        )

    for step in manifest.steps:
        rebuilt = registry[step.id]
        extra_resources = set(step.resources.items()) - set(rebuilt.resources.items())
        if extra_resources:
            raise PipelineMismatchError(
                "manifest resource edges are missing from the rebuilt step: "
                f"step = {step.id}, missing = {sorted(extra_resources)}"
            )


def load_run_manifest(
    path: str | Path, *, config: str | Path | None = None
) -> RunManifest:
    """Load, validate, and settings-resolve a manifest (load order steps 1-5).

    Validates the root ``task`` first and re-injects its settings into every
    step (``parse_manifest``), rejects a non-runnable envelope and an explicit
    empty ``steps: []``, then, if ``config`` is given, replaces the manifest's
    settings with the resolved override and re-injects it.
    """
    manifest = parse_manifest(load_payload(path))
    ensure_runnable(manifest)

    if manifest.steps is not None and not manifest.steps:
        raise click.ClickException(
            "manifest carries an empty 'steps' list: an explicit empty selection "
            "is not a whole-graph run. Remove 'steps' to run every registered "
            "step, or list the steps to run."
        )

    if config is not None:
        settings = resolve(Path(config))
        manifest.task.settings = settings
        if manifest.steps is not None:
            for step in manifest.steps:
                step.task.settings = settings

    return manifest


def build_runner(
    manifest: RunManifest,
    *,
    overwrite: bool = False,
    select: tuple[str, ...] = (),
) -> PipelineRunner:
    """Reconstruct the graph and build the runner (load order steps 6-9).

    Installs the manifest's (possibly overridden) settings, instantiates the
    named pipeline through the factory, verifies recovery against the rebuilt
    registry *before* the runner is constructed (so an unknown id reports a
    :class:`PipelineMismatchError` rather than a selection error), and builds the
    runner over the eligible set the manifest names.
    """
    set_config(manifest.task.settings)
    pipeline = PipelineFactory.make(manifest.pipeline, manifest.task)

    verify_recovery(manifest, StepRegistry())

    eligible = (
        frozenset(step.id for step in manifest.steps)
        if manifest.steps is not None
        else None
    )
    return PipelineRunner(
        manifest.task,
        [pipeline],
        overwrite=manifest.overwrite or overwrite,
        eligible=eligible,
        select=select,
        runtime=LocalRuntime(),
    )


@click.command()
@click.argument("path", type=click.Path(exists=True, readable=True, dir_okay=False))
@click.option(
    "--select",
    "select",
    multiple=True,
    help="Glob over step ids to run (repeatable); dependencies are pulled in.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Rebuild artifacts even where they exist (ORs with the manifest).",
)
@click.pass_context
def run(
    ctx: click.Context, path: str, select: tuple[str, ...], overwrite: bool
) -> None:
    """Execute the run described by the manifest at ``PATH``."""
    root = ctx.find_root()
    obj = root.obj or {}
    if obj.get("describe") or obj.get("dry_run"):
        raise click.UsageError(
            "--describe/--dry-run describe an invocation; they cannot be combined "
            "with 'run', which consumes an existing description.",
            ctx=ctx,
        )

    config = None
    config_source = root.params.get("config")
    if config_source is not None:
        config = config_source

    manifest = load_run_manifest(path, config=config)
    runner = build_runner(manifest, overwrite=overwrite, select=select)

    from gnomad_meb.hail import hail_init

    hail_init()
    runner.run()

    if runner.has_error():
        raise click.ClickException(
            "run finished with errored steps: see the report above."
        )
