"""Serialization of task and run manifests for ``--describe`` and ``--dry-run``.

The two inspection flags share one serialization concern: both emit the same
:class:`~gnomad_meb.pipeline.spec.RunManifest` envelope, rendered as JSON or
YAML on stdout. The envelope is built with pydantic ``model_dump(mode="json")``,
so the bytes are the same validated, declarative description the ``run`` command
consumes; only the encoder differs. ``--describe`` omits ``steps`` (every
registered step is eligible), ``--dry-run`` includes the dependency-ordered
step list.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
import yaml

from gnomad_meb.cmd.task import Task
from gnomad_meb.pipeline.spec import RunManifest

if TYPE_CHECKING:
    from gnomad_meb.pipeline import PipelineRunner


def emit(payload: dict[str, Any], output_format: str) -> None:
    """Print ``payload`` to stdout as ``json`` or ``yaml``."""
    if output_format.lower() == "yaml":
        click.echo(yaml.safe_dump(payload, sort_keys=False))
    else:
        click.echo(json.dumps(payload, indent=2))


def emit_dry_run(runner: PipelineRunner, output_format: str) -> None:
    """Print ``runner``'s plan and manifest without executing any step."""
    manifest = runner.dry_run()
    emit(manifest, output_format)


def manifest_payload(
    pipeline: str | None,
    task: Task,
    *,
    steps: list[dict[str, Any]] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """The run envelope: one model for ``--describe`` and ``--dry-run``.

    ``pipeline`` names the top-level pipeline, or is ``None`` for a command that
    runs no pipeline (``runnable`` is then ``False``). ``steps`` is ``None`` for
    a description and the dependency-ordered step dumps for a dry-run manifest;
    each dump already omits ``task.settings`` (the root task is authoritative).
    """
    return RunManifest.build(pipeline, task, steps=steps, overwrite=overwrite)


def parse_manifest(payload: dict[str, Any]) -> RunManifest:
    """Validate a raw manifest payload into a :class:`RunManifest`.

    A step dump omits ``task.settings`` (the root ``task`` is the one
    authoritative copy), and ``Task.settings`` is required, so the root task is
    validated first, its settings are re-injected into every ``steps[i].task``,
    and only then is the whole envelope validated. A dump that names no steps
    skips the re-injection.
    """
    task = Task.model_validate(payload["task"])
    payload = {**payload, "task": task}
    payload = RunManifest.reinject_settings(payload, task)

    return RunManifest.model_validate(payload)


def load_manifest(path: str | Path) -> RunManifest:
    """Read and validate a manifest file (JSON or YAML)."""
    return parse_manifest(load_payload(path))


def load_payload(path: str | Path) -> dict[str, Any]:
    """Read a manifest file, sniffing JSON then YAML.

    The file extension is not trusted: the bytes are tried as JSON first (the
    ``emit`` output for a JSON payload) and fall back to YAML (which is a JSON
    superset, but a JSON payload round-trips more cheaply through ``json``).
    """
    text = Path(path).read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return yaml.safe_load(text)
