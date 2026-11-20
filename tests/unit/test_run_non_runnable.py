"""Ticket 09: ``run`` refuses a ``runnable: false`` manifest.

Commands that do their work directly without registering a pipeline
(``vds inspect``, ``vds find``) still support ``--describe``, but their payload
carries ``pipeline: null`` / ``runnable: false``. ``run`` must reject such a
manifest with a clear error rather than silently running nothing. A
pipeline-backed description remains runnable, so the two describe payloads are
told apart by the gate that ``run`` applies before it builds anything.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import click
import pytest
from click.testing import CliRunner

from gnomad_meb.cmd import main
from gnomad_meb.cmd.output import parse_manifest
from gnomad_meb.cmd.run import ensure_runnable
from gnomad_meb.cmd.task import PipelineStep, Task
from gnomad_meb.config import Settings


def _write_config(tmp_path: Path) -> Path:
    path = tmp_path / "gnomad_meb.toml"
    path.write_text(
        dedent("""\
        project = "run-09"
    """)
    )
    return path


def _describe(tmp_path: Path, *args: str) -> dict:
    config = _write_config(tmp_path)
    result = CliRunner().invoke(main, ["--config", str(config), "--describe", *args])
    assert result.exit_code == 0, result.output
    return json.loads(result.output)


def _write_payload(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))
    return path


def _manifest(pipeline: str | None) -> dict:
    task = Task(
        settings=Settings(validate_paths=False, project="demo"),
        command=PipelineStep(name="test"),
    )
    return {
        "pipeline": pipeline,
        "task": task.model_dump(mode="json"),
        "steps": None,
        "overwrite": False,
        "runnable": pipeline is not None,
    }


def test_vds_inspect_describe_is_not_runnable(tmp_path: Path) -> None:
    payload = _describe(tmp_path, "vds", "inspect", "-V", "a.vds")

    assert payload["pipeline"] is None
    assert payload["runnable"] is False


def test_vds_find_describe_is_not_runnable(tmp_path: Path) -> None:
    gvcf = tmp_path / "a.g.vcf.gz"
    gvcf.write_text("x")
    payload = _describe(tmp_path, "vds", "find", "-g", str(gvcf))

    assert payload["pipeline"] is None
    assert payload["runnable"] is False


def test_run_rejects_a_vds_inspect_describe(tmp_path: Path) -> None:
    payload = _describe(tmp_path, "vds", "inspect", "-V", "a.vds")
    path = _write_payload(tmp_path, payload)
    config = _write_config(tmp_path)

    result = CliRunner().invoke(main, ["--config", str(config), "run", str(path)])

    assert result.exit_code != 0
    assert "not runnable" in result.output.lower()


def test_run_rejects_a_vds_find_describe(tmp_path: Path) -> None:
    gvcf = tmp_path / "a.g.vcf.gz"
    gvcf.write_text("x")
    payload = _describe(tmp_path, "vds", "find", "-g", str(gvcf))
    path = _write_payload(tmp_path, payload)
    config = _write_config(tmp_path)

    result = CliRunner().invoke(main, ["--config", str(config), "run", str(path)])

    assert result.exit_code != 0
    assert "not runnable" in result.output.lower()


def test_gate_admits_a_pipeline_backed_manifest() -> None:
    ensure_runnable(parse_manifest(_manifest("sample_qc")))  # does not raise


def test_gate_refuses_a_pipelineless_manifest() -> None:
    with pytest.raises(click.ClickException, match="not runnable"):
        ensure_runnable(parse_manifest(_manifest(None)))
