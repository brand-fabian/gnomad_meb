"""Tests for the browser-export scaffolding (ticket 01).

These assert the external seams the scaffolding establishes: the
``browser_export`` pipeline is factory-discoverable, the ``browser run`` command
is reachable from the root CLI and parses ``--mt``, ``BrowserExportConfig``
resolves with and without a ``[browser_export]`` section, ``--describe`` stays
Hail-free, ``get_input()`` resolves the annotation checkpoint, and importing the
package does not import hail. The reshape step itself is covered by
``test_browser_steps.py``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest
from click.testing import CliRunner

from gnomad_meb.cmd import main
from gnomad_meb.cmd.run import build_runner, load_run_manifest
from gnomad_meb.cmd.task import BrowserExportRun, Task
from gnomad_meb.config import BrowserExportConfig, Settings, resolve
from gnomad_meb.constants import DEFAULT_FIELDS, MANDATORY_FIELDS
from gnomad_meb.lib.browser import BrowserExportPipeline
from gnomad_meb.lib.browser import constants as browser_constants
from gnomad_meb.lib.browser.constants import (
    BROWSER_PREFIX,
    CONTIG_FILENAME_TEMPLATE,
    CONTIG_ORDER,
    CONTIG_PROVENANCE_TEMPLATE,
    PARQUET_DATASET_NAME,
)
from gnomad_meb.pipeline import PipelineFactory, PipelineRunner, StepRegistry

REPO_ROOT = Path(__file__).resolve().parents[2]


def _settings(*, project: str = "proj", checkpoint: str = "/ckpt") -> Settings:
    return Settings(validate_paths=False, project=project, checkpoint=checkpoint)


def _write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "gnomad_meb.toml"
    path.write_text(dedent(body).lstrip())
    return path


def _describe(tmp_path: Path, *args: str) -> object:
    config = _write_config(tmp_path, 'project = "describe"\n')
    return CliRunner().invoke(main, ["--config", str(config), "--describe", *args])


# --- constants ---------------------------------------------------------------


def test_mandatory_fields_are_the_browser_column_order() -> None:
    """The fixed physical columns, in the order the browser SQL reads them."""
    assert MANDATORY_FIELDS == (
        "locus",
        "alleles",
        "variant_id",
        "xpos",
        "rsids",
        "caid",
        "exome",
        "genome",
        "joint",
        "transcript_consequences",
        "in_silico_predictors",
        "coverage",
    )


def test_default_fields_are_the_passthrough_defaults() -> None:
    assert DEFAULT_FIELDS == ("in_silico_predictors",)


def test_browser_constants_are_stable() -> None:
    assert BROWSER_PREFIX == "{project}/browser"
    assert PARQUET_DATASET_NAME == "variants.parquet"
    assert CONTIG_FILENAME_TEMPLATE == "locus.contig={contig}"
    assert CONTIG_PROVENANCE_TEMPLATE == "locus.contig={contig}.provenance.json"
    assert CONTIG_ORDER == tuple(
        [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM"]
    )


def test_browser_constants_reexport_the_field_lists() -> None:
    assert browser_constants.MANDATORY_FIELDS is MANDATORY_FIELDS
    assert browser_constants.DEFAULT_FIELDS is DEFAULT_FIELDS


# --- pipeline declaration and factory ----------------------------------------


def test_browser_export_pipeline_name_is_stable() -> None:
    assert BrowserExportPipeline.name() == "browser_export"


def test_browser_export_is_factory_discoverable() -> None:
    assert "browser_export" in PipelineFactory.list()


def test_factory_make_returns_the_pipeline() -> None:
    task = Task(settings=_settings(), command=BrowserExportRun(mt="in.mt"))

    pipeline = PipelineFactory.make("browser_export", task)

    assert isinstance(pipeline, BrowserExportPipeline)


def test_get_input_defers_to_the_annotation_checkpoint_without_mt() -> None:
    BrowserExportPipeline(Task(settings=_settings(), command=BrowserExportRun()))

    step = StepRegistry().get("browser_export.mt")

    assert step is not None
    assert step.source == "annotation.merge"
    assert step.value is None


def test_get_input_returns_the_mt_override() -> None:
    BrowserExportPipeline(
        Task(settings=_settings(), command=BrowserExportRun(mt="in.mt"))
    )

    assert StepRegistry().get("browser_export.mt").value == "in.mt"


def test_dry_run_manifest_round_trips(tmp_path: Path) -> None:
    task = Task(settings=_settings(), command=BrowserExportRun(mt="in.mt"))
    runner = PipelineRunner(task, [BrowserExportPipeline(task)])
    manifest = runner.manifest()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))

    rebuilt = build_runner(load_run_manifest(path))

    def _ids(payload: dict) -> list[str]:
        return [
            f"{step['group']}.{step['name']}" if step["group"] else step["name"]
            for step in payload["steps"]
        ]

    assert _ids(manifest) == [
        "browser_export.mt",
        "browser_export.browser_sites",
        *[f"browser_export.export_contig.{contig}" for contig in CONTIG_ORDER],
    ]
    assert _ids(rebuilt.manifest()) == _ids(manifest)


# --- configuration -----------------------------------------------------------


def test_browser_export_config_defaults_to_default_fields() -> None:
    assert Settings(validate_paths=False).browser_export.fields == list(DEFAULT_FIELDS)


def test_browser_export_section_absent_defaults(tmp_path: Path) -> None:
    config = _write_config(tmp_path, 'project = "partial"\n')

    settings = resolve(config, overrides={"validate_paths": False})

    assert settings.project == "partial"
    assert settings.browser_export.fields == list(DEFAULT_FIELDS)


def test_browser_export_section_resolves_from_toml(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path,
        """
        [browser_export]
        fields = ["gnomad_freq", "clinvar"]
        """,
    )

    settings = resolve(config, overrides={"validate_paths": False})

    assert settings.browser_export.fields == ["gnomad_freq", "clinvar"]


def test_browser_export_config_round_trips_through_settings() -> None:
    settings = Settings(
        validate_paths=False, browser_export=BrowserExportConfig(fields=["grpmax"])
    )

    reloaded = Settings.model_validate_json(settings.model_dump_json())

    assert reloaded.browser_export == settings.browser_export
    assert reloaded.browser_export.fields == ["grpmax"]


def test_browser_export_config_is_reexported() -> None:
    from gnomad_meb.config import BrowserExportConfig as Reexported

    assert Reexported is BrowserExportConfig


# --- CLI ---------------------------------------------------------------------


def test_root_cli_exposes_browser_run_and_parses_mt(tmp_path: Path) -> None:
    result = _describe(tmp_path, "browser", "run", "--mt", "a.mt")

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["task"]["command"]["kind"] == "browser.run"
    assert payload["task"]["command"]["mt"] == "a.mt"
    assert payload["pipeline"] == "browser_export"
    assert payload["runnable"] is True


def test_browser_run_mt_is_optional(tmp_path: Path) -> None:
    result = _describe(tmp_path, "browser", "run")

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["task"]["command"]["mt"] is None


def test_browser_describe_does_not_initialize_hail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom() -> None:
        raise AssertionError("hail_init must not run for --describe")

    monkeypatch.setattr("gnomad_meb.hail.hail_init", _boom)

    result = _describe(tmp_path, "browser", "run", "--mt", "a.mt")

    assert result.exit_code == 0, result.output


# --- import purity -----------------------------------------------------------


def _run_clean_probe(code: str, tmp_path: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    for key in [key for key in env if key.startswith("GMEB_")]:
        del env[key]
    return subprocess.run(  # noqa: S603 - test-owned subprocess, fixed argv
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(tmp_path),
        timeout=300,
    )


def test_importing_the_browser_package_does_not_import_hail(tmp_path: Path) -> None:
    probe = (
        "import sys\n"
        "import gnomad_meb.lib.browser\n"
        "assert 'hail' not in sys.modules, sorted(\n"
        "    m for m in sys.modules if m == 'hail' or m.startswith('hail.'))\n"
        "import gnomad_meb.lib.browser.constants\n"
        "assert 'gnomad_meb.lib.browser.export' not in sys.modules\n"
        "assert 'hail' not in sys.modules\n"
        "print('ok')\n"
    )

    result = _run_clean_probe(probe, tmp_path)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
