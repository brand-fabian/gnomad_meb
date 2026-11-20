"""In-process tests for the merged task description.

These assert the external behaviour of the CLI's task layer: that a resolved
``Task`` serializes and reloads unchanged without consulting the environment or
a config file, and that ``--describe`` prints the task and exits before hail is
initialized. See ``docs/adr/0005`` and ``docs/adr/0006``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from textwrap import dedent
from typing import Any

import pytest
import yaml
from click.testing import CliRunner
from pydantic import ValidationError

import gnomad_meb.config as config_module
from gnomad_meb.cmd import main
from gnomad_meb.cmd.output import parse_manifest
from gnomad_meb.cmd.task import (
    AnnotationRun,
    CommandModel,
    FilterRun,
    GnomadWorkflowRun,
    GwasPipeline,
    SampleQcRun,
    Task,
    VariantQcExport,
    VariantQcLoadVqsr,
    VdsCreate,
    VdsFind,
    VdsInspect,
    VqsrRun,
)
from gnomad_meb.config import INVALID_TOKEN, Settings, resolve


@pytest.fixture(autouse=True)
def _clear_ambient_gnomeb(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop any ambient ``GMEB_`` variables and cached config around each test."""
    for key in [k for k in os.environ if k.startswith("GMEB_")]:
        monkeypatch.delenv(key, raising=False)
    config_module._config = None
    config_module._config_path = None


def _write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "gnomad_meb.toml"
    path.write_text(dedent(body).lstrip())
    return path


def _settings() -> Settings:
    return Settings(validate_paths=False)


# --- the command union -------------------------------------------------------

_COMMANDS: tuple[CommandModel, ...] = (
    VdsCreate(
        gvcf=("a.g.vcf.gz",),
        bucket=("bucket",),
        search=("/search",),
        vds_name="chunked",
        output_base="./out",
        reference_genome="GRCh37",
        n_splits=8,
        interval_size=500_000,
    ),
    GnomadWorkflowRun(
        gvcf=("a.g.vcf.gz",),
        output_base="./out",
        vds_name="workflow",
        overwrite=True,
    ),
    VdsInspect(vds="a.vds", format="yaml"),
    VdsFind(bucket=("bucket",), format="text"),
    SampleQcRun(vds="a.vds"),
    VariantQcExport(
        vds="a.vds",
        as_info="as_info.vcf",
        quasi_info="quasi_info.vcf",
        infer_as_annotations=False,
        cdf_k=10,
        retain_cdfs=True,
    ),
    VariantQcLoadVqsr(vds="a.vds", vcf="vqsr.vcf", as_info="as_info.ht"),
    VqsrRun(
        vds="a.vds",
        as_info="as_info",
        output_dir="./out",
        calling_intervals="/r/calling.interval_list",
        snakemake=("--profile", "slurm"),
    ),
    AnnotationRun(vds="a.vds"),
    FilterRun(mt="a.mt"),
    GwasPipeline(
        dataset="a.dataset",
        output_dir="out",
        snakemake=("--executor", "slurm"),
    ),
)


@pytest.mark.parametrize("command", _COMMANDS, ids=lambda c: c.kind)
def test_task_serialization_round_trip(command: CommandModel) -> None:
    task = Task(settings=_settings(), command=command)

    reloaded = Task.model_validate_json(task.model_dump_json())

    assert reloaded.model_dump() == task.model_dump()
    assert reloaded.command.kind == command.kind


def test_command_fields_survive_the_round_trip() -> None:
    task = Task(
        settings=_settings(),
        command=VqsrRun(
            vds="a.vds",
            as_info="as_info",
            output_dir="./out",
            calling_intervals="/r/calling.interval_list",
            snakemake=("--profile", "slurm"),
        ),
    )

    reloaded = Task.model_validate_json(task.model_dump_json())

    assert reloaded.command.vds == "a.vds"
    assert reloaded.command.as_info == "as_info"
    assert reloaded.command.output_dir == "./out"
    assert reloaded.command.calling_intervals == "/r/calling.interval_list"
    assert reloaded.command.snakemake == ("--profile", "slurm")


def test_kind_is_the_discriminator_and_dumps() -> None:
    task = Task(settings=_settings(), command=VdsInspect(vds="a.vds"))

    payload = json.loads(task.model_dump_json())

    assert payload["command"]["kind"] == "vds.inspect"


def test_unknown_command_kind_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Task.model_validate(
            {"settings": _settings().model_dump(), "command": {"kind": "bogus"}}
        )


# --- reload is independent of environment and config -------------------------


def test_reload_does_not_consult_env_or_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write_config(tmp_path, 'project = "from-file"\n')
    settings = resolve(config, overrides={"validate_paths": False})
    task = Task(settings=settings, command=VdsInspect(vds="a.vds"))
    dump = task.model_dump_json()

    monkeypatch.setenv("GMEB_PROJECT", "from-env")
    config.write_text('project = "changed"\n')

    reloaded = Task.model_validate_json(dump)

    assert reloaded.settings.project == "from-file"
    assert reloaded.command.vds == "a.vds"


def test_settings_keep_the_s3_sentinel() -> None:
    task = Task(settings=_settings(), command=SampleQcRun(vds="a.vds"))

    assert task.settings.s3.access_key == INVALID_TOKEN


# --- --describe --------------------------------------------------------------


def _describe(tmp_path: Path, *args: Any) -> Any:
    config = _write_config(tmp_path, 'project = "describe"\n')
    return CliRunner().invoke(main, ["--config", str(config), "--describe", *args])


def test_describe_prints_the_task_without_initializing_hail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom() -> None:
        raise AssertionError("hail_init must not run for --describe")

    monkeypatch.setattr("gnomad_meb.hail.hail_init", _boom)

    result = _describe(tmp_path, "vds", "inspect", "-V", "a.vds")

    assert result.exit_code == 0
    payload = json.loads(result.output)
    # A command that runs no pipeline emits the envelope with pipeline=None and
    # runnable=False (D9); ticket 07 names the pipeline-backed ones.
    assert payload["pipeline"] is None
    assert payload["runnable"] is False
    assert payload["steps"] is None
    assert payload["task"]["settings"]["project"] == "describe"
    assert payload["task"]["command"]["kind"] == "vds.inspect"
    assert payload["task"]["command"]["vds"] == "a.vds"


def test_describe_names_the_pipeline_of_a_pipeline_backed_command(
    tmp_path: Path,
) -> None:
    result = _describe(tmp_path, "sample-qc", "run", "-V", "a.vds")

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["pipeline"] == "sample_qc"
    assert payload["runnable"] is True
    assert payload["steps"] is None


def test_describe_names_the_pipeline_of_every_pipeline_backed_command(
    tmp_path: Path,
) -> None:
    gvcf = tmp_path / "a.g.vcf.gz"
    gvcf.write_text("x")
    expected = {
        "vds.create": (["vds", "create", "-g", str(gvcf)], "vds"),
        "gnomad_workflow.run": (
            ["gnomad-workflow", "-g", str(gvcf)],
            "gnomad-workflow",
        ),
        "sample_qc.run": (["sample-qc", "run", "-V", "a.vds"], "sample_qc"),
        "variant_qc.export": (
            ["variant-qc", "export", "-V", "a.vds", "-a", "as.vcf"],
            "variant_qc_export",
        ),
        "variant_qc.load_vqsr": (
            [
                "variant-qc",
                "load-vqsr",
                "-V",
                "a.vds",
                "--vcf",
                "vqsr_dir",
                "--as-info",
                "as_info_dir",
            ],
            "variant_qc",
        ),
        "vqsr.pipeline": (
            [
                "vqsr",
                "pipeline",
                "-V",
                "a.vds",
                "-a",
                "as_info",
                "-o",
                "out",
                "-c",
                "/r/calling.interval_list",
            ],
            "vqsr",
        ),
        "annotation.run": (["annotation", "run", "-V", "a.vds"], "annotation"),
        "filter.run": (["filter", "run", "-M", "a.mt"], "filter"),
        "gwas.export": (["gwas", "export", "--mt", "a.mt"], "gwas_export"),
        "gwas.pipeline": (
            ["gwas", "pipeline", "--dataset", "a.dataset", "--output-dir", "out"],
            "gwas",
        ),
        "gwas.load": (
            [
                "gwas",
                "load",
                "--dataset",
                "a.dataset",
                "--gwas",
                "gwas_dir",
                "--burden",
                "burden_dir",
            ],
            "gwas_load",
        ),
        "gwas.workflow": (["gwas", "workflow", "--mt", "a.mt"], "gwas_workflow"),
    }

    for kind, (args, pipeline) in expected.items():
        result = _describe(tmp_path, *args)
        assert result.exit_code == 0, f"{kind}: {result.output}"
        payload = json.loads(result.output)
        assert payload["task"]["command"]["kind"] == kind
        assert payload["pipeline"] == pipeline, kind
        assert payload["runnable"] is True, kind
        assert payload["steps"] is None, kind
        # The envelope is the durable form ``run`` validates; a describe payload
        # carries no steps, so every registered step is eligible on replay.
        manifest = parse_manifest(payload)
        assert manifest.pipeline == pipeline, kind
        assert manifest.runnable is True, kind
        assert manifest.steps is None, kind


def test_flow_describe_names_the_top_level_pipeline(tmp_path: Path) -> None:
    gvcf = tmp_path / "a.g.vcf.gz"
    gvcf.write_text("x")

    result = _describe(tmp_path, "gnomad-workflow", "-g", str(gvcf))

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["pipeline"] == "gnomad-workflow"
    assert payload["runnable"] is True
    assert payload["steps"] is None


def test_describe_records_the_overwrite_flag(tmp_path: Path) -> None:
    gvcf = tmp_path / "a.g.vcf.gz"
    gvcf.write_text("x")

    result = _describe(tmp_path, "gnomad-workflow", "-g", str(gvcf), "--overwrite")

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["pipeline"] == "gnomad-workflow"
    assert payload["overwrite"] is True


def test_describe_captures_the_snakemake_passthrough(tmp_path: Path) -> None:
    result = _describe(
        tmp_path,
        "vqsr",
        "pipeline",
        "-V",
        "a.vds",
        "-a",
        "as_info",
        "-o",
        "out",
        "-c",
        "/r/calling.interval_list",
        "--profile",
        "slurm",
        "--executor",
        "slurm",
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["task"]["command"]["kind"] == "vqsr.pipeline"
    assert payload["task"]["command"]["vds"] == "a.vds"
    assert payload["task"]["command"]["snakemake"] == [
        "--profile",
        "slurm",
        "--executor",
        "slurm",
    ]


def test_describe_captures_the_vds_create_splitting_options(tmp_path: Path) -> None:
    gvcf = tmp_path / "a.g.vcf.gz"
    gvcf.write_text("x")

    result = _describe(
        tmp_path,
        "vds",
        "create",
        "-g",
        str(gvcf),
        "--n-splits",
        "8",
        "--interval-size",
        "500000",
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["task"]["command"]["kind"] == "vds.create"
    assert payload["task"]["command"]["n_splits"] == 8
    assert payload["task"]["command"]["interval_size"] == 500000


@pytest.mark.parametrize("retired", ["--input-batch-size", "--vds"])
def test_vds_create_retired_options_are_gone(tmp_path: Path, retired: str) -> None:
    gvcf = tmp_path / "a.g.vcf.gz"
    gvcf.write_text("x")

    result = _describe(tmp_path, "vds", "create", "-g", str(gvcf), retired, "x")

    assert result.exit_code != 0


def test_describe_works_for_every_leaf_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom() -> None:
        raise AssertionError("hail_init must not run for --describe")

    monkeypatch.setattr("gnomad_meb.hail.hail_init", _boom)

    gvcf = tmp_path / "a.g.vcf.gz"
    gvcf.write_text("x")
    invocations = {
        "vds.create": ["vds", "create", "-g", str(gvcf)],
        "gnomad_workflow.run": ["gnomad-workflow", "-g", str(gvcf), "--overwrite"],
        "vds.inspect": ["vds", "inspect", "-V", "a.vds"],
        "vds.find": ["vds", "find"],
        "sample_qc.run": ["sample-qc", "run", "-V", "a.vds"],
        "variant_qc.export": ["variant-qc", "export", "-V", "a.vds", "-a", "as.vcf"],
        "variant_qc.load_vqsr": [
            "variant-qc",
            "load-vqsr",
            "-V",
            "a.vds",
            "--vcf",
            "vqsr_dir",
            "--as-info",
            "as_info_dir",
        ],
        "vqsr.pipeline": [
            "vqsr",
            "pipeline",
            "-V",
            "a.vds",
            "-a",
            "as_info",
            "-o",
            "out",
            "-c",
            "/r/calling.interval_list",
        ],
        "annotation.run": ["annotation", "run", "-V", "a.vds"],
        "filter.run": ["filter", "run", "-M", "a.mt"],
        "gwas.pipeline": [
            "gwas",
            "pipeline",
            "--dataset",
            "a.dataset",
            "--output-dir",
            "out",
        ],
    }

    for kind, args in invocations.items():
        result = _describe(tmp_path, *args)
        assert result.exit_code == 0, f"{kind}: {result.output}"
        payload = json.loads(result.output)
        assert payload["task"]["command"]["kind"] == kind


def test_describe_output_format_yaml_matches_json(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        main,
        [
            "--config",
            str(_write_config(tmp_path, 'project = "describe"\n')),
            "--output-format",
            "yaml",
            "--describe",
            "vds",
            "find",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = yaml.safe_load(result.output)
    assert payload["pipeline"] is None
    assert payload["task"]["command"]["kind"] == "vds.find"


# --- --dry-run ---------------------------------------------------------------


def test_dry_run_prints_the_manifest_after_booting_hail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--dry-run`` builds the pipeline, so Hail is booted on the way in.

    Since ADR-0018's "dry-run does not initialize hail" constraint was dropped,
    the CLI boots Hail once (guarded, idempotent) right after config settles;
    the contract that matters is that stdout stays a pure payload, asserted at
    the fd level in ``test_cli_output_contract`` and ``test_import_time_boot``.
    """
    monkeypatch.setattr("gnomad_meb.hail.hail_init", lambda: None)

    gvcf = tmp_path / "a.g.vcf.gz"
    gvcf.write_text("x")
    config = _write_config(tmp_path, 'project = "dry"\n')

    result = CliRunner().invoke(
        main,
        ["--config", str(config), "--dry-run", "vds", "create", "-g", str(gvcf)],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["task"]["command"]["kind"] == "vds.create"
    assert payload["pipeline"] == "vds"
    assert payload["runnable"] is True
    assert [step["name"] for step in payload["steps"]] == ["create_variant_dataset"]


def test_dry_run_output_format_yaml(tmp_path: Path) -> None:
    gvcf = tmp_path / "a.g.vcf.gz"
    gvcf.write_text("x")
    config = _write_config(tmp_path, 'project = "dry"\n')

    result = CliRunner().invoke(
        main,
        [
            "--config",
            str(config),
            "--output-format",
            "yaml",
            "--dry-run",
            "vds",
            "create",
            "-g",
            str(gvcf),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = yaml.safe_load(result.stdout)
    assert [step["name"] for step in payload["steps"]] == ["create_variant_dataset"]


def test_dry_run_on_a_pipelineless_command_is_a_non_runnable_envelope(
    tmp_path: Path,
) -> None:
    config = _write_config(tmp_path, 'project = "dry"\n')

    result = CliRunner().invoke(
        main, ["--config", str(config), "--dry-run", "vds", "find"]
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["pipeline"] is None
    assert payload["runnable"] is False
    assert payload["steps"] is None
    assert payload["task"]["command"]["kind"] == "vds.find"


def test_describe_and_dry_run_are_mutually_exclusive(tmp_path: Path) -> None:
    config = _write_config(tmp_path, 'project = "dry"\n')

    result = CliRunner().invoke(
        main, ["--config", str(config), "--describe", "--dry-run", "vds", "find"]
    )

    assert result.exit_code != 0
    assert "mutually exclusive" in result.output


# --- verbosity is a setting, with normal precedence --------------------------


def test_verbose_defaults_to_info(tmp_path: Path) -> None:
    result = _describe(tmp_path, "vds", "find")

    assert json.loads(result.output)["task"]["settings"]["verbose"] == "info"


def test_verbose_comes_from_the_config_file(tmp_path: Path) -> None:
    config = _write_config(tmp_path, 'project = "v"\nverbose = "warning"\n')

    result = CliRunner().invoke(
        main, ["--config", str(config), "--describe", "vds", "find"]
    )

    assert json.loads(result.output)["task"]["settings"]["verbose"] == "warning"


def test_env_verbose_beats_the_config_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write_config(tmp_path, 'project = "v"\nverbose = "warning"\n')
    monkeypatch.setenv("GMEB_VERBOSE", "debug")

    result = CliRunner().invoke(
        main, ["--config", str(config), "--describe", "vds", "find"]
    )

    assert json.loads(result.output)["task"]["settings"]["verbose"] == "debug"


def test_cli_verbose_beats_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write_config(tmp_path, 'project = "v"\n')
    monkeypatch.setenv("GMEB_VERBOSE", "debug")

    result = CliRunner().invoke(
        main,
        ["--config", str(config), "-v", "error", "--describe", "vds", "find"],
    )

    assert json.loads(result.output)["task"]["settings"]["verbose"] == "error"


def test_invalid_verbose_is_rejected(tmp_path: Path) -> None:
    config = _write_config(tmp_path, 'project = "v"\nverbose = "loud"\n')

    result = CliRunner().invoke(
        main, ["--config", str(config), "--describe", "vds", "find"]
    )

    assert result.exit_code != 0
    assert "verbose" in str(result.exception)


# --- --config is required and explicit --------------------------------------


def test_config_is_required() -> None:
    result = CliRunner().invoke(main, ["vds", "inspect", "-V", "a.vds"])

    assert result.exit_code != 0
    assert "--config" in result.output or "--config" in (result.stderr or "")


def test_config_must_name_an_existing_file(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.toml"

    result = CliRunner().invoke(
        main, ["--config", str(missing), "vds", "inspect", "-V", "a"]
    )

    assert result.exit_code != 0


def test_config_rejects_a_directory(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        main, ["--config", str(tmp_path), "vds", "inspect", "-V", "a"]
    )

    assert result.exit_code != 0
