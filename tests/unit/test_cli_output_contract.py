"""The stdout/stderr contract: stdout is the payload, stderr is the log.

``--describe`` and ``--dry-run`` emit exactly one machine-readable payload (JSON
or YAML) on stdout, with every log line on stderr. The reported bug was a raw
JVM/log4j banner prepended to stdout on a pipeline whose ``prepare()`` imports
Hail (``annotation``, ``vds create``, the composed workflow); ``SampleQc`` was
clean only because Hail is imported lazily inside its step bodies.

Both flags share one purity standard: their only difference is the richness of
the envelope (``steps`` omitted vs. present). The regression case must therefore
drive a **Hail-importing ``prepare()``**; a ``vds create`` \u2014 let alone a
SampleQc \u2014 only dry-run passes vacuously.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import pytest
import yaml
from click.testing import CliRunner

import gnomad_meb.cmd.task  # noqa: F401  (breaks the execution/cmd.run cycle)
from gnomad_meb.cmd import main

BANNER_STRINGS = ("NativeCodeLoader", "SparkConf")


# --- a minimal annotation pipeline stub --------------------------------------


class _FakeTaskIntervals:
    """One VDS task interval; ``prepare()`` reads only ``.name``."""

    def __init__(self, name: str):
        self.name = name


class _FakeParams:
    def __init__(self, names):
        self.tasks = tuple(_FakeTaskIntervals(n) for n in names)


class _FakeVariantDataset:
    def __init__(self, names):
        self.params = _FakeParams(names)

    @classmethod
    def open(cls, path):
        return cls(["task-0"])


def _write_config(tmp_path: Path) -> Path:
    path = tmp_path / "gnomad_meb.toml"
    path.write_text(dedent('project = "contract"\n\n[annotation]\n').lstrip())
    return path


@pytest.fixture
def clean_resources(monkeypatch):
    """Give the stage pipelines an empty resource table.

    The real resource tables need on-disk reference data; this contract is about
    the *stream*, so the pipeline only needs to build.
    """
    from gnomad_meb.lib.sample_qc.pipeline import SampleQcPipeline

    monkeypatch.setattr(
        SampleQcPipeline, "resources", property(lambda self: ()), raising=False
    )


@pytest.fixture
def annotated_pipeline(monkeypatch):
    """Make the Annotation pipeline buildable without the VDS/hail stack.

    ``AnnotationPipeline.prepare()`` opens the ``VariantDataset`` (which imports
    ``gnomad_meb.hail`` at module scope) and reads its task list; the real
    resource table needs on-disk reference data. Stub both so the dry-run
    reaches the import.
    """
    monkeypatch.setattr(
        "gnomad_meb.lib.vds.VariantDataset", _FakeVariantDataset, raising=False
    )
    monkeypatch.setattr(
        "gnomad_meb.lib.vds.dataset.VariantDataset", _FakeVariantDataset, raising=False
    )
    from gnomad_meb.lib.annotation.pipeline import AnnotationPipeline

    monkeypatch.setattr(
        AnnotationPipeline, "resources", property(lambda self: ()), raising=False
    )


def _invoke(tmp_path: Path, *args: str):
    config = _write_config(tmp_path)
    return CliRunner().invoke(main, ["--config", str(config), *args])


# --- --describe ---------------------------------------------------------------


def test_describe_stdout_is_exactly_the_serialized_payload(tmp_path: Path) -> None:
    result = _invoke(
        tmp_path,
        "--describe",
        "--output-format",
        "json",
        "vds",
        "inspect",
        "-V",
        "a.vds",
    )

    assert result.exit_code == 0, result.output
    # The whole stream is the payload: it starts with the JSON object and has no
    # trailing text beyond the newline ``click.echo`` adds.
    assert result.stdout.startswith("{")
    assert result.stdout.endswith("}\n")
    json.loads(result.stdout)  # parses the WHOLE stream
    for banner in BANNER_STRINGS:
        assert banner not in result.stdout


def test_describe_puts_the_info_line_on_stderr(tmp_path: Path) -> None:
    result = _invoke(tmp_path, "--describe", "vds", "inspect", "-V", "a.vds")

    # A describe emits only the payload; any log line is on stderr, not stdout.
    assert result.stdout.startswith("{")
    assert "starting" not in result.stdout


def test_describe_yaml_stdout_parses_whole(tmp_path: Path) -> None:
    result = _invoke(
        tmp_path,
        "--describe",
        "--output-format",
        "yaml",
        "vds",
        "inspect",
        "-V",
        "a.vds",
    )

    assert result.exit_code == 0, result.output
    payload = yaml.safe_load(result.stdout)
    assert payload["pipeline"] is None


# --- --dry-run ----------------------------------------------------------------


def test_dry_run_json_parses_the_whole_stdout(tmp_path: Path) -> None:
    gvcf = tmp_path / "a.g.vcf.gz"
    gvcf.write_text("x")

    result = _invoke(
        tmp_path,
        "--dry-run",
        "--output-format",
        "json",
        "vds",
        "create",
        "-g",
        str(gvcf),
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)  # whole stream, not a substring
    assert payload["pipeline"] == "vds"


def test_dry_run_yaml_parses_the_whole_stdout(tmp_path: Path) -> None:
    gvcf = tmp_path / "a.g.vcf.gz"
    gvcf.write_text("x")

    result = _invoke(
        tmp_path,
        "--dry-run",
        "--output-format",
        "yaml",
        "vds",
        "create",
        "-g",
        str(gvcf),
    )

    assert result.exit_code == 0, result.output
    payload = yaml.safe_load(result.stdout)
    assert payload["pipeline"] == "vds"


@pytest.mark.parametrize("flag", ["--dry-run", "--describe"])
def test_annotation_inspection_stdout_has_no_jvm_banner(
    tmp_path: Path, annotated_pipeline, flag
) -> None:
    """The contaminated path: Annotation ``prepare()`` imports Hail.

    ``--describe`` does not build the pipeline today, but it is held to the same
    purity standard and must not regress if that changes.
    """
    result = _invoke(
        tmp_path, flag, "--output-format", "json", "annotation", "run", "-V", "a.vds"
    )

    assert result.exit_code == 0, result.output
    json.loads(result.stdout)  # whole stream
    for banner in BANNER_STRINGS:
        assert banner not in result.stdout


@pytest.mark.parametrize("flag", ["--dry-run", "--describe"])
def test_sample_qc_dry_run_is_the_clean_negative_control(
    tmp_path: Path, clean_resources, flag
) -> None:
    """SampleQc never imports Hail in ``prepare()``; it must stay clean."""
    result = _invoke(
        tmp_path, flag, "--output-format", "json", "sample-qc", "run", "-V", "a.vds"
    )

    assert result.exit_code == 0, result.output
    json.loads(result.stdout)
    for banner in BANNER_STRINGS:
        assert banner not in result.stdout


def test_dry_run_manifest_differs_from_describe_only_by_steps(tmp_path: Path) -> None:
    """One purity standard, one envelope: the flags differ only in ``steps``."""
    gvcf = tmp_path / "a.g.vcf.gz"
    gvcf.write_text("x")
    config = _write_config(tmp_path)

    describe = CliRunner().invoke(
        main,
        ["--config", str(config), "--describe", "vds", "create", "-g", str(gvcf)],
    )
    dry_run = CliRunner().invoke(
        main,
        ["--config", str(config), "--dry-run", "vds", "create", "-g", str(gvcf)],
    )

    described = json.loads(describe.stdout)
    planned = json.loads(dry_run.stdout)
    assert described["steps"] is None
    assert planned["steps"] is not None
    assert described["task"] == planned["task"]
    assert described["pipeline"] == planned["pipeline"]
