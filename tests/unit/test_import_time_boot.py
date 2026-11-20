"""Regression tests for the JVM-boot stdout contamination.

Importing ``gnomad_meb.hail`` is cheap and does not boot the JVM; the boot
happens when Hail's backend first comes up, which is inside :func:`hail_init`.
``hail_init`` points OS fd 1 at fd 2 for its whole body, so the log4j startup
banner (``NativeCodeLoader``/``SparkConf``) lands on stderr and stdout stays a
clean payload stream for ``--dry-run``.

The fd swap is a process-wide, OS-level redirect, so the probes that observe it
run in a subprocess.
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

import gnomad_meb.cmd.task  # noqa: F401  (breaks the execution/cmd.run cycle)
from gnomad_meb.cmd import main

REPO_ROOT = Path(__file__).resolve().parents[2]

BANNER_STRINGS = (b"NativeCodeLoader", b"SparkConf")


def _run_probe(code: str, tmp_path: Path) -> subprocess.CompletedProcess:
    stdout_path = tmp_path / "child-stdout.txt"
    stderr_path = tmp_path / "child-stderr.txt"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
        result = subprocess.run(  # noqa: S603 - test-owned subprocess, fixed argv
            [sys.executable, "-c", code],
            stdout=out,
            stderr=err,
            env=env,
            cwd=str(tmp_path),
            timeout=300,
        )
    result.stdout_bytes = stdout_path.read_bytes()  # type: ignore[attr-defined]
    result.stderr_bytes = stderr_path.read_bytes()  # type: ignore[attr-defined]
    return result


def test_importing_gnomad_meb_hail_does_not_boot(tmp_path: Path) -> None:
    """A plain import must not boot the JVM (so unit tests stay Hail-free)."""
    probe = "import gnomad_meb.hail as H\nprint(H.is_initialized())\n"
    result = _run_probe(probe, tmp_path)

    assert result.stdout_bytes.strip() == b"False"


def test_hail_init_keeps_a_guarded_boot_off_stdout(tmp_path: Path) -> None:
    """A native write to fd 1 inside ``intercept_jvm_output`` lands on stderr.

    This is the mechanism ``hail_init`` relies on: whatever Hail writes during
    startup is redirected, not dropped.
    """
    probe = (
        "import os\n"
        "from gnomad_meb.utils.logging import intercept_jvm_output\n"
        "with intercept_jvm_output():\n"
        "    os.write(1, b'JVM-BANNER\\n')\n"
        "os.write(1, b'AFTER\\n')\n"
    )
    result = _run_probe(probe, tmp_path)

    assert b"JVM-BANNER\n" in result.stderr_bytes
    assert b"JVM-BANNER\n" not in result.stdout_bytes
    assert b"AFTER\n" in result.stdout_bytes


def test_hail_init_restores_fd1_when_the_boot_raises(tmp_path: Path) -> None:
    probe = (
        "from gnomad_meb.utils.logging import intercept_jvm_output\n"
        "import os\n"
        "try:\n"
        "    with intercept_jvm_output():\n"
        "        raise RuntimeError('boom')\n"
        "except RuntimeError:\n"
        "    pass\n"
        "os.write(1, b'RECOVERED\\n')\n"
    )
    result = _run_probe(probe, tmp_path)

    assert b"RECOVERED\n" in result.stdout_bytes


# --- CLI-level purity for a Hail-importing prepare() -------------------------


class _FakeTaskIntervals:
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


def _install_fake_vds(monkeypatch) -> None:
    monkeypatch.setattr(
        "gnomad_meb.lib.vds.VariantDataset", _FakeVariantDataset, raising=False
    )
    monkeypatch.setattr(
        "gnomad_meb.lib.vds.dataset.VariantDataset", _FakeVariantDataset, raising=False
    )
    # The annotation pipeline's real resource table needs on-disk reference data;
    # this test only cares that prepare() reaches (and imports) Hail, so give it
    # an empty resource tuple.
    from gnomad_meb.lib.annotation.pipeline import AnnotationPipeline

    monkeypatch.setattr(
        AnnotationPipeline, "resources", property(lambda self: ()), raising=False
    )


@pytest.mark.parametrize("flag", ["--dry-run", "--describe"])
def test_annotation_inspection_stdout_is_pure(
    tmp_path: Path, monkeypatch, flag
) -> None:
    """``annotation run`` prepare() imports Hail (via VariantDataset); its stdout
    must still be a single clean payload."""
    _install_fake_vds(monkeypatch)
    config = _write_config(tmp_path)

    result = CliRunner().invoke(
        main,
        [
            "--config",
            str(config),
            flag,
            "--output-format",
            "json",
            "annotation",
            "run",
            "-V",
            "a.vds",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)  # parses the WHOLE stream
    assert payload["pipeline"] == "annotation"
    assert "NativeCodeLoader" not in result.stdout
    assert "SparkConf" not in result.stdout
