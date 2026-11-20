"""Unit tests for the genohype CLI seam (ticket 03).

The subprocess is the only thing mocked -- it is the system boundary. A real
pyarrow parquet is written by the fake exporter so that the footer validation
and row count are exercised for real. No test executes the genohype binary;
that is deferred to integration.
"""

from __future__ import annotations

import dataclasses
import subprocess
from collections.abc import Callable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gnomad_meb.lib.browser import genohype

AGENTS_MD = Path(__file__).resolve().parents[2] / "AGENTS.md"


def _write_parquet(path: Path, num_rows: int) -> None:
    """Write a minimal real parquet file with ``num_rows`` rows."""
    table = pa.table({"xpos": pa.array(range(num_rows), type=pa.int64())})
    pq.write_table(table, path)


def _install_binary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the repo root at a temp dir holding an executable fake binary."""
    repo_root = tmp_path / "repo"
    binary = repo_root / genohype.GENOHYPE_BINARY_RELPATH
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    monkeypatch.setattr(genohype, "REPO_ROOT", repo_root)
    return binary


def _fake_run(
    recorded: list[list[str]],
    version_stdout: str,
    export_returncode: int = 0,
    export_stderr: str = "",
    export_writer: Callable[[Path], None] | None = None,
):
    """A ``subprocess.run`` stand-in for the version and export calls."""

    def run(argv, **kwargs):
        recorded.append([str(arg) for arg in argv])
        if argv[1] == "--version":
            return subprocess.CompletedProcess(
                argv, 0, stdout=version_stdout, stderr=""
            )
        if export_writer is not None:
            export_writer(Path(argv[4]))
        return subprocess.CompletedProcess(
            argv, export_returncode, stdout="", stderr=export_stderr
        )

    return run


def test_genohype_binary_resolves_relative_to_the_repo_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = _install_binary(monkeypatch, tmp_path)

    assert genohype.genohype_binary() == binary


def test_missing_binary_fails_with_the_documented_build_step(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(genohype, "REPO_ROOT", tmp_path / "empty")

    with pytest.raises(genohype.GenohypeNotFound) as excinfo:
        genohype.export_parquet("in.ht", "out.parquet")

    assert genohype.GENOHYPE_BUILD_COMMAND in str(excinfo.value)


def test_export_parquet_invokes_the_documented_arguments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = _install_binary(monkeypatch, tmp_path)
    input_ht = str(tmp_path / "in.ht")
    output_parquet = str(tmp_path / "out.parquet")
    recorded: list[list[str]] = []
    monkeypatch.setattr(
        subprocess,
        "run",
        _fake_run(
            recorded,
            "genohype 0.1.0 (abc1234)\n",
            export_writer=lambda path: _write_parquet(path, 0),
        ),
    )

    genohype.export_parquet(input_ht, output_parquet)

    assert recorded[0] == [str(binary), "--version"]
    assert recorded[1] == [
        str(binary),
        "export",
        "parquet",
        input_ht,
        output_parquet,
        "--width",
        "full",
    ]


def test_export_parquet_returns_version_arguments_and_row_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binary(monkeypatch, tmp_path)
    input_ht = str(tmp_path / "in.ht")
    output_parquet = str(tmp_path / "out.parquet")
    monkeypatch.setattr(
        subprocess,
        "run",
        _fake_run(
            [],
            "genohype 0.1.0 (abc1234)\n",
            export_writer=lambda path: _write_parquet(path, 7),
        ),
    )

    result = genohype.export_parquet(input_ht, output_parquet)

    assert result.version == "genohype 0.1.0 (abc1234)"
    assert result.args == (
        "export",
        "parquet",
        input_ht,
        output_parquet,
        "--width",
        "full",
    )
    assert result.num_rows == 7


def test_export_parquet_raises_on_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binary(monkeypatch, tmp_path)
    monkeypatch.setattr(
        subprocess,
        "run",
        _fake_run([], "genohype 0.1.0\n", export_returncode=1, export_stderr="boom"),
    )

    with pytest.raises(genohype.GenohypeExportError) as excinfo:
        genohype.export_parquet("in.ht", tmp_path / "out.parquet")

    assert "boom" in str(excinfo.value)


def test_export_parquet_rejects_an_unreadable_footer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binary(monkeypatch, tmp_path)
    monkeypatch.setattr(
        subprocess, "run", _fake_run([], "genohype 0.1.0\n")
    )

    with pytest.raises(genohype.GenohypeExportError):
        genohype.export_parquet("in.ht", tmp_path / "out.parquet")


def test_result_is_a_frozen_value() -> None:
    result = genohype.GenohypeExportResult(version="v", args=(), num_rows=0)

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.num_rows = 1  # type: ignore[misc]


def test_build_step_is_documented_in_the_module_docstring() -> None:
    assert genohype.GENOHYPE_BUILD_COMMAND in (genohype.__doc__ or "")


def test_build_step_is_documented_in_agents_md() -> None:
    assert genohype.GENOHYPE_BUILD_COMMAND in AGENTS_MD.read_text()
