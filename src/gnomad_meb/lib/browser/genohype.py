"""The genohype CLI seam: run one Hail-Table -> Parquet export.

The per-contig parquet is produced by the ``genohype`` CLI, vendored as the
``third_party/genohype`` git submodule. A submodule records an exact gitlink
commit, and *that commit* is the compatibility contract: there is no branch, tag
or version configuration here, and none anywhere else in the code.

The binary is not checked in. Build it once after cloning::

    git submodule update --init third_party/genohype
    cd third_party/genohype
    cargo build --release --locked --package genohype-cli --bin genohype

which leaves the executable at ``third_party/genohype/target/release/genohype``.
:func:`genohype_binary` locates that path relative to the repository root, and
:func:`export_parquet` invokes::

    genohype export parquet <input.ht> <output.parquet> --width full

No ``--where``/``--interval``/``--limit`` is passed, so genohype stays on its
whole-table parallel path: it preserves our materialized ``xpos`` instead of
appending its own on the sequential path.

Importing this module imports neither hail nor the genohype binary.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

#: The repository root, derived from this file's location (never hardcoded).
REPO_ROOT = Path(__file__).resolve().parents[4]

#: The built genohype executable, relative to :data:`REPO_ROOT`.
GENOHYPE_BINARY_RELPATH = Path("third_party/genohype/target/release/genohype")

#: The command that builds the binary, run from ``third_party/genohype``.
GENOHYPE_BUILD_COMMAND = (
    "cargo build --release --locked --package genohype-cli --bin genohype"
)


class GenohypeNotFound(FileNotFoundError):
    """The genohype binary is missing or not executable; build it as documented."""


class GenohypeExportError(RuntimeError):
    """A genohype export failed, or produced no readable parquet footer."""


@dataclass(frozen=True)
class GenohypeExportResult:
    """One successful export: the CLI version, the argv and the output row count."""

    version: str
    args: tuple[str, ...]
    num_rows: int

    def to_dict(self):
        return {
            "version": self.version,
            "args": self.args,
            "num_rows": self.num_rows,
        }


def genohype_binary() -> Path:
    """The built genohype executable, or raise :class:`GenohypeNotFound`.

    The path is resolved relative to :data:`REPO_ROOT`, so it depends on neither
    the working directory nor any configuration.
    """
    binary = REPO_ROOT / GENOHYPE_BINARY_RELPATH
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise GenohypeNotFound(
            f"genohype binary not found or not executable at {binary}. "
            "Build it from third_party/genohype with: "
            f"{GENOHYPE_BUILD_COMMAND}"
        )
    return binary


def genohype_version(binary: Path) -> str:
    """The ``genohype --version`` output (stripped)."""
    completed = subprocess.run(  # noqa: S603 - fixed, repo-resolved executable
        [str(binary), "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise GenohypeExportError(
            f"`{binary} --version` failed with exit code "
            f"{completed.returncode}: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def export_parquet(
    input_ht: str | Path, output_parquet: str | Path
) -> GenohypeExportResult:
    """Export one Hail Table to a parquet file with genohype.

    Writes directly to ``output_parquet`` (the caller owns the path, e.g. a temp
    file), fails loudly on a non-zero exit or an unreadable footer, and returns a
    :class:`GenohypeExportResult` carrying the version, argv and row count.
    """
    binary = genohype_binary()
    version = genohype_version(binary)
    args = (
        "export",
        "parquet",
        str(input_ht),
        str(output_parquet),
        "--width",
        "full",
    )
    completed = subprocess.run(  # noqa: S603 - fixed, repo-resolved executable
        [str(binary), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise GenohypeExportError(
            f"genohype export failed with exit code {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    try:
        num_rows = pq.ParquetFile(output_parquet).metadata.num_rows
    except Exception as error:
        raise GenohypeExportError(
            f"genohype produced no readable parquet footer at {output_parquet}: "
            f"{error}"
        ) from error
    return GenohypeExportResult(version=version, args=args, num_rows=num_rows)
