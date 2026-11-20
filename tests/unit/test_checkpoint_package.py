"""Tests for the ``gnomad_meb.lib.checkpoint`` package surface.

The split from a flat module must be invisible to callers: every name they
imported still resolves, the serializer registry answers for every kind, and the
package (and ``gnomad_meb.config``) can be imported in a clean interpreter
without booting hail or tripping a circular import.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import gnomad_meb.config as config_module
from gnomad_meb.lib.checkpoint import (
    CHECKPOINT_EXTENSION,
    CHECKPOINT_TYPE,
    Artifact,
    CheckpointMetadata,
    CheckpointSpec,
    Serializer,
    SingleFileSerializer,
    SuccessFileInfo,
    callable_source,
    checkpoint,
    checkpoint_overwrite,
    checkpoint_static,
    declares_step,
    get_checkpoint_path,
    get_serializer,
    has_success_file,
    is_incomplete_hail_table,
    is_s3_path,
    md5_file,
    path_exists,
    s3_base_path,
)

#: The public names the codebase (and this test) imports from the package root.
PUBLIC_NAMES = (
    "CHECKPOINT_EXTENSION",
    "CHECKPOINT_TYPE",
    "Artifact",
    "CheckpointMetadata",
    "CheckpointSpec",
    "Serializer",
    "SingleFileSerializer",
    "SuccessFileInfo",
    "callable_source",
    "checkpoint",
    "checkpoint_overwrite",
    "checkpoint_static",
    "declares_step",
    "get_checkpoint_path",
    "get_serializer",
    "has_success_file",
    "is_incomplete_hail_table",
    "is_s3_path",
    "md5_file",
    "path_exists",
    "s3_base_path",
)


@pytest.fixture(autouse=True)
def _clear_ambient_config(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in [k for k in os.environ if k.startswith("GMEB_")]:
        monkeypatch.delenv(key, raising=False)
    config_module._config = None
    config_module._config_path = None


def test_every_public_name_resolves() -> None:
    import gnomad_meb.lib.checkpoint as pkg

    for name in PUBLIC_NAMES:
        assert hasattr(pkg, name), name
        assert getattr(pkg, name) is not None, name


def test_the_names_above_are_actually_bound_here() -> None:
    # Reference the imported symbols so the import block itself is the test.
    assert Artifact and CheckpointSpec and CheckpointMetadata and SuccessFileInfo
    assert Serializer and SingleFileSerializer
    assert checkpoint and declares_step and checkpoint_overwrite
    assert callable_source and checkpoint_static
    assert get_serializer and get_checkpoint_path
    assert is_s3_path and s3_base_path and path_exists
    assert has_success_file and is_incomplete_hail_table and md5_file


def test_get_serializer_returns_a_class_for_every_kind() -> None:
    for kind in CHECKPOINT_TYPE:
        serializer = get_serializer(kind)
        assert isinstance(serializer, type), kind
        assert issubclass(serializer, Serializer), kind


def test_checkpoint_extension_covers_every_kind() -> None:
    assert set(CHECKPOINT_EXTENSION) == set(CHECKPOINT_TYPE)


def test_unknown_kind_returns_none() -> None:
    assert get_serializer("not-a-kind") is None


def _run_in_clean_interpreter(code: str, src: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(src)}
    for key in [k for k in env if k.startswith("GMEB_")]:
        del env[key]
    return subprocess.run(  # noqa: S603 - a clean interpreter from sys.executable
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(src.parent),
    )


def test_importing_checkpoint_does_not_import_hail_or_cycle() -> None:
    src = Path(__file__).resolve().parents[2] / "src"
    result = _run_in_clean_interpreter(
        "import sys; import gnomad_meb.lib.checkpoint as p; "
        "assert 'hail' not in sys.modules, sorted(m for m in sys.modules "
        "if m == 'hail' or m.startswith('hail.')); "
        "print('ok')",
        src,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_importing_config_does_not_import_hail_or_cycle() -> None:
    src = Path(__file__).resolve().parents[2] / "src"
    result = _run_in_clean_interpreter(
        "import sys; import gnomad_meb.config; "
        "assert 'hail' not in sys.modules, sorted(m for m in sys.modules "
        "if m == 'hail' or m.startswith('hail.')); "
        "print('ok')",
        src,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def _imported_from_checkpoint(tree_dir: Path) -> set[str]:
    """Every name imported from ``gnomad_meb.lib.checkpoint`` under ``tree_dir``."""
    import ast

    names: set[str] = set()
    target = "gnomad_meb.lib.checkpoint"
    for path in tree_dir.rglob("*.py"):
        if "lib/checkpoint" in str(path):
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == target:
                for alias in node.names:
                    names.add(alias.name)
    return names


def test_every_name_the_source_tree_imports_resolves() -> None:
    src = Path(__file__).resolve().parents[2] / "src"
    imported = _imported_from_checkpoint(src)
    assert imported, "expected the source tree to import from the checkpoint package"

    import gnomad_meb.lib.checkpoint as pkg

    missing = sorted(name for name in imported if not hasattr(pkg, name))
    assert missing == [], missing


def test_the_public_surface_has_no_dead_names() -> None:
    # Every name in ``__all__`` resolves (no dangling re-export).
    import gnomad_meb.lib.checkpoint as pkg

    assert sorted(pkg.__all__) == sorted(
        name for name in pkg.__all__ if getattr(pkg, name, None) is not None
    )
