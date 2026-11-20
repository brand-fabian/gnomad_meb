"""Shared hail-free scaffolding for the Variant QC stage unit tests.

The three stage tests (export, load, VQSR) all build their pipelines from a
stubbed ``VariantDataset`` handle -- construction reads the task list from
metadata only -- and the same minimal :class:`Settings`. That scaffolding lives
here so each test module keeps only its stage-specific declarations.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gnomad_meb.config import Settings
from gnomad_meb.lib.vds import VariantDataset

#: The split task names the stub handle exposes (metadata only, no hail session).
TASK_NAMES = ("left", "right")


def stub_handle(names: tuple[str, ...] = TASK_NAMES) -> SimpleNamespace:
    """A ``VariantDataset`` stand-in: ``.params.tasks[i].name`` is all we read."""
    return SimpleNamespace(
        params=SimpleNamespace(
            tasks=tuple(SimpleNamespace(name=name) for name in names)
        )
    )


def stub_vds(
    monkeypatch: pytest.MonkeyPatch, names: tuple[str, ...] = TASK_NAMES
) -> None:
    """Keep pipeline construction hail-free (ADR-0002): stub ``VariantDataset.open``."""
    monkeypatch.setattr(
        VariantDataset, "open", staticmethod(lambda path, *a, **k: stub_handle(names))
    )


def missing_vds(path: object, *a: object, **k: object) -> object:
    """A ``VariantDataset.open`` that fails, for the fail-loudly tests."""
    raise FileNotFoundError(f"invalid vds path: path = {path}")


def stub_missing_vds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``VariantDataset.open`` fail, for the fail-loudly tests."""
    monkeypatch.setattr(VariantDataset, "open", staticmethod(missing_vds))


def make_settings(**overrides: object) -> Settings:
    """The minimal valid :class:`Settings` for a Variant QC stage test."""
    return Settings(
        validate_paths=False,
        checkpoint="/ckpt",
        project="proj",
        **overrides,
    )
