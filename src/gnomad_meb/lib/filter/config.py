"""Hail-free filter and region declaration loaders.

The declaration models live in the settings layer (``gnomad_meb.config``) so
the global config file validates them at resolve time. This module is a
backward-compatible shim: it re-exports the models for the hail-dependent
checkpoint modules and exposes the pure normalizers as free functions. It
performs no file I/O and imports no hail, so
:class:`~gnomad_meb.lib.filter.pipeline.FilterPipeline` can enumerate the
declarations without importing the hail stack.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from gnomad_meb.config import (
    FilterCheckpoint,
    RegionCheckpoint,
)


def load_filters(entries: Iterable[Any]) -> list[FilterCheckpoint]:
    """Normalize filter declarations (models or mappings) to checkpoints."""
    return FilterCheckpoint.load_filters(entries)


def load_regions(entries: Iterable[Any]) -> list[RegionCheckpoint]:
    """Normalize region declarations (models or mappings) to checkpoints."""
    return RegionCheckpoint.load_regions(entries)
