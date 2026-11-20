"""The GWAS load manifest: which datasets were loaded and where.

The load stage writes one manifest next to its terminal tables. It is the single
index a downstream burden/analysis stage reads to discover which ``(table,
region)`` datasets were loaded, what each loaded ``gwas``/``burden`` hail table
is, and its provenance (regenie version, response set and filter globals).

It is the GWAS analog of :class:`~gnomad_meb.lib.filter.manifest.FilterManifest`.
The reader is pure Python (stdlib ``gzip``/``json`` plus pydantic, no hail), so a
downstream stage can resolve its inputs during ``prepare`` before any hail
session.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from gnomad_meb.lib.gwas.constants import (
    GWAS_MANIFEST_SCHEMA_VERSION,
    REGENIE_MANIFEST_PREFIX,
)

if TYPE_CHECKING:
    from gnomad_meb.config import Settings


class GwasTableEntry(BaseModel):
    """One loaded dataset: its identity, paths, responses and provenance."""

    label: str
    source: str
    name: str | None = None
    region: str
    dataset: str
    gwas: str
    burden: str
    responses: list[str] = Field(default_factory=list)
    regenie_version: dict[str, Any] | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    effect_mask_info: dict[str, Any] = Field(default_factory=dict)


class GwasManifest(BaseModel):
    """The GWAS load stage's index of loaded tables and their provenance."""

    schema_version: int = GWAS_MANIFEST_SCHEMA_VERSION
    project: str
    reference_genome: str
    checkpoint_root: str | None = None
    datasets: list[GwasTableEntry] = Field(default_factory=list)


def gwas_manifest_path(settings: Settings) -> str | None:
    """The manifest path for ``settings``, or ``None`` when checkpointing is off.

    Mirrors :meth:`~gnomad_meb.lib.checkpoint.Artifact.path` for the
    :data:`REGENIE_MANIFEST_PREFIX` declaration, so the reader and the producing
    step agree on one path.
    """
    if settings.checkpoint is None:
        return None
    prefix = REGENIE_MANIFEST_PREFIX.format(project=settings.project)
    return f"{settings.checkpoint}/{prefix}.json.gz"


def read_gwas_manifest(path: str | Path) -> GwasManifest:
    """Read and validate the manifest at ``path`` (a gzip JSON checkpoint)."""
    with gzip.open(path, "rb") as f:
        return GwasManifest.model_validate(json.load(f))
