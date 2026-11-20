"""The Filter stage's manifest: which filters ran, where the outputs live.

The Filter stage writes one manifest next to its checkpoints. It is the single
index a downstream stage (GWAS export/run/load, burden) reads to discover which
filtered matrix tables exist and what each filter's runtime result was -- so it
can bind its ``mt`` input without knowing the checkpoint naming convention.

The manifest records the output paths and the runtime results each filter
returned (the applied sub-filter list and the variant/sample tallies). The paths
come from the same constants the producing step declares, so the manifest cannot
disagree with the DAG.

The reader is pure Python (stdlib ``gzip``/``json`` plus pydantic, no hail), so
a downstream stage can resolve its input path during ``prepare`` before any hail
session, mirroring :mod:`gnomad_meb.lib.gwas.metadata`.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from gnomad_meb.lib.filter.constants import (
    MANIFEST_PREFIX,
    MANIFEST_SCHEMA_VERSION,
)

if TYPE_CHECKING:
    from gnomad_meb.config import Settings


def final_n_variants(summary: dict[str, Any] | None) -> int | None:
    """The final variant count a variant-info summary records, else ``None``.

    ``n_variants.final`` when the summary nests the count (the filter stage's
    schema); a flat ``n_variants`` number when it does not; ``None`` when the
    summary records no real count.
    """
    value = (summary or {}).get("n_variants")
    if isinstance(value, dict):
        value = value.get("final")
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return int(value)


class FilterRegion(BaseModel):
    """A region one filter was applied to: its name and matrix path."""

    name: str
    mt: str
    region_info_summary: dict[str, Any] = Field(default_factory=dict)


class FilterEntry(BaseModel):
    """One configured filter: its matrix, result dictionary and regions."""

    name: str
    mt: str
    variant_info: str
    variant_info_summary: dict[str, Any] = Field(default_factory=dict)
    regions: list[FilterRegion] = Field(default_factory=list)

    def get_final_n_variants(self) -> int | None:
        """The last applied sub-filter's final count (sub-filters narrow in order)."""
        summary = self.variant_info_summary or {}
        names = summary.get("filters") or []
        if not names:
            return None
        sub_filter = summary.get(names[-1])
        if not isinstance(sub_filter, dict):
            return None
        return final_n_variants(sub_filter)


class HighQuality(BaseModel):
    """The fixed high-quality step's paths and runtime result."""

    mt: str
    variant_info: str
    sample_info: str
    variant_info_summary: dict[str, Any] = Field(default_factory=dict)
    sample_info_summary: dict[str, Any] = Field(default_factory=dict)


class FilterManifest(BaseModel):
    """The Filter stage's index of output paths and per-filter results.

    Read it to bind a downstream ``mt`` (a ``filters[i].mt`` or one of its
    regions' ``mt``) or to inspect what a filter did
    (``filters[i].variant_info_summary``).
    """

    schema_version: int = MANIFEST_SCHEMA_VERSION
    project: str
    reference_genome: str
    checkpoint_root: str | None = None
    input_mt: str | None = None
    constraint_mt: str | None = None
    high_quality: HighQuality
    filters: list[FilterEntry] = Field(default_factory=list)


def manifest_path(settings: Settings) -> str | None:
    """The manifest path for ``settings``, or ``None`` when checkpointing is off.

    Mirrors :meth:`~gnomad_meb.lib.checkpoint.Artifact.path` for the
    :data:`MANIFEST_PREFIX` declaration, so the reader and the producing step
    agree on one path.
    """
    if settings.checkpoint is None:
        return None
    prefix = MANIFEST_PREFIX.format(
        project=settings.project, checkpoint=settings.checkpoint
    )
    return f"{settings.checkpoint}/{prefix}.json.gz"


def read_manifest(path: str | Path) -> FilterManifest:
    """Read and validate the manifest at ``path`` (a gzip JSON checkpoint)."""
    with gzip.open(path, "rb") as f:
        return FilterManifest.model_validate(json.load(f))
