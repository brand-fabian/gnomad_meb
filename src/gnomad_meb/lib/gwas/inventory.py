"""Resolve a GWAS input into the ordered table inventory the stages fan out over.

The GWAS commands accept either a single MatrixTable (``--mt``) or the Filter
stage's manifest (``--manifest``), optionally narrowed by ``--include`` /
``--exclude``. :class:`GwasInventory` turns that command into an ordered
``list[GwasTable]`` -- the one input every stage (export, run, load, workflow)
builds its step set from -- dropping any table below the ``[gwas].min_n_snps``
floor so the size gate applies deterministically to all of them.

It is pure Python (stdlib ``re`` plus pydantic, no hail): :func:`read_manifest`
is a gzip JSON read, so the inventory resolves during ``prepare`` before any hail
session and a serialized run reconstructs the same tables.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

from gnomad_meb.config import Settings
from gnomad_meb.lib.filter.constants import MANIFEST_SCHEMA_VERSION
from gnomad_meb.lib.filter.manifest import (
    FilterManifest,
    final_n_variants,
    read_manifest,
)
from gnomad_meb.logging import get_logger

#: The filesystem-/step-id-safe shape a table label must take.
LABEL_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")

#: The label of the fixed high-quality basis table.
HIGH_QUALITY_LABEL = "high_quality"

#: The ``GwasTable.source`` values.
SOURCE_MT = "mt"
SOURCE_HIGH_QUALITY = "high_quality"
SOURCE_FILTER = "filter"
SOURCE_FILTER_REGION = "filter_region"


class GwasInventoryException(ValueError):
    """The GWAS input, its manifest or its selection is invalid."""


class GwasTable(BaseModel):
    """One input MatrixTable the GWAS stages fan out over.

    ``label`` is the empty string for a raw ``--mt`` (so its paths reproduce the
    single-table ones exactly) and a table label otherwise; ``source``,
    ``filter_name`` and ``region_name`` record provenance -- the filter (and,
    for a region table, region) a manifest entry came from, empty for a raw
    ``--mt``; ``n_variants`` is the manifest summary's count, when the manifest
    carries one.
    """

    label: str
    mt: str
    source: str
    filter_name: str | None = None
    region_name: str | None = None
    n_variants: int | None = None

    def matches(self, token: str) -> bool:
        """Whether ``token`` names a table: its label, filter name or region name."""
        if token == self.label:
            return True

        if token == self.filter_name:
            return True

        if self.region_name is not None and token == self.region_name:
            return True

        return False


class GwasInventory:
    """A GWAS command's ordered table inventory.

    Build one with :meth:`from_command`; then read the ordered tables, their
    labels, or look a single table up by label. A raw ``--mt`` is a one-element
    inventory with the empty label; a ``--manifest`` is its tables in manifest
    order (the high-quality basis, then each filter, then each filter's regions),
    narrowed by the command's ``include``/``exclude``. Include/exclude are
    manifest-only and exactly one of ``mt``/``manifest`` is required.
    """

    def __init__(self, tables: list[GwasTable]) -> None:
        self._tables = list(tables)

    @classmethod
    def from_command(cls, settings: Settings, command: Any) -> GwasInventory:
        """Resolve a GWAS command's input into its ordered table inventory."""
        mt = getattr(command, "mt", None)
        manifest = getattr(command, "manifest", None)
        include = list(getattr(command, "include", ()) or ())
        exclude = list(getattr(command, "exclude", ()) or ())

        if (mt is None) == (manifest is None):
            raise GwasInventoryException(
                "exactly one of 'mt' or 'manifest' must be provided"
            )

        if mt is not None:
            if include or exclude:
                raise GwasInventoryException(
                    "'include'/'exclude' are only valid with a manifest"
                )
            return cls([GwasTable(label="", mt=str(mt), source=SOURCE_MT)])

        tables = cls._tables_from_manifest(settings, read_manifest(manifest))
        min_n_snps = settings.gwas.min_n_snps if settings.gwas is not None else 0
        return cls(cls._select_tables(tables, include, exclude, min_n_snps=min_n_snps))

    @property
    def tables(self) -> list[GwasTable]:
        """The ordered tables the stages fan out over."""
        return list(self._tables)

    @property
    def labels(self) -> list[str]:
        """The labels of :attr:`tables`, in order."""
        return [table.label for table in self._tables]

    def by_label(self, label: str) -> GwasTable:
        """The table with ``label``; raise :class:`GwasInventoryException` if none."""
        for table in self._tables:
            if table.label == label:
                return table
        raise GwasInventoryException(f"no table with label: {label}")

    @classmethod
    def _tables_from_manifest(
        cls, settings: Settings, manifest: FilterManifest
    ) -> list[GwasTable]:
        """The ordered inventory a validated manifest resolves to."""
        if manifest.schema_version != MANIFEST_SCHEMA_VERSION:
            raise GwasInventoryException(
                "unknown filter manifest schema version: "
                f"manifest = {manifest.schema_version}, "
                f"expected = {MANIFEST_SCHEMA_VERSION}"
            )
        if manifest.reference_genome != settings.reference_genome:
            raise GwasInventoryException(
                "filter manifest reference genome does not match the run: "
                f"manifest = {manifest.reference_genome}, "
                f"run = {settings.reference_genome}"
            )

        tables = [
            GwasTable(
                label=HIGH_QUALITY_LABEL,
                mt=manifest.high_quality.mt,
                source=SOURCE_HIGH_QUALITY,
                filter_name=HIGH_QUALITY_LABEL,
                n_variants=final_n_variants(
                    manifest.high_quality.variant_info_summary
                ),
            )
        ]
        for entry in manifest.filters:
            tables.append(
                GwasTable(
                    label=cls._validate_label(f"filter_{entry.name}"),
                    mt=entry.mt,
                    source=SOURCE_FILTER,
                    filter_name=entry.name,
                    n_variants=entry.get_final_n_variants(),
                )
            )
            for region in entry.regions:
                tables.append(
                    GwasTable(
                        label=cls._validate_label(
                            f"filter_{entry.name}_region_{region.name}"
                        ),
                        mt=region.mt,
                        source=SOURCE_FILTER_REGION,
                        filter_name=entry.name,
                        region_name=region.name,
                        n_variants=final_n_variants(region.region_info_summary),
                    )
                )
        return tables

    @staticmethod
    def _validate_label(label: str) -> str:
        """Reject labels outside ``LABEL_PATTERN``: they name steps and paths."""
        if not LABEL_PATTERN.match(label):
            raise GwasInventoryException(
                f"unsafe table label (must match {LABEL_PATTERN.pattern}): {label}"
            )
        return label

    @classmethod
    def _select_tables(
        cls,
        tables: list[GwasTable],
        include: list[str],
        exclude: list[str],
        min_n_snps: int = 0,
    ) -> list[GwasTable]:
        """Apply the allowlist, then the denylist, then the size floor.

        The floor is the deterministic ``[gwas].min_n_snps`` gate: a table whose
        recorded variant count is below it, or unknown, is dropped here before
        any stage builds steps for it. Both cases are logged; unknown counts are
        dropped because a dataset that small cannot be judged testable.
        """
        selected = tables
        if include:
            selected = [
                table
                for table in selected
                if any(table.matches(token) for token in include)
            ]
        if exclude:
            selected = [
                table
                for table in selected
                if not any(
                    table.matches(token) for token in exclude
                )
            ]
        if min_n_snps > 0:
            logger = get_logger()
            kept = []
            for table in selected:
                if table.n_variants is None:
                    logger.warning(
                        "ignoring min_snps filter for table with unknown variant count: "
                        f"label = {table.label}, min_n_snps = {min_n_snps}"
                    )
                elif table.n_variants < min_n_snps:
                    logger.warning(
                        "dropping table below min_n_snps: "
                        f"label = {table.label}, "
                        f"n_variants = {table.n_variants}, "
                        f"min_n_snps = {min_n_snps}"
                    )
                    continue
                kept.append(table)
            selected = kept
        return selected
