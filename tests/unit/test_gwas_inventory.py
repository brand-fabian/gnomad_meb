"""Tests for the GWAS dataset inventory and its selection.

The inventory is pure Python: a ``FilterManifest`` fixture (or a raw ``--mt``)
resolves, through :meth:`GwasInventory.from_command`, to an ordered
:class:`GwasTable` list, each with the label, path, provenance and (when present)
manifest variant count the stages fan out over.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gnomad_meb.config import Settings
from gnomad_meb.lib.filter.manifest import (
    FilterEntry,
    FilterManifest,
    FilterRegion,
    HighQuality,
)
from gnomad_meb.lib.gwas.inventory import (
    GwasInventory,
    GwasInventoryException,
    GwasTable,
)
from gnomad_meb.lib.gwas.names import DatasetKey


def _settings(reference_genome: str = "GRCh38") -> Settings:
    return Settings(
        validate_paths=False,
        project="study",
        checkpoint="/ckpt",
        reference_genome=reference_genome,
    )


def _settings_with_floor(min_n_snps: int, reference_genome: str = "GRCh38") -> Settings:
    from pathlib import Path

    from gnomad_meb.config.settings import GwasConfig

    return Settings(
        validate_paths=False,
        project="study",
        checkpoint="/ckpt",
        reference_genome=reference_genome,
        gwas=GwasConfig(metadata_csv=Path("/data/study.md.csv"), min_n_snps=min_n_snps),
    )


def _manifest(
    *,
    schema_version: int = 1,
    reference_genome: str = "GRCh38",
    filters: list[FilterEntry] | None = None,
) -> FilterManifest:
    return FilterManifest(
        schema_version=schema_version,
        project="study",
        reference_genome=reference_genome,
        high_quality=HighQuality(
            mt="/data/high_quality.mt",
            variant_info="/data/high_quality.variant_info.json.gz",
            sample_info="/data/high_quality.sample_info.json.gz",
            variant_info_summary={"n_variants": {"final": 100}},
        ),
        filters=filters
        if filters is not None
        else [
            FilterEntry(
                name="rare_lof",
                mt="/data/rare_lof.mt",
                variant_info="/data/rare_lof.variant_info.json.gz",
                variant_info_summary={
                    "filters": ["gnomad_freq", "lof"],
                    "gnomad_freq": {"n_variants": {"final": 30}},
                    "lof": {"n_variants": {"final": 7}},
                },
                regions=[
                    FilterRegion(name="integrated", mt="/data/rintegrated.mt")
                ],
            ),
            FilterEntry(
                name="clinvar",
                mt="/data/clinvar.mt",
                variant_info="/data/clinvar.variant_info.json.gz",
                variant_info_summary={
                    "filters": ["clinvar"],
                    "clinvar": {"n_variants": {"final": 5}},
                },
            ),
        ],
    )


def _command(**overrides) -> SimpleNamespace:
    base = {"mt": None, "manifest": None, "include": (), "exclude": ()}
    base.update(overrides)
    return SimpleNamespace(**base)


def _resolve(monkeypatch, manifest: FilterManifest, **overrides) -> GwasInventory:
    monkeypatch.setattr(
        "gnomad_meb.lib.gwas.inventory.read_manifest", lambda path: manifest
    )
    return GwasInventory.from_command(
        _settings(), _command(manifest="/m.json.gz", **overrides)
    )


def test_raw_mt_is_a_one_element_inventory(monkeypatch) -> None:
    tables = GwasInventory.from_command(_settings(), _command(mt="in.mt")).tables

    assert tables == [GwasTable(label="", mt="in.mt", source="mt")]


def test_manifest_resolves_in_order_with_labels_and_counts(monkeypatch) -> None:
    tables = _resolve(monkeypatch, _manifest()).tables

    assert [(t.label, t.source, t.name, t.n_variants) for t in tables] == [
        ("high_quality", "high_quality", None, 100),
        ("filter_rare_lof", "filter", "rare_lof", 7),
        ("filter_rare_lof_region_integrated", "filter_region", "rare_lof", None),
        ("filter_clinvar", "filter", "clinvar", 5),
    ]
    assert tables[0].mt == "/data/high_quality.mt"


def test_filter_count_uses_the_last_sub_filter(monkeypatch) -> None:
    tables = _resolve(monkeypatch, _manifest()).tables

    # ``lof`` is applied after ``gnomad_freq``, so its ``final`` (7) wins over
    # the earlier ``gnomad_freq`` final (30).
    rare_lof = next(t for t in tables if t.label == "filter_rare_lof")
    assert rare_lof.n_variants == 7


def test_filter_without_a_summary_has_no_count(monkeypatch) -> None:
    entry = FilterEntry(
        name="bare", mt="/data/bare.mt", variant_info="/data/bare.json.gz"
    )
    tables = _resolve(monkeypatch, _manifest(filters=[entry])).tables

    assert next(t for t in tables if t.label == "filter_bare").n_variants is None


def test_labels_and_by_label(monkeypatch) -> None:
    inventory = _resolve(monkeypatch, _manifest())

    assert inventory.labels == [
        "high_quality",
        "filter_rare_lof",
        "filter_rare_lof_region_integrated",
        "filter_clinvar",
    ]
    assert inventory.by_label("filter_clinvar") == inventory.tables[3]
    with pytest.raises(GwasInventoryException):
        inventory.by_label("no_such_label")


def test_include_selects_by_label_filter_and_region_name(monkeypatch) -> None:
    by_label = _resolve(monkeypatch, _manifest(), include=["filter_rare_lof"]).tables
    assert [t.label for t in by_label] == ["filter_rare_lof"]

    by_filter = _resolve(monkeypatch, _manifest(), include=["clinvar"]).tables
    assert [t.label for t in by_filter] == ["filter_clinvar"]

    by_region = _resolve(monkeypatch, _manifest(), include=["integrated"]).tables
    assert [t.label for t in by_region] == ["filter_rare_lof_region_integrated"]

    basis = _resolve(monkeypatch, _manifest(), include=["high_quality"]).tables
    assert [t.label for t in basis] == ["high_quality"]


def test_exclude_wins_over_include(monkeypatch) -> None:
    tables = _resolve(
        monkeypatch,
        _manifest(),
        include=["rare_lof"],
        exclude=["integrated"],
    ).tables

    assert [t.label for t in tables] == ["filter_rare_lof"]


def test_include_exclude_rejected_with_mt() -> None:
    with pytest.raises(GwasInventoryException):
        GwasInventory.from_command(
            _settings(), _command(mt="in.mt", include=["x"])
        )
    with pytest.raises(GwasInventoryException):
        GwasInventory.from_command(
            _settings(), _command(mt="in.mt", exclude=["x"])
        )


def test_exactly_one_of_mt_or_manifest_is_required() -> None:
    with pytest.raises(GwasInventoryException):
        GwasInventory.from_command(_settings(), _command())
    with pytest.raises(GwasInventoryException):
        GwasInventory.from_command(
            _settings(), _command(mt="in.mt", manifest="/m.json.gz")
        )


def test_unknown_schema_version_fails_loudly(monkeypatch) -> None:
    with pytest.raises(GwasInventoryException):
        _resolve(monkeypatch, _manifest(schema_version=99))


def test_mismatched_reference_genome_fails_loudly(monkeypatch) -> None:
    with pytest.raises(GwasInventoryException):
        _resolve(monkeypatch, _manifest(reference_genome="GRCh37"))


def test_unsafe_filter_name_fails_loudly(monkeypatch) -> None:
    entry = FilterEntry(
        name="bad name", mt="/data/bad.mt", variant_info="/data/bad.json.gz"
    )
    with pytest.raises(GwasInventoryException):
        _resolve(monkeypatch, _manifest(filters=[entry]))


def test_min_n_snps_drops_below_floor_and_unknown_counts(monkeypatch) -> None:
    monkeypatch.setattr(
        "gnomad_meb.lib.gwas.inventory.read_manifest", lambda path: _manifest()
    )
    tables = GwasInventory.from_command(
        _settings_with_floor(10), _command(manifest="/m.json.gz")
    ).tables

    # high_quality (100) and rare_lof (7): only the former clears the floor.
    # The region table (unknown count) and clinvar (5) are both dropped.
    assert [t.label for t in tables] == ["high_quality"]


def test_min_n_snps_keeps_count_at_the_floor(monkeypatch) -> None:
    monkeypatch.setattr(
        "gnomad_meb.lib.gwas.inventory.read_manifest", lambda path: _manifest()
    )
    tables = GwasInventory.from_command(
        _settings_with_floor(7), _command(manifest="/m.json.gz")
    ).tables

    # rare_lof has exactly 7 variants: kept (drop is strictly below the floor).
    assert "filter_rare_lof" in [t.label for t in tables]


def test_min_n_snps_composes_with_include(monkeypatch) -> None:
    monkeypatch.setattr(
        "gnomad_meb.lib.gwas.inventory.read_manifest", lambda path: _manifest()
    )
    tables = GwasInventory.from_command(
        _settings_with_floor(10),
        _command(manifest="/m.json.gz", include=["rare_lof"]),
    ).tables

    # The included table is below the floor, so the result is empty.
    assert tables == []


def test_no_floor_when_gwas_unconfigured(monkeypatch) -> None:
    tables = _resolve(monkeypatch, _manifest()).tables

    assert len(tables) == 4


def test_dataset_key_helpers() -> None:
    assert DatasetKey.of("", "gene").label == "gene"
    assert DatasetKey.of("filter_rare_lof", "gene").label == "filter_rare_lof.gene"
    assert DatasetKey.parse("filter_rare_lof.gene") == DatasetKey(
        table="filter_rare_lof", region="gene"
    )
    assert DatasetKey.parse("gene") == DatasetKey(region="gene")
    assert DatasetKey.parse("") == DatasetKey()
