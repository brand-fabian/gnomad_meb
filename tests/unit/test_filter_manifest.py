"""Tests for the Filter stage's central manifest.

The manifest is a validated pydantic model projected from the declared artifact
paths and the runtime results each filter returned. These tests pin the declared
path to the manifest reader's path and exercise the model a downstream
GWAS/burden stage would read, without a hail session.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from gnomad_meb.cmd.task import PipelineStep, Task
from gnomad_meb.config import (
    FilterCheckpoint,
    FilterConfig,
    FilterResources,
    RegionCheckpoint,
    Settings,
    set_config,
)
from gnomad_meb.lib.filter.manifest import (
    FilterEntry,
    FilterManifest,
    FilterRegion,
    HighQuality,
    manifest_path,
    read_manifest,
)
from gnomad_meb.lib.filter.pipeline import FilterPipeline
from gnomad_meb.pipeline import StepRegistry

_FILTERS = [
    FilterCheckpoint(name="rare_lof", min_pLI=0.9),
    FilterCheckpoint(name="all_clinvar_pathogenic", filter_clinvar_pathogenic=True),
]
_REGIONS = [RegionCheckpoint(name="integrated", intervals="/data/integrated.bed")]


def _settings() -> Settings:
    return Settings(
        validate_paths=False,
        project="proj",
        checkpoint="/tmp/opencode/filter-manifest/ckpt",  # noqa: S108
        filter=FilterConfig(
            filters=list(_FILTERS),
            regions=list(_REGIONS),
            resources=FilterResources(constraints="/data/constraints.ht"),
        ),
    )


def _pipeline(settings: Settings) -> FilterPipeline:
    return FilterPipeline(
        Task(settings=settings, command=PipelineStep(name="filter"))
    )


def _manifest(settings: Settings) -> FilterManifest:
    """A representative manifest, built the way the stage builds its own."""
    return FilterManifest(
        project=settings.project,
        reference_genome=settings.reference_genome,
        checkpoint_root=settings.checkpoint,
        input_mt="/data/annotated.mt",
        constraint_mt="/ckpt/proj/intermediate/filter/constraint.mt",
        high_quality=HighQuality(
            mt="/ckpt/proj/intermediate/filter/high_quality_gnomad.mt",
            variant_info=(
                "/ckpt/proj/intermediate/filter/high_quality_gnomad.variant_info.json.gz"
            ),
            sample_info=(
                "/ckpt/proj/intermediate/filter/high_quality_gnomad.sample_info.json.gz"
            ),
            variant_info_summary={
                "cutoffs": {"min_inbreeding_coef": -0.3},
                "n_variants": {"final": 10},
            },
            sample_info_summary={"sample_info": {"apply_sample_filters": True}},
        ),
        filters=[
            FilterEntry(
                name="rare_lof",
                mt="/ckpt/proj/intermediate/filter/rare_lof.mt",
                variant_info=(
                    "/ckpt/proj/intermediate/filter/rare_lof.variant_info.json.gz"
                ),
                variant_info_summary={
                    "filters": ["lof"],
                    "n_variants": {"final": 3},
                },
                regions=[
                    FilterRegion(
                        name="integrated",
                        mt="/ckpt/proj/intermediate/filter/rintegrated/frare_lof.mt",
                    )
                ],
            )
        ],
    )


def test_manifest_path_matches_the_declared_artifact() -> None:
    settings = _settings()
    _pipeline(settings)
    declared = StepRegistry().get("get_manifest").artifacts[0].path(settings)

    assert declared == manifest_path(settings)
    assert declared.endswith("/ckpt/proj/intermediate/filter/manifest.json.gz")


def test_manifest_path_is_none_without_a_checkpoint_root() -> None:
    settings = _settings().model_copy(update={"checkpoint": None})

    assert manifest_path(settings) is None


def test_get_manifest_records_the_declared_artifact_paths() -> None:
    settings = _settings()
    pipeline = _pipeline(settings)
    set_config(settings)
    outputs = {
        "high_quality_variant_info": {},
        "high_quality_sample_info": {},
        **{f"variant_info_{flt.name}": {} for flt in _FILTERS},
        **{
            f"region_{reg.name}_{flt.name}": {}
            for flt in _FILTERS
            for reg in _REGIONS
        },
    }
    manifest = FilterManifest.model_validate(pipeline.get_manifest(**outputs))

    registry = StepRegistry()
    high_quality = registry.get("filter.filter_high_quality_variants_gnomad")
    assert manifest.high_quality.mt == high_quality.artifacts[0].path(settings)
    assert manifest.high_quality.variant_info == (
        high_quality.artifacts[1].path(settings)
    )
    assert manifest.high_quality.sample_info == (
        high_quality.artifacts[2].path(settings)
    )
    constant = registry.get("filter.annotate_constraints_table")
    assert manifest.constraint_mt == constant.artifacts[0].path(settings)

    for flt in _FILTERS:
        entry = next(f for f in manifest.filters if f.name == flt.name)
        artifacts = registry.get(f"filter.{flt.name}").artifacts
        assert entry.mt == artifacts[0].path(settings)
        assert entry.variant_info == artifacts[1].path(settings)
        for reg in _REGIONS:
            region_step = registry.get(f"filter.filter_r{reg.name}_f{flt.name}")
            region = next(r for r in entry.regions if r.name == reg.name)
            assert region.mt == region_step.artifacts[0].path(settings)


def test_read_manifest_round_trips_a_gzip_checkpoint(tmp_path: Path) -> None:
    manifest = _manifest(_settings())
    path = tmp_path / "manifest.json.gz"
    with gzip.open(path, "wt") as out_f:
        json.dump(manifest.model_dump(mode="json"), out_f)

    restored = read_manifest(path)

    assert restored == manifest
    assert [f.name for f in restored.filters] == ["rare_lof"]
    assert restored.filters[0].variant_info_summary["filters"] == ["lof"]
    assert restored.filters[0].regions[0].mt.endswith("frare_lof.mt")
