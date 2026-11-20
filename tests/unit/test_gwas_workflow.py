"""Tests for the composed GWAS workflow and its cross-stage edges.

``GwasWorkflowPipeline`` composes export, run and load over one dataset
inventory in a single central registry. These pin the composition, the
``source`` edges it patches, its determinism across constructions and its
durability (ADR-0017) -- without a hail session.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas
import pytest
from pydantic import ValidationError

from gnomad_meb.cmd.task import GwasWorkflowRun, Task
from gnomad_meb.config import GwasConfig, GwasRegion, Settings
from gnomad_meb.lib.filter.manifest import (
    FilterEntry,
    FilterManifest,
    FilterRegion,
    HighQuality,
)
from gnomad_meb.lib.gwas.workflow import GwasWorkflowPipeline
from gnomad_meb.pipeline import PipelineRunner, StepRegistry


def _study_csv(tmp_path: Path) -> Path:
    path = tmp_path / "study.md.csv"
    pandas.DataFrame(
        {"sample_id": ["1", "2", "3"], "cohort": ["case", "case", "control"]}
    ).to_csv(path, index=False)
    return path


def _settings(tmp_path: Path, regions: list[GwasRegion] | None = None) -> Settings:
    return Settings(
        validate_paths=False,
        project="study",
        checkpoint=str(tmp_path / "ckpt"),
        gwas=GwasConfig(
            metadata_csv=_study_csv(tmp_path),
            case_cohort="case",
            control_cohorts=["control"],
            min_case_count=2,
            min_n_snps=0,
            regions=regions or [GwasRegion(kind="gene")],
        ),
    )


def _write_manifest(tmp_path: Path) -> Path:
    manifest = FilterManifest(
        project="study",
        reference_genome="GRCh38",
        high_quality=HighQuality(
            mt=str(tmp_path / "high_quality.mt"),
            variant_info=str(tmp_path / "high_quality.variant_info.json.gz"),
            sample_info=str(tmp_path / "high_quality.sample_info.json.gz"),
            variant_info_summary={"n_variants": {"final": 100}},
        ),
        filters=[
            FilterEntry(
                name="rare_lof",
                mt=str(tmp_path / "rare_lof.mt"),
                variant_info=str(tmp_path / "rare_lof.variant_info.json.gz"),
                variant_info_summary={
                    "filters": ["lof"],
                    "lof": {"n_variants": {"final": 7}},
                },
                regions=[
                    FilterRegion(name="integrated", mt=str(tmp_path / "r.mt"))
                ],
            )
        ],
    )
    path = tmp_path / "manifest.json.gz"
    with gzip.open(path, "wt") as out_f:
        json.dump(manifest.model_dump(mode="json"), out_f)
    return path


def _task(tmp_path: Path, **overrides) -> Task:
    manifest = overrides.pop("manifest", _write_manifest(tmp_path))
    return Task(
        settings=_settings(tmp_path),
        command=GwasWorkflowRun(
            manifest=str(manifest),
            output_base="/out",
            output_dir="/run",
            **overrides,
        ),
    )


def _build(tmp_path: Path, **overrides) -> GwasWorkflowPipeline:
    StepRegistry._reset()
    return GwasWorkflowPipeline(_task(tmp_path, **overrides))


def test_name_is_stable() -> None:
    assert GwasWorkflowPipeline.name() == "gwas_workflow"


def test_composes_every_stage(tmp_path: Path) -> None:
    workflow = _build(tmp_path, include=("high_quality",))

    assert [pipeline.name() for pipeline in workflow.composed_pipelines] == [
        "gwas_export",
        "gwas",
        "gwas_load",
    ]


def test_registers_every_stage_step(tmp_path: Path) -> None:
    _build(tmp_path, include=("high_quality", "rare_lof"))
    registry = StepRegistry()

    for expected in (
        "gwas_export.gwas_base.high_quality",
        "gwas_export.export_regenie.high_quality.gene",
        "gwas_export.export_regenie.filter_rare_lof_region_integrated.gene",
        "gwas.run_gwas_pipeline.high_quality.gene",
        "gwas_load.load_gwas.high_quality.gene.case",
        "gwas_load.merge_gwas.high_quality.gene",
        "gwas_load.get_manifest",
    ):
        assert registry.get(expected) is not None, expected


def test_wires_the_cross_stage_source_edges(tmp_path: Path) -> None:
    _build(tmp_path, include=("high_quality",))
    registry = StepRegistry()

    label = "high_quality.gene"
    assert registry[f"gwas.dataset.{label}"].source == (
        "gwas_export.export_regenie.high_quality.gene"
    )
    assert registry[f"gwas_load.dataset.{label}"].source == (
        "gwas.run_gwas_pipeline.high_quality.gene"
    )
    assert registry[f"gwas_load.gwas.{label}"].source == (
        "gwas.run_gwas_pipeline.high_quality.gene"
    )
    assert registry[f"gwas_load.burden.{label}"].source == (
        "gwas.run_gwas_pipeline.high_quality.gene"
    )


def test_topological_order_places_stages_in_sequence(tmp_path: Path) -> None:
    _build(tmp_path, include=("high_quality",))
    order = [step.id for step in StepRegistry().topological()]

    assert order.index("gwas_export.export_regenie.high_quality.gene") < order.index(
        "gwas.run_gwas_pipeline.high_quality.gene"
    )
    assert order.index("gwas.run_gwas_pipeline.high_quality.gene") < order.index(
        "gwas_load.merge_gwas.high_quality.gene"
    )


def test_step_set_is_identical_across_constructions(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    first = GwasWorkflowPipeline(_task(tmp_path, manifest=manifest))
    first_steps = set(StepRegistry().steps())
    second = GwasWorkflowPipeline(_task(tmp_path, manifest=manifest))

    assert set(StepRegistry().steps()) == first_steps
    assert first.current_state()["steps"] == second.current_state()["steps"]


def test_mt_workflow_builds_per_region_datasets(tmp_path: Path) -> None:
    StepRegistry._reset()
    task = Task(
        settings=_settings(tmp_path, regions=[GwasRegion(kind="gene")]),
        command=GwasWorkflowRun(mt="in.mt", output_dir="/run"),
    )
    GwasWorkflowPipeline(task)
    registry = StepRegistry()

    # A raw --mt is labelled by region, not table.
    assert registry.get("gwas_export.export_regenie.gene") is not None
    assert registry.get("gwas.run_gwas_pipeline.gene") is not None
    assert registry["gwas.dataset.gene"].source == "gwas_export.export_regenie.gene"


def test_workflow_restores_as_one_run(tmp_path: Path) -> None:
    task = _task(tmp_path, include=("high_quality",))
    workflow = GwasWorkflowPipeline(task)
    runner = PipelineRunner(task, [workflow])
    dumped = json.loads(json.dumps(runner.current_state()))

    restored = PipelineRunner.load(dumped)

    assert set(restored.current_state()["steps"]) == set(dumped["steps"])


def test_workflow_run_requires_exactly_one_input() -> None:
    with pytest.raises(ValidationError):
        GwasWorkflowRun()
    with pytest.raises(ValidationError):
        GwasWorkflowRun(mt="in.mt", manifest="/m.json.gz")


def test_workflow_run_rejects_selection_with_mt() -> None:
    with pytest.raises(ValidationError):
        GwasWorkflowRun(mt="in.mt", include=("high_quality",))


def test_workflow_is_factory_discoverable() -> None:
    from gnomad_meb.pipeline import PipelineFactory

    assert "gwas_workflow" in PipelineFactory.list()
