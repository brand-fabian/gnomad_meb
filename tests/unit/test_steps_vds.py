"""Tests for the VDS build declared as a Pipeline.

``gnomad_meb.lib.vds.pipeline.VdsCreatePipeline`` declares one
``@checkpoint``-decorated step: the per-split fan-out and its resume live inside
:meth:`VariantDataset.process`. These tests exercise the declared step and its
artifact path without running hail.
"""

from __future__ import annotations

from pathlib import Path

from gnomad_meb.cmd.task import Task, VdsCreate
from gnomad_meb.config import Settings
from gnomad_meb.lib.checkpoint import CHECKPOINT_TYPE
from gnomad_meb.lib.vds import VdsCreatePipeline
from gnomad_meb.pipeline import StepRegistry


def _settings(tmp_path: Path) -> Settings:
    return Settings(validate_paths=False, project="demo", checkpoint=str(tmp_path))


def _task(tmp_path: Path, **overrides) -> Task:
    cmd = VdsCreate(output_base=str(tmp_path / "output"), **overrides)
    return Task(settings=_settings(tmp_path), command=cmd)


def test_the_build_is_one_step(tmp_path: Path) -> None:
    VdsCreatePipeline(_task(tmp_path))
    registry = StepRegistry()

    assert registry.get("create_variant_dataset").id == "vds.create_variant_dataset"


def test_the_artifact_is_the_output_metadata_handle(tmp_path: Path) -> None:
    task = _task(tmp_path, vds_name="cohort")
    VdsCreatePipeline(task)
    step = StepRegistry().get("create_variant_dataset")

    assert len(step.artifacts) == 1
    artifact = step.artifacts[0]
    assert artifact.kind == CHECKPOINT_TYPE.NATIVE_VARIANT_DATASET
    assert artifact.path(_settings(tmp_path)) == (
        f"{tmp_path}/output/cohort.metadata.json"
    )


def test_the_build_declares_no_resources(tmp_path: Path) -> None:
    pipeline = VdsCreatePipeline(_task(tmp_path))

    assert pipeline.resources == ()
