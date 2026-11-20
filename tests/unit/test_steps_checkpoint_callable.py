"""Tests for ``Step.checkpoint_callable`` (ticket 05).

The durable form stays declarative: ``checkpoint_callable`` is carried on the
``Step`` for the runner to query, but excluded from every dump and rebuilt from
the pipeline class on restore (ADR-0012/0013).
"""

from __future__ import annotations

from pathlib import Path

from gnomad_meb.cmd.task import Task
from gnomad_meb.config import Settings
from gnomad_meb.lib.checkpoint import CHECKPOINT_TYPE, CheckpointedCallable, checkpoint
from gnomad_meb.pipeline import Pipeline, PipelineRunner
from gnomad_meb.pipeline.spec import Step
from gnomad_meb.pipeline.step import StepRegistry


class _Stage(Pipeline):
    @classmethod
    def name(cls) -> str:
        return "stage"

    @property
    def resources(self):
        return ()

    def get_input(self):
        return ()

    @checkpoint("stage/out", CHECKPOINT_TYPE.DICTIONARY)
    def produce(self):
        return {"a": 1}


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        validate_paths=False, project="demo", checkpoint=str(tmp_path)
    )


def _build(tmp_path: Path) -> PipelineRunner:
    StepRegistry._reset()
    from gnomad_meb.pipeline.registry import ResourceRegistry

    ResourceRegistry._reset()
    task = Task(settings=_settings(tmp_path), command=__import__(
        "gnomad_meb.cmd.task", fromlist=["PipelineStep"]
    ).PipelineStep(name="stage"))
    stage = _Stage(task)
    return PipelineRunner(task, [stage])


def test_a_registered_decorated_step_carries_its_callable(tmp_path: Path) -> None:
    runner = _build(tmp_path)
    step = runner._step_registry["stage.produce"]

    assert isinstance(step.checkpoint_callable, CheckpointedCallable)
    assert step.checkpoint_callable.spec.artifacts == step.artifacts


def test_a_bare_step_without_callable_serializes_no_callable(tmp_path: Path) -> None:
    step = Step(
        name="bare",
        task=Task(settings=_settings(tmp_path), command=__import__(
            "gnomad_meb.cmd.task", fromlist=["PipelineStep"]
        ).PipelineStep(name="bare")),
    )
    assert step.checkpoint_callable is None

    dumped = step.model_dump(mode="json")
    assert "run" not in dumped
    assert "checkpoint_callable" not in dumped


def test_a_decorated_step_dump_excludes_the_callable(tmp_path: Path) -> None:
    runner = _build(tmp_path)
    step = runner._step_registry["stage.produce"]

    dumped = step.model_dump(mode="json")
    assert "run" not in dumped
    assert "checkpoint_callable" not in dumped


def test_run_manifest_dump_step_excludes_the_callable(tmp_path: Path) -> None:
    from gnomad_meb.pipeline.spec import RunManifest

    runner = _build(tmp_path)
    step = runner._step_registry["stage.produce"]

    envelope = RunManifest.dump_step(step)
    assert "run" not in envelope
    assert "checkpoint_callable" not in envelope


def test_load_rebuilds_the_callable(tmp_path: Path) -> None:
    runner = _build(tmp_path)
    dumped = runner.current_state()

    restored = PipelineRunner.load(dumped)
    step = restored._step_registry["stage.produce"]
    assert isinstance(step.checkpoint_callable, CheckpointedCallable)
    # Queryable without invoking.
    assert step.checkpoint_callable.status(restored.settings) is not None
