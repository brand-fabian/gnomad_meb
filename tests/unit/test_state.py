"""The round-trip test: dump a run, restore it, assert it came back.

This is the load-bearing test of the restorable run state. It is the reason
``Pipeline.name`` is an abstract classmethod (the factory keys on it to rebuild
the pipeline class) and the reason the status model is dedicated (it is dumped
and restored whole). A run is constructed and executed, its ``current_state`` is
turned into JSON and back, ``PipelineRunner.load`` rebuilds it, and the task,
pipeline names, step ids, resource names, and per-step status/timing are all
asserted to survive. A drifted step/resource set raises
``PipelineMismatchError``; an unknown pipeline name raises ``ValueError``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gnomad_meb.cmd.task import PipelineStep, Task
from gnomad_meb.config import Settings
from gnomad_meb.lib.checkpoint import CHECKPOINT_TYPE, checkpoint
from gnomad_meb.pipeline import (
    FromStep,
    InputStep,
    Pipeline,
    PipelineRunner,
    Resource,
    ResourceRef,
    RunInput,
    StepRegistry,
    StepStatus,
)
from gnomad_meb.pipeline.execution import PipelineMismatchError


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        validate_paths=False,
        project="demo",
        checkpoint=str(tmp_path / "ckpt"),
    )


def _task(tmp_path: Path) -> Task:
    return Task(settings=_settings(tmp_path), command=PipelineStep(name="test"))


class RoundTrip(Pipeline):
    """A small stage with a resource, an input endpoint, and two work steps."""

    @classmethod
    def name(cls) -> str:
        return "round_trip"

    @property
    def resources(self) -> tuple[Resource, ...]:
        return (
            Resource(name="res", path="/data/res.ht", loader=lambda p: f"loaded:{p}"),
        )

    def get_input(self) -> tuple[InputStep, ...]:
        return (
            InputStep(name="input", task=self.task, group=type(self).name(), value=10),
        )

    @checkpoint("{project}/first", CHECKPOINT_TYPE.FILE)
    def first(self, res=ResourceRef("res")):  # noqa: B008
        return res

    @checkpoint("{project}/second", CHECKPOINT_TYPE.FILE)
    def second(self, value=FromStep("first"), extra=RunInput("input")):  # noqa: B008
        return f"{value}+{extra}"


class Splits(Pipeline):
    """A stage that generates one step per split in ``prepare`` (hail-free).

    Overriding ``prepare`` to append loop-built ``@checkpoint`` methods is the
    scenario this test guards: the generated step set must be reproducible from
    the same inputs so ``current_state()`` -> ``PipelineRunner.load`` rebuilds it
    without a ``PipelineMismatchError``.
    """

    SPLITS = ("left", "right")

    @classmethod
    def name(cls) -> str:
        return "splits"

    @property
    def resources(self) -> tuple[Resource, ...]:
        return ()

    def get_input(self) -> tuple[InputStep, ...]:
        return ()

    @checkpoint("{project}/splits/base", CHECKPOINT_TYPE.FILE)
    def base(self):
        return "base"

    def prepare(self) -> None:
        super().prepare()
        for split in self.SPLITS:

            @checkpoint(f"{{project}}/splits/{split}", CHECKPOINT_TYPE.FILE)
            def gen(value=FromStep("base"), _split=split):  # noqa: B008
                return f"{value}:{_split}"

            self._methods.append((f"gen_{split}", gen, split))


def test_a_generated_step_set_is_restored_identically(tmp_path: Path) -> None:
    task = _task(tmp_path)
    runner = PipelineRunner(task, [Splits(task)])
    dumped = _dumped(runner)

    # The generated steps are in the serialized step set to begin with.
    assert set(dumped["steps"]) >= {"splits.gen_left", "splits.gen_right"}

    # Reconstruction rebuilds them from the same inputs -- no mismatch.
    restored = PipelineRunner.load(dumped)

    assert set(restored.current_state()["steps"]) == set(
        runner.current_state()["steps"]
    )
    rebuilt = StepRegistry().get("splits.gen_left")
    assert rebuilt is not None
    assert rebuilt.task.command.args == {"task": "left"}
    assert rebuilt.depends_on == {"value": "splits.base"}


class Selecting(Pipeline):
    """A stage whose consumer selects an output index from a tuple producer."""

    @classmethod
    def name(cls) -> str:
        return "selecting"

    @property
    def resources(self) -> tuple[Resource, ...]:
        return ()

    def get_input(self) -> tuple[InputStep, ...]:
        return ()

    @checkpoint("{project}/selecting/make", CHECKPOINT_TYPE.FILE)
    def make(self):
        return ("left", "right")

    @checkpoint("{project}/selecting/pick", CHECKPOINT_TYPE.FILE)
    def pick(self, value=FromStep("make", output=1)):  # noqa: B008
        return value


def test_a_dumped_run_preserves_the_fromstep_output_selection(tmp_path: Path) -> None:
    task = _task(tmp_path)
    runner = PipelineRunner(task, [Selecting(task)])
    dumped = _dumped(runner)

    assert dumped["steps"]["selecting.pick"]["outputs"] == {"value": 1}

    PipelineRunner.load(dumped)
    rebuilt = StepRegistry().get("selecting.pick")
    assert rebuilt is not None
    assert rebuilt.outputs == {"value": 1}


def _executed_run(tmp_path: Path) -> PipelineRunner:
    task = _task(tmp_path)
    pipeline = RoundTrip(task)
    runner = PipelineRunner(task, [pipeline])
    runner.run()
    return runner


def _dumped(runner: PipelineRunner) -> dict[str, Any]:
    """The runner's state as JSON-then-Python, as a file round-trip would be."""
    return json.loads(json.dumps(runner.current_state()))


def test_a_dumped_run_restores_task_pipelines_steps_resources_and_timing(
    tmp_path: Path,
) -> None:
    runner = _executed_run(tmp_path)
    restored = PipelineRunner.load(_dumped(runner))

    # The task comes back whole.
    assert restored.task == runner.task

    # The pipeline classes are rebuilt by name from the class, not the payload.
    assert [type(p).name() for p in restored.pipelines] == ["round_trip"]

    # The step ids and resource names come back unchanged.
    assert set(restored.execution_state) == set(runner.execution_state)
    assert set(restored.current_state()["resources"]) == {"res"}

    # Every step's status and timing survive the JSON round-trip.
    for step_id, original in runner.execution_state.items():
        rebuilt = restored.execution_state[step_id]
        assert rebuilt.status == original.status
        assert rebuilt.started_at == original.started_at
        assert rebuilt.finished_at == original.finished_at

    # The work actually ran, so the assertions above are not vacuous.
    assert runner.execution_state["round_trip.second"].status == StepStatus.FINISHED
    assert runner.execution_state["round_trip.second"].finished_at is not None


def test_a_drifted_step_set_raises_pipeline_mismatch(tmp_path: Path) -> None:
    data = _dumped(_executed_run(tmp_path))
    del data["steps"]["round_trip.second"]

    with pytest.raises(PipelineMismatchError):
        PipelineRunner.load(data)


def test_a_drifted_resource_set_raises_pipeline_mismatch(tmp_path: Path) -> None:
    data = _dumped(_executed_run(tmp_path))
    data["resources"]["res"]["path"] = "/data/other.ht"

    with pytest.raises(PipelineMismatchError):
        PipelineRunner.load(data)


def test_an_unknown_pipeline_name_raises_value_error(tmp_path: Path) -> None:
    data = _dumped(_executed_run(tmp_path))
    data["pipelines"][0]["pipeline"] = "does_not_exist"

    with pytest.raises(ValueError):
        PipelineRunner.load(data)
