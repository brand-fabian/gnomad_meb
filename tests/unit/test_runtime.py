"""Tests for the runtime seam (:mod:`gnomad_meb.pipeline.runtime`).

The runner keeps ordering, skipping, dependency resolution, and terminal-state
ownership; a :class:`StepRuntime` owns *how and where* a step's callable runs and
reports intermediate status through the runner's :class:`StepContext` handle.
These tests exercise the seam through the runner's public surface (the
``RecordingObserver`` and ``execution_state``) and through stub runtimes.
"""

from __future__ import annotations

from typing import Any

from gnomad_meb.cmd.task import PipelineStep, Task
from gnomad_meb.config import Settings
from gnomad_meb.lib.checkpoint import CHECKPOINT_TYPE, checkpoint
from gnomad_meb.pipeline import (
    FromStep,
    InputStep,
    Pipeline,
    PipelineRunner,
    RecordingObserver,
    Resource,
    RunInput,
    StepStatus,
)
from gnomad_meb.pipeline.runtime import RUNTIME_PLACEHOLDER, LocalRuntime


def _settings() -> Settings:
    return Settings(validate_paths=False, project="demo")


def _task() -> Task:
    return Task(settings=_settings(), command=PipelineStep(name="test"))


class Chain(Pipeline):
    @classmethod
    def name(cls) -> str:
        return "chain"

    @property
    def resources(self) -> tuple[Resource, ...]:
        return ()

    def get_input(self) -> tuple[InputStep, ...]:
        return (
            InputStep(name="input", task=self.task, group=type(self).name(), value=10),
        )

    @checkpoint("{project}/first", CHECKPOINT_TYPE.FILE)
    def first(self):
        return "one"

    @checkpoint("{project}/second", CHECKPOINT_TYPE.FILE)
    def second(self, value=FromStep("first"), extra=RunInput("input")):  # noqa: B008
        return f"{value}+{extra}"


def test_local_runtime_reproduces_the_started_then_terminal_sequence() -> None:
    task = _task()
    observer = RecordingObserver()
    runner = PipelineRunner(
        task, [Chain(task)], observer=observer, runtime=LocalRuntime()
    )

    runner.run()

    assert observer.calls[1:-1] == [
        ("step_changed", "chain.input", StepStatus.STARTED),
        ("step_changed", "chain.input", StepStatus.FINISHED),
        ("step_changed", "chain.first", StepStatus.STARTED),
        ("step_changed", "chain.first", StepStatus.FINISHED),
        ("step_changed", "chain.second", StepStatus.STARTED),
        ("step_changed", "chain.second", StepStatus.FINISHED),
    ]
    assert runner.execution_state["chain.second"].status is StepStatus.FINISHED
    assert runner.execution_state["chain.second"].result == "one+10"


class PendingRuntime:
    """A stub runtime that publishes ``pending`` before returning the call result."""

    def prepare(self, step, call, task) -> None:
        pass

    def run(self, step, call, context) -> Any:
        context.record(StepStatus.PENDING, msg="queued")
        return call()

    def postprocess(self, step, context) -> Any:
        raise AssertionError("postprocess must not be called when run returns a result")


def test_a_runtime_pending_record_reaches_the_observer_then_finalizes() -> None:
    task = _task()
    observer = RecordingObserver()
    runner = PipelineRunner(
        task, [Chain(task)], observer=observer, runtime=PendingRuntime()
    )

    runner.run()

    assert ("step_changed", "chain.input", StepStatus.PENDING) in observer.calls
    assert runner.execution_state["chain.input"].status is StepStatus.FINISHED
    # The pending line precedes the terminal finish for that step.
    pending_at = observer.calls.index(
        ("step_changed", "chain.input", StepStatus.PENDING)
    )
    finished_at = observer.calls.index(
        ("step_changed", "chain.input", StepStatus.FINISHED)
    )
    assert pending_at < finished_at


class BoomRuntime:
    """A stub runtime whose ``run`` raises for one named step."""

    def __init__(self, fail_on: str) -> None:
        self._fail_on = fail_on

    def prepare(self, step, call, task) -> None:
        pass

    def run(self, step, call, context) -> Any:
        if step.id == self._fail_on:
            raise RuntimeError("kaboom")
        return call()

    def postprocess(self, step, context) -> Any:
        raise AssertionError("postprocess must not be called on a normal step")


def test_a_runtime_that_raises_is_recorded_error_and_not_propagated() -> None:
    task = _task()
    runner = PipelineRunner(task, [Chain(task)], runtime=BoomRuntime("chain.first"))

    runner.run()  # must not raise

    record = runner.execution_state["chain.first"]
    assert record.status is StepStatus.ERROR
    assert "kaboom" in record.msg
    # The walk continues with the remaining steps.
    assert runner.execution_state["chain.second"].status in (
        StepStatus.FINISHED,
        StepStatus.ERROR,
    )


class PostprocessRuntime:
    """A runtime that returns the placeholder for one step and postprocesses it."""

    def prepare(self, step, call, task) -> None:
        pass

    def run(self, step, call, context) -> Any:
        if step.id == "chain.first":
            return RUNTIME_PLACEHOLDER
        return call()

    def postprocess(self, step, context) -> Any:
        return "from-postprocess"


def test_a_postprocess_result_is_used_when_run_returns_the_placeholder() -> None:
    task = _task()
    runner = PipelineRunner(task, [Chain(task)], runtime=PostprocessRuntime())

    runner.run()

    record = runner.execution_state["chain.first"]
    assert record.status is StepStatus.FINISHED
    assert record.result == "from-postprocess"


class ClosedContextRuntime:
    """A stub runtime that keeps the context and records once the run is closed."""

    def __init__(self) -> None:
        self.kept: Any = None

    def prepare(self, step, call, task) -> None:
        pass

    def run(self, step, call, context) -> Any:
        if step.id == "chain.first":
            self.kept = context
        return call()

    def postprocess(self, step, context) -> Any:
        raise AssertionError("no placeholder path in this test")


def test_a_record_on_a_closed_context_does_not_change_the_terminal_result() -> None:
    task = _task()
    runtime = ClosedContextRuntime()
    runner = PipelineRunner(task, [Chain(task)], runtime=runtime)

    runner.run()

    before = runner.execution_state["chain.first"].status
    # The context was closed after the runner finalized the step.
    runtime.kept.record(StepStatus.ERROR, msg="late")

    after = runner.execution_state["chain.first"]
    assert before is StepStatus.FINISHED
    assert after.status is StepStatus.FINISHED
    assert after.msg is None
