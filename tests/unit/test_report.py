"""Tests for the run report's pure rendering core.

The rendered text is the contract: these tests call the four free functions of
:mod:`gnomad_meb.pipeline.report` and assert exact text, without executing a
pipeline. The report reads only a :class:`Step` and its
:class:`~gnomad_meb.pipeline.StepResult`, so it stays hail-free and runnable on
a machine that cannot drive the hail stack.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta

from gnomad_meb.cmd.task import PipelineStep, Task
from gnomad_meb.config import Settings
from gnomad_meb.lib.checkpoint import CheckpointStatus, StalenessReason
from gnomad_meb.logging import get_logger
from gnomad_meb.pipeline import (
    InputStep,
    RecordingObserver,
    RunObserver,
    Step,
    StepResult,
    StepStatus,
    TerminalObserver,
    format_duration,
    render_plan,
    render_step_line,
    render_summary,
)


def _task() -> Task:
    return Task(
        settings=Settings(validate_paths=False, project="demo", checkpoint=None),
        command=PipelineStep(name="test"),
    )


def _step(
    name: str,
    *,
    group: str | None = None,
    artifacts: tuple[object, ...] = (),
    depends_on: dict[str, str] | None = None,
    resources: dict[str, str] | None = None,
) -> Step:
    return Step(
        name=name,
        task=_task(),
        group=group,
        artifacts=artifacts,
        depends_on=depends_on or {},
        resources=resources or {},
        run=lambda: None,
    )


def _at(seconds: float) -> datetime:
    return datetime(2026, 1, 1, 12, 0, tzinfo=UTC) + timedelta(seconds=seconds)


def _record(
    step_id: str,
    status: StepStatus,
    *,
    msg: str | None = None,
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
    checkpoint_status: CheckpointStatus = CheckpointStatus.UNKNOWN,
    reused: bool = False,
    stale: bool = False,
    staleness: tuple[StalenessReason, ...] = (),
) -> StepResult:
    return StepResult(
        step_id=step_id,
        status=status,
        msg=msg,
        started_at=started_at,
        finished_at=finished_at,
        checkpoint_status=checkpoint_status,
        reused=reused,
        stale=stale,
        staleness=staleness,
    )


# --- format_duration --------------------------------------------------------


def test_format_duration_buckets_by_magnitude() -> None:
    assert format_duration(0.0) == "0ms"
    assert format_duration(0.842) == "842ms"
    assert format_duration(0.999) == "999ms"

    assert format_duration(1.0) == "1.0s"
    assert format_duration(12.4) == "12.4s"
    assert format_duration(59.9) == "59.9s"

    assert format_duration(60.0) == "1m 00s"
    assert format_duration(185.0) == "3m 05s"
    assert format_duration(3599.0) == "59m 59s"

    assert format_duration(3600.0) == "1h 00m"
    assert format_duration(3725.0) == "1h 02m"


# --- render_plan ------------------------------------------------------------


def _plan_steps() -> tuple[Step, ...]:
    load = _step("load", group="a")
    make = _step("make", group="a", artifacts=("x",), depends_on={"res": "a.load"})
    use = _step("use", group="b", resources={"v": "a.make"})
    return (load, make, use)


def test_render_plan_is_a_header_and_indexed_aligned_rows() -> None:
    steps = _plan_steps()
    state = {step.id: _record(step.id, StepStatus.INIT) for step in steps}

    assert render_plan(steps, state) == (
        "pipeline run planned: steps = 3, groups = a, b\n"
        "dropped by selection: 0\n"
        "  1. a.load  artifacts = 0  after = (none)  unknown\n"
        "  2. a.make  artifacts = 1  after = a.load  unknown\n"
        "  3. b.use   artifacts = 0  after = a.make  unknown"
    )


def test_render_plan_header_counts_dropped_steps_from_state() -> None:
    steps = _plan_steps()
    state = {step.id: _record(step.id, StepStatus.INIT) for step in steps}
    state["b.use"] = _record("b.use", StepStatus.DROPPED)
    # A dropped step is not rendered as a row, but the header still counts it.
    rendered = render_plan(steps[:2], state)

    assert rendered.splitlines()[1] == "dropped by selection: 1"


def test_render_plan_lists_a_registered_optional_dependency_as_an_edge() -> None:
    make = _step("make", group="a")
    vds = InputStep(name="vds", group="b", task=_task(), source="a.make")
    state = {step.id: _record(step.id, StepStatus.INIT) for step in (make, vds)}

    rendered = render_plan((make, vds), state)

    assert rendered.splitlines()[3].endswith("after = a.make  unknown")


def test_render_plan_of_no_steps_is_just_the_header() -> None:
    assert render_plan((), {}) == (
        "pipeline run planned: steps = 0, groups =\n"
        "dropped by selection: 0"
    )


def test_render_plan_annotates_a_dropped_step() -> None:
    steps = _plan_steps()
    state = {step.id: _record(step.id, StepStatus.INIT) for step in steps}
    state["a.make"] = _record("a.make", StepStatus.DROPPED)

    rendered = render_plan(steps, state)

    assert rendered.splitlines()[3].endswith("after = a.load  unknown  dropped")


def test_render_plan_annotates_a_pending_step() -> None:
    steps = _plan_steps()
    state = {step.id: _record(step.id, StepStatus.INIT) for step in steps}
    state["b.use"] = _record("b.use", StepStatus.PENDING)

    rendered = render_plan(steps, state)

    assert rendered.splitlines()[4].endswith("after = a.make  unknown  pending")


# --- render_step_line -------------------------------------------------------


def test_render_step_line_for_a_started_step_carries_only_the_id() -> None:
    step = _step("create_variant_dataset", group="vds")
    record = _record(step.id, StepStatus.STARTED)

    assert (
        render_step_line(step, record)
        == "running pipeline step: step = vds.create_variant_dataset"
    )


def test_render_step_line_for_a_finished_step_carries_status_and_duration() -> None:
    step = _step("create_variant_dataset", group="vds")
    record = _record(
        step.id,
        StepStatus.FINISHED,
        started_at=_at(0.0),
        finished_at=_at(18.2),
    )

    assert render_step_line(step, record) == (
        "finished pipeline step: step = vds.create_variant_dataset, "
        "status = finished, duration = 18.2s"
    )


def test_render_step_line_for_a_finished_step_without_a_start_has_no_duration() -> None:
    step = _step("after_annotation", group="vds")
    record = _record(step.id, StepStatus.FINISHED)

    assert render_step_line(step, record) == (
        "finished pipeline step: step = vds.after_annotation, status = finished"
    )


def test_render_step_line_for_a_skip_reads_reason_reused() -> None:
    step = _step("sample_qc", group="sample_qc")
    record = _record(
        step.id,
        StepStatus.SKIPPED,
        msg="artifacts exist",
        reused=True,
        finished_at=_at(0.0),
    )

    assert render_step_line(step, record) == (
        "skipped pipeline step: step = sample_qc.sample_qc, "
        "status = skipped, reason = reused"
    )


def test_render_step_line_for_a_stale_skip_reads_reason_stale() -> None:
    step = _step("sample_qc", group="sample_qc")
    record = _record(
        step.id,
        StepStatus.SKIPPED,
        reused=True,
        stale=True,
        staleness=(StalenessReason.LINEAGE_INVALID,),
        finished_at=_at(0.0),
    )

    assert render_step_line(step, record) == (
        "skipped pipeline step: step = sample_qc.sample_qc, "
        "status = skipped, reason = stale"
    )


def test_render_step_line_for_an_error_carries_status_duration_and_msg() -> None:
    step = _step("export", group="variant_qc")
    msg = "failed to process step: step_id = variant_qc.export, error = kaboom"
    record = _record(
        step.id,
        StepStatus.ERROR,
        msg=msg,
        started_at=_at(0.0),
        finished_at=_at(0.8),
    )

    assert render_step_line(step, record) == (
        "failed pipeline step: step = variant_qc.export, status = error, "
        "duration = 800ms, msg = failed to process step: step_id = "
        "variant_qc.export, error = kaboom"
    )


def test_render_step_line_is_empty_for_an_init_record() -> None:
    step = _step("never_ran", group="vds")

    assert render_step_line(step, _record(step.id, StepStatus.INIT)) == ""


def test_render_step_line_is_empty_for_a_dropped_record() -> None:
    step = _step("never_ran", group="vds")

    assert render_step_line(step, _record(step.id, StepStatus.DROPPED)) == ""


# --- render_summary ---------------------------------------------------------


def _summary_state() -> tuple[tuple[Step, ...], dict[str, StepResult]]:
    load = _step("load", group="a")
    make = _step("make", group="a")
    use = _step("use", group="b")
    extra = _step("extra", group="b")
    steps = (load, make, use, extra)
    state = {
        "a.load": _record(
            "a.load",
            StepStatus.FINISHED,
            started_at=_at(0.0),
            finished_at=_at(18.2),
        ),
        "a.make": _record(
            "a.make",
            StepStatus.SKIPPED,
            msg="artifacts exist",
            finished_at=_at(18.3),
        ),
        "b.use": _record(
            "b.use",
            StepStatus.ERROR,
            msg="failed to process step: step_id = b.use, error = kaboom",
            started_at=_at(19.0),
            finished_at=_at(19.8),
        ),
        "b.extra": _record("b.extra", StepStatus.INIT),
    }
    return steps, state


def test_render_summary_tallies_rows_and_lists_failures() -> None:
    steps, state = _summary_state()

    assert render_summary(steps, state) == (
        "pipeline run finished: steps = 4, finished = 1, skipped = 1, "
        "error = 1, init = 1, duration = 19.8s, slowest = a.load (18.2s)\n"
        "  a.load   finished  18.2s\n"
        "  a.make   skipped\n"
        "  b.use    error     800ms\n"
        "  b.extra  init\n"
        "failed steps:\n"
        "  b.use: failed to process step: step_id = b.use, error = kaboom"
    )


def test_render_summary_of_an_all_skipped_run_omits_duration_and_slowest() -> None:
    load = _step("load", group="a")
    make = _step("make", group="a")
    steps = (load, make)
    state = {
        "a.load": _record("a.load", StepStatus.SKIPPED, msg="artifacts exist"),
        "a.make": _record("a.make", StepStatus.SKIPPED, msg="artifacts exist"),
    }

    assert render_summary(steps, state) == (
        "pipeline run finished: steps = 2, skipped = 2\n"
        "  a.load  skipped\n"
        "  a.make  skipped"
    )


def test_render_summary_of_no_steps_is_just_the_tally() -> None:
    assert render_summary((), {}) == "pipeline run finished: steps = 0"


def test_render_summary_tally_counts_dropped_steps_after_error() -> None:
    load = _step("load", group="a")
    make = _step("make", group="a")
    steps = (load, make)
    state = {
        "a.load": _record(
            "a.load",
            StepStatus.ERROR,
            msg="kaboom",
            started_at=_at(0.0),
            finished_at=_at(0.8),
        ),
        "a.make": _record("a.make", StepStatus.DROPPED),
    }

    assert render_summary(steps, state) == (
        "pipeline run finished: steps = 2, error = 1, dropped = 1, "
        "duration = 800ms, slowest = a.load (800ms)\n"
        "  a.load  error    800ms\n"
        "  a.make  dropped\n"
        "failed steps:\n"
        "  a.load: kaboom"
    )


# --- TerminalObserver -------------------------------------------------------


def test_terminal_observer_logs_the_plan_at_info(capturing_logger) -> None:
    steps = _plan_steps()
    state = {step.id: _record(step.id, StepStatus.INIT) for step in steps}
    logger, stream = capturing_logger

    TerminalObserver(logger=logger).run_started(steps, state)

    assert stream.getvalue() == f"INFO {render_plan(steps, state)}\n"


def test_terminal_observer_logs_each_step_line_at_the_level_of_its_status(
    capturing_logger,
) -> None:
    logger, stream = capturing_logger
    observer = TerminalObserver(logger=logger)

    started = _step("create_variant_dataset", group="vds")
    observer.step_changed(started, _record(started.id, StepStatus.STARTED))

    finished = _step("annotate", group="annotation")
    observer.step_changed(
        finished,
        _record(
            finished.id,
            StepStatus.FINISHED,
            started_at=_at(0.0),
            finished_at=_at(18.2),
        ),
    )

    skipped = _step("sample_qc", group="sample_qc")
    observer.step_changed(
        skipped,
        _record(skipped.id, StepStatus.SKIPPED, reused=True),
    )

    failed = _step("export", group="variant_qc")
    observer.step_changed(
        failed,
        _record(failed.id, StepStatus.ERROR, msg="kaboom"),
    )

    assert stream.getvalue() == (
        "INFO running pipeline step: step = vds.create_variant_dataset\n"
        "INFO finished pipeline step: step = annotation.annotate, "
        "status = finished, duration = 18.2s\n"
        "INFO skipped pipeline step: step = sample_qc.sample_qc, "
        "status = skipped, reason = reused\n"
        "ERROR failed pipeline step: step = variant_qc.export, "
        "status = error, msg = kaboom\n"
    )


def test_terminal_observer_is_silent_for_an_init_record(capturing_logger) -> None:
    logger, stream = capturing_logger
    step = _step("never_ran", group="vds")

    TerminalObserver(logger=logger).step_changed(
        step, _record(step.id, StepStatus.INIT)
    )

    assert stream.getvalue() == ""


def test_terminal_observer_logs_the_summary_at_info(capturing_logger) -> None:
    steps, state = _summary_state()
    logger, stream = capturing_logger

    TerminalObserver(logger=logger).run_finished(steps, state)

    assert stream.getvalue() == f"INFO {render_summary(steps, state)}\n"


def test_terminal_observer_defaults_to_the_process_logger() -> None:
    assert TerminalObserver().logger is get_logger()


# --- RecordingObserver ------------------------------------------------------


def test_recording_observer_records_the_call_sequence() -> None:
    observer = RecordingObserver()
    step = _step("load", group="a")
    started = _record(step.id, StepStatus.STARTED)
    finished = _record(step.id, StepStatus.FINISHED)

    observer.run_started((step,), {step.id: started})
    observer.step_changed(step, started)
    observer.step_changed(step, finished)
    observer.run_finished((step,), {step.id: finished})

    assert observer.calls == [
        ("run_started", None, None),
        ("step_changed", "a.load", StepStatus.STARTED),
        ("step_changed", "a.load", StepStatus.FINISHED),
        ("run_finished", None, None),
    ]


def test_recording_observer_keeps_the_plan_and_summary_arguments() -> None:
    observer = RecordingObserver()
    steps = _plan_steps()
    state = {step.id: _record(step.id, StepStatus.INIT) for step in steps}

    observer.run_started(steps, state)
    observer.run_finished(steps, state)

    assert observer.started == (steps, state)
    assert observer.finished == (steps, state)


# --- fan-out test double ----------------------------------------------------


class _FanOutObserver:
    """A test double that forwards every observer call to each wrapped observer."""

    def __init__(self, *observers: RunObserver) -> None:
        self._observers = observers

    def run_started(
        self, steps: Sequence[Step], state: Mapping[str, StepResult]
    ) -> None:
        for observer in self._observers:
            observer.run_started(steps, state)

    def step_changed(self, step: Step, record: StepResult) -> None:
        for observer in self._observers:
            observer.step_changed(step, record)

    def run_finished(
        self, steps: Sequence[Step], state: Mapping[str, StepResult]
    ) -> None:
        for observer in self._observers:
            observer.run_finished(steps, state)


def test_fan_out_observer_double_sends_every_call_to_each_observer() -> None:
    first = RecordingObserver()
    second = RecordingObserver()
    observer = _FanOutObserver(first, second)

    steps = _plan_steps()
    state = {step.id: _record(step.id, StepStatus.INIT) for step in steps}
    step = steps[0]
    record = _record(step.id, StepStatus.STARTED)

    observer.run_started(steps, state)
    observer.step_changed(step, record)
    observer.run_finished(steps, state)

    expected = [
        ("run_started", None, None),
        ("step_changed", step.id, StepStatus.STARTED),
        ("run_finished", None, None),
    ]
    assert first.calls == expected
    assert second.calls == expected
    assert first.started == (steps, state)
    assert second.finished == (steps, state)


# --- checkpoint status / staleness rendering (ticket 08) ---------------------


def test_render_plan_annotates_a_present_status() -> None:
    steps = _plan_steps()
    state = {step.id: _record(step.id, StepStatus.INIT) for step in steps}
    state["a.make"] = _record(
        "a.make", StepStatus.INIT, checkpoint_status=CheckpointStatus.PRESENT
    )

    rendered = render_plan(steps, state)

    assert rendered.splitlines()[3].endswith("after = a.load  present")


def test_render_plan_annotates_absent_and_unknown_status() -> None:
    steps = _plan_steps()
    state = {step.id: _record(step.id, StepStatus.INIT) for step in steps}
    state["a.make"] = _record(
        "a.make", StepStatus.INIT, checkpoint_status=CheckpointStatus.ABSENT
    )
    state["b.use"] = _record(
        "b.use", StepStatus.INIT, checkpoint_status=CheckpointStatus.UNKNOWN
    )

    rendered = render_plan(steps, state)

    assert rendered.splitlines()[3].endswith("after = a.load  absent")
    assert rendered.splitlines()[4].endswith("after = a.make  unknown")


def test_render_plan_marks_a_stale_row() -> None:
    steps = _plan_steps()
    state = {step.id: _record(step.id, StepStatus.INIT) for step in steps}
    state["a.make"] = _record(
        "a.make",
        StepStatus.INIT,
        checkpoint_status=CheckpointStatus.STALE,
        stale=True,
        staleness=(StalenessReason.LINEAGE_INVALID,),
    )

    rendered = render_plan(steps, state)

    assert rendered.splitlines()[3].endswith("after = a.load  stale  stale")


def test_render_plan_row_annotates_an_unknown_step() -> None:
    steps = _plan_steps()
    state = {step.id: _record(step.id, StepStatus.INIT) for step in steps}

    assert render_plan(steps, state).splitlines()[2] == (
        "  1. a.load  artifacts = 0  after = (none)  unknown"
    )


def test_render_summary_tallies_stale_and_marks_the_row() -> None:
    load = _step("load", group="a")
    make = _step("make", group="a")
    steps = (load, make)
    state = {
        "a.load": _record(
            "a.load",
            StepStatus.FINISHED,
            started_at=_at(0.0),
            finished_at=_at(1.0),
        ),
        "a.make": _record(
            "a.make",
            StepStatus.FINISHED,
            stale=True,
            staleness=(StalenessReason.LINEAGE_INVALID,),
            started_at=_at(1.0),
            finished_at=_at(2.0),
        ),
    }

    rendered = render_summary(steps, state)

    assert "stale = 1" in rendered.splitlines()[0]
    assert any(line.rstrip().endswith("stale") for line in rendered.splitlines()[1:])


def test_render_summary_omits_the_stale_tally_when_zero() -> None:
    steps, state = _summary_state()

    assert "stale" not in render_summary(steps, state)


def test_rendering_a_stale_record_needs_no_state_but_the_record() -> None:
    # Purely from the record: no artifact, sidecar or runner involved.
    step = _step("x", group="a", artifacts=("a/x",))
    record = _record(
        step.id,
        StepStatus.SKIPPED,
        reused=True,
        stale=True,
        staleness=(StalenessReason.LINEAGE_INVALID,),
        checkpoint_status=CheckpointStatus.STALE,
        finished_at=_at(0.0),
    )

    assert render_step_line(step, record).endswith("reason = stale")
