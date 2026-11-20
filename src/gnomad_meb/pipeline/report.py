"""The human-readable run report: a pure renderer and the observer seam.

A report describes a run without holding any state of its own: the plan, the
per-step lines, and the summary are all projections of the :class:`Step`s the
runner walks and the :class:`~gnomad_meb.pipeline.status.StepResult`s it records
(ADR-0013). This module imports nothing hail-bound, so its rendering is
unit-testable without executing a pipeline.

The four free functions are the renderer's public surface. :class:`RunObserver`
is the one seam the runner speaks to; :class:`TerminalObserver` is the default
production adapter and :class:`RecordingObserver` the test adapter.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Protocol

from ..lib.checkpoint import CheckpointStatus
from ..logging import get_logger
from .spec import Step
from .status import StepResult, StepStatus

#: A pathological step id is not allowed to push the other columns off-screen.
_STEP_ID_WIDTH_CAP = 48

#: The order statuses are tallied in a summary header; only nonzero counts
#: appear, so an all-finished run reads ``finished = n`` first and a run that
#: never reached a step still names it.
_COUNT_ORDER: tuple[StepStatus, ...] = (
    StepStatus.FINISHED,
    StepStatus.SKIPPED,
    StepStatus.ERROR,
    StepStatus.DROPPED,
    StepStatus.INIT,
    StepStatus.STARTED,
)

#: The event verb a step's status reads as on its own line.
_EVENT_VERB: dict[StepStatus, str] = {
    StepStatus.STARTED: "running",
    StepStatus.FINISHED: "finished",
    StepStatus.SKIPPED: "skipped",
    StepStatus.ERROR: "failed",
}

#: The plan's own verbs for a non-``init`` step status: a pruned step reads
#: ``dropped`` and a queued one ``pending`` on its plan row, distinct from the
#: lifecycle-change verbs in :data:`_EVENT_VERB`.
_PLAN_VERB: dict[StepStatus, str] = {
    StepStatus.DROPPED: "dropped",
    StepStatus.PENDING: "pending",
}

#: The plan/summary mark for an evaluated checkpoint status.
CHECKPOINT_MARK: dict[CheckpointStatus, str] = {
    CheckpointStatus.PRESENT: "present",
    CheckpointStatus.ABSENT: "absent",
    CheckpointStatus.STALE: "stale",
    CheckpointStatus.UNKNOWN: "unknown",
}


def format_duration(seconds: float) -> str:
    """Format ``seconds`` human-first: ``842ms``, ``12.4s``, ``3m 05s``, ``1h 02m``."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        minutes, remainder = divmod(int(seconds), 60)
        return f"{minutes}m {remainder:02d}s"
    hours, remainder = divmod(int(seconds), 3600)
    return f"{hours}h {remainder // 60:02d}m"


def render_plan(steps: Sequence[Step], state: Mapping[str, StepResult]) -> str:
    """Render the planned graph: a header, then one indexed row per step.

    Rows follow the given (topological) order and name each step's declared
    artifact count and incoming edges. The plan is structural — it describes the
    graph and, from the ``state`` mapping, reports which steps were pruned by a
    selection (``dropped``) or are queued (``pending``); it never predicts which
    steps will be skipped. Because a dropped step is by definition excluded from
    ``steps``, the count of dropped steps is read from the whole ``state``
    mapping, not from the rows below.
    """
    header = _plan_header(steps)
    dropped = sum(
        1 for record in state.values() if record.status == StepStatus.DROPPED
    )
    width = _id_width(steps)
    index_width = len(str(len(steps)))

    lines = [header, f"dropped by selection: {dropped}"]
    for index, step in enumerate(steps, start=1):
        edges = step.incoming_edges(state)
        after = ", ".join(edges) if edges else "(none)"
        line = (
            f"  {index:>{index_width}}. {step.id:<{width}}  "
            f"artifacts = {len(step.artifacts)}  after = {after}"
        )
        # The evaluated checkpoint status annotates the row (``present`` /
        # ``absent`` / ``stale`` / ``unknown``).
        record = state[step.id]
        line += f"  {CHECKPOINT_MARK[record.checkpoint_status]}"
        if record.stale:
            line += "  stale"
        verb = _PLAN_VERB.get(record.status)
        if verb is not None:
            line += f"  {verb}"
        lines.append(line)
    return "\n".join(lines)


def render_step_line(step: Step, record: StepResult) -> str:
    """Render one lifecycle change as a greppable line, or ``""`` for ``init``.

    The line leads with an event verb and then a fixed, ordered set of
    ``key = value`` fields. ``step =`` comes first; ``status =`` second on a
    terminal line; ``duration =`` on finished/error when both timestamps exist;
    and ``reason =`` (skip) or ``msg =`` (error) last, so the fields before them
    never move.
    """
    verb = _EVENT_VERB.get(record.status)
    if verb is None:
        return ""

    fields = [f"step = {step.id}"]
    if record.status != StepStatus.STARTED:
        fields.append(f"status = {record.status.value}")

    duration = _duration(record)
    terminal = (StepStatus.FINISHED, StepStatus.ERROR)
    if record.status in terminal and duration is not None:
        fields.append(f"duration = {format_duration(duration)}")

    if record.status == StepStatus.SKIPPED:
        # A skipped step reads the reason it was skipped: ``stale`` when it was
        # invalidated before being reused, else ``reused``.
        reason = "stale" if record.stale else "reused"
        fields.append(f"reason = {reason}")
    if record.status == StepStatus.ERROR and record.msg is not None:
        fields.append(f"msg = {record.msg}")

    return f"{verb} pipeline step: {', '.join(fields)}"


def render_summary(steps: Sequence[Step], state: Mapping[str, StepResult]) -> str:
    """Render the run's outcome: a tally, one row per step, then its failures.

    The tally counts the steps by status and carries the wall clock (the
    earliest start to the latest finish) and the slowest step, both omitted when
    no step has a duration. Rows follow the given order so the summary reads
    like the run, and every ``error`` step is repeated as a final list.
    """
    records = [state[step.id] for step in steps]
    lines = [_summary_tally(steps, records)]

    width = _id_width(steps)
    status_width = max((len(record.status.value) for record in records), default=0)
    for step, record in zip(steps, records, strict=True):
        duration = _duration(record)
        if duration is None:
            lines.append(f"  {step.id:<{width}}  {record.status.value}")
        else:
            lines.append(
                f"  {step.id:<{width}}  {record.status.value:<{status_width}}  "
                f"{format_duration(duration)}"
            )
        if record.stale:
            lines[-1] += "  stale"

    failures = [step for step in steps if state[step.id].status == StepStatus.ERROR]
    if failures:
        lines.append("failed steps:")
        lines.extend(f"  {step.id}: {state[step.id].msg}" for step in failures)
    return "\n".join(lines)


class RunObserver(Protocol):
    """Everything the runner needs to say about a run, at three moments."""

    def run_started(
        self, steps: Sequence[Step], state: Mapping[str, StepResult]
    ) -> None:
        """The walk is about to begin; ``steps`` is its planned order."""

    def step_changed(self, step: Step, record: StepResult) -> None:
        """A step's status changed (started, or reached a terminal status)."""

    def run_finished(
        self, steps: Sequence[Step], state: Mapping[str, StepResult]
    ) -> None:
        """The walk is over; ``steps`` is the order actually walked."""


class TerminalObserver:
    """Render the run to the process log as human-readable lines.

    It delegates all text to the free renderers and chooses only the level:
    the plan, the start/finish/skip lines, and the summary are emitted at
    ``INFO`` (skip promoted from ``DEBUG``); failures at ``ERROR``. An ``init``
    record renders nothing and emits nothing.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or get_logger()

    @property
    def logger(self) -> logging.Logger:
        """The logger the observer writes to (the process log by default)."""
        return self._logger

    def run_started(
        self, steps: Sequence[Step], state: Mapping[str, StepResult]
    ) -> None:
        self._logger.info(render_plan(steps, state))

    def step_changed(self, step: Step, record: StepResult) -> None:
        line = render_step_line(step, record)
        if not line:
            return
        level = logging.ERROR if record.status is StepStatus.ERROR else logging.INFO
        self._logger.log(level, line)

    def run_finished(
        self, steps: Sequence[Step], state: Mapping[str, StepResult]
    ) -> None:
        self._logger.info(render_summary(steps, state))


class RecordingObserver:
    """Record the observer calls a run makes, for tests through the same seam.

    ``calls`` is the ``(method, step_id, status)`` sequence — ``step_id`` and
    ``status`` are ``None`` for the two run-level calls — and ``started`` /
    ``finished`` keep the plan and summary arguments intact.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None, StepStatus | None]] = []
        self.started: tuple[Sequence[Step], Mapping[str, StepResult]] | None = None
        self.finished: tuple[Sequence[Step], Mapping[str, StepResult]] | None = None

    def run_started(
        self, steps: Sequence[Step], state: Mapping[str, StepResult]
    ) -> None:
        self.calls.append(("run_started", None, None))
        self.started = (steps, state)

    def step_changed(self, step: Step, record: StepResult) -> None:
        self.calls.append(("step_changed", step.id, record.status))

    def run_finished(
        self, steps: Sequence[Step], state: Mapping[str, StepResult]
    ) -> None:
        self.calls.append(("run_finished", None, None))
        self.finished = (steps, state)


def _plan_header(steps: Sequence[Step]) -> str:
    """The plan's one-line header: the step count and the distinct groups."""
    groups = _distinct_groups(steps)
    group_list = ", ".join(groups)
    suffix = f"groups = {group_list}" if group_list else "groups ="
    return f"pipeline run planned: steps = {len(steps)}, {suffix}"


def _summary_tally(steps: Sequence[Step], records: Sequence[StepResult]) -> str:
    """The summary's one-line tally: counts, wall clock, and the slowest step."""
    tally = f"pipeline run finished: steps = {len(steps)}"

    counts = _status_counts(records)
    for status in _COUNT_ORDER:
        count = counts.get(status, 0)
        if count:
            tally += f", {status.value} = {count}"

    stale = sum(1 for record in records if record.stale)
    if stale:
        tally += f", stale = {stale}"

    timed = [
        (step.id, duration)
        for step, record in zip(steps, records, strict=True)
        if (duration := _duration(record)) is not None
    ]
    if timed:
        starts = [record.started_at for record in records if record.started_at]
        finishes = [record.finished_at for record in records if record.finished_at]
        wall = (max(finishes) - min(starts)).total_seconds()
        slowest_id, slowest = max(timed, key=lambda pair: pair[1])
        tally += (
            f", duration = {format_duration(wall)}, "
            f"slowest = {slowest_id} ({format_duration(slowest)})"
        )
    return tally


def _status_counts(records: Sequence[StepResult]) -> dict[StepStatus, int]:
    """Count ``records`` by status."""
    counts: dict[StepStatus, int] = {}
    for record in records:
        counts[record.status] = counts.get(record.status, 0) + 1
    return counts


def _duration(record: StepResult) -> float | None:
    """The recorded elapsed seconds, or ``None`` when a timestamp is missing."""
    if record.started_at is None or record.finished_at is None:
        return None
    return (record.finished_at - record.started_at).total_seconds()


def _distinct_groups(steps: Sequence[Step]) -> list[str]:
    """The groups in first-seen order, with ungrouped steps ignored."""
    groups: list[str] = []
    for step in steps:
        if step.group is not None and step.group not in groups:
            groups.append(step.group)
    return groups


def _id_width(steps: Sequence[Step]) -> int:
    """The step-id column width: the longest id present, capped."""
    return min(max((len(step.id) for step in steps), default=0), _STEP_ID_WIDTH_CAP)
