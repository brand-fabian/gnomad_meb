"""The runtime seam: where and how a step's callable runs, across a job boundary.

The runner keeps ordering, skipping, dependency resolution, and ownership of the
terminal :class:`~gnomad_meb.pipeline.status.StepResult`; a :class:`StepRuntime`
owns *how and where* a step's callable runs. The runner hands the runtime a
:class:`StepContext` — the runner-owned status handle for that step — through
which a runtime may publish intermediate statuses (a ``pending`` line for a
remote job, or a runtime-detected error). The runner finalizes the step and
closes the context, so a runtime can never overwrite a terminal record.

:class:`LocalRuntime` is the default: it runs the callable in process and
publishes nothing, reproducing today's observable behaviour.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from gnomad_meb.cmd.task import Task

from .constants import RUNTIME_PLACEHOLDER, TERMINAL_STATUSES
from .spec import Step
from .status import StepResult, StepStatus


class StepContext(Protocol):
    """A runtime's handle on one step's runner-owned status record."""

    def record(
        self,
        status: StepStatus,
        *,
        msg: str | None = None,
        result: Any = RUNTIME_PLACEHOLDER,
    ) -> None:
        """Set the step's status/timestamps and notify the observer."""


class StepRuntime(Protocol):
    """Where and how a step's callable runs, across a job boundary."""

    def prepare(self, step: Step, call: Callable[[], Any], task: Task) -> None: ...

    def run(
        self,
        step: Step,
        call: Callable[[], Any],
        context: StepContext,
    ) -> Any: ...

    def postprocess(self, step: Step, context: StepContext) -> Any: ...


class LocalRuntime:
    """Run each step's callable in process, publishing nothing.

    ``prepare`` and ``postprocess`` are no-ops and ``run`` calls ``call()``
    directly; the runner's own ``STARTED`` and terminal records are the only
    ones this runtime produces. It is the default and reproduces the pre-seam
    behaviour.
    """

    def prepare(self, step: Step, call: Callable[[], Any], task: Task) -> None:
        return None

    def run(
        self,
        step: Step,
        call: Callable[[], Any],
        context: StepContext,
    ) -> Any:
        return call()

    def postprocess(self, step: Step, context: StepContext) -> Any:
        return None


class RunnerStepContext:
    """The runner's concrete :class:`StepContext` for one step.

    It is the single writer of the step's :class:`StepResult` — status, the
    matching timestamp, an optional ``msg`` and in-memory ``result`` — and
    notifies the observer. :meth:`finalize` is the runner's terminal write: it
    stamps the record through the same status -> timestamp mapping but leaves
    the notification to the runner's ``finally``, so a step has exactly one
    terminal observer call. The runner closes the context after finalizing; a
    ``record`` on a closed context is a no-op, so a late or abusive call cannot
    clobber a terminal record.
    """

    def __init__(
        self,
        step: Step,
        record: StepResult,
        notify: Callable[[Step, StepResult], None],
        now: Callable[[], Any],
    ) -> None:
        self._step = step
        self._record = record
        self._notify = notify
        self._now = now
        self._closed = False

    def close(self) -> None:
        """Close the context; subsequent :meth:`record` calls are no-ops."""
        self._closed = True

    def _write(
        self,
        status: StepStatus,
        *,
        msg: str | None,
        result: Any,
    ) -> None:
        """The one status -> timestamp mapping shared by record/finalize."""
        self._record.status = status
        if status in (StepStatus.STARTED, StepStatus.PENDING):
            self._record.started_at = self._now()
        elif status in TERMINAL_STATUSES:
            self._record.finished_at = self._now()

        if msg is not None:
            self._record.msg = msg
        if result is not RUNTIME_PLACEHOLDER:
            self._record.result = result

    def record(
        self,
        status: StepStatus,
        *,
        msg: str | None = None,
        result: Any = RUNTIME_PLACEHOLDER,
    ) -> None:
        if self._closed:
            return
        self._write(status, msg=msg, result=result)
        self._notify(self._step, self._record)

    def finalize(
        self,
        status: StepStatus,
        *,
        msg: str | None = None,
        result: Any = RUNTIME_PLACEHOLDER,
    ) -> None:
        """The runner's terminal write; stamps the record without notifying.

        The runner's ``finally`` is the single terminal ``step_changed`` call, so
        this updates status/timestamp/``msg``/``result`` via the same mapping as
        :meth:`record` and stays silent.
        """
        if self._closed:
            return
        self._write(status, msg=msg, result=result)
