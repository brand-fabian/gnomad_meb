"""Shared constants for the pipeline core."""

from typing import Any

from .status import StepStatus

RUNTIME_PLACEHOLDER: Any = object()
"""The sentinel a :class:`StepRuntime` returns when its ``run`` deferred the
step's real result to ``postprocess`` (a job boundary reads it back later).

A runtime that runs the callable in process (``LocalRuntime``) never returns it;
the runner only calls ``postprocess`` when ``run`` returned this sentinel.
"""

TERMINAL_STATUSES: frozenset[StepStatus] = frozenset(
    {
        StepStatus.FINISHED,
        StepStatus.ERROR,
        StepStatus.SKIPPED,
        StepStatus.DROPPED,
    }
)
"""The statuses that end a step's lifecycle; a terminal record stamps
``finished_at``."""
