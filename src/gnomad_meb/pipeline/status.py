"""The lifecycle status of a step in a run."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from gnomad_meb.lib.checkpoint.status import CheckpointStatus, StalenessReason


class StepStatus(StrEnum):
    """Where a step is in its lifecycle within a run."""

    INIT = "init"
    STARTED = "started"
    PENDING = "pending"
    SKIPPED = "skipped"
    FINISHED = "finished"
    ERROR = "error"
    DROPPED = "dropped"


class StepResult(BaseModel):
    """The observable record of one step in a run.

    ``status`` and the ``started_at``/``finished_at`` timestamps are the run's
    timing record; ``msg`` carries the reason a step failed (or was skipped).
    The in-memory ``result`` is excluded from serialization, so a dumped run
    stays declarative JSON (ADR-0013).
    """

    model_config = ConfigDict(extra="ignore")

    step_id: str
    status: StepStatus = StepStatus.INIT
    msg: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: Any = Field(default=None, exclude=True)
    #: The step's evaluated checkpoint status (ticket 06), recorded by the run
    #: and dry-run prepasses. ``UNKNOWN`` until evaluated.
    checkpoint_status: CheckpointStatus = CheckpointStatus.UNKNOWN
    #: True when the artifact was found reusable and the step skipped.
    reused: bool = False
    #: True when the artifact exists but its provenance disagrees with the run
    #: (ticket 07); ``staleness`` carries the reasons.
    stale: bool = False
    staleness: tuple[StalenessReason, ...] = ()
