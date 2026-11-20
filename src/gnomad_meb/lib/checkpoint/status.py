"""The checkpoint status vocabulary, shared by the callable and the runner.

Deliberately free of hail and ``gnomad_meb.config`` imports so
:mod:`gnomad_meb.config.settings` can import :class:`StalenessPolicy` without a
cycle. Everything here is pure: the policy table (:func:`verdict`), the aggregate
precedence (:func:`aggregate_status`) and the aggregator
(:func:`evaluate_resolutions`). Reading sidecars and resolving paths is the
callable's job (ticket 04).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .artifact import Artifact
    from .metadata import CheckpointMetadata


class CheckpointStatus(StrEnum):
    """The verdict for one checkpoint's artifacts."""

    ABSENT = "absent"  # no artifact, or data incomplete (_SUCCESS missing)
    PRESENT = "present"  # artifact + sidecar present and provenance agrees
    STALE = "stale"  # artifact present, provenance disagrees with the run
    UNKNOWN = "unknown"  # artifact present, no sidecar to compare (legacy/foreign)


class StalenessPolicy(StrEnum):
    """Which provenance signals make a present artifact stale."""

    COMPLETENESS = "completeness"  # today: data presence only
    SOURCE = "source"  # recompute when source_md5 differs
    SOURCE_OR_GIT = "source_or_git"  # ... or git_revision / version differs


class StalenessReason(StrEnum):
    """Why a present artifact is stale for this run."""

    SOURCE_CHANGED = "source_changed"
    GIT_CHANGED = "git_changed"
    VERSION_CHANGED = "version_changed"
    LINEAGE_INVALID = "lineage_invalid"  # an upstream producer is absent this run


@dataclass(frozen=True)
class ArtifactResolution:
    """One artifact's resolved status, with the provenance behind the verdict."""

    artifact: Artifact
    path: str
    metadata: CheckpointMetadata | None
    status: CheckpointStatus
    reasons: tuple[StalenessReason, ...] = ()


@dataclass(frozen=True)
class CheckpointEvaluation:
    """One status evaluation, shared by the runner and the wrapper."""

    status: CheckpointStatus
    reasons: tuple[StalenessReason, ...]
    resolutions: tuple[ArtifactResolution, ...]
    reusable: bool  # True when this run may reuse the artifact
    empty: bool = False  # no declared artifacts (never cached)


def verdict(
    metadata: CheckpointMetadata | None,
    *,
    stored: bool,
    policy: StalenessPolicy,
    invalidated_by_lineage: bool,
    current_source_md5: str | None = None,
    current_git_revision: str | None = None,
    current_version: str | None = None,
) -> tuple[CheckpointStatus, tuple[StalenessReason, ...]]:
    """Apply the policy table to one artifact.

    ``stored`` is the data-completeness signal (the artifact's bytes are present
    and complete). ``metadata`` is the sidecar, or ``None`` when none exists.
    ``invalidated_by_lineage`` is set by the caller for a step whose artifact is
    present but whose upstream producer is absent this run; it never resurrects a
    missing artifact. A ``None`` current value skips that comparison.
    """
    if not stored:
        return CheckpointStatus.ABSENT, ()

    if metadata is None:
        return CheckpointStatus.UNKNOWN, ()

    if invalidated_by_lineage:
        return CheckpointStatus.STALE, (StalenessReason.LINEAGE_INVALID,)

    reasons: list[StalenessReason] = []
    if (
        policy in (StalenessPolicy.SOURCE, StalenessPolicy.SOURCE_OR_GIT)
        and current_source_md5 is not None
        and metadata.source_md5 != current_source_md5
    ):
        reasons.append(StalenessReason.SOURCE_CHANGED)
    if policy is StalenessPolicy.SOURCE_OR_GIT:
        if (
            current_git_revision is not None
            and metadata.git_revision != current_git_revision
        ):
            reasons.append(StalenessReason.GIT_CHANGED)
        if current_version is not None and metadata.version != current_version:
            reasons.append(StalenessReason.VERSION_CHANGED)

    if reasons:
        return CheckpointStatus.STALE, tuple(reasons)
    return CheckpointStatus.PRESENT, ()


#: Aggregate precedence: a missing artifact dominates, then an untrusted one.
_PRECEDENCE = (
    CheckpointStatus.ABSENT,
    CheckpointStatus.STALE,
    CheckpointStatus.UNKNOWN,
    CheckpointStatus.PRESENT,
)


def aggregate_status(resolutions: tuple[ArtifactResolution, ...]) -> CheckpointStatus:
    """The step's aggregate status: ``ABSENT > STALE > UNKNOWN > PRESENT``."""
    statuses = {resolution.status for resolution in resolutions}
    for status in _PRECEDENCE:
        if status in statuses:
            return status
    return CheckpointStatus.PRESENT


def evaluate_resolutions(
    resolutions: tuple[ArtifactResolution, ...],
    *,
    policy: StalenessPolicy,
    lineage_invalidated: bool,
) -> CheckpointEvaluation:
    """Aggregate per-artifact resolutions into one evaluation.

    ``resolutions`` already carry their per-artifact status (the caller applies
    :func:`verdict`). This sets the aggregate status, the union of reasons, and
    ``reusable``: a step is reusable when the aggregate is ``PRESENT``, or
    ``UNKNOWN`` under ``COMPLETENESS`` (today's behaviour). No resolutions means
    ``empty=True`` and never reusable, so a pass-through is never a cache hit.
    """
    empty = len(resolutions) == 0
    resolutions = tuple(resolutions)
    status = aggregate_status(resolutions)
    reasons = tuple(
        reason for resolution in resolutions for reason in resolution.reasons
    )
    reusable = status is CheckpointStatus.PRESENT or (
        status is CheckpointStatus.UNKNOWN and policy is StalenessPolicy.COMPLETENESS
    )
    if empty:
        reusable = False
    return CheckpointEvaluation(
        status=status,
        reasons=reasons,
        resolutions=resolutions,
        reusable=reusable,
        empty=empty,
    )
