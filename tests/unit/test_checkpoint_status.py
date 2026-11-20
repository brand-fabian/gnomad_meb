"""Tests for the checkpoint status vocabulary and its pure rules.

``status.py`` is the one vocabulary the callable (04) and the runner (06) share:
the ``CheckpointStatus`` / ``StalenessPolicy`` / ``StalenessReason`` enums, the
``verdict`` policy table, the ``aggregate_status`` precedence, and the pure
``evaluate_resolutions`` aggregator. It must be importable without hail or
``gnomad_meb.config`` (the enum is read from ``config.settings``).
"""

from __future__ import annotations

import pytest

from gnomad_meb.lib.checkpoint.artifact import Artifact
from gnomad_meb.lib.checkpoint.metadata import CheckpointMetadata
from gnomad_meb.lib.checkpoint.status import (
    ArtifactResolution,
    CheckpointEvaluation,
    CheckpointStatus,
    StalenessPolicy,
    StalenessReason,
    aggregate_status,
    evaluate_resolutions,
    verdict,
)
from gnomad_meb.lib.checkpoint.types import CHECKPOINT_TYPE

POLICIES = list(StalenessPolicy)


def _md(**overrides) -> CheckpointMetadata:
    base = {
        "path": "/ckpt/a.ht",
        "exists": True,
        "source_md5": "src",
        "git_revision": "git",
        "version": "ver",
    }
    return CheckpointMetadata(**{**base, **overrides})


def _art() -> Artifact:
    return Artifact(prefix="a", kind=CHECKPOINT_TYPE.HAIL_TABLE)


def _resolution(status: CheckpointStatus, **kwargs) -> ArtifactResolution:
    return ArtifactResolution(
        artifact=_art(),
        path=kwargs.pop("path", "/ckpt/a.ht"),
        metadata=kwargs.pop("metadata", None),
        status=status,
        reasons=kwargs.pop("reasons", ()),
    )


# --- the verdict policy table -------------------------------------------------


@pytest.mark.parametrize("policy", POLICIES)
def test_missing_or_incomplete_artifact_is_absent(policy: StalenessPolicy) -> None:
    status, reasons = verdict(
        _md(), stored=False, policy=policy, invalidated_by_lineage=False
    )
    assert status is CheckpointStatus.ABSENT
    assert reasons == ()


@pytest.mark.parametrize("policy", POLICIES)
def test_present_artifact_without_sidecar_is_unknown(policy: StalenessPolicy) -> None:
    status, reasons = verdict(
        None, stored=True, policy=policy, invalidated_by_lineage=False
    )
    assert status is CheckpointStatus.UNKNOWN
    assert reasons == ()


@pytest.mark.parametrize(
    "policy,expected_status,expected_reason",
    [
        (StalenessPolicy.COMPLETENESS, CheckpointStatus.PRESENT, None),
        (
            StalenessPolicy.SOURCE,
            CheckpointStatus.STALE,
            StalenessReason.SOURCE_CHANGED,
        ),
        (
            StalenessPolicy.SOURCE_OR_GIT,
            CheckpointStatus.STALE,
            StalenessReason.SOURCE_CHANGED,
        ),
    ],
)
def test_source_md5_mismatch(policy, expected_status, expected_reason) -> None:
    stored = _md(source_md5="old")
    status, reasons = verdict(
        stored,
        stored=True,
        policy=policy,
        invalidated_by_lineage=False,
        current_source_md5="new",
    )
    assert status is expected_status
    if expected_reason is None:
        assert reasons == ()
    else:
        assert reasons == (expected_reason,)


@pytest.mark.parametrize(
    "policy,expected_status,expected_reason",
    [
        (StalenessPolicy.COMPLETENESS, CheckpointStatus.PRESENT, None),
        (StalenessPolicy.SOURCE, CheckpointStatus.PRESENT, None),
        (
            StalenessPolicy.SOURCE_OR_GIT,
            CheckpointStatus.STALE,
            StalenessReason.GIT_CHANGED,
        ),
    ],
)
def test_git_revision_mismatch(policy, expected_status, expected_reason) -> None:
    stored = _md(git_revision="old")
    status, reasons = verdict(
        stored,
        stored=True,
        policy=policy,
        invalidated_by_lineage=False,
        current_source_md5="src",
        current_git_revision="new",
        current_version="ver",
    )
    assert status is expected_status
    assert reasons == (() if expected_reason is None else (expected_reason,))


@pytest.mark.parametrize(
    "policy,expected_status,expected_reason",
    [
        (StalenessPolicy.COMPLETENESS, CheckpointStatus.PRESENT, None),
        (StalenessPolicy.SOURCE, CheckpointStatus.PRESENT, None),
        (
            StalenessPolicy.SOURCE_OR_GIT,
            CheckpointStatus.STALE,
            StalenessReason.VERSION_CHANGED,
        ),
    ],
)
def test_version_mismatch(policy, expected_status, expected_reason) -> None:
    stored = _md(version="old")
    status, reasons = verdict(
        stored,
        stored=True,
        policy=policy,
        invalidated_by_lineage=False,
        current_source_md5="src",
        current_git_revision="git",
        current_version="new",
    )
    assert status is expected_status
    assert reasons == (() if expected_reason is None else (expected_reason,))


@pytest.mark.parametrize("policy", POLICIES)
def test_lineage_invalidation_forces_stale_on_a_present_artifact(
    policy: StalenessPolicy,
) -> None:
    status, reasons = verdict(
        _md(),
        stored=True,
        policy=policy,
        invalidated_by_lineage=True,
        current_source_md5="src",
        current_git_revision="git",
        current_version="ver",
    )
    assert status is CheckpointStatus.STALE
    assert reasons == (StalenessReason.LINEAGE_INVALID,)


@pytest.mark.parametrize("policy", POLICIES)
def test_lineage_never_resurrects_an_absent_artifact(policy: StalenessPolicy) -> None:
    status, reasons = verdict(
        _md(), stored=False, policy=policy, invalidated_by_lineage=True
    )
    assert status is CheckpointStatus.ABSENT
    assert reasons == ()


@pytest.mark.parametrize("policy", POLICIES)
def test_matching_provenance_is_present(policy: StalenessPolicy) -> None:
    status, reasons = verdict(
        _md(),
        stored=True,
        policy=policy,
        invalidated_by_lineage=False,
        current_source_md5="src",
        current_git_revision="git",
        current_version="ver",
    )
    assert status is CheckpointStatus.PRESENT
    assert reasons == ()


def test_source_and_git_can_accumulate_reasons() -> None:
    stored = _md(source_md5="old", git_revision="old", version="old")
    status, reasons = verdict(
        stored,
        stored=True,
        policy=StalenessPolicy.SOURCE_OR_GIT,
        invalidated_by_lineage=False,
        current_source_md5="new",
        current_git_revision="new",
        current_version="new",
    )
    assert status is CheckpointStatus.STALE
    assert reasons == (
        StalenessReason.SOURCE_CHANGED,
        StalenessReason.GIT_CHANGED,
        StalenessReason.VERSION_CHANGED,
    )


# --- aggregate precedence -----------------------------------------------------


def test_aggregate_precedence() -> None:
    assert aggregate_status([]) is CheckpointStatus.PRESENT
    assert (
        aggregate_status([_resolution(CheckpointStatus.PRESENT)])
        is CheckpointStatus.PRESENT
    )
    assert (
        aggregate_status(
            [
                _resolution(CheckpointStatus.PRESENT),
                _resolution(CheckpointStatus.UNKNOWN),
            ]
        )
        is CheckpointStatus.UNKNOWN
    )
    assert (
        aggregate_status(
            [_resolution(CheckpointStatus.UNKNOWN), _resolution(CheckpointStatus.STALE)]
        )
        is CheckpointStatus.STALE
    )
    assert (
        aggregate_status(
            [_resolution(CheckpointStatus.STALE), _resolution(CheckpointStatus.ABSENT)]
        )
        is CheckpointStatus.ABSENT
    )


# --- evaluate_resolutions -----------------------------------------------------


def test_empty_declarations_are_empty_and_never_reusable() -> None:
    evaluation = evaluate_resolutions(
        [], policy=StalenessPolicy.COMPLETENESS, lineage_invalidated=False
    )
    assert evaluation == CheckpointEvaluation(
        status=CheckpointStatus.PRESENT,
        reasons=(),
        resolutions=(),
        reusable=False,
        empty=True,
    )


def test_unknown_is_reusable_under_completeness_but_not_under_source() -> None:
    resolutions = [_resolution(CheckpointStatus.UNKNOWN)]
    soft = evaluate_resolutions(
        resolutions, policy=StalenessPolicy.COMPLETENESS, lineage_invalidated=False
    )
    strict = evaluate_resolutions(
        resolutions, policy=StalenessPolicy.SOURCE, lineage_invalidated=False
    )
    assert soft.reusable is True
    assert strict.reusable is False
    assert soft.status is CheckpointStatus.UNKNOWN


def test_a_present_artifact_is_reusable_under_every_policy() -> None:
    resolutions = [_resolution(CheckpointStatus.PRESENT)]
    for policy in POLICIES:
        evaluation = evaluate_resolutions(
            resolutions, policy=policy, lineage_invalidated=False
        )
        assert evaluation.reusable is True
        assert evaluation.empty is False
