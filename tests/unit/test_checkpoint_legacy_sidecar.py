"""Integration sweep: a pre-change sidecar reads back as PRESENT (ticket 11).

The sidecar format is frozen: a sidecar written by the pre-split code (the eight
historical keys, no ``schema_version``) must still read as ``PRESENT`` under the
default policy, and an artifact with no sidecar must read ``UNKNOWN`` (reusable
under ``COMPLETENESS``).
"""

from __future__ import annotations

import json
from pathlib import Path

from gnomad_meb.config import Settings, set_config
from gnomad_meb.lib.checkpoint import (
    CHECKPOINT_TYPE,
    CheckpointedCallable,
    CheckpointStatus,
    StalenessPolicy,
    checkpoint,
)

#: A sidecar exactly as the pre-change code wrote it: eight keys, no version.
LEGACY_SIDECAR = {
    "path": "{path}",
    "exists": True,
    "mtime": "2024-01-01 00:00:00",
    "data_md5": "deadbeef",
    "source_md5": "{source_md5}",
    "host": "legacy-host",
    "version": "0.0.0-legacy",
    "git_revision": "deadbeef",
}


def _cfg(tmp_path: Path) -> Settings:
    return Settings(validate_paths=False, project="demo", checkpoint=str(tmp_path))


def _dict_fn() -> CheckpointedCallable:
    @checkpoint("demo/x", CHECKPOINT_TYPE.DICTIONARY)
    def fn():
        return {"a": 1}

    return fn


def test_a_pre_change_sidecar_reads_present(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    set_config(cfg)
    fn = _dict_fn()
    path = Path(fn.spec.artifacts[0].path(cfg))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")

    from hashlib import md5

    source_md5 = md5(fn.source().encode()).hexdigest()  # noqa: S324
    sidecar = {**LEGACY_SIDECAR}
    sidecar["path"] = str(path)
    sidecar["source_md5"] = source_md5
    Path(f"{path}.gnomad_meb.md.json").write_text(json.dumps(sidecar))

    assert fn.status(cfg, policy=StalenessPolicy.COMPLETENESS) is (
        CheckpointStatus.PRESENT
    )


def test_an_artifact_without_a_sidecar_reads_unknown_and_is_reusable(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    set_config(cfg)
    fn = _dict_fn()
    path = Path(fn.spec.artifacts[0].path(cfg))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")

    assert fn.status(cfg) is CheckpointStatus.UNKNOWN
    assert fn.evaluate(cfg).reusable is True  # COMPLETENESS treats UNKNOWN as reusable
    assert fn.evaluate(cfg, policy=StalenessPolicy.SOURCE).reusable is False
