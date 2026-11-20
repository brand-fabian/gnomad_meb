"""Tests for the versioned ``CheckpointMetadata`` sidecar model.

The on-disk contract is unchanged: the same eight keys, absent keys default to
current values, unknown keys are ignored, and new sidecars additionally carry a
``schema_version``. These pin the compatibility the split's callers rely on.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import gnomad_meb.config as config_module
from gnomad_meb.lib.checkpoint.metadata import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointMetadata,
    UnsupportedSchemaVersion,
)

OLD_KEYS = (
    "path",
    "exists",
    "mtime",
    "data_md5",
    "source_md5",
    "host",
    "version",
    "git_revision",
)


@pytest.fixture(autouse=True)
def _clear_ambient_config(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in [k for k in os.environ if k.startswith("GMEB_")]:
        monkeypatch.delenv(key, raising=False)
    config_module._config = None
    config_module._config_path = None


def test_old_sidecar_dict_without_schema_version_parses() -> None:
    old = {
        "path": "/ckpt/a.ht",
        "exists": True,
        "mtime": "2024-01-01 00:00:00",
        "data_md5": "d",
        "source_md5": "s",
        "host": "hosty",
        "version": "1.2.3",
        "git_revision": "abc123",
    }

    md = CheckpointMetadata.from_dict(old)

    assert md.schema_version == CHECKPOINT_SCHEMA_VERSION
    assert md.path == "/ckpt/a.ht"
    assert md.host == "hosty"
    assert md.version == "1.2.3"
    assert md.git_revision == "abc123"


def test_absent_manual_fields_default_to_current_values() -> None:
    md = CheckpointMetadata.from_dict({"path": "/ckpt/a.ht", "exists": False})

    assert md.schema_version == CHECKPOINT_SCHEMA_VERSION
    assert md.mtime is None
    assert md.data_md5 is None
    assert md.source_md5 is None
    # host/version/git_revision default to the current run's values.
    assert md.host
    assert md.version
    assert md.git_revision is not None


def test_to_dict_has_schema_version_and_all_historical_keys(tmp_path: Path) -> None:
    md = CheckpointMetadata(path="/ckpt/a.ht", exists=True)

    d = md.to_dict()

    assert d["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    for key in OLD_KEYS:
        assert key in d, key


def test_new_sidecar_round_trips(tmp_path: Path) -> None:
    md = CheckpointMetadata(
        path="/ckpt/a.ht",
        exists=True,
        mtime="2024-01-01 00:00:00",
        data_md5="d",
        source_md5="s",
    )
    dest = tmp_path / "a.ht.gnomad_meb.md.json"

    md.to_json_file(dest)
    back = CheckpointMetadata.from_json_file(dest)

    assert back == md
    assert back.schema_version == CHECKPOINT_SCHEMA_VERSION


def test_unknown_extra_key_is_ignored(tmp_path: Path) -> None:
    dest = tmp_path / "a.md.json"
    dest.write_text(
        json.dumps(
            {
                "path": "/ckpt/a.ht",
                "exists": True,
                "host": "h",
                "version": "v",
                "git_revision": "g",
                "future_field": "ignored",
            }
        )
    )

    md = CheckpointMetadata.from_json_file(dest)

    assert md.path == "/ckpt/a.ht"
    assert not hasattr(md, "future_field")


def test_corrupt_json_fails_loudly(tmp_path: Path) -> None:
    dest = tmp_path / "bad.md.json"
    dest.write_text("{not json")

    with pytest.raises(ValueError) as excinfo:
        CheckpointMetadata.from_json_file(dest)
    # The message must name the file so a caller can find it.
    assert "bad.md.json" in str(excinfo.value)


def test_unsupported_future_schema_version_fails_loudly(tmp_path: Path) -> None:
    dest = tmp_path / "future.md.json"
    dest.write_text(
        json.dumps(
            {
                "path": "/ckpt/a.ht",
                "exists": True,
                "schema_version": CHECKPOINT_SCHEMA_VERSION + 99,
            }
        )
    )

    with pytest.raises(UnsupportedSchemaVersion) as excinfo:
        CheckpointMetadata.from_json_file(dest)
    assert "schema_version" in str(excinfo.value).lower()


def test_from_dict_with_unsupported_schema_version_fails_loudly() -> None:
    with pytest.raises(UnsupportedSchemaVersion):
        CheckpointMetadata.from_dict(
            {"path": "/ckpt/a.ht", "exists": True, "schema_version": 9999}
        )


def test_path_mismatch_is_reported_when_checked() -> None:
    md = CheckpointMetadata.from_dict(
        {"path": "/ckpt/other.ht", "exists": True, "host": "h", "version": "v",
         "git_revision": "g"}
    )

    # The model itself does not know the artifact path; the caller compares it.
    assert not md.matches_path("/ckpt/a.ht")
    assert md.matches_path("/ckpt/other.ht")


def test_s3_shaped_path_goes_through_the_s3fs(monkeypatch: pytest.MonkeyPatch) -> None:
    written: dict[str, bytes] = {}

    class _FS:
        def open(self, path, mode):
            import io

            if "b" in mode:
                buf = io.BytesIO()

                class _Ctx:
                    def __enter__(self_inner):
                        return buf

                    def __exit__(self_inner, *exc):
                        written[path] = buf.getvalue()
                        return False

                return _Ctx()
            raise AssertionError("read not expected")

    class _S3FS:
        fs = _FS()

    cfg = type("C", (), {"s3fs": _S3FS()})()

    monkeypatch.setattr(config_module, "get_config", lambda *a, **k: cfg)
    import gnomad_meb.lib.checkpoint.metadata as metadata_module

    monkeypatch.setattr(metadata_module, "get_config", lambda *a, **k: cfg)

    md = CheckpointMetadata(path="s3://bucket/a.ht", exists=True)
    md.to_json_file("s3://bucket/a.ht.gnomad_meb.md.json")

    payload = json.loads(written["s3://bucket/a.ht.gnomad_meb.md.json"])
    assert payload["path"] == "s3://bucket/a.ht"
    assert payload["schema_version"] == CHECKPOINT_SCHEMA_VERSION
