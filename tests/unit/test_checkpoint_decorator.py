"""Tests for the ``CheckpointedCallable`` class (ticket 04).

The decorator returns a class that is introspectable without invoking the step:
it carries the declaration, the wrapped body and its keywords, the resolved
metadata and the status query surface the runner uses. ``__call__`` reproduces
the historical cache behaviour, plus a per-instance stale flag honoured on top
of the ambient overwrite.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import pytest

import gnomad_meb.config as config_module
from gnomad_meb.config import Settings, set_config
from gnomad_meb.lib.checkpoint import (
    CHECKPOINT_TYPE,
    CheckpointedCallable,
    CheckpointStatus,
    StalenessPolicy,
    StalenessReason,
    checkpoint,
    declares_step,
)


@pytest.fixture(autouse=True)
def _clear_ambient_config(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in [k for k in os.environ if k.startswith("GMEB_")]:
        monkeypatch.delenv(key, raising=False)
    config_module._config = None
    config_module._config_path = None


def _cfg(tmp_path: Path, **overrides) -> Settings:
    base = {
        "validate_paths": False,
        "project": "demo",
        "checkpoint": str(tmp_path),
    }
    return Settings(**{**base, **overrides})


def _dict_callable(prefix: str = "demo/x") -> CheckpointedCallable:
    @checkpoint(prefix, CHECKPOINT_TYPE.DICTIONARY)
    def fn():
        return {"a": 1}

    return fn


# --- the class contract -------------------------------------------------------


def test_decorator_returns_a_checkpointed_callable() -> None:
    fn = _dict_callable()
    assert isinstance(fn, CheckpointedCallable)
    assert fn.spec.artifacts[0].prefix == "demo/x"
    assert fn.__name__ == "fn"
    assert callable(fn)


def test_metadata_and_status_do_not_invoke_the_body(tmp_path: Path) -> None:
    calls: list[int] = []

    @checkpoint("demo/x", CHECKPOINT_TYPE.DICTIONARY)
    def fn():
        calls.append(1)
        return {"a": 1}

    cfg = _cfg(tmp_path)
    assert fn.status(cfg) is CheckpointStatus.ABSENT
    assert fn.metadata(cfg) == (None,)
    assert fn.evaluate(cfg).empty is False
    assert calls == []


def test_present_after_write_and_readable_without_invoking(tmp_path: Path) -> None:
    set_config(_cfg(tmp_path))
    calls: list[int] = []

    @checkpoint("demo/x", CHECKPOINT_TYPE.DICTIONARY)
    def fn():
        calls.append(1)
        return {"a": 1}

    assert fn() == {"a": 1}
    assert calls == [1]

    assert fn.status() is CheckpointStatus.PRESENT
    md, = fn.metadata()
    assert md is not None and md.path.endswith("demo/x.json.gz")
    assert fn.read() == {"a": 1}
    assert calls == [1]  # read/status never invoke


def test_source_change_is_stale_under_source_policy(tmp_path: Path) -> None:
    set_config(_cfg(tmp_path))

    @checkpoint("demo/x", CHECKPOINT_TYPE.DICTIONARY)
    def fn():
        return {"a": 1}

    fn()
    sidecar = tmp_path / "demo" / "x.json.gz.gnomad_meb.md.json"
    import json

    data = json.loads(sidecar.read_text())
    data["source_md5"] = "not-the-current-source"
    sidecar.write_text(json.dumps(data))

    assert fn.status(policy=StalenessPolicy.COMPLETENESS) is CheckpointStatus.PRESENT
    assert fn.status(policy=StalenessPolicy.SOURCE) is CheckpointStatus.STALE
    assert fn.evaluate(policy=StalenessPolicy.SOURCE).reasons == (
        StalenessReason.SOURCE_CHANGED,
    )


def test_mark_stale_re_executes_instead_of_reading_cache(tmp_path: Path) -> None:
    set_config(_cfg(tmp_path))
    calls: list[int] = []

    @checkpoint("demo/x", CHECKPOINT_TYPE.DICTIONARY)
    def fn():
        calls.append(1)
        return {"run": len(calls)}

    first = fn()
    assert first == {"run": 1}

    fn.mark_stale((StalenessReason.LINEAGE_INVALID,))
    second = fn()
    assert second == {"run": 2}
    assert calls == [1, 1]  # body entered twice

    fn.mark_reusable()
    assert fn() == {"run": 2}
    assert calls == [1, 1]


def test_overwrite_context_manager_still_forces_recompute(tmp_path: Path) -> None:
    from gnomad_meb.lib.checkpoint import checkpoint_overwrite

    set_config(_cfg(tmp_path))
    calls: list[int] = []

    @checkpoint("demo/x", CHECKPOINT_TYPE.DICTIONARY)
    def fn():
        calls.append(1)
        return {"run": len(calls)}

    assert fn() == {"run": 1}
    with checkpoint_overwrite(True):
        assert fn() == {"run": 2}


def test_identity_when_checkpoint_disabled(tmp_path: Path) -> None:
    cfg = Settings(validate_paths=False, project="demo", checkpoint=None)
    set_config(cfg)

    @checkpoint("demo/x", CHECKPOINT_TYPE.DICTIONARY)
    def fn():
        return {"a": 1}

    assert fn() == {"a": 1}


def test_empty_declaration_is_a_pass_through() -> None:
    calls: list[int] = []

    @declares_step
    def fn():
        calls.append(1)
        return "done"

    assert isinstance(fn, CheckpointedCallable)
    assert fn.spec.artifacts == ()
    assert fn.evaluate().empty is True
    assert fn.evaluate().reusable is False
    assert fn() == "done"
    assert calls == [1]


# --- source stability ---------------------------------------------------------


def test_source_is_stable_across_equivalent_partials(tmp_path: Path) -> None:
    def body(value=None):
        return value

    p1 = functools.partial(body, value="a")
    p2 = functools.partial(body, value="a")

    fn1 = checkpoint("demo/x", CHECKPOINT_TYPE.DICTIONARY)(p1)
    fn2 = checkpoint("demo/x", CHECKPOINT_TYPE.DICTIONARY)(p2)

    assert fn1.source() == fn2.source()


def test_source_changes_when_the_body_changes(tmp_path: Path) -> None:
    def body_a():
        return "a"

    def body_b():
        return "b"

    fn_a = checkpoint("demo/x", CHECKPOINT_TYPE.DICTIONARY)(body_a)
    fn_b = checkpoint("demo/x", CHECKPOINT_TYPE.DICTIONARY)(body_b)

    assert fn_a.source() != fn_b.source()


def test_source_of_a_partial_hashes_the_underlying_body() -> None:
    def body(value=None):
        return value

    partial = functools.partial(body, value="a")
    fn = checkpoint("demo/x", CHECKPOINT_TYPE.DICTIONARY)(partial)
    plain = checkpoint("demo/x", CHECKPOINT_TYPE.DICTIONARY)(body)

    # The partial's source follows .func recursively: same body, same source.
    assert fn.source() == plain.source()
    assert "0x" not in fn.source()  # no repr() fallback leaking addresses


# --- keywords / wiring attributes --------------------------------------------


def test_keywords_exposes_the_partials_bound_keywords() -> None:
    def body(**kwargs):
        return kwargs

    partial = functools.partial(body, mt="edge")
    fn = checkpoint("demo/x", CHECKPOINT_TYPE.DICTIONARY)(partial)

    assert fn.keywords.get("mt") == "edge"


def test_a_plain_function_has_empty_keywords() -> None:
    fn = _dict_callable()
    assert dict(fn.keywords) == {}
