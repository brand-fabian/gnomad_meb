"""Tests for the single artifact vocabulary in ``gnomad_meb.lib.checkpoint``.

These pin the pieces the pipeline is built on: :class:`Artifact` resolves the
``{checkpoint}/{prefix}.<ext>`` rule and delegates completeness to the kind's
serializer, ``CHECKPOINT_TYPE.FILE`` is a first-class kind, and the
``checkpoint`` decorator attaches its declaration while resolving paths at call
time (so a method decorated at class-definition time follows the run's
settings). See ``.scratch/pipeline-resources/issues/13-*``.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import pytest

import gnomad_meb.config as config_module
from gnomad_meb.config import Settings, set_config
from gnomad_meb.lib.checkpoint import (
    CHECKPOINT_EXTENSION,
    CHECKPOINT_TYPE,
    Artifact,
    CheckpointSpec,
    checkpoint,
    get_checkpoint_path,
)


@pytest.fixture(autouse=True)
def _clear_ambient_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop any ambient ``GMEB_`` variables and the cached global config."""
    for key in [k for k in os.environ if k.startswith("GMEB_")]:
        monkeypatch.delenv(key, raising=False)
    config_module._config = None
    config_module._config_path = None


def _cfg(**overrides) -> Settings:
    base = {"validate_paths": False, "project": "demo", "checkpoint": "/ckpt"}
    return Settings(**{**base, **overrides})


# --- Artifact.path: the one path rule ----------------------------------------


def test_artifact_path_matches_the_checkpoint_rule_for_every_kind() -> None:
    cfg = _cfg()
    expected = {
        CHECKPOINT_TYPE.VARIANT_DATASET: "/ckpt/demo/x.vds",
        CHECKPOINT_TYPE.MATRIX_TABLE: "/ckpt/demo/x.mt",
        CHECKPOINT_TYPE.HAIL_TABLE: "/ckpt/demo/x.ht",
        CHECKPOINT_TYPE.DATAFRAME: "/ckpt/demo/x.parquet",
        CHECKPOINT_TYPE.DICTIONARY: "/ckpt/demo/x.json.gz",
        CHECKPOINT_TYPE.PICKLE: "/ckpt/demo/x.pickle.gz",
        CHECKPOINT_TYPE.PLOTNINE: "/ckpt/demo/x",
        CHECKPOINT_TYPE.FILE: "/ckpt/demo/x",
    }

    for kind, path in expected.items():
        assert Artifact(prefix="demo/x", kind=kind).path(cfg) == path


def test_artifact_path_suffix_overrides_the_kind_extension() -> None:
    cfg = _cfg()

    assert (
        Artifact(prefix="demo/x", kind=CHECKPOINT_TYPE.FILE, suffix="onnx").path(cfg)
        == "/ckpt/demo/x.onnx"
    )


def test_artifact_path_formats_the_prefix_template() -> None:
    cfg = _cfg()

    assert (
        Artifact(prefix="{project}/x", kind=CHECKPOINT_TYPE.HAIL_TABLE).path(cfg)
        == "/ckpt/demo/x.ht"
    )


def test_file_kind_round_trips_through_get_checkpoint_path() -> None:
    set_config(_cfg())

    assert Artifact(prefix="demo/x", kind=CHECKPOINT_TYPE.FILE).path() == (
        get_checkpoint_path("demo/x", CHECKPOINT_TYPE.FILE)
    )
    assert get_checkpoint_path("demo/x", CHECKPOINT_TYPE.FILE) == "/ckpt/demo/x"
    assert CHECKPOINT_EXTENSION[CHECKPOINT_TYPE.FILE] is None


# --- Artifact.exists: completeness delegated to the serializer ---------------


def test_artifact_exists_is_false_for_an_incomplete_hail_directory(
    tmp_path: Path,
) -> None:
    cfg = _cfg(checkpoint=str(tmp_path))
    directory = tmp_path / "a.ht"
    directory.mkdir()
    artifact = Artifact(prefix="a", kind=CHECKPOINT_TYPE.HAIL_TABLE)

    assert not artifact.exists(cfg)

    (directory / "_SUCCESS").write_text("")
    assert not artifact.exists(cfg)

    (directory / "metadata.json.gz").write_text("{}")
    assert artifact.exists(cfg)


def test_artifact_exists_is_true_for_a_present_single_file(tmp_path: Path) -> None:
    cfg = _cfg(checkpoint=str(tmp_path))
    (tmp_path / "a.bgz").write_text("x")

    assert Artifact(prefix="a.bgz", kind=CHECKPOINT_TYPE.FILE).exists(cfg)
    assert not Artifact(prefix="nope.bgz", kind=CHECKPOINT_TYPE.FILE).exists(cfg)


# --- checkpoint(...) attaches a CheckpointSpec -------------------------------


def test_checkpoint_attaches_a_spec_matching_the_arguments() -> None:
    @checkpoint(
        ("a", "b"),
        (CHECKPOINT_TYPE.HAIL_TABLE, CHECKPOINT_TYPE.DICTIONARY),
    )
    def multi():
        return {"a": 1}, {"b": 2}

    spec = multi.spec

    assert isinstance(spec, CheckpointSpec)
    assert spec.artifacts == (
        Artifact(prefix="a", kind=CHECKPOINT_TYPE.HAIL_TABLE),
        Artifact(prefix="b", kind=CHECKPOINT_TYPE.DICTIONARY),
    )


def test_checkpoint_attaches_a_single_artifact() -> None:
    @checkpoint("p", CHECKPOINT_TYPE.FILE)
    def one():
        return "x"

    assert one.spec.artifacts == (
        Artifact(prefix="p", kind=CHECKPOINT_TYPE.FILE),
    )


# --- path resolution moved from decoration time to call time -----------------


def test_decorating_does_not_resolve_against_an_unset_config(
    tmp_path: Path,
) -> None:
    @checkpoint("{project}/out", CHECKPOINT_TYPE.DICTIONARY)
    def compute():
        return {"a": 1}

    # Decorating must not read the (unset) config or write anything.
    assert compute.spec.artifacts[0].prefix == "{project}/out"
    assert not (tmp_path / "demo").exists()

    # With no checkpoint root, the wrapper is an identity pass-through.
    assert compute() == {"a": 1}


def test_calling_the_wrapper_after_set_config_resolves_the_installed_settings(
    tmp_path: Path,
) -> None:
    @checkpoint("{project}/out", CHECKPOINT_TYPE.DICTIONARY)
    def compute():
        return {"a": 1}

    set_config(_cfg(checkpoint=str(tmp_path)))

    assert compute() == {"a": 1}
    assert (tmp_path / "demo" / "out.json.gz").is_file()


def test_checkpoint_accepts_a_source_less_partial(tmp_path: Path) -> None:
    set_config(_cfg(checkpoint=str(tmp_path)))

    def body(value=None):
        return value

    wrapped = checkpoint("{project}/partial", CHECKPOINT_TYPE.DICTIONARY)(
        functools.partial(body, value="bound")
    )

    # ``inspect.getsource`` cannot see a partial; the wrapper must fall back
    # rather than raise. The call still runs the plain body.
    assert wrapped() == "bound"
    assert (tmp_path / "demo" / "partial.json.gz").is_file()
