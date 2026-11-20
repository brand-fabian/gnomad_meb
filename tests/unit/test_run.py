"""Ticket 06: ``run`` replays a ``--describe``/``--dry-run`` manifest.

``gmeb run PATH`` loads the run envelope (JSON or YAML), reconstructs the graph
through :class:`~gnomad_meb.pipeline.factory.PipelineFactory`, and executes it.
These tests drive the command through the in-process click runner; ``hail_init``
is replaced with a no-op so the reconstruction and execution run without a JVM.
The envelope is produced by a small, hail-free test pipeline so a dry-run
manifest round-trips through ``run`` and executes the same graph.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import click
import pytest
from click.testing import CliRunner

from gnomad_meb.cmd import main
from gnomad_meb.cmd.run import build_runner, load_run_manifest
from gnomad_meb.cmd.task import PipelineStep, Task
from gnomad_meb.config import Settings
from gnomad_meb.lib.checkpoint import CHECKPOINT_TYPE, checkpoint
from gnomad_meb.pipeline import (
    FromStep,
    InputStep,
    Pipeline,
    PipelineRunner,
    Resource,
    ResourceRef,
    RunInput,
)
from gnomad_meb.pipeline.execution import PipelineMismatchError


class ReplayPipeline(Pipeline):
    """A hail-free pipeline whose steps echo their inputs.

    It exists so ``run`` can reconstruct a real graph by name through the
    factory and execute it without hail: ``first`` echoes its resource,
    ``second`` combines the two predecessors.
    """

    @classmethod
    def name(cls) -> str:
        return "replay"

    @property
    def resources(self) -> tuple[Resource, ...]:
        return (
            Resource(name="res", path="/data/res.ht", loader=lambda p: f"loaded:{p}"),
        )

    def get_input(self) -> tuple[InputStep, ...]:
        return (
            InputStep(name="input", task=self.task, group=type(self).name(), value=10),
        )

    @checkpoint("{project}/first", CHECKPOINT_TYPE.FILE)
    def first(self, res=ResourceRef("res")):  # noqa: B008
        return res

    @checkpoint("{project}/second", CHECKPOINT_TYPE.FILE)
    def second(self, value=FromStep("first"), extra=RunInput("input")):  # noqa: B008
        return f"{value}+{extra}"


class FailingPipeline(Pipeline):
    """A pipeline whose only work step raises, so ``run`` must exit non-zero."""

    @classmethod
    def name(cls) -> str:
        return "failing"

    @property
    def resources(self) -> tuple[Resource, ...]:
        return ()

    def get_input(self) -> tuple[InputStep, ...]:
        return ()

    @checkpoint("{project}/boom", CHECKPOINT_TYPE.FILE)
    def boom(self):
        raise RuntimeError("boom")


def _settings(*, project: str = "demo") -> Settings:
    return Settings(validate_paths=False, project=project)


def _task(*, settings: Settings | None = None) -> Task:
    return Task(settings=settings or _settings(), command=PipelineStep(name="test"))


def _write_config(tmp_path: Path, *, project: str) -> Path:
    path = tmp_path / "gnomad_meb.toml"
    path.write_text(
        dedent(f"""\
        project = "{project}"
        validate_paths = false
    """)
    )
    return path


def _write_manifest(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))
    return path


def _noop_hail_init(monkeypatch: pytest.MonkeyPatch) -> None:
    import gnomad_meb.hail as hail_module

    monkeypatch.setattr(hail_module, "hail_init", lambda: None)


def _dry_run_manifest(task: Task, pipeline: type[Pipeline] = ReplayPipeline) -> dict:
    """A dry-run envelope for ``pipeline``, built by the runner."""
    runner = PipelineRunner(task, [pipeline(task)])
    return runner.manifest()


def test_dry_run_manifest_round_trips_through_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _noop_hail_init(monkeypatch)
    manifest = _dry_run_manifest(_task())
    path = _write_manifest(tmp_path, manifest)

    loaded = load_run_manifest(path)
    runner = build_runner(loaded)
    runner.run()

    # The rebuilt graph executed exactly the steps the dry-run described, with
    # the same dependency wiring and results.
    state = runner.execution_state
    assert set(state) == {
        "replay.input",
        "replay.first",
        "replay.second",
        "resources.load-res",
    }
    assert state["replay.first"].result == "loaded:/data/res.ht"
    assert state["replay.second"].result == "loaded:/data/res.ht+10"


def test_run_cli_accepts_a_dry_run_manifest_without_a_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _noop_hail_init(monkeypatch)
    manifest = _dry_run_manifest(_task())
    path = _write_manifest(tmp_path, manifest)

    result = CliRunner().invoke(main, ["run", str(path)])

    assert result.exit_code == 0, result.output


def _describe_manifest(task: Task) -> dict:
    """A ``--describe`` envelope for :class:`ReplayPipeline` (no steps)."""
    from gnomad_meb.cmd.output import manifest_payload

    return manifest_payload("replay", task)


def test_config_override_replaces_the_manifest_settings(tmp_path: Path) -> None:
    manifest = _dry_run_manifest(_task(settings=_settings(project="manifest-proj")))
    path = _write_manifest(tmp_path, manifest)
    config = _write_config(tmp_path, project="override-proj")

    loaded = load_run_manifest(path, config=config)
    runner = build_runner(loaded)

    # The override survives the runner's own set_config(self.task.settings): the
    # step's artifact resolves under the override project, not the manifest's.
    assert runner.task.settings.project == "override-proj"
    assert runner.settings.project == "override-proj"
    first = runner._step_registry["replay.first"]
    path = first.artifacts[0].path(runner.settings)
    assert "override-proj/first" in path
    assert "manifest-proj" not in path


def test_describe_manifest_makes_the_whole_graph_eligible(tmp_path: Path) -> None:
    manifest = _describe_manifest(_task())
    assert manifest["steps"] is None
    path = _write_manifest(tmp_path, manifest)

    loaded = load_run_manifest(path)
    runner = build_runner(loaded)

    # No steps list: every registered step is eligible (and nothing was
    # narrowed away).
    assert runner._selected_ids == frozenset(
        {
            "replay.input",
            "replay.first",
            "replay.second",
            "resources.load-res",
        }
    )


def test_run_rejects_an_explicit_empty_steps_list(tmp_path: Path) -> None:
    manifest = _dry_run_manifest(_task())
    manifest["steps"] = []
    path = _write_manifest(tmp_path, manifest)

    with pytest.raises(click.ClickException, match="empty"):
        load_run_manifest(path)


def test_run_rejects_a_non_runnable_manifest(tmp_path: Path) -> None:
    from gnomad_meb.cmd.output import manifest_payload

    path = _write_manifest(tmp_path, manifest_payload(None, _task()))

    with pytest.raises(click.ClickException, match="not runnable"):
        load_run_manifest(path)


def test_manifest_naming_an_unknown_step_raises_before_hail(tmp_path: Path) -> None:
    manifest = _dry_run_manifest(_task())
    bogus = dict(manifest["steps"][0])
    bogus["name"] = "does_not_exist"
    bogus["group"] = None
    manifest["steps"] = [*manifest["steps"], bogus]
    path = _write_manifest(tmp_path, manifest)

    loaded = load_run_manifest(path)
    with pytest.raises(PipelineMismatchError, match="does_not_exist"):
        build_runner(loaded)


def test_manifest_naming_a_missing_resource_edge_raises_before_hail(
    tmp_path: Path,
) -> None:
    manifest = _dry_run_manifest(_task())
    for step in manifest["steps"]:
        if step["name"] == "first":
            step["resources"] = {"res": "bogus-resource"}
    path = _write_manifest(tmp_path, manifest)

    loaded = load_run_manifest(path)
    with pytest.raises(PipelineMismatchError, match="bogus-resource"):
        build_runner(loaded)


@pytest.mark.parametrize(
    ("manifest_overwrite", "cli_overwrite", "expected"),
    [
        (False, False, False),
        (True, False, True),
        (False, True, True),
        (True, True, True),
    ],
)
def test_overwrite_is_ored_with_the_manifest(
    tmp_path: Path,
    manifest_overwrite: bool,
    cli_overwrite: bool,
    expected: bool,
) -> None:
    manifest = _dry_run_manifest(_task())
    manifest["overwrite"] = manifest_overwrite
    path = _write_manifest(tmp_path, manifest)

    loaded = load_run_manifest(path)
    runner = build_runner(loaded, overwrite=cli_overwrite)

    assert runner.overwrite is expected


def test_a_failing_step_exits_non_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _noop_hail_init(monkeypatch)
    path = _write_manifest(
        tmp_path, _dry_run_manifest(_task(), pipeline=FailingPipeline)
    )

    result = CliRunner().invoke(main, ["run", str(path)])

    assert result.exit_code != 0
    assert "errored steps" in result.output.lower()


def test_run_rejects_a_global_describe_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _noop_hail_init(monkeypatch)
    path = _write_manifest(tmp_path, _dry_run_manifest(_task()))

    result = CliRunner().invoke(main, ["--describe", "run", str(path)])

    assert result.exit_code != 0
    assert "--describe" in result.output or "existing description" in result.output


def test_run_selects_a_step_and_pulls_in_its_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _noop_hail_init(monkeypatch)
    path = _write_manifest(tmp_path, _describe_manifest(_task()))

    result = CliRunner().invoke(main, ["run", "--select", "replay.second", str(path)])

    assert result.exit_code == 0, result.output


def test_run_cli_config_overrides_the_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _noop_hail_init(monkeypatch)
    manifest = _dry_run_manifest(_task(settings=_settings(project="manifest-proj")))
    path = _write_manifest(tmp_path, manifest)
    config = _write_config(tmp_path, project="override-proj")

    result = CliRunner().invoke(main, ["--config", str(config), "run", str(path)])

    assert result.exit_code == 0, result.output


def test_run_select_is_resolved_before_hail(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path, _describe_manifest(_task()))

    loaded = load_run_manifest(path)
    runner = build_runner(loaded, select=("replay.second",))

    # second plus its dependency closure; input and the resource loader are
    # pulled in, and nothing else survives.
    assert runner._selected_ids == frozenset(
        {
            "replay.input",
            "replay.first",
            "replay.second",
            "resources.load-res",
        }
    )
