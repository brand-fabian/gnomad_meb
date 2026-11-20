"""Tests for the run engine (:class:`PipelineRunner`)."""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from gnomad_meb.cmd.task import PipelineStep, Task
from gnomad_meb.config import Settings, set_config
from gnomad_meb.lib.checkpoint import (
    CHECKPOINT_TYPE,
    Artifact,
    CheckpointStatus,
    StalenessPolicy,
    StalenessReason,
    checkpoint,
)
from gnomad_meb.pipeline import (
    FromStep,
    InputStep,
    Pipeline,
    PipelineRunner,
    RecordingObserver,
    Resource,
    ResourceRef,
    RunInput,
    RunObserver,
    Step,
    StepResult,
    StepStatus,
    TerminalObserver,
    run_pipeline,
)
from gnomad_meb.pipeline.runtime import LocalRuntime


def _settings(*, checkpoint: str | None = None, **overrides) -> Settings:
    return Settings(
        validate_paths=False, project="demo", checkpoint=checkpoint, **overrides
    )


def _task(*, settings: Settings | None = None) -> Task:
    return Task(settings=settings or _settings(), command=PipelineStep(name="test"))


class Chain(Pipeline):
    @classmethod
    def name(cls) -> str:
        return "chain"

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


def test_a_dropped_only_run_has_not_started_but_is_finished() -> None:
    task = _task()
    runner = PipelineRunner(task, [Chain(task)])
    for record in runner.execution_state.values():
        record.status = StepStatus.DROPPED

    assert runner.has_started() is False
    assert runner.has_finished() is True
    assert runner.has_error() is False


def test_runner_executes_steps_in_dependency_order() -> None:
    task = _task()
    pipeline = Chain(task)
    runner = PipelineRunner(task, [pipeline])
    runner.run()

    state = runner.execution_state
    assert state["chain.first"].result == "loaded:/data/res.ht"
    assert state["chain.second"].result == "loaded:/data/res.ht+10"
    assert state["chain.second"].status == StepStatus.FINISHED


class MultiArtifact(Pipeline):
    """A tuple producer and consumers that select one of its outputs."""

    @classmethod
    def name(cls) -> str:
        return "multi_artifact"

    @property
    def resources(self) -> tuple[Resource, ...]:
        return ()

    def get_input(self) -> tuple[InputStep, ...]:
        return ()

    @checkpoint("{project}/multi_artifact/make", CHECKPOINT_TYPE.FILE)
    def make(self):
        return ("mt", "variant_info", "sample_info")

    @checkpoint("{project}/multi_artifact/first", CHECKPOINT_TYPE.FILE)
    def first(self, mt=FromStep("make")):  # noqa: B008
        return mt

    @checkpoint("{project}/multi_artifact/second", CHECKPOINT_TYPE.FILE)
    def second(self, mt=FromStep("make", output=1)):  # noqa: B008
        return mt


def test_fromstep_output_selects_the_indexed_element() -> None:
    task = _task()
    runner = PipelineRunner(task, [MultiArtifact(task)])
    runner.run()

    state = runner.execution_state
    assert state["multi_artifact.make"].result == ("mt", "variant_info", "sample_info")
    assert state["multi_artifact.first"].result == "mt"
    assert state["multi_artifact.second"].result == "variant_info"


def test_fromstep_output_default_leaves_a_single_output_unchanged() -> None:
    class SingleArtifact(Pipeline):
        @classmethod
        def name(cls) -> str:
            return "single_artifact"

        @property
        def resources(self) -> tuple[Resource, ...]:
            return ()

        def get_input(self) -> tuple[InputStep, ...]:
            return ()

        @checkpoint("{project}/single_artifact/make", CHECKPOINT_TYPE.FILE)
        def make(self):
            return "only"

        @checkpoint("{project}/single_artifact/use", CHECKPOINT_TYPE.FILE)
        def use(self, mt=FromStep("make")):  # noqa: B008
            return mt

    task = _task()
    runner = PipelineRunner(task, [SingleArtifact(task)])
    runner.run()

    assert runner.execution_state["single_artifact.use"].result == "only"


def test_fromstep_default_leaves_a_single_list_output_unchanged() -> None:
    class ListArtifact(Pipeline):
        @classmethod
        def name(cls) -> str:
            return "list_artifact"

        @property
        def resources(self) -> tuple[Resource, ...]:
            return ()

        def get_input(self) -> tuple[InputStep, ...]:
            return ()

        @checkpoint("{project}/list_artifact/make", CHECKPOINT_TYPE.FILE)
        def make(self):
            return ["only"]

        @checkpoint("{project}/list_artifact/use", CHECKPOINT_TYPE.FILE)
        def use(self, mt=FromStep("make")):  # noqa: B008
            return mt

    task = _task()
    runner = PipelineRunner(task, [ListArtifact(task)])
    runner.run()

    assert runner.execution_state["list_artifact.use"].result == ["only"]


def test_runner_records_an_error_and_continues() -> None:
    class Boom(Pipeline):
        @classmethod
        def name(cls) -> str:
            return "boom"

        @property
        def resources(self) -> tuple[Resource, ...]:
            return ()

        def get_input(self) -> tuple[InputStep, ...]:
            return ()

        @checkpoint("{project}/a", CHECKPOINT_TYPE.FILE)
        def a(self):
            raise RuntimeError("kaboom")

        @checkpoint("{project}/b", CHECKPOINT_TYPE.FILE)
        def b(self):
            return "ok"

    task = _task()
    pipeline = Boom(task)
    runner = PipelineRunner(task, [pipeline])
    runner.run()

    state = runner.execution_state
    assert state["boom.a"].status == StepStatus.ERROR
    assert "kaboom" in state["boom.a"].msg
    assert state["boom.b"].result == "ok"


def test_should_skip_is_false_without_artifacts(tmp_path: Path) -> None:
    settings = _settings(checkpoint=str(tmp_path))
    set_config(settings)
    step = Step(name="a", task=_task(), artifacts=(), run=lambda: None)

    assert PipelineRunner(_task(settings=settings), ()).should_skip(step) is False


def test_should_skip_honours_artifact_existence_and_overwrite(tmp_path: Path) -> None:
    settings = _settings(checkpoint=str(tmp_path))
    set_config(settings)
    artifact = Artifact(prefix="{project}/existing", kind=CHECKPOINT_TYPE.FILE)
    path = Path(artifact.path(settings))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")

    step = Step(name="a", task=_task(), artifacts=(artifact,), run=lambda: None)

    assert PipelineRunner(_task(settings=settings), ()).should_skip(step) is True
    assert (
        PipelineRunner(_task(settings=settings), (), overwrite=True).should_skip(step)
        is False
    )


def test_an_input_step_prefers_a_ran_source() -> None:
    class Producer(Pipeline):
        @classmethod
        def name(cls) -> str:
            return "producer"

        @property
        def resources(self) -> tuple[Resource, ...]:
            return ()

        def get_input(self) -> tuple[InputStep, ...]:
            return ()

        @checkpoint("{project}/vds", CHECKPOINT_TYPE.FILE)
        def make(self):
            return "the-vds"

    class Consumer(Pipeline):
        @classmethod
        def name(cls) -> str:
            return "consumer"

        @property
        def resources(self) -> tuple[Resource, ...]:
            return ()

        def get_input(self) -> tuple[InputStep, ...]:
            return (
                InputStep(
                    name="vds",
                    task=self.task,
                    group=type(self).name(),
                    source="producer.make",
                    value="bound-path",
                ),
            )

        @checkpoint("{project}/use", CHECKPOINT_TYPE.FILE)
        def use(self, vds=RunInput("vds")):  # noqa: B008
            return vds

    task = _task()
    runner = PipelineRunner(task, [Producer(task), Consumer(task)])
    runner.run()

    assert runner.execution_state["consumer.vds"].result == "the-vds"


def test_an_input_step_loads_its_bound_value_without_a_source() -> None:
    class Consumer(Pipeline):
        @classmethod
        def name(cls) -> str:
            return "consumer"

        @property
        def resources(self) -> tuple[Resource, ...]:
            return ()

        def get_input(self) -> tuple[InputStep, ...]:
            return (
                InputStep(
                    name="vds",
                    task=self.task,
                    group=type(self).name(),
                    source="producer.make",
                    value="bound-path",
                    run=lambda value: f"loaded:{value}",
                ),
            )

        @checkpoint("{project}/use", CHECKPOINT_TYPE.FILE)
        def use(self, vds=RunInput("vds")):  # noqa: B008
            return vds

    task = _task()
    runner = PipelineRunner(task, [Consumer(task)])
    runner.run()

    assert runner.execution_state["consumer.vds"].result == "loaded:bound-path"


# --- observer wiring (integration) ------------------------------------------


class _FanOutObserver:
    """A test double that forwards every observer call to each wrapped observer."""

    def __init__(self, *observers: RunObserver) -> None:
        self._observers = observers

    def run_started(
        self, steps: Sequence[Step], state: Mapping[str, StepResult]
    ) -> None:
        for observer in self._observers:
            observer.run_started(steps, state)

    def step_changed(self, step: Step, record: StepResult) -> None:
        for observer in self._observers:
            observer.step_changed(step, record)

    def run_finished(
        self, steps: Sequence[Step], state: Mapping[str, StepResult]
    ) -> None:
        for observer in self._observers:
            observer.run_finished(steps, state)


def test_chain_run_emits_the_plan_the_step_lines_and_the_summary(
    capturing_logger,
) -> None:
    task = _task()
    logger, stream = capturing_logger
    recorder = RecordingObserver()
    observer = _FanOutObserver(TerminalObserver(logger=logger), recorder)

    PipelineRunner(task, [Chain(task)], observer=observer).run()

    # One run_started, a started + terminal change per step in walk order, one
    # run_finished -- nothing else.
    assert recorder.calls[0] == ("run_started", None, None)
    assert recorder.calls[-1] == ("run_finished", None, None)
    assert recorder.calls[1:-1] == [
        ("step_changed", "chain.input", StepStatus.STARTED),
        ("step_changed", "chain.input", StepStatus.FINISHED),
        ("step_changed", "resources.load-res", StepStatus.STARTED),
        ("step_changed", "resources.load-res", StepStatus.FINISHED),
        ("step_changed", "chain.first", StepStatus.STARTED),
        ("step_changed", "chain.first", StepStatus.FINISHED),
        ("step_changed", "chain.second", StepStatus.STARTED),
        ("step_changed", "chain.second", StepStatus.FINISHED),
    ]

    text = stream.getvalue()
    lines = text.splitlines()

    # The plan comes first, in topological order, with each row's edges.
    assert lines[0] == "INFO pipeline run planned: steps = 4, groups = chain, resources"
    assert lines[1] == "dropped by selection: 0"
    assert lines[2] == "  1. chain.input         artifacts = 0  after = (none)  unknown"
    assert lines[4] == (
        "  3. chain.first         artifacts = 1  after = resources.load-res  unknown"
    )
    assert (
        lines[5]
        == "  4. chain.second        artifacts = 1  after = chain.first, chain.input"
        "  unknown"
    )

    # Every step's finish line carries its recorded duration.
    for step_id in ("chain.input", "resources.load-res", "chain.first", "chain.second"):
        assert (
            f"INFO finished pipeline step: step = {step_id}, "
            "status = finished, duration = " in text
        )

    # The summary comes last and tallies the finished steps.
    summary = text[text.index("INFO pipeline run finished:") :]
    assert summary.splitlines()[0].startswith(
        "INFO pipeline run finished: steps = 4, finished = 4"
    )


def test_boom_run_emits_the_failure_line_and_lists_the_failed_step(
    capturing_logger, caplog
) -> None:
    class Boom(Pipeline):
        @classmethod
        def name(cls) -> str:
            return "boom"

        @property
        def resources(self) -> tuple[Resource, ...]:
            return ()

        def get_input(self) -> tuple[InputStep, ...]:
            return ()

        @checkpoint("{project}/a", CHECKPOINT_TYPE.FILE)
        def a(self):
            raise RuntimeError("kaboom")

        @checkpoint("{project}/b", CHECKPOINT_TYPE.FILE)
        def b(self):
            return "ok"

    task = _task()
    logger, stream = capturing_logger

    PipelineRunner(task, [Boom(task)], observer=TerminalObserver(logger=logger)).run()

    text = stream.getvalue()
    assert "ERROR failed pipeline step: step = boom.a, status = error" in text
    assert "msg = failed to process step: step_id = boom.a, error = kaboom" in text

    # The report renders the failure once; the runner must not *also* log the
    # raw message (it used to `logger.error(msg)` before the report existed).
    assert text.count("failed pipeline step") == 1
    raw_msg = "failed to process step: step_id = boom.a, error = kaboom"
    assert not any(record.getMessage() == raw_msg for record in caplog.records)
    # The traceback still comes from the runner.
    assert any(
        "Traceback (most recent call last):" in record.getMessage()
        for record in caplog.records
    )

    summary = text[text.index("INFO pipeline run finished:") :]
    assert "failed steps:" in summary
    assert (
        "  boom.a: failed to process step: step_id = boom.a, error = kaboom" in summary
    )


def test_a_skipped_step_notifies_the_observer(tmp_path: Path) -> None:
    settings = _settings(checkpoint=str(tmp_path))
    task = _task(settings=settings)
    artifact = Artifact(prefix="{project}/first", kind=CHECKPOINT_TYPE.FILE)
    path = Path(artifact.path(settings))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")

    observer = RecordingObserver()
    runner = PipelineRunner(task, [Chain(task)], observer=observer)
    runner.run()

    assert runner.execution_state["chain.first"].status is StepStatus.SKIPPED
    assert ("step_changed", "chain.first", StepStatus.SKIPPED) in observer.calls


def test_a_default_runner_still_runs_with_the_terminal_observer() -> None:
    task = _task()
    runner = PipelineRunner(task, [Chain(task)])

    assert isinstance(runner.observer, TerminalObserver)
    runner.run()

    assert runner.execution_state["chain.second"].status is StepStatus.FINISHED


def test_run_pipeline_forwards_an_observer() -> None:
    task = _task()
    observer = RecordingObserver()

    run_pipeline(task, [Chain(task)], observer=observer)

    assert observer.calls[0] == ("run_started", None, None)
    assert observer.calls[-1] == ("run_finished", None, None)


# --- dry-run and the manifest ------------------------------------------------


def test_manifest_lists_every_step_in_dependency_order() -> None:
    task = _task()
    runner = PipelineRunner(task, [Chain(task)])

    manifest = runner.manifest()

    assert manifest["task"]["command"]["kind"] == "pipeline.step"
    assert [step["name"] for step in manifest["steps"]] == [
        "input",
        "load-res",
        "first",
        "second",
    ]


def test_dry_run_prints_the_plan_and_leaves_every_step_at_init() -> None:
    task = _task()
    observer = RecordingObserver()
    runner = PipelineRunner(task, [Chain(task)], observer=observer)

    manifest = runner.dry_run()

    # Only the plan is observed; no step lifecycle call is made.
    assert observer.calls == [("run_started", None, None)]
    assert [step["name"] for step in manifest["steps"]] == [
        "input",
        "load-res",
        "first",
        "second",
    ]
    assert all(
        record.status is StepStatus.INIT for record in runner.execution_state.values()
    )


class PartialChain(Pipeline):
    """A generated step whose edge is the bound keyword of a ``partial``."""

    @classmethod
    def name(cls) -> str:
        return "partial_chain"

    @property
    def resources(self) -> tuple[Resource, ...]:
        return ()

    def get_input(self) -> tuple[InputStep, ...]:
        return ()

    @checkpoint("{project}/partial_chain/source", CHECKPOINT_TYPE.FILE)
    def source(self):
        return "the-input"

    def consume(self, mt=FromStep("source"), tag=None):  # noqa: B008
        return f"{mt}|{tag}"

    def prepare(self) -> None:
        super().prepare()
        self._methods.append(
            (
                "consume",
                checkpoint("{project}/partial_chain/consume", CHECKPOINT_TYPE.FILE)(
                    functools.partial(self.consume, mt=FromStep("source"), tag="bound")
                ),
                None,
            )
        )


def test_a_partial_step_runs_the_body_with_the_resolved_value() -> None:
    task = _task()
    runner = PipelineRunner(task, [PartialChain(task)])
    runner.run()

    # The runner passes ``mt="the-input"``; the partial's own ``mt=FromStep``
    # marker is overridden, while its non-marker ``tag="bound"`` still applies.
    assert runner.execution_state["partial_chain.consume"].result == "the-input|bound"


# --- runner selection: eligible set and --select globs -----------------------


class G(Pipeline):
    """A three-step chain in group ``g``: ``g.a`` -> ``g.b`` -> ``g.c``."""

    @classmethod
    def name(cls) -> str:
        return "g"

    @property
    def resources(self) -> tuple[Resource, ...]:
        return ()

    def get_input(self) -> tuple[InputStep, ...]:
        return ()

    @checkpoint("{project}/g/a", CHECKPOINT_TYPE.FILE)
    def a(self):
        return "a"

    @checkpoint("{project}/g/b", CHECKPOINT_TYPE.FILE)
    def b(self, a=FromStep("a")):  # noqa: B008
        return f"{a}+b"

    @checkpoint("{project}/g/c", CHECKPOINT_TYPE.FILE)
    def c(self, b=FromStep("b")):  # noqa: B008
        return f"{b}+c"


class H(Pipeline):
    """A single step in group ``h`` that consumes ``g.a``."""

    @classmethod
    def name(cls) -> str:
        return "h"

    @property
    def resources(self) -> tuple[Resource, ...]:
        return ()

    def get_input(self) -> tuple[InputStep, ...]:
        return ()

    @checkpoint("{project}/h/d", CHECKPOINT_TYPE.FILE)
    def d(self, a=FromStep("g.a")):  # noqa: B008
        return f"d({a})"


def _selection_runner(**kwargs) -> PipelineRunner:
    task = _task()
    return PipelineRunner(task, [G(task), H(task)], **kwargs)


def test_eligible_none_executes_the_whole_graph() -> None:
    runner = _selection_runner()
    runner.run()

    assert runner.execution_state["g.c"].result == "a+b+c"
    assert runner.execution_state["h.d"].result == "d(a)"
    assert all(
        record.status is StepStatus.FINISHED
        for record in runner.execution_state.values()
    )


def test_a_subset_eligible_runs_only_those_steps_plus_dependencies() -> None:
    runner = _selection_runner(eligible=frozenset({"g.b"}))
    runner.run()

    # g.b plus its dependency g.a run; g.c and h.d are dropped, never executed.
    assert runner.execution_state["g.a"].status is StepStatus.FINISHED
    assert runner.execution_state["g.b"].status is StepStatus.FINISHED
    assert runner.execution_state["g.c"].status is StepStatus.DROPPED
    assert runner.execution_state["h.d"].status is StepStatus.DROPPED
    assert runner.execution_state["g.b"].result == "a+b"


def test_select_unions_patterns_and_expands_dependencies() -> None:
    runner = _selection_runner(select=("g.a", "g.c"))
    runner.run()

    # The union {g.a} | {g.c} is expanded to its dependency closure
    # (g.c pulls in g.b and g.a). h.d is not selected.
    assert runner.execution_state["g.c"].status is StepStatus.FINISHED
    assert runner.execution_state["g.c"].result == "a+b+c"
    assert runner.execution_state["h.d"].status is StepStatus.DROPPED


def test_select_matches_ids_only_not_bare_names_or_groups() -> None:
    # ``a`` alone is not an id (ids are ``g.a`` etc.), so it matches nothing.
    with pytest.raises(ValueError, match="a"):
        _selection_runner(select=("a",))


def test_a_group_glob_matches_the_group_steps() -> None:
    runner = _selection_runner(select=("g.*",))
    runner.run()

    assert runner.execution_state["g.c"].status is StepStatus.FINISHED
    assert runner.execution_state["h.d"].status is StepStatus.DROPPED


def test_unknown_select_pattern_raises_before_any_step_runs() -> None:
    observer = RecordingObserver()
    with pytest.raises(ValueError, match="nope.*"):
        _selection_runner(select=("nope.*",), observer=observer)

    assert observer.calls == []


def test_select_can_only_shrink_the_eligible_set() -> None:
    # g.c is a valid registered id but lies outside the eligible set, so the
    # intersection is empty and no step can be revived.
    with pytest.raises(ValueError):
        _selection_runner(eligible=frozenset({"g.a"}), select=("g.c",))


def test_an_empty_post_intersection_selection_raises() -> None:
    with pytest.raises(ValueError):
        _selection_runner(eligible=frozenset({"g.a", "g.b"}), select=("h.d",))


def test_an_empty_eligible_frozenset_raises() -> None:
    with pytest.raises(ValueError):
        _selection_runner(eligible=frozenset())


def test_dropped_steps_are_not_started_and_do_not_error() -> None:
    runner = _selection_runner(eligible=frozenset({"g.b"}))
    runner.run()

    assert runner.has_started() is True
    assert runner.has_error() is False
    # has_finished is true because every step is terminal (finished or dropped).
    assert runner.has_finished() is True


def test_a_narrow_selection_does_not_emit_step_changed_for_dropped() -> None:
    observer = RecordingObserver()
    runner = _selection_runner(eligible=frozenset({"g.b"}), observer=observer)
    runner.run()

    changed_ids = {
        step_id for kind, step_id, _ in observer.calls if kind == "step_changed"
    }
    assert changed_ids == {"g.a", "g.b"}


def test_dry_run_projects_the_manifest_to_the_selected_graph() -> None:
    runner = _selection_runner(eligible=frozenset({"g.b"}))

    manifest = runner.dry_run()

    # The returned manifest is projected to the selection's closure (g.a, g.b);
    # resolving the plan marks the rest dropped, which the render header counts.
    assert [step["name"] for step in manifest["steps"]] == ["a", "b"]
    assert runner.execution_state["g.a"].status is StepStatus.INIT
    assert runner.execution_state["g.b"].status is StepStatus.INIT
    assert runner.execution_state["g.c"].status is StepStatus.DROPPED
    assert runner.execution_state["h.d"].status is StepStatus.DROPPED


def test_a_restored_runner_walks_the_restored_selection() -> None:
    runner = _selection_runner(eligible=frozenset({"g.b"}), select=("g.*",))
    restored = PipelineRunner.load(runner.current_state())

    restored.run()

    # g.b's closure (g.a, g.b) runs; g.c and h.d stay dropped as in the original.
    assert restored.execution_state["g.a"].status is StepStatus.FINISHED
    assert restored.execution_state["g.b"].status is StepStatus.FINISHED
    assert restored.execution_state["g.c"].status is StepStatus.DROPPED
    assert restored.execution_state["h.d"].status is StepStatus.DROPPED


# --- provenance-driven skip (mode 1, ticket 06) ------------------------------


class _SpyRuntime(LocalRuntime):
    """A ``LocalRuntime`` that records whether ``prepare`` was entered."""

    def __init__(self) -> None:
        self.prepared: list[str] = []

    def prepare(self, step, call, task) -> None:
        self.prepared.append(step.id)
        super().prepare(step, call, task)


class PresentArtifact(Pipeline):
    @classmethod
    def name(cls) -> str:
        return "present"

    @property
    def resources(self) -> tuple[Resource, ...]:
        return ()

    def get_input(self) -> tuple[InputStep, ...]:
        return ()

    @checkpoint("{project}/present", CHECKPOINT_TYPE.FILE)
    def compute(self):
        return "computed"


def _build_present(
    tmp_path: Path, **settings_overrides
) -> tuple[PipelineRunner, Step]:
    settings = _settings(checkpoint=str(tmp_path), **settings_overrides)
    set_config(settings)
    runner = PipelineRunner(
        _task(settings=settings), [PresentArtifact(_task(settings=settings))]
    )
    return runner, runner._step_registry["present.compute"]


def _write_sidecar(step: Step, path: Path, *, source_md5: str | None = None) -> None:
    from hashlib import md5

    from gnomad_meb.lib.checkpoint.metadata import (
        CheckpointMetadata,
        checkpoint_static,
    )

    source = step.checkpoint_callable.source()
    CheckpointMetadata(
        path=str(path),
        exists=True,
        source_md5=source_md5
        or md5(source.encode()).hexdigest(),  # noqa: S324 - test fixture hash
        **checkpoint_static(),
    ).to_json_file(f"{path}.gnomad_meb.md.json")


def _present_path(step: Step) -> Path:
    path = Path(step.checkpoint_callable.spec.artifacts[0].path(step.task.settings))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


def test_a_present_step_is_skipped_and_reused(tmp_path: Path) -> None:
    runner, step = _build_present(tmp_path)
    _write_sidecar(step, _present_path(step))

    spy = _SpyRuntime()
    runner.runtime = spy
    resolved: list[str] = []
    original_resolve = runner._resolve_dependency

    def _spy_resolve(dep_id: str):
        resolved.append(dep_id)
        return original_resolve(dep_id)

    runner._resolve_dependency = _spy_resolve  # type: ignore[method-assign]
    state = runner.run()

    record = state["present.compute"]
    assert record.status is StepStatus.SKIPPED
    assert record.checkpoint_status is CheckpointStatus.PRESENT
    assert record.reused is True
    # The cluster-cost guarantee: a skipped step is never prepared and its
    # dependency values are never resolved.
    assert spy.prepared == []
    assert resolved == []


def test_a_stale_step_under_source_recomputes(tmp_path: Path) -> None:
    runner, step = _build_present(
        tmp_path, staleness_policy=StalenessPolicy.SOURCE
    )
    _write_sidecar(step, _present_path(step), source_md5="not-current")

    assert runner.should_skip(step) is False
    assert step.checkpoint_callable.force_recompute is True
    # A direct call now re-executes rather than reading the stale artifact.
    assert step.run() == b"computed"
    assert step.checkpoint_callable.status() is CheckpointStatus.PRESENT


def test_unknown_is_skipped_under_completeness(tmp_path: Path) -> None:
    runner, step = _build_present(tmp_path)
    _present_path(step)  # present, no sidecar

    assert step.checkpoint_callable.status() is CheckpointStatus.UNKNOWN
    assert runner.should_skip(step) is True


def test_unknown_is_not_skipped_under_source(tmp_path: Path) -> None:
    runner, step = _build_present(tmp_path, staleness_policy=StalenessPolicy.SOURCE)
    _present_path(step)

    assert runner.should_skip(step) is False


def test_overwrite_runs_every_step(tmp_path: Path) -> None:
    settings = _settings(checkpoint=str(tmp_path))
    set_config(settings)
    runner = PipelineRunner(
        _task(settings=settings),
        [PresentArtifact(_task(settings=settings))],
        overwrite=True,
    )
    step = runner._step_registry["present.compute"]
    _write_sidecar(step, _present_path(step))
    assert runner.should_skip(step) is False


def test_declares_step_is_never_skipped(tmp_path: Path) -> None:
    from gnomad_meb.lib.checkpoint import declares_step

    settings = _settings(checkpoint=str(tmp_path))
    set_config(settings)

    class Marker(Pipeline):
        @classmethod
        def name(cls) -> str:
            return "marker"

        @property
        def resources(self) -> tuple[Resource, ...]:
            return ()

        def get_input(self) -> tuple[InputStep, ...]:
            return ()

        @declares_step
        def side_effect(self):
            return "ran"

    runner = PipelineRunner(
        _task(settings=settings), [Marker(_task(settings=settings))]
    )
    step = runner._step_registry["marker.side_effect"]
    assert runner.should_skip(step) is False


def test_dry_run_records_checkpoint_status(tmp_path: Path) -> None:
    runner, step = _build_present(tmp_path)
    _write_sidecar(step, _present_path(step))

    runner.dry_run()
    record = runner.execution_state["present.compute"]
    assert record.checkpoint_status is CheckpointStatus.PRESENT
    assert record.reused is True


def test_dry_run_records_absent_for_a_missing_artifact(tmp_path: Path) -> None:
    runner, _ = _build_present(tmp_path)

    runner.dry_run()
    record = runner.execution_state["present.compute"]
    assert record.checkpoint_status is CheckpointStatus.ABSENT
    assert record.reused is False


# --- lineage staleness (mode 2, ticket 07) -----------------------------------


class LineageChain(Pipeline):
    """``a -> b -> c`` plus an independent ``d``."""

    @classmethod
    def name(cls) -> str:
        return "lineage"

    @property
    def resources(self) -> tuple[Resource, ...]:
        return ()

    def get_input(self) -> tuple[InputStep, ...]:
        return ()

    @checkpoint("{project}/a", CHECKPOINT_TYPE.FILE)
    def a(self):
        return "a"

    @checkpoint("{project}/b", CHECKPOINT_TYPE.FILE)
    def b(self, value=FromStep("a")):  # noqa: B008
        return f"b:{value}"

    @checkpoint("{project}/c", CHECKPOINT_TYPE.FILE)
    def c(self, value=FromStep("b")):  # noqa: B008
        return f"c:{value}"

    @checkpoint("{project}/d", CHECKPOINT_TYPE.FILE)
    def d(self):
        return "d"


def _lineage_runner(tmp_path: Path, **overrides) -> PipelineRunner:
    settings = _settings(checkpoint=str(tmp_path), **overrides)
    set_config(settings)
    return PipelineRunner(
        _task(settings=settings), [LineageChain(_task(settings=settings))]
    )


def _present(step: Step) -> None:
    _write_sidecar(step, _present_path(step))


def test_an_absent_root_marks_the_whole_downstream_chain_stale(tmp_path: Path) -> None:
    runner = _lineage_runner(tmp_path)
    # Only the leaf ``d`` is present; ``a`` is absent, so ``b`` and ``c`` stale.
    _present(runner._step_registry["lineage.d"])

    runner.dry_run()

    a = runner.execution_state["lineage.a"]
    b = runner.execution_state["lineage.b"]
    c = runner.execution_state["lineage.c"]
    assert a.stale is False
    assert a.checkpoint_status is CheckpointStatus.ABSENT
    assert b.stale is True
    assert b.staleness == (StalenessReason.LINEAGE_INVALID,)
    assert c.stale is True
    assert c.staleness == (StalenessReason.LINEAGE_INVALID,)


def test_an_independent_branch_is_not_marked_stale(tmp_path: Path) -> None:
    runner = _lineage_runner(tmp_path)
    _present(runner._step_registry["lineage.d"])

    runner.dry_run()

    assert runner.execution_state["lineage.d"].stale is False


def test_a_recomputed_producer_makes_dependants_run(tmp_path: Path) -> None:
    runner = _lineage_runner(tmp_path)
    # ``b`` and ``c`` present, ``a`` absent -> b/c must recompute, not reuse.
    _present(runner._step_registry["lineage.b"])
    _present(runner._step_registry["lineage.c"])

    state = runner.run()

    assert state["lineage.a"].status is StepStatus.FINISHED
    assert state["lineage.b"].status is StepStatus.FINISHED
    assert state["lineage.c"].status is StepStatus.FINISHED
    assert state["lineage.b"].reused is False
    assert state["lineage.c"].reused is False


class WithLoader(Pipeline):
    """A producer consuming a resource whose loader runs every time."""

    @classmethod
    def name(cls) -> str:
        return "loader_stage"

    @property
    def resources(self) -> tuple[Resource, ...]:
        return (
            Resource(name="res", path="/data/res.ht", loader=lambda p: f"loaded:{p}"),
        )

    def get_input(self) -> tuple[InputStep, ...]:
        return ()

    @checkpoint("{project}/consume", CHECKPOINT_TYPE.FILE)
    def consume(self, res=ResourceRef("res")):  # noqa: B008
        return res


def test_a_resource_loader_does_not_invalidate_its_consumer(tmp_path: Path) -> None:
    settings = _settings(checkpoint=str(tmp_path))
    set_config(settings)
    runner = PipelineRunner(
        _task(settings=settings), [WithLoader(_task(settings=settings))]
    )
    step = runner._step_registry["loader_stage.consume"]
    _present(step)

    runner.dry_run()

    # The resource load step invalidates nothing: the consumer stays reusable.
    assert step.artifacts  # sanity: it does declare an artifact
    loader_ids = [
        sid for sid in runner._step_registry.steps() if sid.startswith("resources.")
    ]
    assert loader_ids
    for sid in loader_ids:
        assert runner.execution_state[sid].status is not StepStatus.SKIPPED
        dependents = runner._step_registry.dependents_of(sid)
        assert dependents


def test_stale_and_staleness_serialize_on_step_result(tmp_path: Path) -> None:
    resolved = StepResult(
        step_id="x",
        stale=True,
        staleness=(StalenessReason.LINEAGE_INVALID,),
    )
    dumped = resolved.model_dump(mode="json")
    assert dumped["stale"] is True
    assert dumped["staleness"] == ["lineage_invalid"]


def test_an_input_step_does_not_invalidate_its_consumer(tmp_path: Path) -> None:
    settings = _settings(checkpoint=str(tmp_path))
    set_config(settings)

    class WithInput(Pipeline):
        @classmethod
        def name(cls) -> str:
            return "input_stage"

        @property
        def resources(self) -> tuple[Resource, ...]:
            return ()

        def get_input(self) -> tuple[InputStep, ...]:
            return (
                InputStep(
                    name="input", task=self.task, group=type(self).name(), value=10
                ),
            )

        @checkpoint("{project}/consumer", CHECKPOINT_TYPE.FILE)
        def consumer(self, value=RunInput("input")):  # noqa: B008
            return f"c:{value}"

    runner = PipelineRunner(
        _task(settings=settings), [WithInput(_task(settings=settings))]
    )
    step = runner._step_registry["input_stage.consumer"]
    _present(step)

    runner.dry_run()

    assert step.checkpoint_callable.force_recompute is False
    assert runner.execution_state["input_stage.consumer"].stale is False


def test_dry_run_renders_present_absent_and_stale_rows(tmp_path: Path) -> None:
    settings = _settings(checkpoint=str(tmp_path))
    set_config(settings)
    runner = PipelineRunner(
        _task(settings=settings), [LineageChain(_task(settings=settings))]
    )
    # ``d`` present; ``a`` absent -> ``b``/``c`` stale in the rendered plan.
    d = runner._step_registry["lineage.d"]
    _write_sidecar(d, _present_path(d))

    logger_lines: list[str] = []
    from gnomad_meb.pipeline.report import render_plan

    runner.dry_run()
    rendered = render_plan(
        [runner._step_registry[s] for s in
         ("lineage.a", "lineage.b", "lineage.c", "lineage.d")],
        runner.execution_state,
    )

    lines = {
        line.split()[1]: line
        for line in rendered.splitlines()
        if line.startswith("  ")
    }

    assert lines["lineage.a"].endswith("absent")
    assert lines["lineage.b"].endswith("absent  stale")
    assert lines["lineage.c"].endswith("absent  stale")
    assert lines["lineage.d"].endswith("present")
    assert logger_lines == []  # dry_run executed nothing


def test_a_stale_producer_does_not_invalidate_its_dependants(tmp_path: Path) -> None:
    # Lineage is driven only by an ABSENT artifact, not by any step that merely
    # failed to skip: a STALE producer recomputes but leaves its (present,
    # provenance-matching) consumers reusable.
    runner = _lineage_runner(tmp_path, staleness_policy=StalenessPolicy.SOURCE)
    a = runner._step_registry["lineage.a"]
    b = runner._step_registry["lineage.b"]
    # ``a`` present but with a mismatched source -> STALE; ``b`` present+valid.
    _write_sidecar(a, _present_path(a), source_md5="not-current")
    _present(b)

    runner.dry_run()

    assert runner.execution_state["lineage.a"].checkpoint_status is (
        CheckpointStatus.STALE
    )
    assert runner.execution_state["lineage.b"].stale is False
    assert runner.execution_state["lineage.b"].reused is True
