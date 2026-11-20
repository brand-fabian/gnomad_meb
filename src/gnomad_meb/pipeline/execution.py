"""The run engine: walk the central registry, execute each step, record status.

:class:`PipelineRunner` executes every step registered by the pipelines it was
given, in dependency order. A step whose declared artifacts already exist is
recorded ``skipped`` (unless ``overwrite``); the ``@checkpoint`` wrapper is the
cache and the writer. A step that raises is recorded as ``error`` with its
message (the traceback goes to the log) and the walk continues with the remaining
ready steps; a step whose dependency failed is refused and recorded ``error``
too, so a failure propagates rather than poisoning downstream results with
``None``.

The runner also owns the run's durable form and its restoration. Its
``current_state()`` dumps the task, the pipelines' provenance, the two
registries and a per-step :class:`~gnomad_meb.pipeline.status.StepResult`;
``PipelineRunner.load`` rebuilds a run from it by re-instantiating each pipeline
subclass through :class:`~gnomad_meb.pipeline.PipelineFactory` (loaders and
callables are rebuilt from the class, never serialized — ADR-0012/0013).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime
from fnmatch import fnmatch
from typing import Any

from gnomad_meb.cmd.task import Task
from gnomad_meb.config import set_config
from gnomad_meb.lib.checkpoint.status import (
    CheckpointEvaluation,
    CheckpointStatus,
    StalenessReason,
)
from gnomad_meb.logging import get_logger

from .constants import RUNTIME_PLACEHOLDER
from .factory import PipelineFactory
from .registry import ResourceRegistry
from .report import RunObserver, TerminalObserver
from .runtime import LocalRuntime, RunnerStepContext, StepRuntime
from .spec import InputStep, RunManifest, Step
from .status import StepResult, StepStatus
from .step import StepRegistry

logger = get_logger()


Call = Callable[[], Any]


class PipelineMismatchError(ValueError):
    """A restored run's step or resource set differs from the serialized one."""


class DependencyFailed(RuntimeError):
    """A step cannot run because its dependencies failed."""


def _now() -> datetime:
    """The current time, timezone-aware (UTC)."""
    return datetime.now(UTC)


class PipelineRunner:
    """Execute the registered steps in dependency order and observe the run."""

    def __init__(
        self,
        task: Task,
        pipelines: Any,
        overwrite: bool = False,
        *,
        observer: RunObserver | None = None,
        eligible: frozenset[str] | None = None,
        select: tuple[str, ...] = (),
        runtime: StepRuntime | None = None,
    ) -> None:
        self.task = task
        self.pipelines = tuple(pipelines)
        self.overwrite = overwrite
        self.observer: RunObserver = observer or TerminalObserver()
        self.runtime: StepRuntime = runtime or LocalRuntime()

        if eligible is not None and not eligible:
            raise ValueError("eligible step set must not be empty")

        self._resource_registry = ResourceRegistry()
        self._step_registry = StepRegistry()

        #: Per-step status/timing, ``{step_id: StepResult}``. The in-memory
        #: ``result`` is excluded from serialization.
        self.execution_state: dict[str, StepResult] = {
            step_id: StepResult(step_id=step_id)
            for step_id in self._step_registry.steps()
        }

        self._eligible_steps: frozenset[str] | None = eligible
        self._select_steps: tuple[str, ...] = select

    def _resolve_selection(
        self, eligible: frozenset[str] | None, select: tuple[str, ...]
    ) -> frozenset[str]:
        """Resolve the executed subgraph from the eligible set and ``select`` globs.

        Starts from ``eligible`` (default: every registered step), intersects it
        with the union of the registered ids each glob matches (a glob matching
        nothing anywhere in the registry is an error), and expands each survivor
        to itself plus its dependency closure. Selection is monotone
        non-increasing; an empty result is an error rather than a whole-graph
        run.
        """
        registered = self._step_registry.steps()
        constrained = eligible is not None or bool(select)
        selected = set(registered) if eligible is None else set(eligible)

        if select:
            matched: set[str] = set()
            for pattern in select:
                hits = {step_id for step_id in registered if fnmatch(step_id, pattern)}
                if not hits:
                    raise ValueError(
                        "select pattern matched no registered step: "
                        f"pattern = {pattern}"
                    )
                matched |= hits
            selected &= matched

        if constrained and not selected:
            raise ValueError(
                "selection matched no eligible step: "
                f"select = {select}, eligible = {eligible}"
            )

        expanded = set(selected)
        for step_id in selected:
            expanded |= self._step_registry.dependencies_of(step_id)
        return frozenset(expanded)

    def _resolve_steps(self) -> Iterator[Step]:
        """Yield the selected steps in dependency order, marking the rest dropped.

        Resolves this runner's selection on each call, walks the registry's
        topological order, and yields only the steps inside that selection. Every
        step outside it is recorded ``dropped`` in the execution state before the
        generator yields its neighbours, so a plan rendered while walking already
        reflects the pruned graph.
        """
        selected_steps = self._resolve_selection(self._eligible_steps, self._select_steps)
        ordered = self._step_registry.topological()
        for step in ordered:
            if step.id in selected_steps:
                yield step
            else:
                record = self.execution_state[step.id]
                record.status = StepStatus.DROPPED

    @property
    def settings(self):
        """The run's resolved settings."""
        return self.task.settings

    def has_started(self) -> bool:
        """Whether this runner has started any tasks.

        Reads the execution state recorded by this runner to check whether any task has
        been run at any stage by this runner or previous invocations.
        """
        return any(
            record.status not in (StepStatus.INIT, StepStatus.DROPPED)
            for record in self.execution_state.values()
        )

    def has_finished(self) -> bool:
        """Whether every registered step reached a terminal status."""
        terminal = {
            StepStatus.SKIPPED,
            StepStatus.ERROR,
            StepStatus.FINISHED,
            StepStatus.DROPPED,
        }
        return all(
            record.status in terminal for record in self.execution_state.values()
        )

    def has_error(self) -> bool:
        """Check if any step recorded an error state."""
        return any(
            record.status == StepStatus.ERROR
            for record in self.execution_state.values()
        )

    def should_skip(self, step: Step) -> bool:
        """Whether ``step``'s artifacts are reusable under the run's policy.

        The step's :class:`CheckpointedCallable` is the single source of the
        verdict: a ``PRESENT`` artifact is skipped and the callable marked
        reusable; a ``STALE`` artifact is run and the callable marked stale so a
        direct ``__call__`` recomputes too. A step with no callable falls back to
        today's ``Step.artifacts``-existence check; a step with neither is never
        skipped.
        """
        if self.overwrite:
            return False

        callable_ = step.checkpoint_callable
        if callable_ is not None:
            if callable_.force_recompute:
                # A lineage verdict (ticket 07) or a prior stale mark means this
                # step must recompute rather than reuse its artifact.
                return False
            evaluation = callable_.evaluate(self.settings)
            self._apply_evaluation(step, self.execution_state[step.id], evaluation)
            return evaluation.reusable

        if not step.artifacts:
            return False
        return all(artifact.exists(self.settings) for artifact in step.artifacts)

    def _apply_evaluation(
        self, step: Step, record: StepResult, evaluation: CheckpointEvaluation
    ) -> None:
        """Record an evaluation and set the callable's stale/reusable flag once.

        The one place the verdict is written back: the record carries the status
        for observation (ADR-0014), the callable's flag drives the next
        ``__call__``.
        """
        callable_ = step.checkpoint_callable
        record.checkpoint_status = evaluation.status
        record.reused = evaluation.reusable
        if evaluation.reusable:
            if callable_ is not None:
                callable_.mark_reusable()
            return
        if evaluation.status is CheckpointStatus.STALE:
            if callable_ is not None:
                callable_.mark_stale(evaluation.reasons)
            record.stale = True
            record.staleness = evaluation.reasons
        elif callable_ is not None:
            callable_.mark_reusable()

    def _resolve_dependency(self, dep_id: str) -> Any:
        """The value of an already-walked dependency, keyed by its status.

        A failed dependency refuses the consumer; a dependency that did not run
        this session (it was skipped, or restored from a dump without an
        in-memory result) is read back from its declared artifacts.
        """
        record = self.execution_state[dep_id]
        if record.status == StepStatus.ERROR:
            raise DependencyFailed(f"step dependency failed: dependency = {dep_id}")

        if record.result is not None:
            return record.result

        producer = self._step_registry[dep_id]
        if not producer.artifacts:
            raise DependencyFailed(
                f"dependency produced no artifact: dependency = {dep_id}"
            )

        values = tuple(
            artifact.serializer.read(artifact.path(self.settings))
            for artifact in producer.artifacts
        )
        return values[0] if len(values) == 1 else values

    def _get_input_step(self, step: InputStep) -> Call:
        """Execute an input step.

        Resolves an input in order of: (1) the result of the step named by
        ``source`` when it ran this session; (2) that step's declared artifact,
        read from disk via ``step.run(path)``; (3) the bound ``value`` via
        ``step.run(value)``. A failed ``source`` or ``depends_on`` producer
        refuses the input, so the failure propagates instead of falling through
        to a stale value.
        """
        if step.source is not None:
            source = self.execution_state.get(step.source)
            if source is not None and source.status == StepStatus.ERROR:
                raise DependencyFailed(
                    f"step dependency failed: step = {step.id}, "
                    f"dependency = {source.step_id}"
                )
            if source is not None and source.result is not None:
                return lambda: source.result

        for dep_id in step.depends_on.values():
            record = self.execution_state.get(dep_id)
            if record is not None and record.status == StepStatus.ERROR:
                raise DependencyFailed(
                    f"step dependency failed: step = {step.id}, dependency = {dep_id}"
                )

        if step.source is not None and step.source in self._step_registry:
            source_step = self._step_registry[step.source]
            # TODO: Support multi-output checkpoints properly
            if len(source_step.artifacts) > 0:
                path = source_step.artifacts[0].path(self.settings)
                return lambda: step.run(path)

        if step.value is not None:
            return lambda: step.run(step.value)

        raise ValueError(f"input step has no value or source: step = {step.id}")

    def _get_work_step(self, step: Step) -> Any:
        kwargs: dict[str, Any] = {}
        for param, dep in {**step.depends_on, **step.resources}.items():
            value = self._resolve_dependency(dep)
            index = step.outputs.get(param)
            if index is not None and isinstance(value, tuple):
                value = value[index]
            kwargs[param] = value

        def call() -> Any:
            from gnomad_meb.lib.checkpoint import checkpoint_overwrite

            with checkpoint_overwrite(self.overwrite):
                return step.run(**kwargs)

        return call

    def run_step(self, step: Step) -> None:
        """Execute one step through the runtime, recording its result or error.

        The runner owns the lifecycle: it builds the callable and the context,
        records ``STARTED`` inside its protected path, delegates to
        :meth:`StepRuntime.run`, finalizes as ``FINISHED`` (or ``ERROR`` on any
        raised exception) and closes the context. The runtime may publish
        intermediate statuses through the context, but the runner is the only
        writer of terminal states.
        """
        record = self.execution_state[step.id]
        context = RunnerStepContext(step, record, self.observer.step_changed, _now)
        context.record(StepStatus.STARTED)

        try:
            if isinstance(step, InputStep):
                step_fn = self._get_input_step(step)
            else:
                step_fn = self._get_work_step(step)

            self.runtime.prepare(step, step_fn, self.task)
            result = self.runtime.run(step, step_fn, context)
            if result is RUNTIME_PLACEHOLDER:
                result = self.runtime.postprocess(step, context)

            context.finalize(StepStatus.FINISHED, result=result)
        except Exception as exc:
            from traceback import format_exc

            msg = f"failed to process step: step_id = {step.id}, error = {exc}"
            logger.error(format_exc())
            context.finalize(StepStatus.ERROR, msg=msg)
        finally:
            context.close()
            self.observer.step_changed(step, record)

    def manifest(self) -> dict[str, Any]:
        """The run's declarative envelope: pipeline name, task, and steps.

        Each entry of ``steps`` is a :class:`~gnomad_meb.pipeline.spec.Step`
        dump with ``step.task.settings`` omitted (the root ``task`` is the one
        authoritative copy), in dependency order. The envelope is the same
        :class:`~gnomad_meb.pipeline.spec.RunManifest` ``--describe`` emits with
        no steps, so both inspection flags are projections of one model.
        """
        return self._manifest(self._step_registry.topological())

    def _manifest(self, ordered: list[Step]) -> dict[str, Any]:
        """Build the envelope over a given ordered step list."""
        steps = [RunManifest.dump_step(step) for step in ordered]
        return RunManifest.build(
            self.pipelines[0].name(),
            self.task,
            steps=steps,
            overwrite=self.overwrite,
        )

    def dry_run(self) -> dict[str, Any]:
        """Print the plan, return the manifest, and execute nothing.

        The observer renders the human-readable plan over the *selected* graph:
        only the steps that would run appear as rows, while the header's
        ``dropped by selection`` count reports what the graph contained but the
        selection pruned. The returned manifest matches that projected
        granularity. Every walked step keeps its ``init`` status.
        """
        set_config(self.task.settings)
        steps = list(self._resolve_steps())
        self._prepass(steps)
        self.observer.run_started(steps, self.execution_state)
        return self._manifest(steps)

    def _prepass(self, steps: Sequence[Step]) -> None:
        """Evaluate every step's checkpoint status and propagate lineage.

        Records ``checkpoint_status`` / ``reused`` / ``stale`` / ``staleness`` on
        every step's record, and marks the callables stale/reusable, so a plan
        rendered from the records reflects the same status a real run would. An
        ``ABSENT`` producer invalidates its downstream consumers immediately,
        before the walk passes it. Runs no step, resolves no dependency and
        writes nothing.
        """
        for step in steps:
            callable_ = step.checkpoint_callable
            if callable_ is None:
                continue
            record = self.execution_state[step.id]
            evaluation = callable_.evaluate(self.settings)
            self._apply_evaluation(step, record, evaluation)
            if evaluation.status is CheckpointStatus.ABSENT:
                self._invalidate_dependants(step)

    def _propagates_lineage(self, step: Step) -> bool:
        """Whether a recomputed ``step`` invalidates its downstream consumers.

        Only checkpointed artifact producers do. Input endpoints and resource
        loader/modification steps carry a value or load a resource rather than
        producing a checkpointed artifact, so a missing input or unloadable
        resource is not an upstream recomputation.
        """
        if isinstance(step, InputStep):
            return False
        if step.group == "resources":
            return False
        return bool(step.artifacts)

    def _invalidate_dependants(self, step: Step) -> None:
        """Mark every downstream consumer of an ``ABSENT`` step stale.

        Lineage staleness is driven **only by an absent artifact** (the step
        will be recomputed), not by any step that merely did not skip. The
        closure is the forward one -- :meth:`dependents_of` -- not the upstream
        ``dependencies_of``. A consumer that is a checkpointed, not-yet-walked
        step carries the stale flag into its own ``should_skip``, so it
        recomputes rather than reading its (now untrustworthy) artifact.
        """
        if not self._propagates_lineage(step):
            return
        for dependant_id in self._step_registry.dependents_of(step.id):
            dependant = self._step_registry[dependant_id]
            callable_ = dependant.checkpoint_callable
            if callable_ is None:
                continue
            record = self.execution_state[dependant.id]
            if record.status == StepStatus.SKIPPED:
                # Already walked and skipped: keep its record.
                continue
            callable_.mark_stale((StalenessReason.LINEAGE_INVALID,))
            record.stale = True
            record.staleness = (StalenessReason.LINEAGE_INVALID,)

    def run(self) -> dict[str, StepResult]:
        """Run the selected steps in dependency order.

        Steps outside the selection are recorded ``dropped`` while the plan is
        resolved, *before* it is rendered, so the report accounts for the pruned
        graph; only the selected ids are walked, and a dropped step never
        executes.
        """
        set_config(self.task.settings)

        ordered = list(self._resolve_steps())
        self.observer.run_started(ordered, self.execution_state)

        for step in ordered:
            record = self.execution_state[step.id]

            if self.should_skip(step):
                record.status = StepStatus.SKIPPED
                record.msg = "artifacts exist"
                record.reused = True
                record.finished_at = _now()

                self.observer.step_changed(step, record)
                continue

            # An absent producer invalidates its downstream consumers
            # immediately, before the walk continues past it (ticket 07).
            if record.checkpoint_status is CheckpointStatus.ABSENT:
                self._invalidate_dependants(step)
            self.run_step(step)

        self.observer.run_finished(ordered, self.execution_state)
        return self.execution_state

    # --- durable form --------------------------------------------------------

    def current_state(self) -> dict[str, Any]:
        """The run's restorable state: task, pipelines, registries, selection, statuses."""
        return {
            "task": self.task.model_dump(mode="json"),
            "overwrite": self.overwrite,
            "eligibility": (
                None if self._eligible_steps is None else sorted(self._eligible_steps)
            ),
            "select": list(self._select_steps),
            "pipelines": [pipeline.current_state() for pipeline in self.pipelines],
            "resources": self._resource_registry.current_state(),
            "steps": self._step_registry.current_state(),
            "execution_state": {
                step_id: record.model_dump(mode="json")
                for step_id, record in self.execution_state.items()
            },
        }

    @classmethod
    def load(cls, data: dict[str, Any]) -> PipelineRunner:
        """Restore a run dumped by :meth:`current_state`.

        Resets the two registries, reinstalls the run's settings, re-instantiates
        each pipeline through the factory (re-registering its steps and
        resources and rebuilding its loaders from the class), asserts the
        restored step and resource sets equal the serialized ones, and restores
        the per-step status model. The run's selection (its eligible set and
        select patterns) is restored too, so a resumed run walks the same
        subgraph it originally resolved rather than the whole graph.
        """
        StepRegistry._reset()
        ResourceRegistry._reset()

        task = Task.model_validate(data["task"])
        set_config(task.settings)

        pipelines = [
            PipelineFactory.make(
                provenance["pipeline"], Task.model_validate(provenance)
            )
            for provenance in data["pipelines"]
        ]

        runner = cls(
            task,
            pipelines,
            overwrite=data.get("overwrite", False),
            eligible=(
                None
                if data.get("eligibility") is None
                else frozenset(data["eligibility"])
            ),
            select=tuple(data.get("select", ())),
        )
        runner._check_restored(data)
        runner.execution_state = {
            step_id: StepResult.model_validate(record)
            for step_id, record in data["execution_state"].items()
        }
        return runner

    def _check_restored(self, data: dict[str, Any]) -> None:
        """Assert the restored step and resource sets equal the serialized ones."""
        restored_steps = set(self._step_registry.steps())
        serialized_steps = set(data["steps"])
        if restored_steps != serialized_steps:
            raise PipelineMismatchError(
                "restored steps differ from the serialized run: "
                f"missing = {serialized_steps - restored_steps}, "
                f"extra = {restored_steps - serialized_steps}"
            )

        restored_resources = self._resource_registry.current_state()
        serialized_resources = data["resources"]
        if restored_resources != serialized_resources:
            raise PipelineMismatchError(
                "restored resources differ from the serialized run: "
                f"restored = {restored_resources}, "
                f"serialized = {serialized_resources}"
            )


def run_pipeline(
    task: Task,
    pipelines: Any,
    overwrite: bool = False,
    *,
    observer: RunObserver | None = None,
    runtime: StepRuntime | None = None,
) -> dict[str, StepResult]:
    """Construct a runner for ``task`` and ``pipelines`` and execute every step."""
    return PipelineRunner(
        task,
        pipelines,
        overwrite=overwrite,
        observer=observer,
        runtime=runtime,
    ).run()
