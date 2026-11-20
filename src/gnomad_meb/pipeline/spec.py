"""The :class:`Step` unit of work and its :class:`InputStep` endpoint.

A ``Step`` is one method of a :class:`~gnomad_meb.pipeline.Pipeline` projected
into a runnable unit: the callable to execute (``run``), its declared
``artifacts`` (the one vocabulary in :mod:`gnomad_meb.lib.checkpoint`), the
``depends_on`` / ``resources`` edges its parameter defaults declare, and the
``Task`` that carries the run's resolved settings.

An ``InputStep`` is the endpoint a ``RunInput`` parameter resolves to. It
switches between *using* the result of another step (``source``) when it ran,
reading that step's declared artifact, and *loading* a bound value with its
``run`` loader, so a ``RunInput`` works both standalone and when composed with
the step that produces it.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gnomad_meb.cmd.task import Task
from gnomad_meb.lib.checkpoint import CheckpointedCallable


def _identity(value: Any) -> Any:
    """The default input loader: hand the bound value through unchanged."""
    return value


class StepState(BaseModel):
    """The serializable provenance of one registered step.

    ``run`` (and, for a resource, its loader) is never serialized; this
    projection is what ``StepRegistry.current_state`` dumps so the durable form
    stays declarative JSON (ADR-0012).
    """

    model_config = ConfigDict(extra="ignore")

    name: str
    group: str | None = None
    artifacts: list[str] = []
    depends_on: dict[str, str] = {}
    outputs: dict[str, int] = {}
    resources: dict[str, str] = {}


class Step(BaseModel):
    """One unit of work in the pipeline's DAG.

    ``run`` is the callable the runner executes and is never serialized.
    ``depends_on`` maps a parameter name to the id of the step whose result feeds
    it; ``resources`` maps a parameter name to the id of the step that loads (or
    modifies) the resource it consumes. ``outputs`` maps a ``depends_on``
    parameter name to the output index its ``FromStep`` marker selected
    (default ``0``).
    """

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    name: str
    task: Task
    artifacts: tuple[Any, ...] = ()
    depends_on: dict[str, str] = Field(default_factory=dict)
    outputs: dict[str, int] = Field(default_factory=dict)
    resources: dict[str, str] = Field(default_factory=dict)
    group: str | None = None
    run: Callable = Field(exclude=True, default=_identity)
    #: The step's :class:`CheckpointedCallable` when it is a decorated step, so
    #: the runner can query status before dispatch. Never serialized (rebuilt
    #: from the pipeline class on restore, ADR-0012/0013), exactly as ``run`` is.
    checkpoint_callable: CheckpointedCallable | None = Field(exclude=True, default=None)

    @property
    def id(self) -> str:
        """The step's global id: ``<group>.<name>`` when grouped."""
        return f"{self.group}.{self.name}" if self.group else self.name

    def optional_dependencies(self) -> tuple[str, ...]:
        return ()

    def incoming_edges(self, available: Collection[str]) -> tuple[str, ...]:
        """The step's incoming edges, deduplicated in first-seen order.

        The edges are the declared ``depends_on`` and ``resources`` values plus
        the step's registered optional dependencies that are present in
        ``available`` (a missing optional dependency is not an edge the walk
        honours, so it is dropped).
        """
        edges = [*self.depends_on.values(), *self.resources.values()]
        edges.extend(dep for dep in self.optional_dependencies() if dep in available)

        seen: set[str] = set()
        ordered: list[str] = []
        for dep in edges:
            if dep not in seen:
                seen.add(dep)
                ordered.append(dep)
        return tuple(ordered)

    def state(self) -> StepState:
        """The step's serializable provenance (its ``run`` excluded)."""
        return StepState(
            name=self.name,
            group=self.group,
            artifacts=[artifact.prefix for artifact in self.artifacts],
            depends_on=dict(self.depends_on),
            outputs=dict(self.outputs),
            resources=dict(self.resources),
        )


class InputStep(Step):
    """The endpoint a ``RunInput("name")`` parameter resolves to.

    When ``source`` names a step that ran this session, that step's result is
    used directly. Otherwise the endpoint loads its input from ``source``'s
    declared artifact, else from the bound ``value``. A ``source`` that failed
    refuses the input. The default ``run`` is the identity loader, for inputs
    that are already the value a consumer wants. A composed workflow can order
    an endpoint after several producers with ``depends_on``; those edges are
    walked by the graph but never feed the endpoint's value.
    """

    source: str | None = None
    value: Any = None
    run: Callable = Field(exclude=True, default=_identity)

    def optional_dependencies(self) -> tuple[str, ...]:
        return (self.source,) if self.source is not None else ()


class RunManifest(BaseModel):
    """The declarative description of one pipeline run.

    ``steps`` is optional: ``--describe`` emits an envelope with no steps (the
    eligible set is whatever the pipeline registers), ``--dry-run`` emits the
    full step list in dependency order. ``steps[i]`` is a ``Step`` dump **with
    ``steps[i].task.settings`` omitted**; the root ``task.settings`` is the one
    authoritative copy, re-injected into every step on load.

    ``pipeline`` names the top-level pipeline (or is ``None`` for a command that
    runs no pipeline); ``pipeline is None`` **iff** ``runnable`` is ``False``.
    """

    model_config = ConfigDict(extra="ignore")

    pipeline: str | None
    task: Task
    steps: list[Step] | None = None
    overwrite: bool = False
    runnable: bool = True

    @model_validator(mode="after")
    def _runnable_iff_pipeline(self) -> RunManifest:
        if (self.pipeline is None) != (not self.runnable):
            raise ValueError("runnable must be False exactly when pipeline is None")
        return self

    @classmethod
    def build(
        cls,
        pipeline: str | None,
        task: Task,
        *,
        steps: list[dict[str, Any]] | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """The one authoritative run envelope, as a JSON-ready dict.

        ``pipeline`` names the top-level pipeline, or is ``None`` for a command
        that runs no pipeline (``runnable`` is then ``False``). ``steps`` is
        ``None`` for a description and the dependency-ordered step dumps for a
        dry-run manifest; each dump already omits ``task.settings`` (the root
        task is authoritative, see :meth:`dump_step`).
        """
        return {
            "pipeline": pipeline,
            "task": task.model_dump(mode="json"),
            "steps": steps,
            "overwrite": overwrite,
            "runnable": pipeline is not None,
        }

    @staticmethod
    def dump_step(step: Step) -> dict[str, Any]:
        """A step dump with ``task.settings`` omitted (the root is canonical)."""
        return step.model_dump(mode="json", exclude={"task": {"settings"}})

    @staticmethod
    def reinject_settings(payload: dict[str, Any], task: Task) -> dict[str, Any]:
        """Re-inject the root ``task.settings`` into every ``steps[i].task``.

        The inverse of :meth:`dump_step`; a dump that names no steps is
        returned unchanged.
        """
        raw_steps = payload.get("steps")
        if raw_steps is None:
            return payload
        return {
            **payload,
            "steps": [
                {**step, "task": {**step["task"], "settings": task.settings}}
                for step in raw_steps
            ],
        }
