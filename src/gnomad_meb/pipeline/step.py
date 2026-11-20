"""The central :class:`StepRegistry`.

Every step a pipeline instantiates is registered here; the
:class:`~gnomad_meb.pipeline.PipelineRunner` consumes the registry in
dependency order.
"""

from __future__ import annotations

import graphlib
import threading
from traceback import format_exc
from typing import Any

from gnomad_meb.logging import get_logger

from .spec import Step

logger = get_logger()


class StepRegistry:
    """A process-global index of every step registered by the pipelines."""

    _instance: StepRegistry | None = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    @classmethod
    def _reset(cls) -> None:
        """Drop the current registry instance (used between tests)."""
        with cls._lock:
            cls._instance = None

    def __init__(self) -> None:
        if getattr(self, "is_initialized", False):
            return

        self.is_initialized = True
        self._steps: dict[str, Step] = {}
        self._by_name: dict[str, str] = {}

    def add(self, step: Step) -> str:
        """Register ``step`` and return its id (idempotent).

        Two pipelines may each declare a step of the same local name (for
        example an input endpoint ``vds``); they are distinct by their
        group-prefixed id, and the bare-name index keeps the first.
        """
        with self._lock:
            if step.id in self._steps:
                return step.id
            self._steps[step.id] = step
            self._by_name.setdefault(step.name, step.id)
            return step.id

    def __contains__(self, item: str) -> bool:
        return item in self._steps or item in self._by_name

    def names(self) -> tuple[str, ...]:
        return tuple(self._by_name)

    def steps(self) -> tuple[str, ...]:
        return tuple(self._steps)

    def get(self, key: str, default: Step | None = None) -> Step | None:
        if key in self._steps:
            return self._steps[key]
        if key in self._by_name:
            return self._steps[self._by_name[key]]
        return default

    def __getitem__(self, key: str) -> Step:
        value = self.get(key)
        if value is None:
            raise KeyError(f"invalid step key: key = {key}")
        return value

    def _dependencies(self, step: Step) -> list[str]:
        """A step's edges: its declared deps plus any registered optional deps.

        An optional dependency (an :class:`InputStep`'s ``source``) is only an
        edge when the named step is registered; a missing one is logged rather
        than silently dropped.
        """
        optional = step.optional_dependencies()
        missing = [dep for dep in optional if dep not in self]
        if missing:
            logger.warning(
                f"missing optional dependencies: step = {step.id}, "
                f"missing = {missing}"
            )
        return list(step.incoming_edges(self))

    def dependencies_of(self, step_id: str) -> set[str]:
        """The reverse transitive closure of a step's incoming edges.

        Every producer reachable from ``step_id`` -- over declared
        ``depends_on``, ``resources``, and registered optional edges -- is
        returned, excluding ``step_id`` itself. Edges are taken from
        :meth:`_dependencies` so optional-edge presence is honoured exactly as
        the topological walk honours it.
        """
        closure: set[str] = set()
        frontier = list(self._dependencies(self[step_id]))
        while frontier:
            dep = frontier.pop()
            if dep in closure or dep == step_id:
                continue
            closure.add(dep)
            frontier.extend(self._dependencies(self[dep]))
        return closure

    def dependents_of(self, step_id: str) -> set[str]:
        """Every consumer reachable from ``step_id`` (the forward closure).

        The reverse of :meth:`dependencies_of`: the steps whose incoming edges
        reach ``step_id``, transitively, excluding ``step_id`` itself. Used by
        the lineage rule (ticket 07), where an absent producer invalidates its
        downstream consumers. Edges are taken from :meth:`_dependencies`, so a
        registered optional edge is walked exactly as the topological order
        honours it.
        """
        dependents: dict[str, set[str]] = {step_id: set() for step_id in self._steps}
        for candidate in self._steps:
            for producer in self._dependencies(self[candidate]):
                dependents.setdefault(producer, set()).add(candidate)

        closure: set[str] = set()
        frontier = list(dependents.get(step_id, ()))
        while frontier:
            consumer = frontier.pop()
            if consumer in closure or consumer == step_id:
                continue
            closure.add(consumer)
            frontier.extend(dependents.get(consumer, ()))
        return closure

    def topological(self) -> list[Step]:
        """Return every registered step in dependency order."""
        sorter: graphlib.TopologicalSorter = graphlib.TopologicalSorter()
        for step in self._steps.values():
            sorter.add(step.id, *self._dependencies(step))
        try:
            order = list(sorter.static_order())
        except graphlib.CycleError as exc:
            logger.error(f"cycle in registered steps: error = {exc}")
            logger.error(format_exc())
            raise ValueError("steps can not be executed") from exc
        return [self[step_id] for step_id in order]

    def __len__(self) -> int:
        return len(self._steps)

    def current_state(self) -> dict[str, Any]:
        """A serializable provenance record of every registered step."""
        return {
            step_id: step.state().model_dump(mode="json")
            for step_id, step in self._steps.items()
        }
