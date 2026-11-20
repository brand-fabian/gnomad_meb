"""The central :class:`ResourceRegistry`.

Every resource a pipeline declares is registered here. For each managed
resource the registry returns the :class:`~gnomad_meb.pipeline.spec.Step`
instances that provide it: the step that already produces it (a *derived*
resource whose path is another step's declared artifact), or a load step that
reads the resource in place with its own loader, optionally followed by the
modification steps the registry applies to make the resource conform to the
run. The loader is never serialized.
"""

from __future__ import annotations

import functools
import threading
from itertools import chain
from typing import Any, Protocol, runtime_checkable

from gnomad_meb.cmd.task import PipelineStep, Task
from gnomad_meb.config import Settings, get_config

from .resource import Resource
from .spec import Step
from .step import StepRegistry


class ReferenceGenomeMismatch(ValueError):
    """A Resource's reference genome does not match what the run needs."""


@runtime_checkable
class ResourceModification(Protocol):
    """The registry-owned interface for adapting a resource to the run.

    A modification knows whether it applies to a resource on the run's reference
    genome (``applies``) and, if so, the ordered Step(s) that read the resource's
    ``path`` in place and materialise a conformed checkpoint as the final step's
    output (``steps``). No concrete modification (e.g. liftover) lands now.
    """

    def applies(self, resource: Resource, run_reference: str) -> bool: ...

    def steps(self, resource: Resource, settings: Settings) -> tuple[Step, ...]: ...


class ResourceRegistry:
    """The run's resources, keyed by name and by ``(path, reference_genome)``."""

    _instance: ResourceRegistry | None = None
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

        cfg = get_config()
        self.settings = cfg
        self.run_reference = cfg.reference_genome
        self.modifications: tuple[ResourceModification, ...] = ()
        self.is_initialized = True
        self._resources: dict[tuple[str, str], Resource] = {}
        self._by_name: dict[str, tuple[str, str]] = {}

    def add(self, resource: Resource) -> str:
        """Register ``resource`` and return its canonical name (idempotent)."""
        with self._lock:
            if resource.id not in self._resources:
                self._resources[resource.id] = resource
            self._by_name[resource.name] = resource.id
            return resource.name

    def names(self) -> tuple[str, ...]:
        return tuple(self._by_name)

    def get_resource(self, item: str | tuple[str, str]) -> Resource | None:
        if isinstance(item, tuple):
            return self._resources.get(item)
        identity = self._by_name.get(item)
        if identity is None:
            return None
        return self._resources[identity]

    def __contains__(self, item: str | tuple[str, str]) -> bool:
        if isinstance(item, tuple):
            return item in self._resources
        return item in self._by_name

    def _producer_step(self, resource: Resource) -> Step | None:
        """The registered step whose artifact is this resource's path, if any."""
        step_registry = StepRegistry()
        for step_id in step_registry.steps():
            step = step_registry[step_id]
            if any(
                artifact.path(step.task.settings) == str(resource.path)
                for artifact in step.artifacts
            ):
                return step
        return None

    def _modifications_for(self, resource: Resource) -> list[Step]:
        return list(
            chain.from_iterable(
                mod.steps(resource, self.settings)
                for mod in self.modifications
                if mod.applies(resource, self.run_reference)
            )
        )

    def _load_steps(self, resource: Resource) -> list[Step]:
        """The load step plus any modification steps for ``resource``."""
        load = Step(
            name=f"load-{resource.name}",
            task=Task(
                settings=self.settings,
                command=PipelineStep(name=f"load-{resource.name}", group="resources"),
            ),
            group="resources",
            run=functools.partial(resource.loader, resource.path),
        )
        steps = [load, *self._modifications_for(resource)]
        if len(steps) > 1:
            # The first modification consumes the loaded resource by name.
            steps[1].resources = {resource.name: load.id, **steps[1].resources}
        return steps

    def get(self, name: str, default: list[str] | None = None) -> list[str] | None:
        """Return the ids of the steps that provide the resource called ``name``.

        A derived resource resolves to the single step that already produces its
        path; otherwise a load step (and any modification steps) is registered
        and its ids returned.
        """
        resource = self.get_resource(name)
        if resource is None:
            return default

        producer = self._producer_step(resource)
        if producer is not None:
            return [producer.id]

        step_registry = StepRegistry()
        return [step_registry.add(step) for step in self._load_steps(resource)]

    def __getitem__(self, name: str) -> list[str]:
        steps = self.get(name)
        if steps is None:
            raise KeyError(f"unknown resource key: key = {name}")
        return steps

    def current_state(self) -> dict[str, Any]:
        """A serializable provenance record of every registered resource."""
        return {
            name: resource.model_dump(mode="json")
            for name in self._by_name
            if (resource := self.get_resource(name)) is not None
        }
