"""The ``Pipeline`` base class: decorated methods are the DAG declaration.

A stage subclasses :class:`Pipeline` and declares its work as
``@checkpoint``-decorated methods. The parameter *defaults* are the wire: a
``FromStep`` reads a sibling step, a ``ResourceRef`` reads a declared resource,
and a ``RunInput`` reads an :class:`~gnomad_meb.pipeline.spec.InputStep`
endpoint. Constructing a pipeline registers its resources and steps in the two
central registries; :class:`~gnomad_meb.pipeline.PipelineRunner` then executes
the whole registry in dependency order.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from gnomad_meb.cmd.task import PipelineStep, Task
from gnomad_meb.config import Settings, get_config
from gnomad_meb.lib.checkpoint import CheckpointedCallable
from gnomad_meb.logging import get_logger

from .markers import FromStep, ResourceRef, RunInput
from .registry import ReferenceGenomeMismatch, ResourceRegistry
from .resource import Resource
from .spec import InputStep, Step
from .step import StepRegistry

_Method = Callable[..., object]

#: One finalized method: ``(name, callable, task)``. ``task`` is the step's
#: per-task identity (``None`` for a whole-stage or merge step); ``prepare``
#: produces the ordered list and discovery consumes it.
_PreparedMethod = tuple[str, _Method, str | None]

logger = get_logger()


def _firstlineno(method: _Method) -> int:
    """The source line of ``method``'s original definition.

    A :class:`CheckpointedCallable` exposes the original via ``__wrapped__``
    (``functools.update_wrapper``); every wrapper shares one code object, so the
    wrapped function carries the definition-order line number.
    """
    target = getattr(method, "__wrapped__", method)
    code = getattr(target, "__code__", None)
    if code is None:
        # A partial (generated step) has no code object; follow to its func.
        target = getattr(target, "func", target)
        code = getattr(target, "__code__", None)
    return code.co_firstlineno if code is not None else 0


def _decorated_methods(cls: type) -> list[tuple[str, _Method]]:
    """Every ``@checkpoint``-decorated method of ``cls``, in definition order.

    A decorated method is a :class:`CheckpointedCallable` instance, not a plain
    function, so it is discovered by ``isinstance`` rather than
    ``inspect.isfunction``. Accessing it through the class returns the callable
    itself (``__get__`` with no instance), so the definition line is available.
    """
    decorated = [
        (name, member)
        for name, member in inspect.getmembers(cls)
        if isinstance(member, CheckpointedCallable)
    ]
    decorated.sort(key=lambda item: _firstlineno(item[1]))
    return decorated


class Pipeline(ABC):
    """A stage: a set of ``@checkpoint``-decorated methods and its resources.

    On construction the pipeline registers the resources it declares and the
    steps it derives (input endpoints, decorated methods, and any extra steps)
    into the central registries. It holds only the run's provenance metadata
    otherwise.

    ``name``, ``resources`` and ``get_input`` are abstract: a concrete stage
    states its own name (also the step-group prefix), the resources it declares,
    and the input endpoints it provides — an empty tuple where it has none.
    """

    def __init__(self, task: Task) -> None:
        self.task = task
        self.settings: Settings = (
            task.settings if task.settings is not None else get_config()
        )
        self.command = task.command

        self._resource_registry = ResourceRegistry()
        self._step_registry = StepRegistry()

        self._check_reference_genomes()
        self._resource_ids = self._discover_resources()
        self.prepare()
        self._step_ids = self._discover_steps()

    # --- declaration hooks ---------------------------------------------------

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """The pipeline's name; also the step-group prefix."""

    @property
    @abstractmethod
    def resources(self) -> tuple[Resource, ...]:
        """The Resource objects this pipeline declares (empty when none)."""

    @abstractmethod
    def get_input(self) -> tuple[InputStep, ...]:
        """The :class:`InputStep` endpoints this pipeline provides (empty when none)."""

    def prepare(self) -> None:
        """Finalize the pipeline's callable list from its inputs.

        Called once, after the resources are registered and before the steps
        are discovered. It sets :attr:`_methods`, the single ordered list that
        discovery consumes: ``(name, callable, task)`` triples, where ``task``
        is the per-task identity (``None`` for a whole-stage or merge
        step). :meth:`_make_step` carries that identity into the step command's
        ``args`` -- the step's declarative, JSON-serializable input slot -- so a
        generated per-task step records which split task it belongs to.

        The default is exactly the ``@checkpoint``-decorated methods, in
        definition order. A stage that derives methods from its inputs (for
        example one per ``VariantDataset`` split task) overrides this, calling
        ``super().prepare()`` to keep the decorated ones and then appending its
        generated methods, and must produce the complete, deterministic list.
        Reading an input here and failing loudly when it is absent is the
        contract: the same inputs must yield the same step set on every
        construction, so a serialized run reconstructs to the same steps.
        """
        self._methods: list[_PreparedMethod] = [
            (name, getattr(self, name), None)
            for name, _ in _decorated_methods(type(self))
        ]

    def _extra_steps(self) -> list[Step]:
        """Steps not derived from decorated methods (default: none)."""
        return []

    def _artifacts(self, name: str, method: _Method) -> tuple[Any, ...]:
        """The artifacts a step declares; the decorator's, unless overridden."""
        return method.spec.artifacts

    # --- registration --------------------------------------------------------

    def _check_reference_genomes(self) -> None:
        run_reference = self.settings.reference_genome
        for resource in self.resources:
            if resource.reference_genome == run_reference:
                continue
            if not any(
                mod.applies(resource, run_reference)
                for mod in self._resource_registry.modifications
            ):
                raise ReferenceGenomeMismatch(
                    f"resource {resource.name!r} is on {resource.reference_genome} "
                    f"but the run uses {run_reference}"
                )

    def _discover_resources(self) -> list[str]:
        return [self._resource_registry.add(resource) for resource in self.resources]

    def _make_step(
        self, name: str, method: _Method, task: str | None, *, group: str | None
    ) -> Step:
        """Project one finalized ``method`` into a :class:`Step` (edges empty).

        ``method`` is the finalized callable from :meth:`prepare` -- a bound
        method today, possibly a ``functools.partial`` of a plain body later --
        and is used directly as the step's ``run`` (it may have no attribute on
        ``self``). ``task`` is the step's per-task identity, carried into the
        step command's ``args``: the declarative home for a step's
        JSON-serializable inputs. It is ``None`` for whole-stage and merge
        steps, leaving ``args`` empty.
        """
        args = {"task": task} if task is not None else {}
        return Step(
            name=name,
            task=Task(
                settings=self.settings,
                command=PipelineStep(name=name, group=group, args=args),
            ),
            artifacts=self._artifacts(name, method),
            group=group,
            run=method,
            checkpoint_callable=(
                method if isinstance(method, CheckpointedCallable) else None
            ),
        )

    def _wire_step(self, step: Step, method: _Method) -> None:
        """Fill a step's ``depends_on`` / ``resources`` from its parameter markers.

        Edges come from two places. A plain method carries its markers as
        parameter *defaults*, read off ``inspect.signature``. A generated method
        is a ``functools.partial`` of a plain body, and its edges are the
        partial's bound keywords: ``inspect.signature`` surfaces a *named* bound
        keyword (``mt=FromStep("…")``) but cannot expose a terminal ``**parts``
        merge whose per-task kwargs are the edges, so the keywords are read
        directly too. A non-marker keyword (``task="left"``) is ignored.
        """
        defaults = {
            parameter.name: parameter.default
            for parameter in inspect.signature(method).parameters.values()
        }
        # A generated method is a ``functools.partial`` of a plain body; its
        # edges are the partial's bound keywords, exposed as ``method.keywords``
        # (a first-class attribute of the ``CheckpointedCallable``).
        keywords = getattr(method, "keywords", None)
        if keywords:
            defaults.update(keywords)

        for name, default in defaults.items():
            if name == "self":
                continue
            if isinstance(default, FromStep | RunInput):
                group = type(self).name() or None
                key = f"{group}.{default.name}" if group else default.name
                target = (
                    self._step_registry.get(key) or self._step_registry[default.name]
                )
                step.depends_on[name] = target.id
                if isinstance(default, FromStep):
                    step.outputs[name] = default.output
            elif isinstance(default, ResourceRef):
                if default.name in self._resource_registry:
                    step.resources[name] = self._resource_registry[default.name][-1]

    def _discover_steps(self) -> list[str]:
        step_ids: list[str] = []

        for input_step in self.get_input():
            step_ids.append(self._step_registry.add(input_step))

        built: list[tuple[Step, _Method]] = []
        for name, method, task in self._methods:
            step = self._make_step(name, method, task, group=type(self).name() or None)
            self._step_registry.add(step)
            step_ids.append(step.id)
            built.append((step, method))

        for step, method in built:
            self._wire_step(step, method)

        for step in self._extra_steps():
            step_ids.append(self._step_registry.add(step))

        logger.info(
            f"registered pipeline: pipeline = {type(self).name()}, "
            f"n_resources = {len(self._resource_ids)}, n_steps = {len(step_ids)}"
        )
        return step_ids

    # --- provenance ----------------------------------------------------------

    def current_state(self) -> dict[str, Any]:
        """The pipeline's provenance metadata (name, command, settings, steps)."""
        return {
            "pipeline": type(self).name(),
            "command": self.command.model_dump(mode="json"),
            "settings": self.settings.model_dump(mode="json"),
            "resources": list(self._resource_ids),
            "steps": [self._step_registry[step_id].id for step_id in self._step_ids],
        }
