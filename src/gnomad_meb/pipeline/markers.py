"""Markers used as parameter *defaults* on ``@checkpoint``-decorated steps.

They are tiny frozen values: :func:`inspect.signature` reads them off a step
method's parameters to derive its edges and inputs. Typed annotations are
optional documentation; the marker default is authoritative.

A step parameter defaulting to ``FromStep("name")`` reads the named sibling
step's output; ``ResourceRef("name")`` reads a declared resource; ``RunInput("name")``
is a caller-supplied value bound with :meth:`~gnomad_meb.pipeline.Pipeline.bind`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FromStep:
    """Read the output of the sibling step called ``name``.

    ``output`` selects one element of the sibling's result when it is a
    tuple/list (a multi-artifact step); it defaults to the first, so
    single-output consumers are unchanged.
    """

    name: str
    output: int = 0


@dataclass(frozen=True)
class ResourceRef:
    """Read the resource called ``name`` (resolved by the registry)."""

    name: str


@dataclass(frozen=True)
class RunInput:
    """A caller-supplied input called ``name``, bound via ``Pipeline.bind``."""

    name: str
