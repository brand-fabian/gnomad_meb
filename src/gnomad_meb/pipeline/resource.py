"""Resources declared by each pipeline and resolved by the registry.

A resource is *referenced, not copied*: a pipeline declares the resources it
depends on as :class:`Resource` objects (``name``, ``path``, ``reference_genome``,
``loader``). An unmodified resource is read from its declared ``path`` in place
by its own ``loader`` and schedules no checkpoint, so a multi-terabyte table is
never duplicated into the checkpoint root. Only when the **registry** applies a
modification that makes a resource conform to the run is it materialised as a
checkpoint; a resource carries no modification of its own.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from gnomad_meb.constants import REFERENCE_GENOMES

T = TypeVar("T")


class Resource(BaseModel, Generic[T]):
    """An external dataset a run depends on.

    ``loader`` is a function that takes ``path`` as input and returns the actual
    resource instance consumed by pipeline steps. It is never serialized.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    name: str
    path: Path | str
    loader: Callable[[Path | str], T] = Field(exclude=True)
    reference_genome: REFERENCE_GENOMES = "GRCh38"

    @property
    def id(self) -> tuple[str, str]:
        """Resources are identified by their path and reference genome."""
        return (str(self.path), self.reference_genome)
