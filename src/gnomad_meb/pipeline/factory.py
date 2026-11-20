"""The :class:`PipelineFactory`: resolve a pipeline subclass by name.

The factory is what makes a dumped run restorable: a serialized run records each
pipeline's abstract classmethod ``name``, and the factory re-instantiates the
matching subclass — which re-registers its resources and steps and rebuilds its
loaders from the class (never serialized, ADR-0012). Nothing callable is
embedded in the durable form.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gnomad_meb.cmd.task import Task

    from .pipeline import Pipeline


class PipelineFactory:
    """Instantiate a concrete :class:`Pipeline` subclass from its ``name``."""

    @classmethod
    def _get_concrete_subclasses(cls, base_cls: type) -> list[type[Pipeline]]:
        """Yield every concrete (non-abstract) subclass of ``base_cls``."""
        for sub in base_cls.__subclasses__():
            if not getattr(sub, "__abstractmethods__", None):
                yield sub
            yield from cls._get_concrete_subclasses(sub)

    @classmethod
    def make(cls, name: str, task: Task) -> Pipeline:
        """Instantiate the concrete subclass whose ``name()`` matches ``name``."""
        from .pipeline import Pipeline

        cls._register()

        for pipeline_t in cls._get_concrete_subclasses(Pipeline):
            if pipeline_t.name() == name:
                return pipeline_t(task)
        raise ValueError(f"invalid pipeline name: name = {name}")

    @classmethod
    def list(cls) -> list[str]:
        """Every registered pipeline's name."""
        from .pipeline import Pipeline

        cls._register()

        return [
            pipeline_t.name() for pipeline_t in cls._get_concrete_subclasses(Pipeline)
        ]

    @staticmethod
    def _register() -> None:
        """Import the stage modules so their subclasses register with the factory."""
        for module_name in (
            "gnomad_meb.lib.annotation",
            "gnomad_meb.lib.browser.export",
            "gnomad_meb.lib.filter",
            "gnomad_meb.lib.gnomad_workflow",
            "gnomad_meb.lib.gwas",
            "gnomad_meb.lib.sample_qc",
            "gnomad_meb.lib.variant_qc",
            "gnomad_meb.lib.vds",
        ):
            import_module(module_name)
