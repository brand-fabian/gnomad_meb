"""The browser-export stage package.

:class:`~gnomad_meb.lib.browser.export.BrowserExportPipeline` is re-exported
lazily (PEP 562), so importing :mod:`gnomad_meb.lib.browser.constants` (or any
other hail-free helper) does not import the pipeline or Hail. The eager import
:class:`~gnomad_meb.pipeline.factory.PipelineFactory` relies on is triggered by
:meth:`~gnomad_meb.pipeline.factory.PipelineFactory._register`, which imports the
``export`` submodule directly.
"""

from __future__ import annotations

__all__ = ["BrowserExportPipeline"]


def __getattr__(name: str):
    """Resolve the lazily re-exported pipeline (PEP 562)."""
    if name == "BrowserExportPipeline":
        from gnomad_meb.lib.browser.export import BrowserExportPipeline

        return BrowserExportPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
