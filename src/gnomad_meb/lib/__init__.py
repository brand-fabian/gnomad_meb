"""The library: checkpoints, utilities, and every ported pipeline stage.

The stage classes are re-exported lazily (PEP 562). Importing this package does
**not** eagerly import every stage: that would drag hail and ``gnomad_meb.config``
into any ``import gnomad_meb.lib.*`` (notably ``config.settings`` importing the
hail-free ``checkpoint.status``), causing a cycle. The eager registration the
:class:`~gnomad_meb.pipeline.PipelineFactory` relies on is triggered explicitly by
:meth:`~gnomad_meb.pipeline.factory.PipelineFactory._register`, which imports the
stage subpackages directly.
"""

from __future__ import annotations

from importlib import import_module

#: name → the submodule that defines it. Resolved on attribute access.
_EXPORTS = {
    "AnnotationPipeline": "gnomad_meb.lib.annotation",
    "BrowserExportPipeline": "gnomad_meb.lib.browser",
    "FilterPipeline": "gnomad_meb.lib.filter",
    "GnomadWorkflowPipeline": "gnomad_meb.lib.gnomad_workflow",
    "GwasExportPipeline": "gnomad_meb.lib.gwas",
    "GwasInventory": "gnomad_meb.lib.gwas",
    "GwasLoadPipeline": "gnomad_meb.lib.gwas",
    "GwasRunPipeline": "gnomad_meb.lib.gwas",
    "GwasTable": "gnomad_meb.lib.gwas",
    "GwasWorkflowPipeline": "gnomad_meb.lib.gwas",
    "RegenieDataset": "gnomad_meb.lib.gwas",
    "SampleQcPipeline": "gnomad_meb.lib.sample_qc",
    "VariantQcExportPipeline": "gnomad_meb.lib.variant_qc",
    "VariantQcImportPipeline": "gnomad_meb.lib.variant_qc",
    "VqsrPipeline": "gnomad_meb.lib.variant_qc",
    "VariantDataset": "gnomad_meb.lib.vds",
    "VariantDatasetParameters": "gnomad_meb.lib.vds",
    "VariantDatasetTaskIntervals": "gnomad_meb.lib.vds",
    "VdsCreatePipeline": "gnomad_meb.lib.vds",
    "VdsLocation": "gnomad_meb.lib.vds",
    "checkpoint": "gnomad_meb.lib.checkpoint",
    "glob": "gnomad_meb.lib.utils",
}


def __getattr__(name: str):
    """Resolve the lazily re-exported names (PEP 562)."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module_name), name)


__all__ = list(_EXPORTS)
