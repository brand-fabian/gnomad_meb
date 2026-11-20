"""The composed whole-pipeline workflow.

:class:`GnomadWorkflowPipeline` is a composite :class:`Pipeline`: it declares no
work of its own. Constructing it instantiates every ported stage over one shared
VDS handle -- the VDS build, Sample QC, Variant QC export, VQSR, Variant QC
import, Annotation and Filter -- and patches in the cross-stage edges the stages cannot
declare themselves: the shared per-split ``as_info`` directory the export writes
and VQSR/import read, and VQSR's output feeding the import.

Every stage registers its steps in the central
:class:`~gnomad_meb.pipeline.StepRegistry`, so
:class:`~gnomad_meb.pipeline.PipelineRunner` executes the whole graph in
dependency order. Shared resources and the derived QC tables therefore resolve
once, across the composed graph, rather than once per stage.
"""

from __future__ import annotations

import os
from functools import cached_property

from gnomad_meb.cmd.task import (
    AnnotationRun,
    FilterRun,
    SampleQcRun,
    Task,
    VariantQcExport,
    VariantQcLoadVqsr,
    VdsCreate,
    VqsrRun,
)
from gnomad_meb.lib.annotation import AnnotationPipeline
from gnomad_meb.lib.filter import FilterPipeline
from gnomad_meb.lib.sample_qc import SampleQcPipeline
from gnomad_meb.lib.variant_qc.export import VariantQcExportPipeline
from gnomad_meb.lib.variant_qc.load import VariantQcImportPipeline
from gnomad_meb.lib.variant_qc.vqsr import VqsrPipeline
from gnomad_meb.lib.vds import VariantDataset
from gnomad_meb.lib.vds.pipeline import VdsCreatePipeline
from gnomad_meb.pipeline import Pipeline


class GnomadWorkflowPipeline(Pipeline):
    """A complete gnomAD workflow, from GVCFs to an annotated matrix table.

    This pipeline aggregates the ported stages to run an end-to-end QC and
    annotation pipeline on one shared VDS. It owns no steps; the graph is the
    union of its composed stages' steps plus the edges wired in
    :meth:`prepare`.
    """

    @classmethod
    def name(cls) -> str:
        return "gnomad-workflow"

    @property
    def resources(self) -> tuple:
        """No resources of its own; each composed stage declares its own."""
        return ()

    def get_input(self) -> tuple:
        """No endpoints of its own; each composed stage declares its own."""
        return ()

    # --- derived inputs ------------------------------------------------------

    @cached_property
    def project_dir(self) -> str:
        """The run's project directory under the checkpoint (or scratch) root."""
        settings = self.settings
        return f"{settings.checkpoint or settings.scratch_dir}/{settings.project}"

    @cached_property
    def as_info_dir(self) -> str:
        """The shared per-split ``as_info`` directory.

        All three Variant QC commands take a *directory*, not a file: the export
        writes ``task-<task>/as_info.vcf.bgz`` here, VQSR reads those per-split
        VCFs from the same directory, and the import reads
        ``task-<task>/as_info.ht`` from it too (the export's per-task hail
        checkpoints also live here).
        """
        return f"{self.project_dir}/intermediate/variant_qc"

    @cached_property
    def vqsr_output_dir(self) -> str:
        """The VQSR output directory holding per-split ``task-<task>/vqsr.vcf.bgz``."""
        vqsr = self.settings.vqsr
        return str(vqsr.output_dir) if vqsr is not None else "output"

    @cached_property
    def calling_intervals(self) -> str:
        """The global picard intervals VQSR's per-task lists are derived from."""
        resources = self.settings.resources
        return str(resources.calling_intervals) if resources is not None else ""

    @cached_property
    def vds_handle(self) -> str:
        """The metadata handle the VDS build writes (its declared artifact).

        Passed to the downstream stages as the fallback value of their ``vds``
        endpoint; at run time they consume the VDS-build step's result or
        artifact, so this is only the value used when the build is absent.
        """
        command = self.command
        assert isinstance(command, VdsCreate), "the workflow run is a VDS build"
        return os.path.abspath(
            f"{command.output_base}/{command.vds_name}.metadata.json"
        )

    # --- composition ---------------------------------------------------------

    def prepare(self) -> None:
        """Compose the stages, then reconcile their cross-stage edges.

        Called once, before the composite's (empty) step set is discovered. Each
        stage constructed here registers its own resources and steps in the two
        central registries, so by the time the runner walks them the whole graph
        is a single DAG.

        The composite's command states the VDS-build options; the downstream
        stages derive everything else from the resolved settings.
        """
        super().prepare()  # the composite declares no @checkpoint methods

        settings = self.settings
        as_info = self.as_info_dir

        # Seed the metadata the downstream stages read at construction time, so
        # they can enumerate the VDS split tasks before the build step runs.
        # This is a bare handle: metadata only, no split data (see
        # ``VariantDataset.write_metadata``).
        VariantDataset.create_bare(
            os.path.dirname(self.vds_handle),
            os.path.basename(self.vds_handle).removesuffix(".metadata.json"),
            (),
            reference_genome=self.command.reference_genome,
            n_splits=self.command.n_splits,
            interval_size=self.command.interval_size,
            overwrite=self.command.overwrite,
        ).write_metadata()

        self.composed_pipelines = (
            VdsCreatePipeline(self.task),
            SampleQcPipeline(
                Task(settings=settings, command=SampleQcRun(vds=self.vds_handle))
            ),
            VariantQcExportPipeline(
                Task(
                    settings=settings,
                    command=VariantQcExport(
                        vds=self.vds_handle, as_info=as_info, quasi_info=as_info
                    ),
                )
            ),
            VqsrPipeline(
                Task(
                    settings=settings,
                    command=VqsrRun(
                        vds=self.vds_handle,
                        as_info=as_info,
                        output_dir=self.vqsr_output_dir,
                        calling_intervals=self.calling_intervals,
                    ),
                )
            ),
            VariantQcImportPipeline(
                Task(
                    settings=settings,
                    command=VariantQcLoadVqsr(
                        vds=self.vds_handle, vcf=self.vqsr_output_dir, as_info=as_info
                    ),
                )
            ),
            AnnotationPipeline(
                Task(settings=settings, command=AnnotationRun(vds=self.vds_handle))
            ),
            FilterPipeline(Task(settings=settings, command=FilterRun())),
        )

        self._wire_composed()

    def _wire_composed(self) -> None:
        """Patch the edges the stages cannot declare as parameter markers.

        Each cross-stage dependency is an ordering-only ``depends_on`` edge: the
        producers return side-effect payloads (VQSR a success bool, the export a
        single-split file artifact), not the directories the consumers read, so
        the endpoints keep their own bound values. The VQSR stage reads every
        split's exported ``as_info`` VCFs, and the import reads every split's
        VQSR VCFs and ``as_info`` hail tables.
        """
        registry = self._step_registry

        export = VariantQcExportPipeline.name()
        as_info_exports = tuple(
            step_id
            for step_id in registry.steps()
            if step_id.startswith(f"{export}.export_as_info_vcf.")
        )
        as_info_tables = tuple(
            step_id
            for step_id in registry.steps()
            if step_id.startswith(f"{export}.compute_as_info_table.")
        )
        vqsr_step = f"{VqsrPipeline.name()}.run_vqsr_pipeline"

        # VQSR consumes every split's exported as_info VCF.
        registry["vqsr.as_info"].depends_on.update(
            {step_id: step_id for step_id in as_info_exports}
        )
        # The import consumes every split's VQSR VCFs (one aggregated step) and
        # every split's exported as_info table.
        registry["variant_qc.vcf"].depends_on["vqsr"] = vqsr_step
        registry["variant_qc.as_info"].depends_on.update(
            {step_id: step_id for step_id in as_info_tables}
        )

        # The filter stage consumes the annotation stage's terminal merged matrix
        # table. The edge is wired by ``source``; the endpoint reads that step's
        # declared artifact with ``hl.read_matrix_table`` when it is not in memory.
        registry["filter.mt"].source = "annotation.merge"
