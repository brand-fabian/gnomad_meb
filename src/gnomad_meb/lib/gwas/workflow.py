"""The composed GWAS workflow: export -> regenie -> load in one DAG.

:class:`GwasWorkflowPipeline` is a composite :class:`Pipeline`: it declares no
work of its own. Constructing it resolves the input into the dataset inventory
and instantiates the three GWAS stages over it -- export, run and load -- then
patches the cross-stage edges the stages cannot declare themselves: each run
step's ``dataset`` endpoint feeds from its ``export_regenie`` step, and each load
endpoint from its run step.

Because the response gate is deterministic
(:class:`~gnomad_meb.lib.gwas.metadata.GwasMetadata`), all three stages
build their full step sets at construction, so every step registers in the one
central :class:`~gnomad_meb.pipeline.StepRegistry` and a single
:class:`~gnomad_meb.pipeline.PipelineRunner` executes the whole graph in
dependency order. No registries are reset, so the workflow is
factory-discoverable and durably restorable as one run (ADR-0017), exactly like
:class:`~gnomad_meb.lib.gnomad_workflow.GnomadWorkflowPipeline`.
"""

from __future__ import annotations

from pathlib import Path

from gnomad_meb.cmd.task import (
    GwasDatasetRef,
    GwasExport,
    GwasLoad,
    GwasPipeline,
    GwasWorkflowRun,
    Task,
)
from gnomad_meb.lib.gwas.constants import INTERMEDIATE_PREFIX
from gnomad_meb.lib.gwas.export import GwasExportPipeline
from gnomad_meb.lib.gwas.inventory import GwasInventory
from gnomad_meb.lib.gwas.load import GwasLoadPipeline
from gnomad_meb.lib.gwas.names import DatasetKey
from gnomad_meb.lib.gwas.run import GwasRunPipeline
from gnomad_meb.lib.gwas.utils import region_label
from gnomad_meb.pipeline import Pipeline


class GwasWorkflowException(Exception):
    """A required GWAS workflow input or configuration is missing."""


def workflow_export_base(output_base: str | None, settings) -> str:
    """The workflow's export base: the command override, else the default."""
    if output_base:
        return str(Path(output_base))
    if settings.checkpoint is None:
        raise GwasWorkflowException(
            "can not export plink and regenie files without checkpoint path"
        )
    prefix = INTERMEDIATE_PREFIX.format(project=settings.project)
    return str(Path(settings.checkpoint) / prefix / "dataset")


class GwasWorkflowPipeline(Pipeline):
    """The whole ``export -> regenie -> load`` GWAS workflow over one input."""

    @classmethod
    def name(cls) -> str:
        return "gwas_workflow"

    @property
    def resources(self) -> tuple:
        """No resources of its own; each composed stage declares its own."""
        return ()

    def get_input(self) -> tuple:
        """No endpoints of its own; each composed stage declares its own."""
        return ()

    def prepare(self) -> None:
        """Compose the three stages over the inventory and wire their edges."""
        super().prepare()

        settings = self.settings
        gwas = settings.gwas
        if gwas is None:
            raise GwasWorkflowException("missing gwas configuration section")
        command = self.command
        assert isinstance(command, GwasWorkflowRun), "the workflow run is a GWAS run"

        self._tables = GwasInventory.from_command(settings, command).tables
        export_base = workflow_export_base(command.output_base, settings)
        run_root = command.output_dir or str(gwas.output_dir)
        regions = [region_label(region.kind, region.size) for region in gwas.regions]

        refs: list[GwasDatasetRef] = []
        for table in self._tables:
            for region in regions:
                key = DatasetKey.of(table.label, region)
                refs.append(
                    GwasDatasetRef(
                        label=key.label,
                        dataset=f"{export_base}/{key.segment}",
                        gwas=f"{run_root}/{key.segment}/gwas",
                        burden=f"{run_root}/{key.segment}/burden",
                    )
                )
        self._refs = refs

        self.composed_pipelines = (
            GwasExportPipeline(
                Task(
                    settings=settings,
                    command=GwasExport(
                        mt=command.mt,
                        manifest=command.manifest,
                        include=command.include,
                        exclude=command.exclude,
                        base=command.output_base,
                    ),
                )
            ),
            GwasRunPipeline(
                Task(
                    settings=settings,
                    command=GwasPipeline(
                        output_dir=run_root,
                        snakemake=command.snakemake,
                        datasets=refs,
                    ),
                )
            ),
            GwasLoadPipeline(
                Task(settings=settings, command=GwasLoad(datasets=refs))
            ),
        )

        self._wire_composed()

    def _wire_composed(self) -> None:
        """Patch the cross-stage edges the stages cannot declare themselves.

        Each edge is an :class:`~gnomad_meb.pipeline.InputStep` ``source``: the
        run step's ``dataset`` endpoint is ordered after its ``export_regenie``
        step, and each load endpoint after its run step. The endpoints keep their
        own deterministic values, so a missing optional dependency is logged and
        dropped by the registry without changing the resolved inputs.
        """
        registry = self._step_registry
        for ref in self._refs:
            key = DatasetKey.parse(ref.label)
            export_step = f"gwas_export.{key.step('export_regenie')}"
            run_step = f"gwas.{key.step('run_gwas_pipeline')}"
            registry[f"gwas.dataset.{ref.label}"].source = export_step
            registry[f"gwas_load.dataset.{ref.label}"].source = run_step
            registry[f"gwas_load.gwas.{ref.label}"].source = run_step
            registry[f"gwas_load.burden.{ref.label}"].source = run_step
