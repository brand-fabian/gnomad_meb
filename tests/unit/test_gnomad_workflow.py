"""Tests for the composed whole-pipeline workflow.

``gnomad_meb.lib.gnomad_workflow.GnomadWorkflowPipeline`` is a composite
pipeline: constructing it over a VDS handle instantiates every ported stage and
wires the cross-stage edges the stages cannot declare themselves. These tests
pin that structure through the central :class:`StepRegistry` without touching
hail.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from gnomad_meb.cmd.task import GnomadWorkflowRun, Task
from gnomad_meb.config import Settings
from gnomad_meb.config.settings import (
    AnnotationConfig,
    AnnotationResources,
    GwasConfig,
    Resources,
    SampleQcConfig,
    VariantQcConfig,
    VepAnnotationResources,
    VqsrConfig,
)
from gnomad_meb.lib import GnomadWorkflowPipeline
from gnomad_meb.lib.vds import VariantDataset
from gnomad_meb.pipeline import PipelineFactory, PipelineRunner, StepRegistry
from gnomad_meb.pipeline.execution import DependencyFailed
from gnomad_meb.pipeline.status import StepStatus

OUTPUT_BASE = "/out"
VDS_NAME = "vds"
VDS = f"{OUTPUT_BASE}/{VDS_NAME}.metadata.json"


@pytest.fixture(autouse=True)
def _stubbed_vds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the composed workflow build hail-free (ADR-0002).

    ``prepare`` seeds a bare VDS metadata so the downstream stages can enumerate
    the splits; that would mkdir the output base and derive the intervals through
    hail. Stub both ends: the seed becomes a no-op metadata write, and every
    stage that opens the run's handle sees a fixed two-split dataset.
    """
    handle = SimpleNamespace(
        params=SimpleNamespace(
            tasks=(
                SimpleNamespace(name="left"),
                SimpleNamespace(name="right"),
            )
        )
    )
    bare = SimpleNamespace(write_metadata=lambda *a, **k: None)
    monkeypatch.setattr(
        VariantDataset, "open", staticmethod(lambda path, *a, **k: handle)
    )
    monkeypatch.setattr(
        VariantDataset, "create_bare", staticmethod(lambda *a, **k: bare)
    )


def _settings() -> Settings:
    return Settings(
        validate_paths=False,
        project="demo",
        checkpoint="/ckpt",
        resources=Resources(
            calling_intervals=Path("/res/calling_intervals.bed"),
            telomeres_and_centromeres=Path("/res/telomeres_and_centromeres.ht"),
            lcr_intervals=Path("/res/lcr_intervals.ht"),
            seg_dup_intervals=Path("/res/seg_dup_intervals.ht"),
            gnomad_pca_loadings=Path("/res/pca_loadings.ht"),
            gnomad_pca_model=Path("/res/pca_model.onnx"),
            gnomad_v4=Path("/res/gnomad_v4.ht"),
            reference=Path("/res/reference.fa"),
        ),
        sample_qc=SampleQcConfig(),
        variant_qc=VariantQcConfig(),
        vqsr=VqsrConfig(),
        annotation=AnnotationConfig(
            resources=AnnotationResources(clinvar_vcf=Path("/res/clinvar.vcf")),
            vep=VepAnnotationResources(
                vep=Path("/res/vep"),
                cache_dir=Path("/res/vep_cache"),
                cadd=Path("/res/cadd.tsv.gz"),
                cadd_indel=Path("/res/cadd_indel.tsv.gz"),
                alpha_missense=Path("/res/alpha_missense.tsv.gz"),
            ),
        ),
        gwas=GwasConfig(metadata_csv=Path("/res/gwas.md.csv")),
    )


def _task() -> Task:
    """The workflow's task: a VDS build over one stub GVCF."""
    return Task(
        settings=_settings(),
        command=GnomadWorkflowRun(
            gvcf=("/data/a.gvcf.gz",), output_base=OUTPUT_BASE, vds_name=VDS_NAME
        ),
    )


def _build() -> GnomadWorkflowPipeline:
    return GnomadWorkflowPipeline(_task())


def test_name_is_stable() -> None:
    assert GnomadWorkflowPipeline.name() == "gnomad-workflow"


def test_composes_every_stage() -> None:
    workflow = _build()

    assert [type(pipeline).name() for pipeline in workflow.composed_pipelines] == [
        "vds",
        "sample_qc",
        "variant_qc_export",
        "vqsr",
        "variant_qc",
        "annotation",
        "filter",
    ]


def test_registers_every_stage_step() -> None:
    _build()
    names = set(StepRegistry().names())
    for expected in (
        "create_variant_dataset",
        "filter_intervals",
        "compute_info_table.left",
        "load_vqsr.left",
        "run_vqsr_pipeline",
        "merge",
    ):
        assert expected in names


def test_derived_qc_tables_order_annotate_after_their_producers() -> None:
    _build()
    resources = StepRegistry().get("annotate_resources.left")
    variant_qc = StepRegistry().get("annotate_variant_qc.left")

    assert resources.resources["sample_qc_ht"] == "sample_qc.compute_hard_filters"
    assert variant_qc.resources["variant_qc_ht"] == "variant_qc.merge"


def test_vqsr_sits_between_export_and_import() -> None:
    workflow = _build()
    registry = StepRegistry()

    # Every cross-stage edge here is ordering-only (``depends_on``): the
    # producers return side-effect payloads, not the directories the consumers
    # read.
    assert registry["vqsr.as_info"].source is None
    assert set(registry["vqsr.as_info"].depends_on.values()) == {
        "variant_qc_export.export_as_info_vcf.left",
        "variant_qc_export.export_as_info_vcf.right",
    }
    assert registry["variant_qc.vcf"].source is None
    assert set(registry["variant_qc.vcf"].depends_on.values()) == {
        "vqsr.run_vqsr_pipeline"
    }

    # The three Variant QC commands share one per-split ``as_info`` directory.
    as_info_dir = "/ckpt/demo/intermediate/variant_qc"
    assert registry["variant_qc_export.as_info"].value == as_info_dir
    assert registry["variant_qc.as_info"].value == as_info_dir
    assert registry["vqsr.as_info"].value == as_info_dir

    vqsr = next(p for p in workflow.composed_pipelines if p.name() == "vqsr")
    assert vqsr.command.as_info == as_info_dir
    assert vqsr.command.output_dir == "output"


def test_orders_every_split_before_its_consumer() -> None:
    """VQSR and the import wait on *every* split, not just the first.

    One ``source`` cannot express N edges, so the per-split orderings ride on the
    endpoints' ordering-only ``depends_on`` edges.
    """
    _build()
    registry = StepRegistry()

    assert set(registry["vqsr.as_info"].depends_on.values()) == {
        "variant_qc_export.export_as_info_vcf.left",
        "variant_qc_export.export_as_info_vcf.right",
    }
    assert set(registry["variant_qc.as_info"].depends_on.values()) == {
        "variant_qc_export.compute_as_info_table.left",
        "variant_qc_export.compute_as_info_table.right",
    }


def test_composed_qc_inputs_resolve_to_their_own_directories() -> None:
    """A producer payload must not replace a QC endpoint's bound directory.

    ``run_vqsr_pipeline`` returns a success bool and each export step declares a
    single-split file artifact; neither is the per-split *directory* consumed, so
    the ordering-only edges plus the default identity loader must keep each
    endpoint at its own value.
    """
    workflow = _build()
    registry = StepRegistry()
    runner = PipelineRunner(_task(), [workflow])

    record = runner.execution_state["vqsr.run_vqsr_pipeline"]
    record.status = StepStatus.FINISHED
    record.result = True

    vcf = registry["variant_qc.vcf"]
    assert runner._get_input_step(vcf)() == vcf.value
    as_info = registry["variant_qc.as_info"]
    assert runner._get_input_step(as_info)() == as_info.value
    vqsr_as_info = registry["vqsr.as_info"]
    assert runner._get_input_step(vqsr_as_info)() == vqsr_as_info.value


def test_a_failed_ordering_producer_refuses_the_input() -> None:
    """An errored ``depends_on`` producer refuses the endpoint's bound value."""
    workflow = _build()
    registry = StepRegistry()
    runner = PipelineRunner(_task(), [workflow])

    runner.execution_state["vqsr.run_vqsr_pipeline"].status = StepStatus.ERROR

    with pytest.raises(DependencyFailed):
        runner._get_input_step(registry["variant_qc.vcf"])()


def test_filter_consumes_the_annotation_matrix_table() -> None:
    _build()
    registry = StepRegistry()

    # The filter stage reads the annotation stage's terminal merged matrix table;
    # the edge is wired by ``source``.
    assert registry["filter.mt"].source == "annotation.merge"


def test_workflow_vqsr_reads_snakemake_from_settings() -> None:
    """The composed VQSR stage gets its cluster args from ``[vqsr].snakemake``.

    The workflow exposes no VQSR-specific flag, so the setting is the only way
    to configure the Snakemake invocation on an HPC cluster.
    """
    settings = _settings()
    settings.vqsr.snakemake = ["--profile", "slurm", "--executor", "slurm"]
    workflow = GnomadWorkflowPipeline(
        Task(
            settings=settings,
            command=GnomadWorkflowRun(
                gvcf=("/data/a.g.vcf.gz",),
                output_base=OUTPUT_BASE,
                vds_name=VDS_NAME,
            ),
        )
    )

    vqsr = next(p for p in workflow.composed_pipelines if p.name() == "vqsr")
    assert vqsr.snakemake_args == ("--profile", "slurm", "--executor", "slurm")


def test_topological_order_places_producers_before_consumers() -> None:
    _build()
    order = [step.id for step in StepRegistry().topological()]

    assert order.index("vds.create_variant_dataset") < order.index(
        "sample_qc.filter_intervals"
    )
    assert order.index("sample_qc.compute_hard_filters") < order.index(
        "annotation.merge"
    )
    assert order.index("variant_qc.merge") < order.index("annotation.merge")
    assert order.index("variant_qc_export.export_as_info_vcf.left") < order.index(
        "vqsr.run_vqsr_pipeline"
    )
    assert order.index("vqsr.run_vqsr_pipeline") < order.index("variant_qc.merge")


def test_factory_round_trips_the_workflow() -> None:
    StepRegistry._reset()
    restored = PipelineFactory.make("gnomad-workflow", _task())

    assert isinstance(restored, GnomadWorkflowPipeline)
    assert len(StepRegistry()) == 76


def test_run_gnomad_workflow_runs_the_composed_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ``gnomad_workflow`` command's body runs the composite through the runner."""
    from gnomad_meb.cmd.gnomad_workflow import run_gnomad_workflow

    ran: dict[str, object] = {}

    class _RecordingRunner:
        def __init__(self, task: Task, pipelines, overwrite: bool = False) -> None:
            ran["pipelines"] = tuple(pipelines)
            ran["overwrite"] = overwrite

        def run(self) -> None:
            ran["ran"] = True

        def has_finished(self) -> bool:
            return True

        def has_error(self) -> bool:
            return False

    monkeypatch.setattr("gnomad_meb.pipeline.PipelineRunner", _RecordingRunner)

    assert run_gnomad_workflow(_task()) is True

    assert ran["ran"] is True
    assert ran["overwrite"] is False
    assert isinstance(ran["pipelines"][0], GnomadWorkflowPipeline)
