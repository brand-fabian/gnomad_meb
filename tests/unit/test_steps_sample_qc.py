"""Declared-DAG tests for the Sample QC pipeline.

``SampleQcPipeline`` declares its steps as ``@checkpoint``-decorated methods;
constructing it registers the steps in the central
:class:`~gnomad_meb.pipeline.StepRegistry`. These tests pin down the *declared*
graph — step names and order, ``depends_on``, per-step resources, the declared
resources, and artifact paths — without running any hail.
"""

from __future__ import annotations

from pathlib import Path

from gnomad_meb.cmd.task import SampleQcRun, Task
from gnomad_meb.config import Settings
from gnomad_meb.config.settings import Resources, SampleQcConfig
from gnomad_meb.lib.checkpoint import CHECKPOINT_TYPE
from gnomad_meb.lib.sample_qc import SampleQcPipeline
from gnomad_meb.pipeline import StepRegistry

WORK_STEPS = (
    "filter_intervals",
    "base_sample_qc",
    "get_annotated_sample_qc",
    "compute_sex_inference_vds",
    "get_infer_sex",
    "densify_pca_sites",
    "get_contamination_estimate",
    "get_king_estimate",
    "get_relatedness2",
    "get_relatedness",
    "apply_ancestry_classification",
    "get_combined_sample_qc_table",
    "compute_hard_filters",
)

WORK_EDGES = {
    "base_sample_qc": {"vds": "filter_intervals"},
    "get_annotated_sample_qc": {
        "vds": "filter_intervals",
        "sample_qc_ht": "base_sample_qc",
    },
    "compute_sex_inference_vds": {"vds": "filter_intervals"},
    "get_infer_sex": {"vds": "compute_sex_inference_vds"},
    "get_contamination_estimate": {"mt": "densify_pca_sites"},
    "densify_pca_sites": {"vds": "filter_intervals"},
    "get_king_estimate": {"mt": "densify_pca_sites"},
    "get_relatedness2": {
        "mt": "densify_pca_sites",
        "king_mt": "get_king_estimate",
    },
    "get_relatedness": {"mt": "densify_pca_sites"},
    "apply_ancestry_classification": {"mt": "densify_pca_sites"},
    "get_combined_sample_qc_table": {
        "sample_qc_ht": "get_annotated_sample_qc",
        "ploidy_ht": "get_infer_sex",
        "charr_ht": "get_contamination_estimate",
        "ancestry_ht": "apply_ancestry_classification",
    },
    "compute_hard_filters": {
        "sample_qc_ht": "get_combined_sample_qc_table",
        "king_mt": "get_king_estimate",
    },
}

WORK_RESOURCES = {
    "filter_intervals": (
        "telomeres_and_centromeres",
        "lcr_intervals",
        "seg_dup_intervals",
    ),
    "compute_sex_inference_vds": (
        "lcr_intervals",
        "seg_dup_intervals",
    ),
    "get_infer_sex": ("calling_intervals",),
    "get_contamination_estimate": ("gnomad_v4_sites",),
    "densify_pca_sites": ("pca_loadings_ht",),
    "get_relatedness": ("pca_loadings_ht",),
    "apply_ancestry_classification": (
        "fit",
        "pca_loadings_ht",
    ),
}

WORK_ARTIFACTS = {
    "filter_intervals": (
        "{project}/intermediate/sample_qc/filter_intervals",
        CHECKPOINT_TYPE.NATIVE_VARIANT_DATASET,
    ),
    "base_sample_qc": (
        "{project}/intermediate/sample_qc/base_sample_qc",
        CHECKPOINT_TYPE.HAIL_TABLE,
    ),
    "get_annotated_sample_qc": (
        "{project}/intermediate/sample_qc/annotate_adj",
        CHECKPOINT_TYPE.HAIL_TABLE,
    ),
    "compute_sex_inference_vds": (
        "{project}/intermediate/sample_qc/infer_sex",
        CHECKPOINT_TYPE.VARIANT_DATASET,
    ),
    "get_infer_sex": (
        "{project}/intermediate/sample_qc/ploidy",
        CHECKPOINT_TYPE.HAIL_TABLE,
    ),
    "get_contamination_estimate": (
        "{project}/intermediate/sample_qc/charr",
        CHECKPOINT_TYPE.HAIL_TABLE,
    ),
    "densify_pca_sites": (
        "{project}/intermediate/sample_qc/pca_sites",
        CHECKPOINT_TYPE.MATRIX_TABLE,
    ),
    "get_king_estimate": (
        "{project}/intermediate/sample_qc/king",
        CHECKPOINT_TYPE.MATRIX_TABLE,
    ),
    "get_relatedness2": (
        "{project}/intermediate/sample_qc/relatedness2",
        CHECKPOINT_TYPE.HAIL_TABLE,
    ),
    "get_relatedness": (
        "{project}/intermediate/sample_qc/relatedness",
        CHECKPOINT_TYPE.HAIL_TABLE,
    ),
    "apply_ancestry_classification": (
        "{project}/intermediate/sample_qc/ancestry_classification",
        CHECKPOINT_TYPE.HAIL_TABLE,
    ),
    "get_combined_sample_qc_table": (
        "{project}/intermediate/sample_qc/combined",
        CHECKPOINT_TYPE.HAIL_TABLE,
    ),
    "compute_hard_filters": ("{project}/sample_qc", CHECKPOINT_TYPE.HAIL_TABLE),
}

RESOURCES = {
    "calling_intervals": "/res/calling_intervals.bed",
    "telomeres_and_centromeres": "/res/telomeres_and_centromeres.ht",
    "lcr_intervals": "/res/lcr_intervals.ht",
    "seg_dup_intervals": "/res/seg_dup_intervals.ht",
    "gnomad_v4": "/res/gnomad_v4.ht",
    "gnomad_pca_model": "/res/pca_model.onnx",
    "gnomad_pca_loadings": "/res/pca_loadings.ht",
}


def _settings(
    calling_intervals: Path = Path("/res/calling_intervals.bed"),
) -> Settings:
    return Settings(
        validate_paths=False,
        project="demo",
        checkpoint="/ckpt",
        resources=Resources(
            calling_intervals=calling_intervals,
            telomeres_and_centromeres=Path("/res/telomeres_and_centromeres.ht"),
            lcr_intervals=Path("/res/lcr_intervals.ht"),
            seg_dup_intervals=Path("/res/seg_dup_intervals.ht"),
            gnomad_pca_loadings=Path("/res/pca_loadings.ht"),
            gnomad_pca_model=Path("/res/pca_model.onnx"),
            gnomad_v4=Path("/res/gnomad_v4.ht"),
            reference=Path("/res/reference.fa"),
        ),
        sample_qc=SampleQcConfig(),
    )


def _pipeline(vds: str = "in.vds") -> SampleQcPipeline:
    return SampleQcPipeline(Task(settings=_settings(), command=SampleQcRun(vds=vds)))


def test_steps_are_the_work_closures_in_run_order() -> None:
    _pipeline()
    registry = StepRegistry()

    assert [name for name in registry.names() if name in WORK_STEPS] == list(WORK_STEPS)


def test_steps_declare_their_fromstep_edges() -> None:
    _pipeline()
    registry = StepRegistry()

    for name, expected in WORK_EDGES.items():
        got = {
            parameter: dependency.rsplit(".", 1)[-1]
            for parameter, dependency in registry.get(name).depends_on.items()
        }
        assert got == expected, name


def test_steps_declare_the_resources_they_consume() -> None:
    _pipeline()
    registry = StepRegistry()

    for name in WORK_STEPS:
        assert set(registry.get(name).resources) == set(WORK_RESOURCES.get(name, ()))


def test_each_step_declares_exactly_one_output_artifact() -> None:
    _pipeline()
    registry = StepRegistry()

    for name in WORK_STEPS:
        assert len(registry.get(name).artifacts) == 1, name


def test_artifact_prefixes_and_kinds_match_the_original_checkpoints() -> None:
    _pipeline()
    registry = StepRegistry()

    for name, (prefix, kind) in WORK_ARTIFACTS.items():
        artifact = registry.get(name).artifacts[0]
        assert artifact.prefix == prefix, name
        assert artifact.kind == kind, name


def test_artifact_paths_are_project_relative_checkpoints() -> None:
    _pipeline()
    registry = StepRegistry()

    assert (
        registry.get("filter_intervals").artifacts[0].path(_settings())
        == "/ckpt/demo/intermediate/sample_qc/filter_intervals.metadata.json"
    )
    # The prefix carries no extension; the VARIANT_DATASET kind appends ``.vds``
    # exactly once.
    assert (
        registry.get("compute_sex_inference_vds").artifacts[0].path(_settings())
        == "/ckpt/demo/intermediate/sample_qc/infer_sex.vds"
    )
    assert (
        registry.get("compute_hard_filters").artifacts[0].path(_settings())
        == "/ckpt/demo/sample_qc.ht"
    )


def test_the_bound_vds_is_the_input_endpoint_value() -> None:
    _pipeline(vds="gs://bucket/input.vds")

    assert StepRegistry()["sample_qc.vds"].value == "gs://bucket/input.vds"


def test_the_pipeline_declares_resources_from_settings() -> None:
    declared = _pipeline().resources

    assert [resource.name for resource in declared] == list(RESOURCES)
    for resource in declared:
        assert str(resource.path) == RESOURCES[resource.name]
        assert resource.reference_genome == "GRCh38"


def test_resources_come_from_the_pipeline_settings_not_a_catalog() -> None:
    settings = _settings(calling_intervals=Path("/elsewhere/calling.bed"))
    pipeline = SampleQcPipeline(
        Task(settings=settings, command=SampleQcRun(vds="in.vds"))
    )

    declared = {resource.name: resource for resource in pipeline.resources}
    assert str(declared["calling_intervals"].path) == "/elsewhere/calling.bed"
