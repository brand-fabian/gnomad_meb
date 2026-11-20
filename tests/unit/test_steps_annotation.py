"""Tests for the Annotation pipeline's declared resources.

The stage is a :class:`~gnomad_meb.pipeline.Pipeline` subclass: constructing it
registers its ``@checkpoint`` steps and its declared :class:`Resource` objects.
These tests exercise the declaration, not the hail computations: the work steps,
their ``FromStep`` edges, the decorator's artifacts, the VDS input endpoint, and
the resource set (paths from settings, loaders, and derived QC tables). A final
group drives ``AnnotationPipeline.compute_coverage_stats`` on a tiny dense
matrix to cover the telomere/centromere filter and the strata.
"""

from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import pytest

from gnomad_meb.cmd.task import AnnotationRun, Task
from gnomad_meb.config import (
    AnnotationConfig,
    AnnotationResources,
    Resources,
    SampleQcConfig,
    Settings,
    VepAnnotationResources,
)
from gnomad_meb.constants import DEFAULT_REFERENCE_GENOME
from gnomad_meb.lib.annotation import AnnotationPipeline
from gnomad_meb.lib.annotation import resources as annotation_resources
from gnomad_meb.lib.checkpoint import CHECKPOINT_TYPE, Artifact, checkpoint
from gnomad_meb.lib.vds import VariantDataset
from gnomad_meb.pipeline import (
    FromStep,
    InputStep,
    Pipeline,
    PipelineRunner,
    Resource,
    StepRegistry,
)

#: The split task names the stub handle exposes (metadata only, no hail session).
TASK_NAMES = ("left", "right")

#: The in-silico postprocess substep's name (long enough to break many lines).
POSTPROCESS = "annotate_postprocess_gnomad_and_insilico_predictors"


def _stub_handle(names: tuple[str, ...] = TASK_NAMES) -> SimpleNamespace:
    """A ``VariantDataset`` stand-in: ``.params.tasks[i].name`` is all we read."""
    return SimpleNamespace(
        params=SimpleNamespace(
            tasks=tuple(SimpleNamespace(name=name) for name in names)
        )
    )


@pytest.fixture(autouse=True)
def _stubbed_vds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the metadata reader so pipeline construction stays hail-free.

    ``AnnotationPipeline.prepare`` reads the split list through
    ``VariantDataset.open(path)``; the real reader builds ``hl.Locus``/``hl.Interval``
    and needs a hail session, which cannot run here (ADR-0002).
    """
    monkeypatch.setattr(
        VariantDataset, "open", staticmethod(lambda path, *a, **k: _stub_handle())
    )


def _expected_work_steps(tasks: tuple[str, ...] = TASK_NAMES) -> list[str]:
    """The per-split step set: three imports, an eight-step run per task, merge."""
    steps = ["import_clinvar_ht", "import_cadd_ht", "import_alphamissense_ht"]
    for task in tasks:
        steps.append(f"compute_default_info_table.{task}")
        steps.append(f"variant_dataset_to_matrix_table.{task}")
        steps.append(f"annotate_resources.{task}")
        steps.append(f"annotate_variant_qc.{task}")
        steps.append(f"annotate_vep.{task}")
        steps.append(f"{POSTPROCESS}.{task}")
        steps.append(f"annotate_regions_and_freq.{task}")
        steps.append(f"compute_coverage_stats.{task}")
        steps.append(f"annotate_coverage_stats.{task}")
    steps.append("merge")
    return steps


RESOURCE_NAMES = (
    "gnomad_ht",
    "sample_qc_ht",
    "variant_qc_ht",
    "lcr_intervals_ht",
    "seg_dup_intervals_ht",
    "telomeres_and_centromeres",
    "clinvar_vcf",
    "cadd",
    "cadd_indel",
    "alpha_missense",
)


def _settings(checkpoint_root: str | None = "/ckpt") -> Settings:
    return Settings(
        project="proj",
        checkpoint=checkpoint_root,
        validate_paths=False,
        resources=Resources(
            calling_intervals="/r/calling.intervals",
            telomeres_and_centromeres="/r/tc.ht",
            lcr_intervals="/r/lcr.ht",
            seg_dup_intervals="/r/segdup.ht",
            gnomad_pca_loadings="/r/pca.loadings.ht",
            gnomad_pca_model="/r/pca.onnx",
            gnomad_v4="/r/gnomad.ht",
            reference="/r/ref.fa",
        ),
        sample_qc=SampleQcConfig(),
        annotation=AnnotationConfig(
            resources=AnnotationResources(clinvar_vcf="/r/clinvar.vcf.gz"),
            vep=VepAnnotationResources(
                vep="/r/vep",
                cache_dir="/r/cache",
                cadd="/r/cadd.tsv.gz",
                cadd_indel="/r/cadd.indel.tsv.gz",
                alpha_missense="/r/am.tsv.gz",
            ),
        ),
    )


def _pipeline(
    settings: Settings | None = None, vds: str = "/data/callset.vds"
) -> AnnotationPipeline:
    return AnnotationPipeline(
        Task(settings=settings or _settings(), command=AnnotationRun(vds=vds))
    )


def _declared(settings: Settings) -> dict[str, object]:
    return {resource.name: resource for resource in _pipeline(settings).resources}


def _derived_path(prefix: str, settings: Settings) -> str:
    return Artifact(prefix=prefix, kind=CHECKPOINT_TYPE.HAIL_TABLE).path(settings)


def test_declares_the_per_split_work_steps_in_definition_order() -> None:
    _pipeline()
    registry = StepRegistry()
    expected = _expected_work_steps()

    assert [name for name in registry.names() if name in expected] == expected


def test_work_step_artifacts_have_the_expected_kind_and_prefix() -> None:
    _pipeline()
    registry = StepRegistry()
    expected = {
        "import_clinvar_ht": (
            "proj/intermediate/annotation/clinvar.ht",
            CHECKPOINT_TYPE.HAIL_TABLE,
        ),
        "import_cadd_ht": (
            "proj/intermediate/annotation/cadd.ht",
            CHECKPOINT_TYPE.HAIL_TABLE,
        ),
        "import_alphamissense_ht": (
            "proj/intermediate/annotation/alphamissense.ht",
            CHECKPOINT_TYPE.HAIL_TABLE,
        ),
        "merge": ("proj/annotation.mt", CHECKPOINT_TYPE.MATRIX_TABLE),
    }
    for task in TASK_NAMES:
        expected[f"compute_default_info_table.{task}"] = (
            f"proj/intermediate/annotation/task-{task}/default_info.ht",
            CHECKPOINT_TYPE.HAIL_TABLE,
        )
        expected[f"variant_dataset_to_matrix_table.{task}"] = (
            f"proj/intermediate/annotation/task-{task}/dense_callset.mt",
            CHECKPOINT_TYPE.MATRIX_TABLE,
        )
        expected[f"annotate_resources.{task}"] = (
            f"proj/intermediate/annotation/task-{task}/annotate_resources.mt",
            CHECKPOINT_TYPE.MATRIX_TABLE,
        )
        expected[f"annotate_variant_qc.{task}"] = (
            f"proj/intermediate/annotation/task-{task}/annotate_variant_qc.mt",
            CHECKPOINT_TYPE.MATRIX_TABLE,
        )
        expected[f"annotate_vep.{task}"] = (
            f"proj/intermediate/annotation/task-{task}/annotate_vep.mt",
            CHECKPOINT_TYPE.MATRIX_TABLE,
        )
        expected[f"{POSTPROCESS}.{task}"] = (
            f"proj/intermediate/annotation/task-{task}/{POSTPROCESS}.mt",
            CHECKPOINT_TYPE.MATRIX_TABLE,
        )
        expected[f"annotate_regions_and_freq.{task}"] = (
            f"proj/intermediate/annotation/task-{task}/annotate_regions_and_freq.mt",
            CHECKPOINT_TYPE.MATRIX_TABLE,
        )
        expected[f"compute_coverage_stats.{task}"] = (
            f"proj/intermediate/annotation/task-{task}/coverage.ht",
            CHECKPOINT_TYPE.HAIL_TABLE,
        )
        expected[f"annotate_coverage_stats.{task}"] = (
            f"proj/intermediate/annotation/task-{task}/annotate_coverage_stats.mt",
            CHECKPOINT_TYPE.MATRIX_TABLE,
        )
    settings = _settings()
    for name, (path, kind) in expected.items():
        artifacts = registry.get(name).artifacts
        assert len(artifacts) == 1, name
        assert artifacts[0].kind == kind
        assert artifacts[0].path(settings).endswith(path)


def test_the_whole_callset_dense_and_annotate_steps_are_absent() -> None:
    _pipeline()
    registry = StepRegistry()

    assert "annotation.variant_dataset_to_matrix_table" not in registry.steps()
    assert "annotation.annotate" not in registry.steps()
    # The substep bodies are plain methods, so the partials wrap them and no
    # whole-stage step is registered for any of them.
    assert "annotation.annotate_resources" not in registry.steps()
    assert "annotation.annotate_variant_qc" not in registry.steps()
    assert "annotation.annotate_vep" not in registry.steps()
    assert f"annotation.{POSTPROCESS}" not in registry.steps()
    assert "annotation.annotate_regions_and_freq" not in registry.steps()


def test_each_dense_step_slices_the_vds_input_endpoint() -> None:
    _pipeline(vds="/data/callset.vds")
    registry = StepRegistry()

    for task in TASK_NAMES:
        assert registry.get(f"variant_dataset_to_matrix_table.{task}").depends_on == {
            "vds": "annotation.vds",
            "info_ht": f"annotation.compute_default_info_table.{task}",
        }
    assert registry["annotation.vds"].value == "/data/callset.vds"


def test_each_info_step_slices_the_vds_input_endpoint() -> None:
    _pipeline(vds="/data/callset.vds")
    registry = StepRegistry()

    for task in TASK_NAMES:
        assert registry.get(f"compute_default_info_table.{task}").depends_on == {
            "vds": "annotation.vds"
        }
    assert registry["annotation.vds"].value == "/data/callset.vds"


def test_each_substep_wires_the_linear_chain_and_the_imports() -> None:
    _pipeline()
    registry = StepRegistry()

    for task in TASK_NAMES:
        resources = registry.get(f"annotate_resources.{task}")
        assert resources.depends_on == {
            "mt": f"annotation.variant_dataset_to_matrix_table.{task}",
            "clinvar_ht": "annotation.import_clinvar_ht",
        }
        assert registry.get(f"annotate_variant_qc.{task}").depends_on == {
            "mt": f"annotation.annotate_resources.{task}",
        }
        assert registry.get(f"annotate_vep.{task}").depends_on == {
            "mt": f"annotation.annotate_variant_qc.{task}",
        }
        # The postprocess step owns the in-silico finalisation (VEP reassigns
        # in_silico_predictors for cluster-VEP'd rows), so it consumes the
        # cadd/alphamissense imports.
        assert registry.get(f"{POSTPROCESS}.{task}").depends_on == {
            "mt": f"annotation.annotate_vep.{task}",
            "cadd_ht": "annotation.import_cadd_ht",
            "alphamissense_ht": "annotation.import_alphamissense_ht",
        }
        assert registry.get(f"annotate_regions_and_freq.{task}").depends_on == {
            "mt": f"annotation.{POSTPROCESS}.{task}",
        }


def test_annotation_steps_returns_the_seven_ordered_per_task_substeps() -> None:
    pipeline = _pipeline()

    steps = pipeline._annotation_steps("left")

    assert [name for name, _, _ in steps] == [
        "annotate_resources.left",
        "annotate_variant_qc.left",
        "annotate_vep.left",
        f"{POSTPROCESS}.left",
        "annotate_regions_and_freq.left",
        "compute_coverage_stats.left",
        "annotate_coverage_stats.left",
    ]
    assert [task for _, _, task in steps] == ["left"] * 7
    expected_edges = [
        {"mt": FromStep("variant_dataset_to_matrix_table.left")},
        {"mt": FromStep("annotate_resources.left")},
        {"mt": FromStep("annotate_variant_qc.left")},
        {"mt": FromStep("annotate_vep.left")},
        {"mt": FromStep(f"{POSTPROCESS}.left")},
        {"mt": FromStep("annotate_resources.left")},
        {
            "mt": FromStep("annotate_regions_and_freq.left"),
            "coverage_ht": FromStep("compute_coverage_stats.left"),
        },
    ]
    for (_, method, _), expected in zip(steps, expected_edges, strict=True):
        assert inspect.unwrap(method).keywords == expected


def test_annotation_steps_leak_no_whole_stage_marker() -> None:
    """Every marker the helper produces names a per-task predecessor.

    The pre-port chain declared whole-stage methods (``annotate`` and the dense
    step) and wired them by bare name. The substeps must instead wire their
    per-task predecessor: a ``FromStep`` naming ``annotate_resources``
    with no task suffix would wire the whole-callset step -- which no longer
    exists -- instead of ``annotate_resources.{task}``, dropping the
    linear per-split chain.
    """
    task = "left"
    steps = _pipeline()._annotation_steps(task)

    # The whole-stage names the pre-port chain used, which the registry
    # qualifies as ``annotation.<name>``. A returned step name or ``FromStep``
    # target reaching one of these would be a marker leaking to whole-stage.
    whole_stage = {
        "variant_dataset_to_matrix_table",
        "annotate",
        "annotate_resources",
        "annotate_variant_qc",
        "annotate_vep",
        POSTPROCESS,
        "annotate_regions_and_freq",
        "compute_coverage_stats",
        "annotate_coverage_stats",
    }
    per_task_predecessors = {
        f"variant_dataset_to_matrix_table.{task}",
        f"annotate_resources.{task}",
        f"annotate_variant_qc.{task}",
        f"annotate_vep.{task}",
        f"{POSTPROCESS}.{task}",
        f"annotate_regions_and_freq.{task}",
        f"compute_coverage_stats.{task}",
    }

    returned = {name for name, _, _ in steps}
    assert returned.isdisjoint(whole_stage)
    assert all(name.endswith(f".{task}") for name in returned)

    markers = [
        marker
        for _, method, _ in steps
        for marker in inspect.unwrap(method).keywords.values()
    ]
    assert all(isinstance(marker, FromStep) for marker in markers)
    assert {marker.name for marker in markers} == per_task_predecessors
    assert all(marker.name.endswith(f".{task}") for marker in markers)
    assert all(marker.name not in whole_stage for marker in markers)


def test_generated_steps_carry_their_task_identity() -> None:
    _pipeline()
    registry = StepRegistry()

    for task in TASK_NAMES:
        dense = registry.get(f"variant_dataset_to_matrix_table.{task}")
        assert dense.task.command.args == {"task": task}
        for name in (
            "annotate_resources",
            "annotate_variant_qc",
            "annotate_vep",
            POSTPROCESS,
            "annotate_regions_and_freq",
            "compute_coverage_stats",
            "annotate_coverage_stats",
        ):
            assert registry.get(f"{name}.{task}").task.command.args == {"task": task}
    assert registry.get("merge").task.command.args == {}


def test_merge_wires_the_coverage_join_and_the_terminal_artifact() -> None:
    settings = _settings()
    _pipeline(settings)
    step = StepRegistry().get("merge")

    assert step.depends_on == {
        task: f"annotation.annotate_coverage_stats.{task}" for task in TASK_NAMES
    }
    assert len(step.artifacts) == 1
    assert step.artifacts[0].kind == CHECKPOINT_TYPE.MATRIX_TABLE
    assert step.artifacts[0].path(settings).endswith("proj/annotation.mt")


def test_prepare_fails_loudly_when_the_vds_handle_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _missing(path: object, *a: object, **k: object) -> object:
        raise FileNotFoundError(f"invalid vds path: path = {path}")

    monkeypatch.setattr(VariantDataset, "open", staticmethod(_missing))

    with pytest.raises(FileNotFoundError):
        _pipeline(vds="/data/does_not_exist.vds")


def test_the_per_task_step_set_round_trips_through_json() -> None:
    task = Task(settings=_settings(), command=AnnotationRun(vds="/data/callset.vds"))
    runner = PipelineRunner(task, [AnnotationPipeline(task)])
    dumped = json.loads(json.dumps(runner.current_state()))

    restored = PipelineRunner.load(dumped)

    # The same inputs rebuilt the identical step set -- no PipelineMismatchError.
    assert set(restored.current_state()["steps"]) == set(dumped["steps"])
    registry = StepRegistry()
    for task_name in TASK_NAMES:
        info = registry.get(f"compute_default_info_table.{task_name}")
        dense = registry.get(f"variant_dataset_to_matrix_table.{task_name}")
        resources = registry.get(f"annotate_resources.{task_name}")
        variant_qc = registry.get(f"annotate_variant_qc.{task_name}")
        vep = registry.get(f"annotate_vep.{task_name}")
        postprocess = registry.get(f"{POSTPROCESS}.{task_name}")
        regions_and_freq = registry.get(f"annotate_regions_and_freq.{task_name}")
        coverage = registry.get(f"compute_coverage_stats.{task_name}")
        annotate_coverage = registry.get(f"annotate_coverage_stats.{task_name}")
        assert resources is not None and variant_qc is not None
        assert vep is not None
        assert postprocess is not None and regions_and_freq is not None
        assert coverage is not None and annotate_coverage is not None
        assert info is not None and dense is not None

        # The rebuilt run callables must bind the same task identity and edges,
        # not merely be callable. ``checkpoint`` wraps a ``functools.partial`` and
        # deliberately preserves it under ``__wrapped__`` (see
        # ``lib/checkpoint._callable_source``), so ``prepare``'s bindings stay
        # inspectable -- a reconstruction that silently rebuilt different
        # per-task callables would fail here.
        assert inspect.unwrap(info.run).keywords == {"task": task_name}
        assert inspect.unwrap(dense.run).keywords == {
            "info_ht": FromStep(f"compute_default_info_table.{task_name}"),
            "task": task_name,
        }
        assert inspect.unwrap(resources.run).keywords == {
            "mt": FromStep(f"variant_dataset_to_matrix_table.{task_name}")
        }
        assert inspect.unwrap(variant_qc.run).keywords == {
            "mt": FromStep(f"annotate_resources.{task_name}")
        }
        assert inspect.unwrap(vep.run).keywords == {
            "mt": FromStep(f"annotate_variant_qc.{task_name}")
        }
        assert inspect.unwrap(postprocess.run).keywords == {
            "mt": FromStep(f"annotate_vep.{task_name}")
        }
        assert inspect.unwrap(regions_and_freq.run).keywords == {
            "mt": FromStep(f"{POSTPROCESS}.{task_name}")
        }
        assert inspect.unwrap(coverage.run).keywords == {
            "mt": FromStep(f"annotate_resources.{task_name}")
        }
        assert inspect.unwrap(annotate_coverage.run).keywords == {
            "mt": FromStep(f"annotate_regions_and_freq.{task_name}"),
            "coverage_ht": FromStep(f"compute_coverage_stats.{task_name}"),
        }

        assert info.task.command.args == {"task": task_name}
        assert dense.task.command.args == {"task": task_name}
        assert resources.task.command.args == {"task": task_name}
        assert variant_qc.task.command.args == {"task": task_name}
        assert vep.task.command.args == {"task": task_name}
        assert postprocess.task.command.args == {"task": task_name}
        assert regions_and_freq.task.command.args == {"task": task_name}
        assert coverage.task.command.args == {"task": task_name}
        assert annotate_coverage.task.command.args == {"task": task_name}
        assert dense.depends_on["info_ht"] == (
            f"annotation.compute_default_info_table.{task_name}"
        )
        assert resources.depends_on["mt"] == (
            f"annotation.variant_dataset_to_matrix_table.{task_name}"
        )
        assert variant_qc.depends_on["mt"] == (
            f"annotation.annotate_resources.{task_name}"
        )
        assert vep.depends_on["mt"] == (
            f"annotation.annotate_variant_qc.{task_name}"
        )
        assert postprocess.depends_on["mt"] == f"annotation.annotate_vep.{task_name}"
        assert regions_and_freq.depends_on["mt"] == (
            f"annotation.{POSTPROCESS}.{task_name}"
        )
        assert coverage.depends_on["mt"] == (
            f"annotation.annotate_resources.{task_name}"
        )
        assert annotate_coverage.depends_on == {
            "mt": f"annotation.annotate_regions_and_freq.{task_name}",
            "coverage_ht": f"annotation.compute_coverage_stats.{task_name}",
        }

    merge = registry.get("merge")
    assert merge is not None
    assert inspect.unwrap(merge.run).keywords == {
        task_name: FromStep(f"annotate_coverage_stats.{task_name}")
        for task_name in TASK_NAMES
    }
    assert merge.depends_on == {
        task_name: f"annotation.annotate_coverage_stats.{task_name}"
        for task_name in TASK_NAMES
    }


def test_the_pipeline_declares_the_ten_resources() -> None:
    declared = _pipeline().resources

    assert tuple(resource.name for resource in declared) == RESOURCE_NAMES


def test_resource_paths_are_read_from_settings() -> None:
    declared = _declared(_settings())

    assert declared["gnomad_ht"].path == "/r/gnomad.ht"
    assert declared["lcr_intervals_ht"].path == "/r/lcr.ht"
    assert declared["seg_dup_intervals_ht"].path == "/r/segdup.ht"
    assert declared["telomeres_and_centromeres"].path == "/r/tc.ht"
    assert declared["clinvar_vcf"].path == "/r/clinvar.vcf.gz"
    assert declared["cadd"].path == "/r/cadd.tsv.gz"
    assert declared["cadd_indel"].path == "/r/cadd.indel.tsv.gz"
    assert declared["alpha_missense"].path == "/r/am.tsv.gz"
    for resource in declared.values():
        assert resource.reference_genome == "GRCh38"


def test_the_substeps_consume_the_six_whole_stage_resources() -> None:
    _pipeline()
    registry = StepRegistry()

    for task in TASK_NAMES:
        assert set(registry.get(f"annotate_resources.{task}").resources) == {
            "gnomad_ht",
            "sample_qc_ht",
        }
        assert set(registry.get(f"annotate_variant_qc.{task}").resources) == {
            "variant_qc_ht",
        }
        assert set(registry.get(f"annotate_vep.{task}").resources) == set()
        assert set(registry.get(f"{POSTPROCESS}.{task}").resources) == set()
        assert set(registry.get(f"annotate_regions_and_freq.{task}").resources) == {
            "lcr_intervals_ht",
            "seg_dup_intervals_ht",
        }
        assert set(registry.get(f"compute_coverage_stats.{task}").resources) == {
            "telomeres_and_centromeres",
        }
        assert set(registry.get(f"annotate_coverage_stats.{task}").resources) == set()


def test_the_coverage_step_reads_the_dense_matrix_not_the_vds() -> None:
    """The coverage step's only matrix edge is the dense ``annotate_resources``.

    It must not reach the VDS input or the ``sample_qc_ht`` resource: the strata
    come from the dense columns (``hard_filters`` / ``cohort``) already joined by
    the preceding steps, and the site set is the dense rows, not a ``ref_ht``.
    """
    _pipeline()
    registry = StepRegistry()

    for task in TASK_NAMES:
        step = registry.get(f"compute_coverage_stats.{task}")
        assert step.depends_on == {
            "mt": f"annotation.annotate_resources.{task}",
        }
        assert set(step.resources) == {"telomeres_and_centromeres"}
        assert "vds" not in step.depends_on
        assert "sample_qc_ht" not in step.resources
        assert inspect.unwrap(step.run).keywords == {
            "mt": FromStep(f"annotate_resources.{task}"),
        }


def test_import_source_loaders_read_the_declared_source_path(monkeypatch) -> None:
    calls: list[str] = []

    def _record(path):
        calls.append(str(path))
        return f"loaded:{path}"

    monkeypatch.setattr(annotation_resources, "get_alphamissense_ht", _record)
    monkeypatch.setattr(annotation_resources, "read_clinvar_vcf", _record)
    monkeypatch.setattr(
        annotation_resources,
        "get_cadd_ht",
        lambda source: calls.append("+".join(map(str, source))) or "loaded:cadd",
    )

    declared = _declared(_settings())

    assert (
        declared["alpha_missense"].loader(declared["alpha_missense"].path)
        == "loaded:/r/am.tsv.gz"
    )
    assert (
        declared["clinvar_vcf"].loader(declared["clinvar_vcf"].path)
        == "loaded:/r/clinvar.vcf.gz"
    )
    assert declared["cadd"].loader(declared["cadd"].path) == "loaded:cadd"
    assert calls == [
        "/r/am.tsv.gz",
        "/r/clinvar.vcf.gz",
        "/r/cadd.tsv.gz+/r/cadd.indel.tsv.gz",
    ]


def test_unconfigured_quality_control_resources_are_derived() -> None:
    settings = _settings()
    declared = _declared(settings)

    assert declared["sample_qc_ht"].path == _derived_path(
        "{project}/sample_qc", settings
    )
    assert declared["variant_qc_ht"].path == _derived_path(
        "{project}/variant_qc", settings
    )


def test_configured_quality_control_paths_override_the_derived_ones() -> None:
    settings = _settings()
    assert settings.annotation is not None
    assert settings.annotation.resources is not None
    settings.annotation.resources.sample_qc = "/ext/sample_qc.ht"
    settings.annotation.resources.variant_qc = "/ext/variant_qc.ht"

    declared = _declared(settings)

    assert declared["sample_qc_ht"].path == "/ext/sample_qc.ht"
    assert declared["variant_qc_ht"].path == "/ext/variant_qc.ht"


class _QcProducer(Pipeline):
    @classmethod
    def name(cls) -> str:
        return "qc"

    @property
    def resources(self) -> tuple[Resource, ...]:
        return ()

    def get_input(self) -> tuple[InputStep, ...]:
        return ()

    @checkpoint("{project}/sample_qc", CHECKPOINT_TYPE.HAIL_TABLE)
    def sample_qc(self):
        return None

    @checkpoint("{project}/variant_qc", CHECKPOINT_TYPE.HAIL_TABLE)
    def variant_qc(self):
        return None


def test_derived_qc_resources_order_annotate_after_their_producers() -> None:
    settings = _settings()
    _QcProducer(Task(settings=settings, command=AnnotationRun(vds="x")))
    _pipeline(settings)

    step = StepRegistry().get(f"annotate_variant_qc.{TASK_NAMES[0]}")

    assert step.resources["variant_qc_ht"] == "qc.variant_qc"


def test_derived_sample_qc_resource_orders_annotate_after_its_producer() -> None:
    settings = _settings()
    _QcProducer(Task(settings=settings, command=AnnotationRun(vds="x")))
    _pipeline(settings)

    step = StepRegistry().get(f"annotate_resources.{TASK_NAMES[0]}")

    assert step.resources["sample_qc_ht"] == "qc.sample_qc"


# --- coverage aggregation over the dense matrix (ticket 02) -------------------

#: s1: pass / cohort A, s2: filtered / cohort B, s3: pass / cohort A.
COVERAGE_SAMPLE_ANNOTATIONS = {
    "s1": {"hard_filters": [], "cohort": "A"},
    "s2": {"hard_filters": ["FAIL"], "cohort": "B"},
    "s3": {"hard_filters": [], "cohort": "A"},
}

#: Dense rows at chr1:1..4; the telomere/centromere interval covers chr1:1..2.
COVERAGE_POSITIONS = (1, 2, 3, 4)
COVERAGE_DP = {"s1": 10, "s2": 20, "s3": 30}


def _coverage_dense_mt(hl):
    """A tiny dense split matrix with ``DP``, ``hard_filters`` and ``cohort``."""
    schema = hl.tstruct(
        locus=hl.tlocus(DEFAULT_REFERENCE_GENOME),
        alleles=hl.tarray(hl.tstr),
        s=hl.tstr,
        DP=hl.tint32,
        hard_filters=hl.tarray(hl.tstr),
        cohort=hl.tstr,
    )
    rows = [
        {
            "locus": hl.Locus("chr1", position, DEFAULT_REFERENCE_GENOME),
            "alleles": ["A", "C"],
            "s": sample,
            "DP": dp,
            "hard_filters": COVERAGE_SAMPLE_ANNOTATIONS[sample]["hard_filters"],
            "cohort": COVERAGE_SAMPLE_ANNOTATIONS[sample]["cohort"],
        }
        for position in COVERAGE_POSITIONS
        for sample, dp in COVERAGE_DP.items()
    ]
    return hl.Table.parallelize(rows, schema=schema).to_matrix_table(
        row_key=["locus", "alleles"],
        col_key=["s"],
        col_fields=["hard_filters", "cohort"],
    )


def _telomeres_and_centromeres_table(hl):
    """A one-interval telomere/centromere table covering chr1:1..2."""
    interval = hl.Interval(
        hl.Locus("chr1", 1, DEFAULT_REFERENCE_GENOME),
        hl.Locus("chr1", 2, DEFAULT_REFERENCE_GENOME),
        includes_start=True,
        includes_end=True,
    )
    return hl.Table.parallelize(
        [{"interval": interval}],
        schema=hl.tstruct(interval=hl.tinterval(hl.tlocus(DEFAULT_REFERENCE_GENOME))),
    )


def test_coverage_filters_telomeres_and_centromeres(hl) -> None:
    """The coverage site set is the dense rows minus the telomere intervals."""
    ht = _pipeline().compute_coverage_stats(
        mt=_coverage_dense_mt(hl),
        telomeres_and_centromeres=_telomeres_and_centromeres_table(hl),
    )

    assert tuple(ht.key) == ("locus",)
    assert sorted(row.locus.position for row in ht.collect()) == [3, 4]


@pytest.mark.parametrize(
    ("group_by_cohort", "expected_keys"),
    [(True, {"filtered", "cohort"}), (False, {"filtered"})],
)
def test_coverage_strata_follow_the_group_by_cohort_setting(
    hl, group_by_cohort: bool, expected_keys: set[str]
) -> None:
    """``cohort`` is a strata group only when ``group_by_cohort`` is set."""
    settings = _settings()
    assert settings.annotation is not None
    settings.annotation.group_by_cohort = group_by_cohort

    ht = _pipeline(settings).compute_coverage_stats(
        mt=_coverage_dense_mt(hl),
        telomeres_and_centromeres=_telomeres_and_centromeres_table(hl),
    )

    meta = hl.eval(ht.coverage_stats_meta)
    assert all("group" not in entry for entry in meta)
    assert {key for entry in meta for key in entry} == expected_keys


# --- coverage join onto the annotation matrix and merge (ticket 04) -----------

def _coverage_stats_table(hl):
    """The per-locus coverage table for the tiny dense matrix (keyed by locus)."""
    return _pipeline().compute_coverage_stats(
        mt=_coverage_dense_mt(hl),
        telomeres_and_centromeres=_telomeres_and_centromeres_table(hl),
    )


def _coverage_annotated_mt(hl, coverage_ht):
    """A dense split matrix with the coverage join and a ``vep_globals`` global.

    ``merge`` sums the per-split VEP ``defined``/``missing`` counts, so a real
    part carries ``vep_globals``; the stub values only need the two fields.
    """
    mt = _coverage_dense_mt(hl)
    mt = mt.annotate_globals(vep_globals=hl.struct(defined=0, missing=0))
    return _pipeline().annotate_coverage_stats(mt=mt, coverage_ht=coverage_ht)


def test_annotate_coverage_stats_carries_only_the_meta_global(hl) -> None:
    """The join adds ``coverage_stats`` and only the meta global.

    The per-stratum sample counts (``coverage_stats_meta_sample_count``) are
    provenance for the standalone coverage table, not for every annotation
    matrix, so they must not ride along.
    """
    coverage_ht = _coverage_stats_table(hl)

    annotated = _pipeline().annotate_coverage_stats(
        mt=_coverage_dense_mt(hl), coverage_ht=coverage_ht
    )

    assert "coverage_stats" in annotated.row.dtype.fields
    assert "coverage_stats_meta" in annotated.globals.dtype.fields
    assert "coverage_stats_meta_sample_count" not in annotated.globals.dtype.fields


def test_merge_preserves_the_coverage_join_across_parts(hl) -> None:
    """Two coverage-annotated parts union with an identical schema.

    ``coverage_stats`` and ``coverage_stats_meta`` survive ``merge``; the merged
    matrix does not require ``coverage_stats_meta_sample_count``.
    """
    coverage_ht = _coverage_stats_table(hl)
    part = _coverage_annotated_mt(hl, coverage_ht)

    merged = _pipeline().merge(left=part, right=part)

    assert "coverage_stats" in merged.row.dtype.fields
    assert "coverage_stats_meta" in merged.globals.dtype.fields
    assert "coverage_stats_meta_sample_count" not in merged.globals.dtype.fields

