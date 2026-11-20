"""Tests for the Variant QC export pipeline.

The stage derives one ``compute_info_table`` / ``compute_as_info_table`` /
``compute_quasi_info_table`` step and one ``as_info`` / ``quasi_info`` VCF export
per ``VariantDataset`` split task. These tests exercise the declaration, not the
hail computations: the per-task work steps, their ``FromStep`` edges, the
checkpoint prefixes, the bound per-task VCF export targets, the VDS input
endpoint, and the reconstruction round-trip. ``VariantDataset.open`` is stubbed
with a two-task handle (metadata only) as in ``test_gnomad_workflow.py``.
"""

from __future__ import annotations

import inspect
import json

import pytest
from variant_qc_helpers import (
    TASK_NAMES,
    make_settings,
    stub_missing_vds,
    stub_vds,
)

from gnomad_meb.cmd.task import Task, VariantQcExport
from gnomad_meb.config import Settings
from gnomad_meb.lib.checkpoint import CHECKPOINT_TYPE, Artifact
from gnomad_meb.lib.variant_qc.export import (
    VariantQcExportPipeline,
)
from gnomad_meb.pipeline import FromStep, PipelineRunner, StepRegistry


@pytest.fixture(autouse=True)
def _stubbed_vds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep pipeline construction hail-free (ADR-0002): two stub split tasks."""
    stub_vds(monkeypatch)


def _pipeline(**overrides) -> VariantQcExportPipeline:
    return VariantQcExportPipeline(
        Task(settings=make_settings(), command=VariantQcExport(**overrides))
    )


def _expected_work_steps(tasks: tuple[str, ...] = TASK_NAMES) -> list[str]:
    steps: list[str] = []
    for task in tasks:
        steps.append(f"compute_info_table.{task}")
        steps.append(f"compute_as_info_table.{task}")
        steps.append(f"compute_quasi_info_table.{task}")
        steps.append(f"export_as_info_vcf.{task}")
        steps.append(f"export_quasi_info_vcf.{task}")
    return steps


def _expected(settings: Settings, task: str, step: str, extension: str) -> str:
    prefix = (
        f"{settings.checkpoint}/{settings.project}"
        f"/intermediate/variant_qc/task-{task}"
    )
    return f"{prefix}/{step}.{extension}"


def test_group_name_is_stable() -> None:
    assert VariantQcExportPipeline.name() == "variant_qc_export"


def test_declares_one_step_per_split_in_definition_order() -> None:
    _pipeline(vds="in.vds")
    expected = _expected_work_steps()

    assert [name for name in StepRegistry().names() if name in expected] == expected


def test_the_whole_callset_info_step_is_absent() -> None:
    _pipeline(vds="in.vds")
    registry = StepRegistry()

    assert "variant_qc_export.compute_info_table" not in registry.steps()
    assert "variant_qc_export.compute_as_info_table" not in registry.steps()


def test_dependencies_match_the_per_task_closure_edges() -> None:
    _pipeline(vds="in.vds")
    registry = StepRegistry()

    for task in TASK_NAMES:
        assert registry.get(f"compute_info_table.{task}").depends_on == {
            "vds": "variant_qc_export.vds",
            "infer_as_annotations": "variant_qc_export.infer_as_annotations",
            "retain_cdfs": "variant_qc_export.retain_cdfs",
            "cdf_k": "variant_qc_export.cdf_k",
        }
        assert registry.get(f"compute_as_info_table.{task}").depends_on == {
            "info_ht": f"variant_qc_export.compute_info_table.{task}"
        }
        assert registry.get(f"compute_quasi_info_table.{task}").depends_on == {
            "info_ht": f"variant_qc_export.compute_info_table.{task}"
        }
        assert registry.get(f"export_as_info_vcf.{task}").depends_on == {
            "as_info_ht": f"variant_qc_export.compute_as_info_table.{task}",
            "target": "variant_qc_export.as_info",
        }
        assert registry.get(f"export_quasi_info_vcf.{task}").depends_on == {
            "quasi_info_ht": f"variant_qc_export.compute_quasi_info_table.{task}",
            "target": "variant_qc_export.quasi_info",
        }


def test_declares_no_external_resources() -> None:
    pipeline = _pipeline(vds="in.vds")

    assert pipeline.resources == ()


def test_info_table_checkpoints_use_the_per_task_prefixes() -> None:
    settings = make_settings()
    _pipeline(vds="in.vds")
    registry = StepRegistry()

    for task in TASK_NAMES:
        expected = {
            f"compute_info_table.{task}": "info",
            f"compute_as_info_table.{task}": "as_info",
            f"compute_quasi_info_table.{task}": "quasi_info",
        }
        for name, step in expected.items():
            artifacts = registry.get(name).artifacts
            assert artifacts == (
                Artifact(
                    prefix=(
                        f"{{project}}/intermediate/variant_qc/task-{task}/{step}"
                    ),
                    kind=CHECKPOINT_TYPE.HAIL_TABLE,
                ),
            )
            assert artifacts[0].path(settings) == _expected(
                settings, task, step, "ht"
            )


def test_vcf_exports_declare_the_per_task_bound_target() -> None:
    _pipeline(vds="in.vds", as_info="out", quasi_info="qout")
    registry = StepRegistry()

    for task in TASK_NAMES:
        assert registry.get(f"export_as_info_vcf.{task}").artifacts == (
            Artifact(
                prefix=f"out/task-{task}/as_info",
                kind=CHECKPOINT_TYPE.FILE,
                suffix="vcf.bgz",
            ),
        )
        assert registry.get(f"export_quasi_info_vcf.{task}").artifacts == (
            Artifact(
                prefix=f"qout/task-{task}/quasi_info",
                kind=CHECKPOINT_TYPE.FILE,
                suffix="vcf.bgz",
            ),
        )


def test_unbound_target_declares_no_export_output() -> None:
    _pipeline(vds="in.vds", as_info="out")
    registry = StepRegistry()

    assert registry.get("export_as_info_vcf.left").artifacts != ()
    assert registry.get("export_quasi_info_vcf.left").artifacts == ()


def test_run_inputs_resolve_to_endpoints() -> None:
    _pipeline(
        vds="in.vds",
        as_info="out",
        infer_as_annotations=False,
        retain_cdfs=True,
        cdf_k=100,
    )
    registry = StepRegistry()
    step = registry.get("compute_info_table.left")

    assert set(step.depends_on) == {
        "vds",
        "infer_as_annotations",
        "retain_cdfs",
        "cdf_k",
    }
    assert registry["variant_qc_export.infer_as_annotations"].value is False
    assert registry["variant_qc_export.retain_cdfs"].value is True
    assert registry["variant_qc_export.cdf_k"].value == 100


def test_option_inputs_default() -> None:
    _pipeline(vds="in.vds")
    registry = StepRegistry()

    assert registry["variant_qc_export.infer_as_annotations"].value is True
    assert registry["variant_qc_export.retain_cdfs"].value is False
    assert registry["variant_qc_export.cdf_k"].value == 200


def test_generated_steps_carry_their_task_identity() -> None:
    _pipeline(vds="in.vds")
    registry = StepRegistry()

    for task in TASK_NAMES:
        for step in _expected_work_steps((task,)):
            assert registry.get(step).task.command.args == {"task": task}


def test_prepare_fails_loudly_when_the_vds_handle_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_missing_vds(monkeypatch)

    with pytest.raises(FileNotFoundError):
        _pipeline(vds="/data/does_not_exist.vds")


def test_the_per_task_step_set_round_trips_through_json() -> None:
    task = Task(settings=make_settings(), command=VariantQcExport(vds="in.vds"))
    runner = PipelineRunner(task, [VariantQcExportPipeline(task)])
    dumped = json.loads(json.dumps(runner.current_state()))

    restored = PipelineRunner.load(dumped)

    # The same inputs rebuilt the identical step set -- no PipelineMismatchError.
    assert set(restored.current_state()["steps"]) == set(dumped["steps"])
    registry = StepRegistry()
    for task_name in TASK_NAMES:
        info = registry.get(f"compute_info_table.{task_name}")
        export = registry.get(f"export_as_info_vcf.{task_name}")
        assert info is not None and export is not None

        # The rebuilt run callables must bind the same task identity and edges,
        # not merely be callable. ``checkpoint`` wraps a ``functools.partial`` and
        # deliberately preserves it under ``__wrapped__`` (see
        # ``lib/checkpoint._callable_source``), so ``prepare``'s bindings stay
        # inspectable -- a reconstruction that silently rebuilt different
        # per-task callables would fail here.
        assert inspect.unwrap(info.run).keywords == {"task": task_name}
        assert inspect.unwrap(export.run).keywords == {
            "task": task_name,
            "as_info_ht": FromStep(f"compute_as_info_table.{task_name}"),
        }

        assert info.task.command.args == {"task": task_name}
        assert export.task.command.args == {"task": task_name}
        assert export.depends_on["as_info_ht"] == (
            f"variant_qc_export.compute_as_info_table.{task_name}"
        )
