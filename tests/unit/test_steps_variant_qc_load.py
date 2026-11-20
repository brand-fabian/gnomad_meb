"""Tests for the Variant QC import pipeline.

The stage derives one ``load_vqsr`` step per ``VariantDataset`` split task, plus
a single terminal ``merge``. Each per-task load imports that split's own VQSR
VCF and attaches ``AS_lowqual`` from that split's own ``as_info`` table; ``merge``
unions the row-disjoint per-split tables once and stays the producer of
``{project}/variant_qc``. These tests exercise the declaration, not the hail
computations: the per-task work steps, their ``RunInput`` edges, the checkpoint
prefixes, the terminal artifact, and the reconstruction round-trip.
``VariantDataset.open`` is stubbed with a two-task handle (metadata only) as in
``test_gnomad_workflow.py``.
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

from gnomad_meb.cmd.task import Task, VariantQcLoadVqsr
from gnomad_meb.config import Settings
from gnomad_meb.lib.checkpoint import CHECKPOINT_TYPE, Artifact
from gnomad_meb.lib.variant_qc.load import (
    VariantQcImportPipeline,
)
from gnomad_meb.pipeline import FromStep, PipelineRunner, StepRegistry


@pytest.fixture(autouse=True)
def _stubbed_vds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the import stage's per-task ``prepare`` hail-free (ADR-0002)."""
    stub_vds(monkeypatch)


def _task(
    settings: Settings,
    vcf: str = "vqsr_dir",
    as_info: str = "as_info_dir",
    vds: str = "in.vds",
) -> Task:
    return Task(
        settings=settings,
        command=VariantQcLoadVqsr(vds=vds, vcf=vcf, as_info=as_info),
    )


def _pipeline(**overrides: object) -> VariantQcImportPipeline:
    settings = make_settings()
    return VariantQcImportPipeline(
        _task(
            settings,
            vcf=overrides.get("vcf", "vqsr_dir"),
            as_info=overrides.get("as_info", "as_info_dir"),
            vds=overrides.get("vds", "in.vds"),
        )
    )


def _expected_work_steps(tasks: tuple[str, ...] = TASK_NAMES) -> list[str]:
    steps = [f"load_vqsr.{task}" for task in tasks]
    steps.append("merge")
    return steps


def test_group_and_terminal_are_stable() -> None:
    assert VariantQcImportPipeline.name() == "variant_qc"


def test_declares_one_load_step_per_split_plus_merge_in_definition_order() -> None:
    _pipeline()
    expected = _expected_work_steps()

    assert [name for name in StepRegistry().names() if name in expected] == expected


def test_the_whole_callset_load_step_is_absent() -> None:
    _pipeline()
    registry = StepRegistry()

    assert "variant_qc.load_vqsr" not in registry.steps()


def test_load_checkpoints_use_the_per_task_prefixes() -> None:
    settings = make_settings()
    _pipeline()
    registry = StepRegistry()

    for task in TASK_NAMES:
        artifacts = registry.get(f"load_vqsr.{task}").artifacts
        assert artifacts == (
            Artifact(
                prefix=f"{{project}}/intermediate/variant_qc/task-{task}/load_vqsr",
                kind=CHECKPOINT_TYPE.HAIL_TABLE,
            ),
        )
        assert artifacts[0].path(settings) == (
            f"{settings.checkpoint}/{settings.project}"
            f"/intermediate/variant_qc/task-{task}/load_vqsr.ht"
        )


def test_merge_declares_the_terminal_artifact() -> None:
    settings = make_settings()
    _pipeline()
    step = StepRegistry().get("merge")

    assert step.id == "variant_qc.merge"
    assert step.artifacts == (
        Artifact(prefix="{project}/variant_qc", kind=CHECKPOINT_TYPE.HAIL_TABLE),
    )
    assert step.artifacts[0].path(settings) == "/ckpt/proj/variant_qc.ht"


def test_each_load_depends_on_the_vcf_and_as_info_run_inputs() -> None:
    _pipeline(vcf="/data/vqsr", as_info="/data/as_info")
    registry = StepRegistry()

    for task in TASK_NAMES:
        assert registry.get(f"load_vqsr.{task}").depends_on == {
            "vcf": "variant_qc.vcf",
            "as_info": "variant_qc.as_info",
        }
    assert registry["variant_qc.vcf"].value == "/data/vqsr"
    assert registry["variant_qc.as_info"].value == "/data/as_info"


def test_merge_depends_on_every_load_step() -> None:
    _pipeline()
    step = StepRegistry().get("merge")

    assert step.depends_on == {
        task: f"variant_qc.load_vqsr.{task}" for task in TASK_NAMES
    }


def test_generated_steps_carry_their_task_identity() -> None:
    _pipeline()
    registry = StepRegistry()

    for task in TASK_NAMES:
        assert registry.get(f"load_vqsr.{task}").task.command.args == {"task": task}
    assert registry.get("merge").task.command.args == {}


def test_the_import_declares_no_resources() -> None:
    pipeline = _pipeline()

    assert pipeline.resources == ()


def test_prepare_fails_loudly_when_the_vds_handle_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_missing_vds(monkeypatch)

    with pytest.raises(FileNotFoundError):
        _pipeline(vds="/data/does_not_exist.vds")


def test_the_per_task_step_set_round_trips_through_json() -> None:
    task = _task(make_settings(), vcf="/data/vqsr", as_info="/data/as_info")
    runner = PipelineRunner(task, [VariantQcImportPipeline(task)])
    dumped = json.loads(json.dumps(runner.current_state()))

    restored = PipelineRunner.load(dumped)

    # The same inputs rebuilt the identical step set -- no PipelineMismatchError.
    assert set(restored.current_state()["steps"]) == set(dumped["steps"])
    registry = StepRegistry()
    for task_name in TASK_NAMES:
        load = registry.get(f"load_vqsr.{task_name}")
        assert load is not None

        # The rebuilt run callables must bind the same task identity, not merely
        # be callable. ``checkpoint`` wraps a ``functools.partial`` and preserves
        # it under ``__wrapped__`` (see ``lib/checkpoint._callable_source``), so
        # ``prepare``'s bindings stay inspectable.
        assert inspect.unwrap(load.run).keywords == {"task": task_name}
        assert load.task.command.args == {"task": task_name}

    merge = registry.get("merge")
    assert merge is not None
    assert inspect.unwrap(merge.run).keywords == {
        task_name: FromStep(f"load_vqsr.{task_name}") for task_name in TASK_NAMES
    }
    assert merge.depends_on == {
        task_name: f"variant_qc.load_vqsr.{task_name}" for task_name in TASK_NAMES
    }
