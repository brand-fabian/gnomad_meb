"""Tests for the VQSR stage scaffold and the per-task calling-intervals helper.

The stage is a :class:`~gnomad_meb.pipeline.Pipeline` subclass registering one
Snakemake-backed step whose declared artifacts are every per-task
``{output_dir}/task-{task}/vqsr.vcf.bgz``. Each task's restricted calling list is
computed in plain Python with ``intervaltree`` (no hail, no Snakemake rule). Both
are exercised hail-free: the VDS handle is read for metadata only and stubbed
(ADR-0002).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from variant_qc_helpers import TASK_NAMES, make_settings, stub_missing_vds, stub_vds

from gnomad_meb.cmd.task import Task, VqsrRun
from gnomad_meb.config import Settings, VqsrConfig
from gnomad_meb.lib.checkpoint import Artifact
from gnomad_meb.lib.variant_qc.vqsr import VqsrPipeline, _per_task_intervals
from gnomad_meb.pipeline import PipelineRunner, StepRegistry


@pytest.fixture(autouse=True)
def _stubbed_vds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the metadata reader so pipeline construction stays hail-free."""
    stub_vds(monkeypatch)


def _run(vds: str = "/data/callset.vds") -> VqsrRun:
    return VqsrRun(
        vds=vds,
        as_info="/data/as_info",
        output_dir="out/vqsr",
        calling_intervals="/r/calling.interval_list",
    )


def _pipeline(settings: Settings | None = None, vds: str = "/data/callset.vds"):
    return VqsrPipeline(Task(settings=settings or make_settings(), command=_run(vds)))


# --- the per-task calling intervals ------------------------------------------

#: A small picard interval_list: two overlapping chr1 intervals, one adjacent,
#: and one chr2 interval.
CALLING = (
    "@HD\tVN:1.6\tSO:coordinate\n"
    "@SQ\tSN:chr1\tLN:1000\n"
    "@SQ\tSN:chr2\tLN:500\n"
    "chr1\t100\t200\t+\ta\n"
    "chr1\t180\t300\t+\tb\n"
    "chr1\t301\t400\t+\tc\n"
    "chr2\t50\t60\t+\td\n"
)


def _write_calling(tmp_path) -> object:
    path = tmp_path / "calling.interval_list"
    path.write_text(CALLING)
    return path


def _task(name: str, intervals: list[tuple[str, int, int]]) -> SimpleNamespace:
    """A hail-free stand-in for ``VariantDatasetTaskIntervals`` (inclusive ends)."""
    return SimpleNamespace(
        name=name,
        intervals=tuple(
            SimpleNamespace(
                start=SimpleNamespace(contig=contig, position=start),
                end=SimpleNamespace(contig=contig, position=end),
                includes_start=True,
                includes_end=True,
            )
            for contig, start, end in intervals
        ),
    )


def test_per_task_intervals_intersects_merges_and_writes_picard(tmp_path) -> None:
    calling = _write_calling(tmp_path)
    target = tmp_path / "task-left" / "calling_intervals.interval_list"
    task = _task("left", [("chr1", 50, 310), ("chr2", 1, 100)])

    result = _per_task_intervals(str(calling), task, str(target))

    assert result == str(target)
    lines = target.read_text().splitlines()
    # the @SQ lengths come from the input header, in input order
    assert "@SQ\tSN:chr1\tLN:1000" in lines
    assert "@SQ\tSN:chr2\tLN:500" in lines
    rows = [line for line in lines if not line.startswith("@")]
    # chr1's three overlapping/adjacent pieces collapse to one interval; chr2's
    # single interval survives
    assert rows == [
        "chr1\t100\t310\t+\tchr1:100-310",
        "chr2\t50\t60\t+\tchr2:50-60",
    ]


def test_per_task_intervals_with_no_overlap_keeps_the_header(tmp_path) -> None:
    calling = _write_calling(tmp_path)
    target = tmp_path / "task-right" / "calling_intervals.interval_list"
    task = _task("right", [("chr1", 500, 600)])

    _per_task_intervals(str(calling), task, str(target))

    lines = target.read_text().splitlines()
    assert any(line.startswith("@SQ") for line in lines)
    assert [line for line in lines if not line.startswith("@")] == []


# --- the declared DAG --------------------------------------------------------


def test_registers_a_single_snakemake_step() -> None:
    _pipeline()

    assert set(StepRegistry().steps()) == {
        "vqsr.vds",
        "vqsr.as_info",
        "vqsr.run_vqsr_pipeline",
    }


def test_declares_the_vcfs_tranches_and_provenance() -> None:
    settings = make_settings()
    _pipeline(settings)
    step = StepRegistry().get("vqsr.run_vqsr_pipeline")

    expected = (
        [
            Artifact.from_path(f"out/vqsr/task-{task}/vqsr.vcf.bgz").path(settings)
            for task in TASK_NAMES
        ]
        + [
            Artifact.from_path(f"out/vqsr/vqsr.{mode}.tranches.txt").path(settings)
            for mode in ("snp", "indel")
        ]
        + [Artifact.from_path("out/vqsr/gatk.version.json").path(settings)]
    )
    assert [artifact.path(settings) for artifact in step.artifacts] == expected


def test_step_depends_on_its_input_endpoints() -> None:
    _pipeline(vds="/data/callset.vds")
    registry = StepRegistry()

    # ``as_info`` is consumed by path (the workflow wires its ``source`` to the
    # export step), so it is a declared edge like the VDS handle.
    assert registry.get("vqsr.run_vqsr_pipeline").depends_on == {
        "vds": "vqsr.vds",
        "as_info": "vqsr.as_info",
    }
    assert registry["vqsr.vds"].value == "/data/callset.vds"
    assert registry["vqsr.as_info"].value == "/data/as_info"


def test_step_carries_no_task_identity() -> None:
    _pipeline()

    assert StepRegistry().get("vqsr.run_vqsr_pipeline").task.command.args == {}


# --- the snakemake passthrough -----------------------------------------------


def test_snakemake_args_fall_back_to_the_config() -> None:
    """With no command override, ``[vqsr].snakemake`` supplies the cluster args.

    This is how the composed ``gnomad-workflow`` configures VQSR: it builds a
    ``VqsrRun`` without a passthrough and relies on the setting.
    """
    settings = make_settings(vqsr=VqsrConfig(snakemake=["--profile", "slurm"]))

    assert _pipeline(settings).snakemake_args == ("--profile", "slurm")


def test_snakemake_args_prefer_the_command_override() -> None:
    settings = make_settings(vqsr=VqsrConfig(snakemake=["--profile", "slurm"]))
    command = _run()
    command.snakemake = ("--profile", "kubernetes")

    pipeline = VqsrPipeline(Task(settings=settings, command=command))

    assert pipeline.snakemake_args == ("--profile", "kubernetes")


def test_snakemake_args_are_empty_without_a_config_or_override() -> None:
    assert _pipeline().snakemake_args == ()


def test_prepare_fails_loudly_when_the_vds_handle_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_missing_vds(monkeypatch)

    with pytest.raises(FileNotFoundError):
        _pipeline(vds="/data/does_not_exist.vds")


def test_the_step_set_round_trips_through_json() -> None:
    task = Task(settings=make_settings(), command=_run())
    runner = PipelineRunner(task, [VqsrPipeline(task)])
    dumped = json.loads(json.dumps(runner.current_state()))

    restored = PipelineRunner.load(dumped)

    assert set(restored.current_state()["steps"]) == set(dumped["steps"])
    step = StepRegistry().get("vqsr.run_vqsr_pipeline")
    assert step is not None
    assert step.depends_on == {"vds": "vqsr.vds", "as_info": "vqsr.as_info"}
    assert len(step.artifacts) == len(TASK_NAMES) + 3
