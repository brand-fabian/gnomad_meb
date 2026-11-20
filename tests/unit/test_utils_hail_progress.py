"""Tests for the Spark progress reporter (``gnomad_meb.utils.hail_progress``).

The reporter polls ``sc.statusTracker()`` from a background thread and renders
stage progress through the process's rich console (live) or as log lines
(non-terminal, the SLURM case). Hail-free: importing the module must not boot a
JVM, so these tests drive a fake tracker through ``_poll`` directly.
"""

from __future__ import annotations

import importlib
import sys

import gnomad_meb.cmd.task  # noqa: F401  (breaks the execution/cmd.run cycle)
from gnomad_meb.utils import hail_progress


class _FakeStageInfo:
    def __init__(
        self, name: str, total: int, done: int, running: int = 0, failed: int = 0
    ):
        self.name = name
        self.numTasks = total
        self.numCompletedTasks = done
        self.numActiveTasks = running
        self.numFailedTasks = failed


class _FakeTracker:
    def __init__(self, stages: dict[int, _FakeStageInfo]):
        self._stages = stages

    def getActiveStageIds(self):
        return list(self._stages)

    def getStageInfo(self, sid):
        return self._stages.get(sid)


def _reporter(monkeypatch, tracker, **kwargs):
    reporter = hail_progress.SparkProgressReporter(live=False, **kwargs)
    monkeypatch.setattr(
        reporter, "_tracker", lambda: tracker, raising=False
    )
    # hail is imported lazily inside _poll; substitute the tracker lookup.
    monkeypatch.setattr(
        hail_progress, "get_status_tracker", lambda: tracker, raising=False
    )
    return reporter


def test_importing_the_module_does_not_import_hail() -> None:
    sys.modules.pop("hail", None)
    importlib.reload(hail_progress)
    assert "hail" not in sys.modules


def test_non_live_mode_logs_start_and_finish(monkeypatch, capturing_logger) -> None:
    logger, stream = capturing_logger
    tracker = _FakeTracker({1: _FakeStageInfo("scan", total=10, done=3, running=1)})
    reporter = hail_progress.SparkProgressReporter(
        live=False, logger=logger, poll_interval=0.0, min_seconds=0.0
    )
    monkeypatch.setattr(hail_progress, "get_status_tracker", lambda: tracker)

    reporter._poll()  # stage 1 active
    tracker._stages = {}
    reporter._poll()  # stage 1 finished

    output = stream.getvalue()
    assert "stage 1" in output
    assert "scan" in output
    assert output.count("INFO") >= 1
    assert "finished" in output


def test_live_mode_adds_and_removes_a_rich_task(monkeypatch) -> None:
    from rich.console import Console

    console = Console(file=open("/dev/null", "w"), force_terminal=False)
    tracker = _FakeTracker({1: _FakeStageInfo("scan", total=10, done=3)})
    reporter = hail_progress.SparkProgressReporter(
        live=True, console=console, min_seconds=0.0
    )
    reporter._progress = hail_progress.Progress(console=console, transient=True)
    reporter._progress.start()
    monkeypatch.setattr(hail_progress, "get_status_tracker", lambda: tracker)

    reporter._poll()  # adds a task
    assert reporter._stages[1].task_id is not None

    tracker._stages = {}
    reporter._poll()  # removes it
    assert 1 not in reporter._stages

    reporter._progress.stop()


def test_stage_under_min_seconds_is_skipped(monkeypatch, capturing_logger) -> None:
    logger, stream = capturing_logger
    tracker = _FakeTracker({1: _FakeStageInfo("quick", total=10, done=3)})
    reporter = hail_progress.SparkProgressReporter(
        live=False, logger=logger, min_seconds=10_000.0
    )
    monkeypatch.setattr(hail_progress, "get_status_tracker", lambda: tracker)

    reporter._poll()

    assert stream.getvalue() == ""
