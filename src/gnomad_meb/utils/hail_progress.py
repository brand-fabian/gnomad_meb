"""Spark progress, reported through the process's rich console / logger.

Spark's own console progress bar writes raw carriage-return frames directly to
stderr, outside rich, so it interleaves badly with ``RichHandler`` output. Hail
is initialised with ``show_progress=False`` and this reporter polls the same
``sc.statusTracker()`` Spark's bar uses, so the numbers are the same but the
rendering is ours. Two modes:

- live (interactive terminal): a rich ``Progress`` bar per active stage, on the
  same console as the log handler, so log lines print *above* the bars.
- log lines (non-terminal, e.g. a SLURM output file): one INFO line when a stage
  starts, whenever it crosses another ``log_step`` percent, at least every
  ``log_interval`` seconds, and when it finishes.

``hail`` is imported lazily inside :func:`get_status_tracker`, so importing this
module never boots a JVM (unit tests can drive the reporter directly).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

HAIL_PROGRESS_LOGGER = "gnomad_meb.hail.progress"


def get_status_tracker():
    """Return Spark's status tracker, importing hail only when first needed."""
    import hail as hl

    return hl.spark_context().statusTracker()


@dataclass
class Stage:
    first_seen: float
    name: str = ""
    total: int = 0
    done: int = 0
    running: int = 0
    failed: int = 0
    task_id: int | None = None
    reported: bool = False
    last_bucket: int = -1
    last_log: float = 0.0


class SparkProgressReporter:
    def __init__(
        self,
        console: Console | None = None,
        logger: logging.Logger | None = None,
        *,
        poll_interval: float = 0.5,
        min_seconds: float = 1.0,
        log_step: int = 10,
        log_interval: float = 60.0,
        live: bool | None = None,
        max_name: int = 40,
    ):
        from gnomad_meb.utils.logging import get_console

        self.console = console or get_console()
        self.logger = logger or logging.getLogger(HAIL_PROGRESS_LOGGER)
        self.poll_interval = poll_interval
        self.min_seconds = min_seconds
        self.log_step = log_step
        self.log_interval = log_interval
        self.live = self.console.is_terminal if live is None else live
        self.max_name = max_name
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._progress: Progress | None = None
        self._stages: dict[int, Stage] = {}

    # -- lifecycle -------------------------------------------------------
    def start(self) -> SparkProgressReporter:
        if self._thread is not None:
            return self
        if self.live:
            self._progress = Progress(
                TextColumn("{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=self.console,
                transient=True,
            )
            self._progress.start()
        self._thread = threading.Thread(
            target=self._run, name="spark-progress", daemon=True
        )
        self._thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None
        if self._progress is not None:
            self._progress.stop()
            self._progress = None

    def __enter__(self) -> SparkProgressReporter:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- internals -------------------------------------------------------
    def _run(self) -> None:
        while not self._stop_event.wait(self.poll_interval):
            try:
                self._poll()
            except Exception:  # e.g. SparkContext stopped while we poll
                self.logger.debug("progress poll failed", exc_info=True)

    def _label(self, sid: int, st: Stage) -> str:
        name = (st.name or "").strip()
        if len(name) > self.max_name:
            name = name[: self.max_name - 1] + "…"
        return f"stage {sid} [{name}]"

    def _poll(self) -> None:
        tracker = get_status_tracker()
        now = time.monotonic()
        active = set(tracker.getActiveStageIds())

        for sid in active:
            info = tracker.getStageInfo(sid)
            if info is None:
                continue
            st = self._stages.setdefault(sid, Stage(first_seen=now))
            st.name, st.total = info.name, info.numTasks
            st.done = info.numCompletedTasks
            st.running = info.numActiveTasks
            st.failed = info.numFailedTasks
            if now - st.first_seen >= self.min_seconds:
                self._report(sid, st, now)

        for sid in [s for s in self._stages if s not in active]:
            self._finish(sid, self._stages.pop(sid), tracker, now)

    def _report(self, sid: int, st: Stage, now: float) -> None:
        run_state = f"{st.running} running" + (
            f", {st.failed} failed)" if st.failed else ")"
        )
        desc = f"{self._label(sid, st)} ({run_state}"
        if self.live and self._progress is not None:
            if st.task_id is None:
                st.task_id = self._progress.add_task(
                    desc, total=max(st.total, 1), completed=st.done
                )
            else:
                self._progress.update(st.task_id, description=desc, completed=st.done)
            st.reported = True
            return

        pct = 100.0 if st.total == 0 else 100.0 * st.done / st.total
        bucket = int(pct // self.log_step)
        crossed = bucket > st.last_bucket or now - st.last_log >= self.log_interval
        if not st.reported or crossed:
            self._log_line(sid, st, pct)
            st.last_bucket, st.last_log = bucket, now
        st.reported = True

    def _log_line(self, sid: int, st: Stage, pct: float) -> None:
        msg = (
            f"{self._label(sid, st)}: {st.done}/{st.total} tasks "
            f"({pct:.0f}%), {st.running} running"
        )
        if st.failed:
            msg += f", {st.failed} failed"
        self.logger.info(msg, extra={"markup": False})

    def _finish(self, sid: int, st: Stage, tracker, now: float) -> None:
        info = tracker.getStageInfo(sid)
        if info is not None:
            st.done = info.numCompletedTasks
            st.failed = info.numFailedTasks
            st.total = info.numTasks
        if st.task_id is not None and self._progress is not None:
            self._progress.remove_task(st.task_id)
        if st.reported:
            self.logger.info(
                "%s finished: %d/%d tasks%s in ~%.0f s",
                self._label(sid, st), st.done, st.total,
                f", {st.failed} failed" if st.failed else "",
                now - st.first_seen,
                extra={"markup": False},
            )
