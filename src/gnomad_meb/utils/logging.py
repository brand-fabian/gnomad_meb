"""Logging plumbing: one stderr console for every runtime in the process.

The contract this module implements:

**stdout = payload** (the ``--dry-run``/``--describe`` manifest, ``vds inspect``
JSON); **stderr = everything else**. A log line can never corrupt a piping
consumer, so the console here is stderr by design and every log handler is built
over it.

This module owns its own module-level logger, addressed by name
(``logging.getLogger("gnomad_meb")``). It must never import
:mod:`gnomad_meb.logging` (the facade), which imports it back; the import graph
is one-way: ``gnomad_meb.logging`` → ``gnomad_meb.utils.logging`` → stdlib
``logging``.

It also owns the **stdlib root logger**. Third-party libraries (``gnomad``,
``gnomad_qc``) log through root: each module calls ``logging.basicConfig`` at
import, which installs a plain ``StreamHandler`` on root (first import wins;
later calls are no-ops). :func:`configure_logging` clears that handler and puts
the shared console on root too, so those records use the same rich console and
run log. See :func:`configure_logging` for why the root handler is a separate
object at a different level.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from shutil import get_terminal_size

from rich.console import Console
from rich.logging import RichHandler

from gnomad_meb.utils.constants import HAIL_LOG_LEVELS, HAIL_LOG_LINE_PATTERN

FORMAT = "%(message)s"
DATEFORMAT = "%Y-%m-%d %H:%M:%S"

# Use a wider terminal size by default, when writing the log file:
WIDTH = get_terminal_size((200, 24)).columns

ROOT_LOGGER_NAME = "gnomad_meb"
HAIL_LOGGER_NAME = "gnomad_meb.hail"

LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

CONSOLE = Console(
    width=WIDTH,
    tab_size=4,
    stderr=True,
)
HANDLER = RichHandler(console=CONSOLE)
HANDLER.setFormatter(logging.Formatter(fmt=FORMAT, datefmt=DATEFORMAT))

#: The stdlib root logger's rich handler. Third-party libraries (``gnomad``,
#: ``gnomad_qc``) log through root, not ``gnomad_meb``. A separate instance lets
#: a handler-level gate enforce ``-v`` on those records without touching
#: ``HANDLER`` or the independent ``gnomad_meb.hail`` gate -- a level on the
#: shared ``HANDLER`` would bound the hail subtree too.
ROOT_HANDLER = RichHandler(console=CONSOLE)
ROOT_HANDLER.setFormatter(logging.Formatter(fmt=FORMAT, datefmt=DATEFORMAT))

FILE_HANDLER: logging.Handler | None = None
ROOT_FILE_HANDLER: logging.Handler | None = None

#: Filename of the log4j2 configuration written into the scratch dir.
LOG4J_PROPERTIES_NAME = "log4j2.properties"

#: Java logger prefixes routed to the stderr appender by the log4j config.
LOG4J_LOGGERS = ("org.apache.hadoop", "org.apache.spark", "is.hail")


def get_console() -> Console:
    """The process's one rich console.

    stderr by design: stdout is reserved for machine payloads (``--dry-run``/
    ``--describe`` manifests, inspect JSON), so a log line can never corrupt a
    piping consumer.
    """
    return CONSOLE


def get_default_handler() -> RichHandler:
    """The shared ``RichHandler`` over :func:`get_console`."""
    return HANDLER


class HailLogHandler(logging.Handler):
    """Tail hail's log4j file and re-emit each **record** through the rich logger.

    Hail writes java logs (``SparkBackend$``, ``RegionPool``, ``HadoopFS`` ...)
    to a file by design (``hail.init(log=...)``), outside Python's logging. A
    daemon thread tails the file and forwards each record to the
    ``gnomad_meb.hail`` logger, so those records honour the hail log level and
    appear in the run log alongside everything else. The level is parsed from the
    ``LEVEL:`` field and the java logger name becomes the record's logger
    argument; a record whose first line is unparseable is forwarded at WARNING.

    Tailing starts at the **beginning** of the file, not the end, so the boot
    lines log4j writes during ``hail.init`` (which runs before the bridge is
    attached) are captured too.

    A record is not a physical line: log4j's ``%m`` may contain newlines (Hail
    dumps whole IR trees on one record), so physical lines are grouped -- a line
    matching the header pattern starts a new record, every following
    non-matching line is a continuation of the current one. A trailing line with
    no newline yet is held back until it completes.

    Duplication is intentional: ``hail.log`` on disk survives a crash better than
    a tailer, and the same bytes also land in the run log / stderr.
    """

    def __init__(self, path: Path | str, *, poll_interval: float = 0.2):
        super().__init__()
        self.path = Path(path)
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        #: The current record's header plus its continuation lines, held until
        #: the next header (or stop) so a multi-line record is forwarded whole.
        self._pending: list[str] = []

    def start(self) -> HailLogHandler:
        """Start tailing from the beginning of the file on a daemon thread."""
        if self._thread is not None:
            return self
        self._thread = threading.Thread(
            target=self._tail, name="hail-log-tail", daemon=True
        )
        self._thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the tailer (never blocks process exit: the thread is a daemon)."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    def _tail(self) -> None:
        # From the start, not the end: hail truncates/recreates ``hail.log`` per
        # run, and the boot lines log4j writes during ``hail.init`` are exactly
        # what a start-at-EOF tailer would skip. Partial trailing lines are not
        # consumed (``position`` only advances past the last newline) so an
        # in-flight write is not forwarded half-formed.
        position = 0
        while not self._stop_event.wait(self.poll_interval):
            try:
                with self.path.open("r", encoding="utf-8", errors="replace") as fh:
                    fh.seek(position)
                    chunk = fh.read()
            except FileNotFoundError:
                continue
            except Exception:  # pragma: no cover - defensive tailer
                logging.getLogger(HAIL_LOGGER_NAME).debug(
                    "hail log tail failed", exc_info=True
                )
                continue
            complete, newline, rest = chunk.rpartition("\n")
            if newline:
                position += len(complete) + 1
                for line in complete.split("\n"):
                    self._consume(line)
                # Flush at the end of the chunk: hail writes a whole log event
                # (header + any embedded-newline continuations) in one write, so
                # a record is complete here, and flushing bounds the bridge's
                # latency to one poll interval instead of the next header.
                self._flush()
        self._flush()

    def _consume(self, line: str) -> None:
        """Route one complete physical line to the current record or a new one."""
        if HAIL_LOG_LINE_PATTERN.match(line):
            self._flush()
        self._pending.append(line)

    def _flush(self) -> None:
        """Forward the buffered record (header + continuations) as one message."""
        if not self._pending:
            return
        lines, self._pending = self._pending, []
        self.forward_record(lines[0], lines[1:])

    def forward_record(self, header: str, continuation: list[str]) -> None:
        """Re-emit one logical hail record through the ``gnomad_meb.hail`` logger.

        ``header`` is the record's first physical line; ``continuation`` are the
        following lines that belong to the same record (an embedded newline in
        log4j's ``%m``, e.g. a whole IR tree). The continuation lines are joined
        back onto the message so a multi-line record stays one record.
        """
        if not header:
            return
        message = "\n".join([header, *continuation])
        match = HAIL_LOG_LINE_PATTERN.match(header)
        if match is None:
            logging.getLogger(HAIL_LOGGER_NAME).warning(
                message, extra={"markup": False}
            )
            return
        level = HAIL_LOG_LEVELS.get(match.group("level"), logging.WARNING)
        name = f"{HAIL_LOGGER_NAME}.{match.group('logger').strip()}"
        logging.getLogger(name).log(
            level,
            "\n".join([match.group("message"), *continuation]),
            extra={"markup": False},
        )

    def forward(self, line: str) -> None:
        """Re-emit one raw, single-line hail record (kept for direct callers)."""
        if not line:
            return
        self.forward_record(line, [])


def attach_hail_logging(path: Path | str) -> HailLogHandler:
    """Start tailing hail's log file into ``gnomad_meb.hail``."""
    return HailLogHandler(path).start()


@contextmanager
def intercept_jvm_output() -> Iterator[None]:
    """Point OS fd 1 at fd 2 while Hail boots, so no native banner reaches stdout.

    Used around the import of ``gnomad_meb.hail`` (which boots the JVM) and again
    around ``hail.init(...)``. Only the OS file descriptor is swapped
    (``os.dup``/``os.dup2``); Hail's log4j and Spark startup writes go straight to
    fd 1 (unbuffered), so this captures them. Java startup lines are *retained*:
    they land on fd 2 (stderr / the app log) rather than being dropped.

    Note this is not confined to the JVM: Python writes to fd 1 inside the context
    are redirected too, because ``sys.stdout`` normally targets fd 1. Buffered
    Python writes are flushed later, so the reliable rule is to emit the payload
    *after* the context exits. The context must therefore be entered before the
    payload is emitted and exited by the time ``emit`` writes it.
    """
    saved = os.dup(1)
    try:
        os.dup2(2, 1)
        yield
    finally:
        os.dup2(saved, 1)
        os.close(saved)


def write_log4j_properties(
    scratch_dir: Path | str,
    *,
    level: str = "warning",
) -> Path:
    """Write the JVM's ``log4j2.properties`` once and return its path.

    Routes ``org.apache.hadoop``, ``org.apache.spark`` and ``is.hail`` to a
    **stderr** console appender at ``level``, so log4j no longer writes to fd 1
    and its verbosity follows ``Settings.hail.log_level``. Idempotent: an
    identical file is left in place, so every entry point can call it.
    """
    path = Path(scratch_dir) / LOG4J_PROPERTIES_NAME
    path.parent.mkdir(parents=True, exist_ok=True)

    level_name = level.upper()
    lines = [
        "status = error",
        "name = GnomadMebLog4j2",
        "",
        "appender.console.type = Console",
        "appender.console.name = Stderr",
        "appender.console.target = SYSTEM_ERR",
        "appender.console.layout.type = PatternLayout",
        "appender.console.layout.pattern = %d{yyyy-MM-dd HH:mm:ss} %-5p %c{1}: %m%n",
        "",
        "rootLogger.level = " + level_name,
        "rootLogger.appenderRef.stderr.ref = Stderr",
        "",
    ]
    for name in LOG4J_LOGGERS:
        key = name.replace(".", "_").replace("-", "_")
        lines.append(f"logger.{key}.name = {name}")
        lines.append(f"logger.{key}.level = {level_name}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def configure_logging(
    level: str = "info",
    *,
    hail_level: str = "warning",
    log_file: Path | str | None = None,
    console: Console | None = None,
    scratch_dir: Path | str | None = None,
) -> logging.Logger:
    """Install the process's logging: rich stderr handler, optional file handler.

    Idempotent. Sets the root ``gnomad_meb`` level from ``level`` and the
    ``gnomad_meb.hail`` subtree from ``hail_level`` (so hail's java noise can be
    demoted independently). Also takes over the stdlib root logger: it clears
    the plain handler ``gnomad_methods``/``gnomad_qc`` install via
    ``logging.basicConfig`` at import and attaches the shared console (and, when
    ``log_file`` is given, the same file) so those records join our stack. The
    root handlers are separate instances at ``level``: those modules pin their
    loggers to ``INFO``, so only a handler-level gate can enforce ``-v`` on
    them, and not sharing ``HANDLER`` keeps the ``gnomad_meb.hail`` gate
    independent. ``log_file`` adds a file handler at the same level. When
    ``scratch_dir`` is given, the JVM's ``log4j2.properties`` is written there so
    log4j stops writing to fd 1; ``hail_init`` writes it too, so a caller with no
    settings (the config-free ``run``) is still covered. Building here rather
    than at import lets a test construct an isolated logger+console pair without
    touching process globals.
    """
    global HANDLER, FILE_HANDLER, ROOT_HANDLER, ROOT_FILE_HANDLER, CONSOLE

    if console is not None:
        CONSOLE = console
        HANDLER = RichHandler(console=CONSOLE)
        HANDLER.setFormatter(logging.Formatter(fmt=FORMAT, datefmt=DATEFORMAT))
        ROOT_HANDLER = RichHandler(console=CONSOLE)
        ROOT_HANDLER.setFormatter(logging.Formatter(fmt=FORMAT, datefmt=DATEFORMAT))

    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.setLevel(LEVELS[level])
    logger.propagate = False

    if HANDLER not in logger.handlers:
        logger.addHandler(HANDLER)

    if log_file is not None:
        if FILE_HANDLER is not None:
            logger.removeHandler(FILE_HANDLER)
        FILE_HANDLER = logging.FileHandler(str(log_file))
        FILE_HANDLER.setFormatter(logging.Formatter(fmt=FORMAT, datefmt=DATEFORMAT))
        FILE_HANDLER.setLevel(LEVELS[level])
        logger.addHandler(FILE_HANDLER)

    # Third-party libraries log through the stdlib root logger: gnomad_methods
    # and gnomad_qc each call ``logging.basicConfig`` at import, installing a
    # plain StreamHandler on root (first import wins; later calls are no-ops).
    # Clear it and put our handlers there too, so those records use the shared
    # console and run log. The root handler carries ``level`` (gated at the
    # handler, since those modules pin their own loggers to INFO) and stays a
    # separate instance from ``HANDLER`` so the hail subtree gate is unaffected.
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    ROOT_HANDLER.setLevel(LEVELS[level])
    root.addHandler(ROOT_HANDLER)

    if log_file is not None:
        if ROOT_FILE_HANDLER is not None:
            root.removeHandler(ROOT_FILE_HANDLER)
        ROOT_FILE_HANDLER = logging.FileHandler(str(log_file))
        ROOT_FILE_HANDLER.setFormatter(
            logging.Formatter(fmt=FORMAT, datefmt=DATEFORMAT)
        )
        ROOT_FILE_HANDLER.setLevel(LEVELS[level])
        root.addHandler(ROOT_FILE_HANDLER)

    logging.getLogger(HAIL_LOGGER_NAME).setLevel(LEVELS[hail_level])

    if scratch_dir is not None:
        write_log4j_properties(scratch_dir, level=hail_level)

    return logger
