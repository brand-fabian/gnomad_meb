"""Tests for the logging plumbing (``gnomad_meb.utils.logging``).

The contract under test: one shared stderr console, one idempotent
``configure_logging`` that sets the ``gnomad_meb`` root and the
``gnomad_meb.hail`` subtree levels independently, and a facade that keeps the
four historical functions while re-exporting the plumbing. Hail-free: importing
``gnomad_meb.utils.logging`` must not boot a JVM.
"""

from __future__ import annotations

import logging
import os

import pytest

# Import ``cmd.task`` first, as the rest of the suite does: it breaks the
# ``pipeline.execution`` ↔ ``cmd.run`` cycle (a pre-existing ordering
# dependency, not introduced here).
import gnomad_meb.cmd.task  # noqa: F401
from gnomad_meb import logging as facade
from gnomad_meb.utils import logging as plumbing


def test_get_console_is_a_stderr_console() -> None:
    assert plumbing.get_console().stderr is True


def test_rich_handler_uses_the_shared_console() -> None:
    handler = plumbing.get_default_handler()
    assert isinstance(handler, plumbing.RichHandler)
    assert handler.console is plumbing.get_console()


def test_configure_logging_is_idempotent() -> None:
    plumbing.configure_logging(level="info")
    plumbing.configure_logging(level="info")

    root = logging.getLogger("gnomad_meb")
    stderr_handlers = [h for h in root.handlers if isinstance(h, plumbing.RichHandler)]
    assert len(stderr_handlers) == 1


def test_configure_logging_sets_root_and_hail_levels_independently() -> None:
    plumbing.configure_logging(level="info", hail_level="warning")

    assert logging.getLogger("gnomad_meb").level == logging.INFO
    assert logging.getLogger("gnomad_meb.hail").level == logging.WARNING


def test_configure_logging_adds_a_file_handler_at_the_same_level(tmp_path) -> None:
    log_file = tmp_path / "run.log"
    precursor = logging.getLogger("gnomad_meb")
    before = len(precursor.handlers)

    plumbing.configure_logging(level="debug", log_file=log_file)
    after = len(logging.getLogger("gnomad_meb").handlers)

    assert after == before + 1


def test_facade_get_logger_returns_the_root_logger() -> None:
    plumbing.configure_logging(level="error")
    assert facade.get_logger() is logging.getLogger("gnomad_meb")


def test_facade_set_level_and_getters_behave_as_before() -> None:
    facade.set_level("warning")

    assert facade.get_current_level() == "warning"
    assert facade.get_log_level()["info"] == logging.INFO


def test_facade_reexports_the_plumbing_surface() -> None:
    assert facade.configure_logging is plumbing.configure_logging
    assert facade.get_console is plumbing.get_console


def test_utils_logging_imports_without_a_cycle() -> None:
    import importlib

    module = importlib.import_module("gnomad_meb.utils.logging")
    assert module is plumbing


# --- intercept_jvm_output (ticket 02) ----------------------------------------


def test_intercept_jvm_output_sends_fd1_writes_to_fd2(tmp_path) -> None:
    """A raw ``os.write(1, ...)`` inside the window lands on fd 2."""
    capture = tmp_path / "stderr.txt"
    saved_stderr = os.dup(2)
    fd_stderr = os.open(capture, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.dup2(fd_stderr, 2)
    try:
        with plumbing.intercept_jvm_output():
            os.write(1, b"jvm-banner\n")
    finally:
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)
        os.close(fd_stderr)

    assert capture.read_bytes() == b"jvm-banner\n"


def test_intercept_jvm_output_restores_fd1_after_the_window(tmp_path) -> None:
    capture = tmp_path / "stdout.txt"
    saved_stdout = os.dup(1)
    fd_stdout = os.open(capture, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.dup2(fd_stdout, 1)
    try:
        with plumbing.intercept_jvm_output():
            os.write(1, b"inside\n")
        os.write(1, b"after\n")
    finally:
        os.dup2(saved_stdout, 1)
        os.close(saved_stdout)
        os.close(fd_stdout)

    assert capture.read_bytes() == b"after\n"


def test_intercept_jvm_output_restores_fd1_when_the_body_raises(tmp_path) -> None:
    capture = tmp_path / "stdout.txt"
    saved_stdout = os.dup(1)
    fd_stdout = os.open(capture, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.dup2(fd_stdout, 1)
    try:
        with pytest.raises(RuntimeError):
            with plumbing.intercept_jvm_output():
                raise RuntimeError("boom")
        os.write(1, b"recovered\n")
    finally:
        os.dup2(saved_stdout, 1)
        os.close(saved_stdout)
        os.close(fd_stdout)

    assert capture.read_bytes() == b"recovered\n"


def test_facade_reexports_intercept_jvm_output() -> None:
    assert facade.intercept_jvm_output is plumbing.intercept_jvm_output


# --- hail level + log4j properties (ticket 03) -------------------------------


def test_configure_logging_writes_a_log4j_properties_file(tmp_path) -> None:
    path = plumbing.write_log4j_properties(tmp_path, level="info")

    text = path.read_text()
    assert path.name == "log4j2.properties"
    for prefix in ("org.apache.hadoop", "org.apache.spark", "is.hail"):
        assert prefix in text
    assert "Console" in text  # the stderr console appender


def test_facade_does_not_need_to_reexport_log4j(tmp_path) -> None:
    # The writer is plumbing, consumed by hail_init; no facade requirement.
    assert plumbing.write_log4j_properties


# --- HailLogHandler / attach_hail_logging (ticket 04) ------------------------


def _wait_for(predicate, timeout=5.0):
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def _records(caplog_records, message_fragment):
    return [r for r in caplog_records if message_fragment in r.getMessage()]


def test_attach_hail_logging_bridges_java_lines(tmp_path) -> None:
    import logging as _logging

    log_path = tmp_path / "hail.log"
    log_path.write_text("")

    records: list[_logging.LogRecord] = []

    class _Collector(_logging.Handler):
        def emit(self, record):
            records.append(record)

    collector = _Collector()
    hail_logger = _logging.getLogger("gnomad_meb.hail")
    hail_logger.addHandler(collector)
    plumbing.configure_logging(level="info", hail_level="info")
    try:
        handler = plumbing.attach_hail_logging(log_path)
        with log_path.open("a") as fh:
            fh.write("RegionPool: INFO: initialized for thread 1\n")
            fh.flush()
        assert _wait_for(lambda: _records(records, "initialized for thread 1"))
        record = _records(records, "initialized for thread 1")[0]
        assert record.levelno == _logging.INFO
        assert record.name == "gnomad_meb.hail.RegionPool"
        handler.stop()
    finally:
        hail_logger.removeHandler(collector)


def test_attach_hail_logging_defaults_unparseable_lines_to_warning(tmp_path) -> None:
    import logging as _logging

    log_path = tmp_path / "hail.log"
    log_path.write_text("")

    records: list[_logging.LogRecord] = []

    class _Collector(_logging.Handler):
        def emit(self, record):
            records.append(record)

    collector = _Collector()
    hail_logger = _logging.getLogger("gnomad_meb.hail")
    hail_logger.addHandler(collector)
    plumbing.configure_logging(level="info", hail_level="info")
    try:
        handler = plumbing.attach_hail_logging(log_path)
        with log_path.open("a") as fh:
            fh.write("no level field here\n")
            fh.flush()
        assert _wait_for(lambda: _records(records, "no level field here"))
        record = _records(records, "no level field here")[0]
        assert record.levelno == _logging.WARNING
        handler.stop()
    finally:
        hail_logger.removeHandler(collector)


def test_hail_info_is_demoted_by_the_hail_level(tmp_path) -> None:
    import logging as _logging

    plumbing.configure_logging(level="info", hail_level="error")
    try:
        hail_logger = _logging.getLogger("gnomad_meb.hail")
        assert hail_logger.isEnabledFor(_logging.INFO) is False
        assert _logging.getLogger("gnomad_meb.hail.RegionPool").isEnabledFor(
            _logging.INFO
        ) is False
    finally:
        plumbing.configure_logging(level="info", hail_level="warning")


def test_bridge_survives_rich_markup_in_the_line(tmp_path) -> None:
    import logging as _logging

    log_path = tmp_path / "hail.log"
    log_path.write_text("")

    records: list[_logging.LogRecord] = []

    class _Collector(_logging.Handler):
        def emit(self, record):
            records.append(record)

    collector = _Collector()
    hail_logger = _logging.getLogger("gnomad_meb.hail")
    hail_logger.addHandler(collector)
    plumbing.configure_logging(level="info", hail_level="info")
    try:
        handler = plumbing.attach_hail_logging(log_path)
        with log_path.open("a") as fh:
            fh.write("is.hail: INFO: braces {like} [rich] markup\n")
            fh.flush()
        assert _wait_for(lambda: _records(records, "rich"))
        handler.stop()
    finally:
        hail_logger.removeHandler(collector)


def test_facade_reexports_the_hail_bridge() -> None:
    assert facade.HailLogHandler is plumbing.HailLogHandler
    assert facade.attach_hail_logging is plumbing.attach_hail_logging


# --- multi-line records + startup capture (ticket 05) ------------------------


def _collector(records):
    import logging as _logging

    class _Collector(_logging.Handler):
        def emit(self, record):
            records.append(record)

    return _Collector()


def test_tailer_groups_multiline_records_into_one_record(tmp_path) -> None:
    """An IR dump spanning many physical lines is re-emitted as one record."""
    import logging as _logging

    log_path = tmp_path / "hail.log"
    log_path.write_text("")

    records: list[_logging.LogRecord] = []
    collector = _collector(records)
    hail_logger = _logging.getLogger("gnomad_meb.hail")
    hail_logger.addHandler(collector)
    plumbing.configure_logging(level="info", hail_level="info")
    try:
        handler = plumbing.attach_hail_logging(log_path)
        with log_path.open("a") as fh:
            fh.write(
                "2026-11-12 11:20:54.815 LoweringPipeline: INFO: initial IR: "
                "IR size 2: \n"
                "\n"
                "%1 = RNGStateLiteral\n"
                "!mt = MatrixRead [Matrix{global:Struct{}}]\n"
                "  %2 = GetField(%col) [s]\n"
            )
            fh.flush()
        assert _wait_for(lambda: _records(records, "RNGStateLiteral"))
        handler.stop()
    finally:
        hail_logger.removeHandler(collector)

    # Exactly one record, from the header's logger, at the header's level.
    assert len(records) == 1
    record = records[0]
    assert record.name == "gnomad_meb.hail.LoweringPipeline"
    assert record.levelno == _logging.INFO
    # Header message and every continuation line live in the one message.
    message = record.getMessage()
    assert "initial IR: IR size 2:" in message
    assert "%1 = RNGStateLiteral" in message
    assert "!mt = MatrixRead" in message
    assert "%2 = GetField(%col) [s]" in message


def test_tailer_emits_pre_existing_startup_lines(tmp_path) -> None:
    """Boot lines written before the bridge attaches are still captured."""
    import logging as _logging

    log_path = tmp_path / "hail.log"
    log_path.write_text(
        "2026-11-12 11:20:49.647 SparkBackend$: WARN: This Hail JAR was compiled "
        "for Spark 3.5.3, running with Spark 3.5.9.\n"
        "  Compatibility is not guaranteed.\n"
        "2026-11-12 11:20:50.053 py4jutils$: INFO: Hail 0.2.138-58956ebc28fc\n"
    )

    records: list[_logging.LogRecord] = []
    collector = _collector(records)
    hail_logger = _logging.getLogger("gnomad_meb.hail")
    hail_logger.addHandler(collector)
    plumbing.configure_logging(level="info", hail_level="info")
    try:
        handler = plumbing.attach_hail_logging(log_path)
        assert _wait_for(lambda: _records(records, "Hail 0.2.138-58956ebc28fc"))
        handler.stop()
    finally:
        hail_logger.removeHandler(collector)

    startup = _records(records, "Compatibility is not guaranteed")
    assert startup, "startup lines written before attach must be captured"
    assert startup[0].name == "gnomad_meb.hail.SparkBackend$"
    assert startup[0].levelno == _logging.WARNING
    assert _records(records, "0.2.138")


def test_tailer_holds_back_an_incomplete_trailing_line(tmp_path) -> None:
    """A line with no newline yet is not forwarded until it completes."""
    import logging as _logging
    import time

    log_path = tmp_path / "hail.log"
    log_path.write_text("")

    records: list[_logging.LogRecord] = []
    collector = _collector(records)
    hail_logger = _logging.getLogger("gnomad_meb.hail")
    hail_logger.addHandler(collector)
    plumbing.configure_logging(level="info", hail_level="info")
    try:
        handler = plumbing.attach_hail_logging(log_path)
        with log_path.open("a") as fh:
            fh.write("RegionPool: INFO: initiali")  # deliberately incomplete
            fh.flush()
        time.sleep(0.5)
        assert not _records(records, "initiali"), "partial line leaked"
        with log_path.open("a") as fh:
            fh.write("zed for thread 1\n")
            fh.flush()
        assert _wait_for(lambda: _records(records, "initialized for thread 1"))
        handler.stop()
    finally:
        hail_logger.removeHandler(collector)


def test_forward_record_joins_continuations() -> None:
    """The single-record entry point joins continuations onto the header message."""
    import logging as _logging

    records: list[_logging.LogRecord] = []
    collector = _collector(records)
    logger = _logging.getLogger("gnomad_meb.hail.TestLogger")
    logger.addHandler(collector)
    logger.setLevel(_logging.INFO)
    try:
        plumbing.HailLogHandler("unused.log").forward_record(
            "TestLogger: INFO: header line",
            ["continuation one", "continuation two"],
        )
    finally:
        logger.removeHandler(collector)

    assert len(records) == 1
    assert records[0].getMessage() == (
        "header line\ncontinuation one\ncontinuation two"
    )


# --- stdlib root takeover (gnomad_methods' ``logging.basicConfig``) ----------


def test_configure_logging_takes_over_the_stdlib_root_logger() -> None:
    """A ``basicConfig``-style handler on root is replaced by ours, not doubled."""
    root = logging.getLogger()
    stale = logging.StreamHandler()
    root.addHandler(stale)
    try:
        plumbing.configure_logging(level="info")

        assert stale not in root.handlers
        rich = [h for h in root.handlers if isinstance(h, plumbing.RichHandler)]
        assert rich == [plumbing.ROOT_HANDLER]
        assert plumbing.ROOT_HANDLER.level == logging.INFO
    finally:
        root.removeHandler(stale)


def test_configure_logging_does_not_stack_root_handlers() -> None:
    """Repeated calls leave exactly one root handler (idempotent)."""
    plumbing.configure_logging(level="info")
    plumbing.configure_logging(level="info")

    root = logging.getLogger()
    rich = [h for h in root.handlers if isinstance(h, plumbing.RichHandler)]
    assert len(rich) == 1


def test_gnomad_records_reach_the_root_handler() -> None:
    """A ``gnomad.*`` record reaches our root handler; ``gnomad_meb`` does not."""
    import logging as _logging

    records: list[_logging.LogRecord] = []
    collector = _collector(records)

    plumbing.configure_logging(level="info")
    root = _logging.getLogger()
    root.addHandler(collector)
    try:
        gnomad_logger = _logging.getLogger("gnomad.utils.vcf")
        gnomad_logger.setLevel(_logging.INFO)
        gnomad_logger.info("from gnomad")

        _logging.getLogger("gnomad_meb").info("from the app")
    finally:
        root.removeHandler(collector)

    names = [r.name for r in records]
    assert "gnomad.utils.vcf" in names
    # ``gnomad_meb`` sets propagate=False, so its records never reach root.
    assert "gnomad_meb" not in names


def test_root_handler_is_gated_at_the_app_level() -> None:
    """gnomad pins its loggers to INFO; only the handler level can demote them."""
    plumbing.configure_logging(level="warning")
    assert plumbing.ROOT_HANDLER.level == logging.WARNING

    plumbing.configure_logging(level="debug")
    assert plumbing.ROOT_HANDLER.level == logging.DEBUG
