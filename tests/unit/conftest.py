"""Shared fixtures for the in-process unit tests.

A single, local, single-core hail session serves every unit test that needs
reference-contig lengths or hail objects; it is initialised lazily and skipped
when the host has no working JVM (the local dev caveat in ``AGENTS.md``).
"""

from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def capturing_logger() -> tuple[logging.Logger, io.StringIO]:
    """A private logger writing ``<level> <message>`` lines to a string buffer."""
    stream = io.StringIO()
    logger = logging.getLogger("test.capturing")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger, stream


@pytest.fixture(autouse=True)
def _reset_pipeline_registries():
    """Reset the process-global registries and config between tests."""
    import gnomad_meb.config as config_module
    from gnomad_meb.pipeline.registry import ResourceRegistry
    from gnomad_meb.pipeline.step import StepRegistry

    config_module._config = None
    config_module._config_path = None
    StepRegistry._reset()
    ResourceRegistry._reset()
    yield
    StepRegistry._reset()
    ResourceRegistry._reset()


@pytest.fixture(autouse=True)
def _no_hail_boot(request, monkeypatch):
    """Keep the unit suite off the Hail/Spark stack (AGENTS.md).

    Pipeline commands boot Hail through ``hail_init`` (guarded, idempotent). A
    unit test that drives such a command through the in-process click runner must
    not start a JVM, so ``hail_init`` is replaced by a no-op by default. Tests
    that exercise the boot itself (the fd probes, ``test_hail_slurm``) opt out
    with the ``real_hail_init`` marker.
    """
    if request.node.get_closest_marker("real_hail_init"):
        return

    import gnomad_meb.hail as hail_module

    monkeypatch.setattr(hail_module, "hail_init", lambda: None)


@pytest.fixture(scope="session")
def hl():
    """A local, single-core hail session for reference-contig lengths."""
    import hail as hail_module

    tmp_dir = Path(tempfile.mkdtemp(prefix="unit-hail-"))
    try:
        hail_module.init(
            master="local[1]",
            idempotent=True,
            quiet=True,
            tmp_dir=str(tmp_dir / "tmp"),
            local_tmpdir=str(tmp_dir / "local"),
            log=str(tmp_dir / "hail.log"),
        )
    except Exception as exc:  # pragma: no cover - environment without a JVM
        pytest.skip(f"hail session unavailable: {exc}")
    yield hail_module
    hail_module.stop()
