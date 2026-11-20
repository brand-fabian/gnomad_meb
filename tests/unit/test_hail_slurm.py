"""Tests for Slurm-aware CPU detection in ``gnomad_meb.hail``."""

from __future__ import annotations

import pytest

import gnomad_meb.cmd.task  # noqa: F401  (breaks the execution/cmd.run cycle)
from gnomad_meb import hail as hail_module

SLURM_VARS = [
    "SLURM_JOB_ID",
    "SLURM_CPUS_PER_TASK",
    "SLURM_CPUS_ON_NODE",
    "SLURM_JOB_CPUS_PER_NODE",
    "SLURM_TASKS_PER_NODE",
    "SLURM_NTASKS_PER_NODE",
    "SLURM_NTASKS",
    "SLURM_NODELIST",
    "SLURM_JOB_NODELIST",
    "SLURM_NODEID",
    "SLURMD_NODENAME",
]


@pytest.fixture
def slurm_env(monkeypatch):
    """Clear all Slurm variables and the CPU cache so each test is isolated."""
    for name in SLURM_VARS:
        monkeypatch.delenv(name, raising=False)
    hail_module.get_cpu_count.cache_clear()

    def set_env(**values: str) -> None:
        for key, value in values.items():
            monkeypatch.setenv(key, value)

    return set_env


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, []),
        ("", []),
        ("8", [8]),
        ("16(x2),8,16", [16, 16, 8, 16]),
        ("4(x3)", [4, 4, 4]),
    ],
)
def test_expand_slurm_count_list(value, expected):
    assert hail_module._expand_slurm_count_list(value) == expected


@pytest.mark.parametrize(
    ("nodelist", "expected"),
    [
        (None, []),
        ("node01", ["node01"]),
        ("node[01-03]", ["node01", "node02", "node03"]),
        ("node[01-02,05]", ["node01", "node02", "node05"]),
        ("gpu[1-2],cpu07", ["gpu1", "gpu2", "cpu07"]),
    ],
)
def test_expand_slurm_nodelist(nodelist, expected):
    assert hail_module._expand_slurm_nodelist(nodelist) == expected


def test_without_slurm_job_uses_all_worker_cpus(slurm_env, monkeypatch):
    monkeypatch.setattr(hail_module, "cpu_count", lambda: 12)
    assert hail_module.get_cpu_count() == 12


def test_cpus_per_task_wins(slurm_env):
    slurm_env(
        SLURM_JOB_ID="1",
        SLURM_CPUS_PER_TASK="8",
        SLURM_CPUS_ON_NODE="128",
        SLURM_NTASKS_PER_NODE="16",
    )
    assert hail_module.get_cpu_count() == 8


def test_node_specific_lists_use_current_node(slurm_env):
    # Three nodes, CPUs 16/16/8 and tasks 4/4/2. We run on node index 2, so
    # 8 CPUs / 2 tasks = 4.
    slurm_env(
        SLURM_JOB_ID="1",
        SLURM_JOB_CPUS_PER_NODE="16(x2),8",
        SLURM_TASKS_PER_NODE="4(x2),2",
        SLURM_NODELIST="node[01-03]",
        SLURM_NODEID="2",
    )
    assert hail_module.get_cpu_count() == 4


def test_scalar_tasks_per_node_is_broadcast(slurm_env):
    # ``SLURM_NTASKS_PER_NODE`` can only carry a single value.
    slurm_env(
        SLURM_JOB_ID="1",
        SLURM_JOB_CPUS_PER_NODE="16,16",
        SLURM_NTASKS_PER_NODE="4",
        SLURM_NODELIST="node[01-02]",
        SLURM_NODEID="1",
    )
    assert hail_module.get_cpu_count() == 4


def test_compressed_tasks_per_node(slurm_env):
    # Some setups expose the compressed form via ``SLURM_NTASKS_PER_NODE``.
    slurm_env(
        SLURM_JOB_ID="1",
        SLURM_JOB_CPUS_PER_NODE="16(x2),8",
        SLURM_NTASKS_PER_NODE="4(x2),2",
        SLURM_NODEID="2",
    )
    assert hail_module.get_cpu_count() == 4


def test_cpus_on_node_divided_by_tasks(slurm_env):
    # Edge case 2: a node hands out 32 CPUs but four tasks share it.
    slurm_env(
        SLURM_JOB_ID="1",
        SLURM_CPUS_ON_NODE="32",
        SLURM_TASKS_PER_NODE="4",
    )
    assert hail_module.get_cpu_count() == 8


def test_missing_allocation_falls_back_to_worker(slurm_env, monkeypatch):
    monkeypatch.setattr(hail_module, "cpu_count", lambda: 7)
    slurm_env(SLURM_JOB_ID="1")
    assert hail_module.get_cpu_count() == 7


def test_never_returns_zero(slurm_env, monkeypatch):
    monkeypatch.setattr(hail_module, "cpu_count", lambda: 0)
    assert hail_module.get_cpu_count() == 1


class _NoopReporter:
    """Stands in for SparkProgressReporter so the test never boots a JVM."""

    def start(self):
        return self

    def stop(self, timeout: float = 5.0) -> None:
        pass


@pytest.fixture(autouse=True)
def _reset_hail_initialized():
    """``hail_init`` is now idempotent; reset the process flag between tests."""
    hail_module.INITIALIZED = False
    yield
    hail_module.INITIALIZED = False


@pytest.mark.real_hail_init
def test_hail_init_routes_log4j_to_stderr(tmp_path, monkeypatch):
    """``hail_init`` writes the log4j config and points the JVM at it."""
    from gnomad_meb.config import Settings, set_config

    captured: dict = {}

    def _fake_init(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(hail_module.hail, "init", _fake_init)
    monkeypatch.setattr(hail_module.hail, "_set_flags", lambda **_: None)
    monkeypatch.setattr(hail_module, "attach_hail_logging", lambda *a, **k: None)
    monkeypatch.setattr(
        hail_module, "SparkProgressReporter", lambda *a, **k: _NoopReporter()
    )
    monkeypatch.setattr(hail_module, "cpu_count", lambda: 2)
    hail_module.get_cpu_count.cache_clear()
    monkeypatch.setenv("SCRATCH_DIR", str(tmp_path))
    set_config(
        Settings(validate_paths=False, project="demo", scratch_dir=str(tmp_path))
    )

    hail_module.hail_init()

    props = tmp_path / "log4j2.properties"
    assert props.exists()
    conf = captured["spark_conf"]
    for key in ("spark.driver.extraJavaOptions", "spark.executor.extraJavaOptions"):
        assert f"-Dlog4j.configurationFile={props}" in conf[key]


@pytest.mark.real_hail_init
def test_hail_init_is_idempotent(tmp_path, monkeypatch):
    """The second call is a no-op: the JVM is booted exactly once."""
    from gnomad_meb.config import Settings, set_config

    calls: list[int] = []

    def _fake_init(**kwargs):
        calls.append(1)

    monkeypatch.setattr(hail_module.hail, "init", _fake_init)
    monkeypatch.setattr(hail_module.hail, "_set_flags", lambda **_: None)
    monkeypatch.setattr(hail_module, "attach_hail_logging", lambda *a, **k: None)
    monkeypatch.setattr(
        hail_module, "SparkProgressReporter", lambda *a, **k: _NoopReporter()
    )
    monkeypatch.setattr(hail_module, "cpu_count", lambda: 2)
    hail_module.get_cpu_count.cache_clear()
    monkeypatch.setenv("SCRATCH_DIR", str(tmp_path))
    set_config(
        Settings(validate_paths=False, project="demo", scratch_dir=str(tmp_path))
    )

    assert hail_module.is_initialized() is False
    hail_module.hail_init()
    assert hail_module.is_initialized() is True
    hail_module.hail_init()

    assert len(calls) == 1
