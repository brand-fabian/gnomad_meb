"""The subprocess CLI seam the integration tests drive the pipeline through.

The suite invokes ``main.py`` as a subprocess, in a fresh process per call (see
``docs/adr/0001-integration-tests-use-subprocess-cli-seam.md``). This module owns
that seam: the environment a subprocess runs in and the runner that invokes it.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import harness
import pytest


def cli_environment() -> dict[str, str]:
    """Environment for a CLI subprocess.

    ``SPARK_MASTER_IP``/``SPARK_MASTER_PORT`` are removed so hail falls back to
    local mode; no slurm environment variables are required.
    """
    env = dict(os.environ)
    env.pop("SPARK_MASTER_IP", None)
    env.pop("SPARK_MASTER_PORT", None)
    return env


@dataclass
class CliResult:
    """The result of one CLI invocation."""

    command: Sequence[str]
    returncode: int
    stdout: str
    stderr: str
    log: Path | None = None

    def __str__(self) -> str:
        location = f"\nlog: {self.log}" if self.log is not None else ""
        return (
            f"command: {' '.join(self.command)}\n"
            f"exit code: {self.returncode}{location}\n"
            f"--- stdout ---\n{self.stdout}\n"
            f"--- stderr ---\n{self.stderr}"
        )


class CliRunner:
    """Invoke the pipeline CLI as a subprocess, in a fresh process per call."""

    def __init__(self, env: dict[str, str], uv: str = "uv") -> None:
        self.env = env
        self.uv = uv

    def run(
        self,
        *args: object,
        config: str | Path | None = None,
        step: str | None = None,
        check: bool = True,
        timeout: float | None = None,
        log: str | Path | None = None,
    ) -> CliResult:
        """Run ``uv run --project <repo> python3 main.py --config … <args>``.

        The subprocess's stdout and stderr are streamed live to this process's
        streams as they are produced, so long-running builds are not silent, and
        accumulated into the result. When ``log`` is given, the same output is
        also teed to that file.

        On failure (non-zero exit and ``check``) the command, exit code and
        captured output are reported.
        """
        command = [
            self.uv,
            "run",
            "--project",
            str(harness.REPO_ROOT),
            "python3",
            str(harness.MAIN),
        ]
        if config is not None:
            command += ["--config", str(config)]
        command += [str(arg) for arg in args]

        log_path = Path(log) if log is not None else None
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)

        stdout: list[str] = []
        stderr: list[str] = []
        log_handle = log_path.open("w") if log_path is not None else None
        log_lock = threading.Lock()

        def pump(stream: Any, buffer: list[str], live: Any) -> None:
            for line in stream:
                buffer.append(line)
                live.write(line)
                live.flush()
                if log_handle is not None:
                    with log_lock:
                        log_handle.write(line)
                        log_handle.flush()
            stream.close()

        try:
            process = subprocess.Popen(  # noqa: S603
                command,
                cwd=str(harness.REPO_ROOT),
                env=self.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            readers = (
                threading.Thread(
                    target=pump, args=(process.stdout, stdout, sys.stdout), daemon=True
                ),
                threading.Thread(
                    target=pump, args=(process.stderr, stderr, sys.stderr), daemon=True
                ),
            )
            for reader in readers:
                reader.start()
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                raise
            finally:
                for reader in readers:
                    reader.join()
        finally:
            if log_handle is not None:
                log_handle.close()

        result = CliResult(
            tuple(command),
            process.returncode,
            "".join(stdout),
            "".join(stderr),
            log_path,
        )
        if check and process.returncode != 0:
            label = f"step {step!r} failed" if step is not None else "CLI failed"
            pytest.fail(f"{label}\n{result}")
        return result
