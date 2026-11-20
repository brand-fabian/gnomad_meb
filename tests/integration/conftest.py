"""Fixtures for the gnomad-meb integration tests.

Every test runs the pipeline through the CLI as a subprocess, inside the test
container (see docs/adr/0001-... and docs/adr/0002-...).

The example workflow is driven by one session fixture, :func:`gnomad_workflow`,
which runs the ``gnomad-workflow`` operator command through the CLI. Running the
command is the one description *and* execution of the graph: production owns the
steps, the DAG and checkpoint reuse, so there is no test-side step registry.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import harness
import pandas as pd
import pytest
from cli import CliRunner, cli_environment

# --- reference genome parameterization -------------------------------------


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parameterize the session-scoped ``reference`` fixture over ``data/``."""
    if "reference" not in metafunc.fixturenames:
        return

    try:
        references = harness.discover_references()
    except FileNotFoundError as exc:
        raise pytest.UsageError(str(exc)) from exc

    if not references:
        raise pytest.UsageError(
            f"no reference genome datasets found under '{harness.dataset_root()}' "
            "(expected one directory per reference genome)"
        )

    params = []
    for reference in references:
        marks = (
            [pytest.mark.skip(reason=reference.skip_reason)]
            if reference.skip_reason
            else []
        )
        params.append(pytest.param(reference, id=reference.name, marks=marks))
    metafunc.parametrize("reference", params, indirect=True, scope="session")


@pytest.fixture(scope="session", autouse=True)
def enable_gnomad_meb_debug_logging():
    from gnomad_meb.logging import set_level

    set_level("debug")


@pytest.fixture(scope="session")
def reference(request: pytest.FixtureRequest) -> harness.Reference:
    """The reference genome under test (one parameter per dataset under ``data/``)."""
    return request.param


# --- paths and environment --------------------------------------------------


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """The fixture dataset root; a hard error when it is absent."""
    try:
        return harness.require_data_dir()
    except FileNotFoundError as exc:
        pytest.fail(str(exc))


@pytest.fixture(scope="session")
def reference_root() -> Path:
    """The reference-resource root; a hard error when it is absent."""
    try:
        return harness.reference_root()
    except FileNotFoundError as exc:
        pytest.fail(str(exc))


@pytest.fixture(scope="session")
def vep_root() -> Path:
    try:
        return harness.vep_root()
    except FileNotFoundError as exc:
        pytest.fail(str(exc))


@pytest.fixture(scope="session")
def run_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The run's base directory, shared across the session.

    When the entrypoint provides ``GMEB_TEST_RUN_DIR`` (a bind-mounted path),
    use it so the rendered config is valid both inside and outside the
    container, and so a later run over the same directory can skip the
    checkpoints production already declares present. Without it (running pytest
    directly) the session uses pytest's own temporary directory.
    """
    base = os.environ.get(harness.RUN_DIR_ENV)
    if base:
        path = Path(base)
        path.mkdir(parents=True, exist_ok=True)
        return path
    return tmp_path_factory.mktemp("gnomad-meb-tests")


@pytest.fixture(scope="session")
def run_dir(run_root: Path) -> Path:
    """The session's scratch, temp and per-test output directory.

    Kept separate from the pipeline artifacts shared across the session under
    :func:`harness.session_dir`.
    """
    path = run_root / "run"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def session_dir(run_root: Path, reference: harness.Reference) -> Path:
    """The reference-scoped, session-shared directory for pipeline artifacts.

    The VDS, hail checkpoints and VQSR output live here. The path is keyed by
    the reference genome, so a persistent run directory lets production skip
    the checkpoints it already declares **present** rather than recomputing
    them.
    """
    path = harness.session_dir(run_root, reference)
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def output_dir(run_dir: Path, request: pytest.FixtureRequest) -> Path:
    """A per-test output directory under the session's run directory."""
    path = harness.test_output_dir(run_dir, request.node.nodeid)
    path.mkdir(parents=True, exist_ok=True)
    return path


# --- CLI and config ---------------------------------------------------------


@pytest.fixture(scope="session")
def cli() -> CliRunner:
    """A subprocess CLI runner with a clean, slurm-free environment."""
    return CliRunner(cli_environment())


@pytest.fixture(scope="session")
def rendered_config(
    reference: harness.Reference,
    session_dir: Path,
    run_dir: Path,
    reference_root: Path,
    vep_root: Path,
) -> Path:
    """The rendered ``--config`` file for the current reference genome.

    The upstream artifacts (VDS, checkpoints, VQSR output) are scoped to the
    session-shared directory; scratch and hail temp stay in the session's run
    directory.
    """
    return harness.render_config(
        reference,
        run_dir=session_dir,
        checkpoint_dir=session_dir / "checkpoint",
        scratch_dir=run_dir / "scratch",
        hail_tmp=run_dir / "hail-tmp",
        config_dir=run_dir / "config",
        ref_root=reference_root,
        vep_root=vep_root,
    )


# --- fixture oracle ---------------------------------------------------------


@pytest.fixture(scope="session")
def pedigree(reference: harness.Reference) -> pd.DataFrame:
    """The pedigree oracle (``samples.tsv``) for the current reference genome."""
    return harness.load_pedigree(reference)


# --- hail -------------------------------------------------------------------


@pytest.fixture(scope="session")
def hail(run_dir: Path) -> Any:
    """A locally-initialized hail session used to read produced artifacts back.

    Hail is initialized once per session, in a fresh pytest process, against
    the run's temporary directory; no spark cluster is required.
    """
    hl = pytest.importorskip("hail")

    tmp_dir = run_dir / "pytest-hail-tmp"
    local_tmpdir = run_dir / "pytest-hail-local"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    local_tmpdir.mkdir(parents=True, exist_ok=True)

    hl.init(
        master="local[1]",
        idempotent=True,
        tmp_dir=str(tmp_dir),
        local_tmpdir=str(local_tmpdir),
        log=str(run_dir / "pytest-hail.log"),
        quiet=True,
    )
    yield hl
    hl.stop()


# --- the example workflow ---------------------------------------------------


@pytest.fixture(scope="session")
def gnomad_workflow(
    cli: CliRunner,
    rendered_config: Path,
    session_dir: Path,
    reference: harness.Reference,
) -> Path:
    """Run the whole composed workflow once and return a ``Path``.

    ``gnomad-workflow`` is a composite pipeline: constructing it composes every
    stage (VDS build, QC, annotation, filter) and the runner executes the whole
    DAG, skipping **present** checkpoints. Driving it through the CLI seam is
    therefore all the "graph" the suite needs — production owns the steps and
    their dependencies, so no test-side step registry exists to drift.

    The returned ``Path`` points at the Filter stage's high-quality variants
    MatrixTable — not a matrix value — which is the public ``--mt`` input the
    deferred GWAS example consumes. Nothing in this suite reads it yet; the
    composite example asserts its own terminal artifacts. Returning the path
    here avoids re-spelling that internal path.
    """
    cli.run(
        "gnomad-workflow",
        "--search", reference.dataset_dir,
        "--output-base", str(session_dir / "integration.vds"),
        config=rendered_config,
    )
    return (
        harness.project_checkpoint_dir(session_dir, reference)
        / "intermediate/filter/high_quality_gnomad.mt"
    )
