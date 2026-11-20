"""The integration-test fixture environment.

The suite drives the pipeline through the same command-line interface the slurm
wrappers use, invoking ``main.py`` as a subprocess (see
``docs/adr/0001-integration-tests-use-subprocess-cli-seam.md``). The
site-specific ``--config`` file is rendered from a committed per-reference
template (see ``docs/adr/0003-integration-tests-render-config-from-template.md``).

This module owns the fixture *environment*: discovering reference genomes,
loading the Pedigree, and rendering the per-reference config. The Expected
values live in :mod:`oracle` and the subprocess seam in :mod:`cli`.
"""

from __future__ import annotations

import os
import string
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
ORACLE_DIR = Path(__file__).resolve().parent / "data"
MAIN = REPO_ROOT / "main.py"

#: The tracked pedigree file, one per reference genome.
PEDIGREE_FILE = "samples.tsv"

DATA_DIR_ENV = "GMEB_TEST_DATA_DIR"
REF_ROOT_ENV = "GMEB_TEST_REF_ROOT"
VEP_ROOT_ENV = "GMEB_TEST_VEP_ROOT"
RUN_DIR_ENV = "GMEB_TEST_RUN_DIR"

#: Sub-directory under the run root holding the pipeline artifacts shared across
#: the session, kept under ``SHARED_DIR/<project>`` (see :func:`session_dir`).
SHARED_DIR = "shared"

#: Reference builds the pipeline CLI knows how to run.
SUPPORTED_REFERENCES = ("GRCh38", "GRCh37")

#: Recognised-but-unsupported reference builds, mapped to the reason they are
#: skipped rather than tested.
UNSUPPORTED_REFERENCES = {
    "CHM13": (
        "CHM13/T2T is not a runnable parameter yet: the CLI and downstream "
        "modules hard-code GRCh38 (see the integration-test spec, Out of Scope)"
    ),
}

_REFERENCE_ALIASES = {
    "grch38": "GRCh38",
    "hg38": "GRCh38",
    "grch37": "GRCh37",
    "hg19": "GRCh37",
    "b37": "GRCh37",
    "chm13": "CHM13",
    "t2t": "CHM13",
}


@dataclass(frozen=True)
class Reference:
    """A reference genome discoverable from a ``data/`` dataset directory."""

    name: str
    dataset_dir: Path | None = None
    skip_reason: str | None = None


def normalize_reference(name: str) -> str | None:
    """Map a dataset directory name onto a canonical reference name."""
    return _REFERENCE_ALIASES.get(name.strip().lower())


def dataset_root() -> Path:
    """The fixture dataset root, from the environment or ``<repo>/data``."""
    return Path(os.environ.get(DATA_DIR_ENV, REPO_ROOT / "data"))


def require_data_dir() -> Path:
    """The fixture dataset root, or a hard error naming the missing path."""
    path = dataset_root()
    if not path.is_dir():
        raise FileNotFoundError(
            f"missing test data directory: '{path}' (set {DATA_DIR_ENV})"
        )
    return path


def discover_references(data_dir: Path | None = None) -> list[Reference]:
    """Discover reference genomes from the dataset directories under ``data/``.

    Raises ``FileNotFoundError`` naming the path when the dataset root is
    absent, so a missing dataset is a hard error rather than a spurious pass.
    Reference directories that are recognised but unsupported (CHM13) are
    returned with an explicit ``skip_reason``.
    """
    data_dir = Path(data_dir) if data_dir is not None else dataset_root()
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"missing test data directory: '{data_dir}' (set {DATA_DIR_ENV})"
        )

    references: list[Reference] = []
    for entry in sorted(data_dir.iterdir()):
        if not entry.is_dir():
            continue
        canonical = normalize_reference(entry.name)
        if canonical is None:
            reason = f"unrecognised reference genome directory: '{entry.name}'"
            references.append(Reference(entry.name, entry, reason))
        elif canonical not in SUPPORTED_REFERENCES:
            reason = UNSUPPORTED_REFERENCES.get(
                canonical, f"reference genome {canonical} is not supported"
            )
            references.append(Reference(canonical, entry, reason))
        else:
            references.append(Reference(canonical, entry))
    return references


# --- pedigree oracle --------------------------------------------------------


def oracle_dir(reference: Reference) -> Path:
    """The tracked fixture-oracle directory for a reference genome."""
    path = ORACLE_DIR / reference.name.lower()
    if not path.is_dir():
        raise FileNotFoundError(
            f"missing fixture oracle directory for reference {reference.name}: '{path}'"
        )
    return path


def oracle_path(reference: Reference, filename: str) -> Path:
    """A tracked fixture-oracle file, or a hard error naming the missing path."""
    path = oracle_dir(reference) / filename
    if not path.is_file():
        raise FileNotFoundError(f"missing fixture oracle file: '{path}'")
    return path


def load_pedigree(reference: Reference) -> pd.DataFrame:
    """The pedigree oracle (``samples.tsv``) for a reference genome."""
    return pd.read_csv(oracle_path(reference, PEDIGREE_FILE), sep="\t", dtype=str)


# --- paths and layout -------------------------------------------------------


def reference_root() -> Path:
    """The reference-resource root from the environment, or fail hard."""
    value = os.environ.get(REF_ROOT_ENV)
    if not value:
        raise FileNotFoundError(
            f"{REF_ROOT_ENV} is not set; point it at the reference-resource root"
        )
    path = Path(value)
    if not path.is_dir():
        raise FileNotFoundError(
            f"missing reference-resource root: '{path}' ({REF_ROOT_ENV})"
        )
    return path


def vep_root() -> Path:
    """The vep cache root path."""
    value = os.environ.get(VEP_ROOT_ENV)
    if not value:
        raise FileNotFoundError(f"{VEP_ROOT_ENV} is not set.")
    value = Path(value)
    if not value.is_dir():
        raise FileNotFoundError(
            f"invalid vep root directory: root = '{value}' ({VEP_ROOT_ENV})"
        )
    return value


def project_name(reference: Reference) -> str:
    """The project namespace a reference's rendered config scopes paths to."""
    return f"gnomad-meb-test-{reference.name.lower()}"


def session_dir(root: str | Path, reference: Reference) -> Path:
    """The reference-scoped, session-shared directory for pipeline artifacts.

    All upstream artifacts (the VDS, hail checkpoints and the VQSR output) live
    here. The path depends only on the reference genome. Point a later run at
    the same run root and production skips the **present** checkpoints it
    already declares, instead of recomputing them.
    """
    return Path(root) / SHARED_DIR / project_name(reference)


def project_checkpoint_dir(session_path: str | Path, reference: Reference) -> Path:
    """The ``{checkpoint}/{project}`` directory for a reference's hail tables."""
    return Path(session_path) / "checkpoint" / project_name(reference)


def step_log(session_path: str | Path, step: str) -> Path:
    """The log file a step's CLI subprocess writes to, under the session dir."""
    return Path(session_path) / "logs" / f"{step.replace(' ', '-')}.log"


def test_output_dir(run_path: str | Path, nodeid: str) -> Path:
    """A per-test output directory, scoped to the test node id."""
    stem = nodeid.replace("/", "_").replace("::", ".")
    return Path(run_path) / "output" / stem


def static_path(name: str) -> Path:
    """A file in the repository's ``static/`` directory.

    Spelled once here so the resources the pipeline consumes by path (the VEP
    wrapper, the fixed VQSR header) are not re-derived at each call site.
    """
    return REPO_ROOT / "static" / name


# --- config rendering -------------------------------------------------------


def render_config(
    reference: Reference,
    *,
    run_dir: str | Path,
    ref_root: str | Path,
    vep_root: str | Path,
    checkpoint_dir: str | Path | None = None,
    scratch_dir: str | Path | None = None,
    hail_tmp: str | Path | None = None,
    config_dir: str | Path | None = None,
    project: str | None = None,
    validate: bool = True,
) -> Path:
    """Render the per-reference config template into ``run_dir``.

    ``run_dir`` is the reference-scoped session directory: the shared upstream
    artifacts (the VDS and the VQSR output) live there. ``checkpoint_dir`` (the
    hail checkpoints, default ``run_dir/checkpoint``), ``scratch_dir`` (default
    ``run_dir/scratch``), ``hail_tmp`` (default ``run_dir/hail-tmp``) and
    ``config_dir`` (default ``run_dir/config``) are overridden so the session's
    scratch, temp and config files stay separate from the shared artifacts.
    """
    template = TEMPLATES_DIR / f"{reference.name.lower()}.toml"
    if not template.is_file():
        raise FileNotFoundError(
            f"missing config template for reference {reference.name}: '{template}'"
        )

    project = project or project_name(reference)
    run_dir = Path(run_dir)
    checkpoint_dir = (
        Path(checkpoint_dir) if checkpoint_dir is not None else run_dir / "checkpoint"
    )
    scratch_dir = (
        Path(scratch_dir) if scratch_dir is not None else run_dir / "scratch"
    )
    hail_tmp = Path(hail_tmp) if hail_tmp is not None else run_dir / "hail-tmp"
    template_root = TEMPLATES_DIR

    rendered = string.Template(template.read_text()).substitute(
        ref_root=str(ref_root),
        run_dir=str(run_dir),
        checkpoint=str(checkpoint_dir),
        scratch_dir=str(scratch_dir),
        hail_tmp=str(hail_tmp),
        vep_root=str(vep_root),
        vep_exe=str(static_path("vep.sh").resolve()),
        vqsr_header=str(static_path("vqsr.header.fixed").resolve()),
        project=project,
        template_root=str(template_root),
    )

    out_dir = Path(config_dir) if config_dir is not None else run_dir / "config"
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / f"{project}.toml"
    config_path.write_text(rendered)

    # Config.load only warns about a missing scratch dir, so create the
    # directories the rendered config expects.
    for path in (checkpoint_dir, scratch_dir, hail_tmp):
        path.mkdir(parents=True, exist_ok=True)

    if validate:
        validate_config_resources(config_path)
    return config_path


def _input_resource_paths(config: dict) -> Iterable[str]:
    """The pre-existing (input) resource paths a rendered config expects."""
    yield from config.get("resources", {}).values()
    yield from config.get("vqsr", {}).get("resources", {}).values()
    # The filter stage's constraint-metrics table is an input; its filters and
    # regions are declarations, not files.
    filter_resources = config.get("filter", {}).get("resources", {})
    yield from (path for path in filter_resources.values() if path is not None)
    annotation_resources = config.get("annotation", {}).get("resources", {})
    # sample_qc/variant_qc are pipeline outputs, not inputs.
    if "clinvar_vcf" in annotation_resources:
        yield annotation_resources["clinvar_vcf"]
    for value in config.get("annotation", {}).get("vep", {}).values():
        if isinstance(value, str):
            yield value


def validate_config_resources(config_path: str | Path) -> None:
    """Fail hard, naming every missing path, if input resources are absent."""
    config_path = Path(config_path)
    with open(config_path, "rb") as handle:
        config = tomllib.load(handle)

    missing = [
        path for path in _input_resource_paths(config) if not Path(path).exists()
    ]
    if missing:
        listing = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            f"missing reference resources referenced by '{config_path}':\n{listing}"
        )
