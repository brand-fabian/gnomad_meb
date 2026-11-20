"""In-process tests for the integration harness seams.

These tests are fast and pipeline-free: they exercise reference discovery,
config rendering and the Oracle contract without hail and without a subprocess.
Ticket 04 folds the former ``test_smoke.py`` and ``test_oracle.py`` cases in
here, so keep additions small and additive.

The former ``test_cli_help_smoke`` is deliberately not folded in: it spawned the
CLI, and this module stays subprocess-free. The composite example
(``test_gnomad_workflow.py``, ``vqsr``-marked) exercises the CLI seam instead.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import harness
import oracle
import pytest

pytestmark = pytest.mark.integration


# --- config rendering: the Filter prerequisite ------------------------------


def test_render_config_carries_filter_prerequisite(tmp_path: Path) -> None:
    """The rendered config carries ``[filter]`` and its constraint resource.

    ``gnomad-workflow`` composes the Filter stage, whose fixed
    ``annotate_constraints_table`` step resolves ``ResourceRef("constraints")``;
    without this section the workflow fails to run.
    """
    reference = harness.Reference("GRCh38")
    ref_root = tmp_path / "ref-root"
    ref_root.mkdir()
    vep_root = tmp_path / "vep-root"
    vep_root.mkdir()
    run_dir = tmp_path / "run"

    config_path = harness.render_config(
        reference,
        run_dir=run_dir,
        ref_root=ref_root,
        vep_root=vep_root,
        validate=False,
    )

    with open(config_path, "rb") as handle:
        config = tomllib.load(handle)

    # Rendered into the run directory, not the repo.
    assert config_path.is_relative_to(run_dir)

    filter_config = config["filter"]
    assert filter_config["apply_sample_filters"] is False
    assert filter_config.get("regions", []) == []
    assert len(filter_config["filters"]) == 2

    constraints = ref_root / "annotation/hail/gnomad.v4.1.constraint_metrics.ht"
    assert Path(filter_config["resources"]["constraints"]) == constraints


def test_missing_constraint_resource_names_path(tmp_path: Path) -> None:
    """A missing constraint table is a hard error naming the missing path."""
    reference = harness.Reference("GRCh38")
    ref_root = tmp_path / "ref-root"
    (ref_root / "annotation" / "hail").mkdir(parents=True)
    vep_root = tmp_path / "vep-root"
    vep_root.mkdir()

    config_path = harness.render_config(
        reference,
        run_dir=tmp_path / "run",
        ref_root=ref_root,
        vep_root=vep_root,
        validate=False,
    )
    constraints = ref_root / "annotation/hail/gnomad.v4.1.constraint_metrics.ht"

    with pytest.raises(FileNotFoundError, match=str(constraints)):
        harness.validate_config_resources(config_path)


# --- config rendering: scoping and missing inputs ---------------------------


def test_render_config_scopes_paths_to_run_dir(tmp_path: Path) -> None:
    """A rendered config points every output location into the run directory."""
    reference = harness.Reference("GRCh38")
    run_dir = tmp_path / "run"
    ref_root = run_dir / "ref-root"
    ref_root.mkdir(parents=True, exist_ok=True)
    vep_root = run_dir / "vep-root"
    vep_root.mkdir(parents=True, exist_ok=True)

    config_path = harness.render_config(
        reference, run_dir=run_dir, ref_root=ref_root, validate=False, vep_root=vep_root
    )

    with open(config_path, "rb") as handle:
        config = tomllib.load(handle)

    assert config["project"] == "gnomad-meb-test-grch38"
    assert Path(config["scratch_dir"]).is_relative_to(run_dir)
    assert Path(config["checkpoint"]).is_relative_to(run_dir)
    assert Path(config["vqsr"]["output_dir"]).is_relative_to(run_dir)
    assert Path(config["resources"]["reference"]).is_relative_to(ref_root)


def test_missing_config_template_names_path(tmp_path: Path) -> None:
    """An unsupported reference has no template, and the error names it."""
    reference = harness.Reference("CHM13")

    with pytest.raises(FileNotFoundError, match="chm13.toml"):
        harness.render_config(
            reference,
            run_dir=tmp_path,
            ref_root=tmp_path,
            validate=False,
            vep_root=tmp_path,
        )


def test_missing_resource_names_path(tmp_path: Path) -> None:
    """A missing input resource is a hard error naming the missing path."""
    config_path = tmp_path / "missing-resource.toml"
    missing = tmp_path / "does-not-exist.fasta"
    config_path.write_text(f'[resources]\nreference = "{missing}"\n')

    with pytest.raises(FileNotFoundError, match=str(missing)):
        harness.validate_config_resources(config_path)


# --- reference discovery ----------------------------------------------------


def test_discovery_skips_chm13(tmp_path: Path) -> None:
    """CHM13 is discovered but skipped with an explicit reason."""
    data_dir = tmp_path / "data"
    (data_dir / "grch38").mkdir(parents=True)
    (data_dir / "chm13").mkdir()
    (data_dir / "README").write_text("not a dataset")

    references = {ref.name: ref for ref in harness.discover_references(data_dir)}

    assert set(references) == {"GRCh38", "CHM13"}
    assert references["GRCh38"].skip_reason is None
    assert references["CHM13"].skip_reason
    assert references["CHM13"].dataset_dir == data_dir / "chm13"


def test_discovery_missing_data_dir_names_path(tmp_path: Path) -> None:
    """A missing dataset root is a hard error naming the path."""
    missing = tmp_path / "no-such-data"

    with pytest.raises(FileNotFoundError, match=str(missing)):
        harness.discover_references(missing)


# --- the Oracle contract ----------------------------------------------------
#
# The oracle files are filled stage by stage, so the Oracle drops blank cells
# rather than failing on them (commit a40546e); a header-only column compares
# nothing. Only an absent column name is a hard error.


def _oracle(column: str, values: dict[str, str | None]) -> oracle.Oracle:
    return oracle.Oracle(harness.Reference("GRCh38"), {column: values})


def test_blank_cells_are_not_compared() -> None:
    """A blank expected cell is dropped; only the filled keys are compared."""
    fixture = _oracle("sex", {"s1": "M", "s2": None})

    # The differing actual at the blank key is skipped, so this passes...
    fixture.expect("sex", {"s1": "M", "s2": "F"})

    # ...while a mismatch on the filled key still fails.
    with pytest.raises(AssertionError):
        fixture.expect("sex", {"s1": "F", "s2": "F"})


def test_empty_column_compares_nothing() -> None:
    """A header-only column (no rows) has nothing to compare and passes."""
    fixture = _oracle("sex", {})

    fixture.expect("sex", {})


def test_absent_column_names_it() -> None:
    """Requesting a column the oracle does not have fails, naming the column."""
    fixture = _oracle("sex", {"s1": "M"})

    with pytest.raises(pytest.fail.Exception, match="no column 'ancestry'"):
        fixture.expect("ancestry", {"s1": "EUR"})


def test_mismatch_lists_the_offending_keys() -> None:
    """A mismatch names the keys that differ and only those."""
    fixture = _oracle("sex", {"s1": "M", "s2": "F"})

    with pytest.raises(AssertionError) as excinfo:
        fixture.expect("sex", {"s1": "F", "s2": "F"})

    assert "s1" in str(excinfo.value)
    assert "s2" not in str(excinfo.value)


def test_numeric_respects_its_tolerance() -> None:
    """A difference within tolerance passes; one beyond it fails."""
    fixture = _oracle("af", {"v1": "1.0"})

    fixture.expect_numeric("af", {"v1": "1.0005"}, 1e-3)

    with pytest.raises(AssertionError):
        fixture.expect_numeric("af", {"v1": "1.01"}, 1e-3)
