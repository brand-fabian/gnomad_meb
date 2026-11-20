"""In-process tests for the dataset read step's contract and inspect summary.

These pin the external behaviour the CLI relies on: opening a dataset from its
metadata file reconstructs the handle without rewriting the metadata, the dataset
reports its completeness, and the pure union step refuses an incomplete dataset.
A local, single-core hail session supplies the reference-contig lengths the
interval objects need. See ``.scratch/vds-dataset-migration`` (ticket 03).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("hail")

from gnomad_meb.lib.vds import (
    VariantDataset,
    VariantDatasetParameters,
    VariantDatasetTaskIntervals,
)


def _write_dataset(
    base: Path, hl, *, finished: bool
) -> tuple[Path, VariantDatasetParameters]:
    """Write a dataset metadata file by hand and return its path and params."""
    base.mkdir(parents=True, exist_ok=True)
    intervals = (
        hl.Interval(
            hl.Locus("chr1", 1, "GRCh38"),
            hl.Locus("chr1", 100, "GRCh38"),
            includes_start=True,
            includes_end=True,
        ),
    )
    task = VariantDatasetTaskIntervals("chr1", intervals, "GRCh38")
    params = VariantDatasetParameters(
        base=base,
        name="vds",
        gvcfs=("sample.g.vcf.gz",),
        tasks=(task,),
        task_status={"chr1": ("finished" if finished else "init", "")},
    )
    metadata_path = base / "vds.metadata.json"
    params.to_file(metadata_path)
    return metadata_path, params


def test_open_reconstructs_base_and_name(tmp_path: Path, hl) -> None:
    _write_dataset(tmp_path, hl, finished=True)

    dataset = VariantDataset.open(tmp_path / "vds.metadata.json")

    assert dataset.base == tmp_path
    assert dataset.name == "vds"


def test_opening_does_not_rewrite_the_metadata(tmp_path: Path, hl) -> None:
    metadata_path, _ = _write_dataset(tmp_path, hl, finished=False)
    before = metadata_path.read_text()

    VariantDataset.open(metadata_path)

    assert metadata_path.read_text() == before


def test_open_heals_base_and_name_after_a_recursive_copy(tmp_path: Path, hl) -> None:
    """A dataset relocated by ``cp -a``/``rsync -a`` opens at its new root.

    The metadata records the *original* base, but split paths are derived from
    the handle's ``base``/``name`` rather than stored, so ``open`` must re-anchor
    both to the metadata file's actual directory and stem. Otherwise every split
    would be read from the pre-copy location.
    """
    source = tmp_path / "source"
    metadata_path, _ = _write_dataset(source, hl, finished=True)
    # The split rides along with the metadata under a recursive copy.
    (source / "vds.chr1.vds").mkdir(parents=True)
    (source / "vds.chr1.vds" / "marker").write_text("split")

    relocated = tmp_path / "relocated"
    shutil.copytree(source, relocated)

    # The copied metadata still names the old base: the copy is byte-for-byte.
    copied_params = json.loads((relocated / "vds.metadata.json").read_text())
    assert copied_params["base"] == str(source)

    dataset = VariantDataset.open(relocated / "vds.metadata.json")

    assert dataset.base == relocated
    assert dataset.name == "vds"
    assert dataset.params.base == relocated
    assert dataset.params.name == "vds"
    assert dataset.get_metadata_path() == relocated / "vds.metadata.json"
    # Every split resolves under the new root, where the copy actually landed.
    assert all(
        Path(dataset._get_vds_path(task)).is_relative_to(relocated)
        for task in dataset.params.tasks
    )
    assert Path(dataset._get_vds_path(dataset.params.tasks[0])).is_dir()


def test_open_accepts_a_consistently_renamed_dataset(tmp_path: Path, hl) -> None:
    """Renaming the metadata and its splits together re-anchors the handle."""
    source = tmp_path / "source"
    metadata_path, _ = _write_dataset(source, hl, finished=True)
    (source / "vds.chr1.vds").mkdir(parents=True)
    (source / "vds.chr1.vds" / "marker").write_text("split")
    metadata_path.rename(source / "renamed.metadata.json")
    (source / "vds.chr1.vds").rename(source / "renamed.chr1.vds")

    dataset = VariantDataset.open(source / "renamed.metadata.json")

    assert dataset.name == "renamed"
    assert dataset.params.name == "renamed"
    assert dataset.get_metadata_path() == source / "renamed.metadata.json"
    assert Path(dataset._get_vds_path(dataset.params.tasks[0])).is_dir()


def test_is_complete_tracks_task_status(tmp_path: Path, hl) -> None:
    _, params = _write_dataset(tmp_path, hl, finished=True)
    assert VariantDataset(params.base, params.name, params).is_complete

    _, incomplete = _write_dataset(tmp_path / "other", hl, finished=False)
    assert not VariantDataset(incomplete.base, incomplete.name, incomplete).is_complete


def test_pure_union_raises_on_an_incomplete_dataset(tmp_path: Path, hl) -> None:
    _, params = _write_dataset(tmp_path, hl, finished=False)
    dataset = VariantDataset(params.base, params.name, params)

    with pytest.raises(ValueError, match="incomplete"):
        dataset.to_hail()


def test_pure_union_raises_when_no_split_is_finished(tmp_path: Path, hl) -> None:
    _, params = _write_dataset(tmp_path, hl, finished=False)
    dataset = VariantDataset(params.base, params.name, params)

    with pytest.raises(ValueError, match="no valid variant datasets"):
        dataset.to_hail(error_on_missing=False)


def test_inspect_tolerates_a_dataset_with_no_finished_split(tmp_path: Path, hl) -> None:
    metadata_path, _ = _write_dataset(tmp_path, hl, finished=False)

    summary = VariantDataset.open(metadata_path).inspect()

    assert summary["samples"] == []
    assert summary["name"] == "vds"


def _write_multi_task_dataset(base: Path, hl) -> VariantDataset:
    """Write a two-task dataset: ``chr1`` finished, ``chr2`` still pending."""
    base.mkdir(parents=True, exist_ok=True)

    def _interval(contig: str) -> tuple:
        return (
            hl.Interval(
                hl.Locus(contig, 1, "GRCh38"),
                hl.Locus(contig, 100, "GRCh38"),
                includes_start=True,
                includes_end=True,
            ),
        )

    tasks = (
        VariantDatasetTaskIntervals("chr1", _interval("chr1"), "GRCh38"),
        VariantDatasetTaskIntervals("chr2", _interval("chr2"), "GRCh38"),
    )
    params = VariantDatasetParameters(
        base=base,
        name="vds",
        gvcfs=("sample.g.vcf.gz",),
        tasks=tasks,
        task_status={"chr1": ("finished", ""), "chr2": ("init", "")},
    )
    params.to_file(base / "vds.metadata.json")
    return VariantDataset(base, params.name, params)


def test_select_tasks_narrows_tasks_and_scopes_completeness(
    tmp_path: Path, hl
) -> None:
    dataset = _write_multi_task_dataset(tmp_path, hl)
    assert not dataset.is_complete

    selected = dataset.select_tasks(lambda task: task.name == "chr1")

    assert [t.name for t in selected.params.tasks] == ["chr1"]
    assert set(selected.params.task_status) == {"chr1"}
    assert selected.is_complete
    # The parent is untouched: selection is a lazy view, not a mutation.
    assert len(dataset.params.tasks) == 2
    assert not dataset.is_complete


def test_select_tasks_is_read_only(tmp_path: Path, hl) -> None:
    dataset = _write_multi_task_dataset(tmp_path, hl)

    selected = dataset.select_tasks(lambda task: task.name == "chr1")

    with pytest.raises(ValueError, match="no in-memory datasets"):
        selected.write()


def test_filter_contigs_selects_matching_tasks(tmp_path: Path, hl) -> None:
    dataset = _write_multi_task_dataset(tmp_path, hl)

    selected = dataset.filter_contigs(["chr2"])

    assert [t.name for t in selected.params.tasks] == ["chr2"]


def _write_contig_dataset(base: Path, hl, contigs: tuple[str, ...]) -> VariantDataset:
    """Write a dataset with one pending task per contig in ``contigs``."""
    base.mkdir(parents=True, exist_ok=True)

    def _interval(contig: str) -> tuple:
        return (
            hl.Interval(
                hl.Locus(contig, 1, "GRCh38"),
                hl.Locus(contig, 100, "GRCh38"),
                includes_start=True,
                includes_end=True,
            ),
        )

    tasks = tuple(
        VariantDatasetTaskIntervals(contig, _interval(contig), "GRCh38")
        for contig in contigs
    )
    params = VariantDatasetParameters(
        base=base,
        name="vds",
        gvcfs=("sample.g.vcf.gz",),
        tasks=tasks,
        task_status=dict.fromkeys(contigs, ("init", "")),
    )
    params.to_file(base / "vds.metadata.json")
    return VariantDataset(base, params.name, params)


def test_chromosomes_to_keep_mirrors_hail_filter_modes(tmp_path: Path, hl) -> None:
    dataset = _write_contig_dataset(tmp_path, hl, ("chr1", "chrX", "chrY"))

    assert set(dataset._chromosomes_to_keep(keep=("chr1", "chrX"))) == {"chr1", "chrX"}

    removed = dataset._chromosomes_to_keep(remove=("chrX", "chrY"))
    assert "chr1" in removed
    assert "chrX" not in removed
    assert "chrY" not in removed

    autosomes = dataset._chromosomes_to_keep(keep_autosomes=True)
    assert "chr1" in autosomes
    assert "chrX" not in autosomes
    assert "chrY" not in autosomes


def test_filter_chromosomes_prunes_tasks_before_filtering(tmp_path: Path, hl) -> None:
    """The task prune happens on the lazy handle apply() maps over."""
    dataset = _write_contig_dataset(tmp_path, hl, ("chr1", "chrX", "chrY"))

    autosomes = dataset.filter_chromosomes(keep_autosomes=True)

    assert [t.name for t in autosomes.params.tasks] == ["chr1"]
    assert autosomes.is_changed
    # The parent is untouched: pruning is a view, not a mutation.
    assert [t.name for t in dataset.params.tasks] == ["chr1", "chrX", "chrY"]
