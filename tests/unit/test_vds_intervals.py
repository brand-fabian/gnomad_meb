"""In-process tests for the pure interval-splitting and serialization logic.

These pin the external behaviour of ``VariantDatasetTaskIntervals.get_interval_tasks``
and the ``VariantDatasetParameters``/``VariantDatasetTaskIntervals`` metadata
round-trip, so off-by-one errors in interval boundaries and configuration loss
across a metadata round-trip are caught without a full build. A local,
single-core hail session supplies the reference-contig lengths the interval
objects need. See ``.scratch/vds-dataset-migration`` (ticket 06).
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("hail")

from gnomad_meb.constants import CONTIGS
from gnomad_meb.lib.vds import (
    VariantDatasetParameters,
    VariantDatasetTaskIntervals,
)

REFERENCE_GENOME = "GRCh38"
DEFAULT_INTERVAL_SIZE = 1_200_000
SMALL_N_SPLITS = 0
LARGE_N_SPLITS = 4_000
PACKED_INTERVAL_SIZE = 800_000
# A contig-sized interval: the per-contig branch does not depend on
# ``interval_size``, so this keeps a boundary-case task list cheap to build.
COARSE_INTERVAL_SIZE = 300_000_000


@pytest.fixture(scope="module")
def small_tasks(hl):
    """The per-contig tasks: one split per contig when ``n_splits`` is small."""
    return VariantDatasetTaskIntervals.get_interval_tasks(
        reference_genome=REFERENCE_GENOME,
        interval_size=DEFAULT_INTERVAL_SIZE,
        n_splits=SMALL_N_SPLITS,
    )


@pytest.fixture(scope="module")
def large_tasks(hl):
    """The packed tasks: roughly ``n_splits`` splits when ``n_splits`` is large."""
    return VariantDatasetTaskIntervals.get_interval_tasks(
        reference_genome=REFERENCE_GENOME,
        interval_size=PACKED_INTERVAL_SIZE,
        n_splits=LARGE_N_SPLITS,
    )


def _flatten(tasks: list[VariantDatasetTaskIntervals]) -> list:
    """Every interval, in the order the tasks were produced."""
    return [interval for task in tasks for interval in task.intervals]


def _reference_lengths(hl) -> dict[str, int]:
    """Contig lengths for the contigs ``get_interval_tasks`` splits by default."""
    lengths = hl.get_reference(REFERENCE_GENOME).lengths
    return {contig: lengths[contig] for contig in CONTIGS}


def _intervals_by_contig(intervals: list) -> dict[str, list]:
    """Intervals grouped by contig, preserving the order they appear in."""
    by_contig: dict[str, list] = {}
    for interval in intervals:
        assert interval.start.contig == interval.end.contig, (
            f"interval spans contigs: {interval}"
        )
        assert interval.start.position <= interval.end.position, (
            f"interval start is after its end: {interval}"
        )
        by_contig.setdefault(interval.start.contig, []).append(interval)
    return by_contig


def _assert_tiles_contigs(intervals: list, contig_lengths: dict[str, int]) -> None:
    """The intervals tile each contig with no gaps and no overlap.

    Each contig is covered from position 1 to its length: every interval includes
    both its start and its end, every interval's end equals the next interval's
    start minus one, and the contig-final interval ends at the contig length.
    """
    by_contig = _intervals_by_contig(intervals)

    assert set(by_contig) == set(contig_lengths), "not every contig was covered"

    for contig, contig_intervals in by_contig.items():
        length = contig_lengths[contig]
        assert contig_intervals[0].start.position == 1, (
            f"contig {contig} does not start at position 1"
        )
        assert contig_intervals[-1].end.position == length, (
            f"contig {contig} does not end at its length {length}"
        )
        for i, interval in enumerate(contig_intervals):
            assert interval.includes_start, (
                f"contig {contig}: interval {interval} excludes its start"
            )
            assert interval.includes_end, (
                f"contig {contig}: interval {interval} excludes its end"
            )
            if i < len(contig_intervals) - 1:
                assert (
                    interval.end.position == contig_intervals[i + 1].start.position - 1
                ), (
                    f"contig {contig}: gap or overlap between "
                    f"{interval} and {contig_intervals[i + 1]}"
                )


# --- split generation --------------------------------------------------------


def test_small_n_splits_yields_one_task_per_contig(hl, small_tasks) -> None:
    """With ``n_splits`` small, each contig is its own split."""
    contig_lengths = _reference_lengths(hl)

    assert [task.name for task in small_tasks] == list(contig_lengths)
    assert len(small_tasks) == len(contig_lengths)
    assert all(task.reference_genome == REFERENCE_GENOME for task in small_tasks)
    assert all(len(task.intervals) >= 1 for task in small_tasks)


def test_n_splits_equal_to_contig_count_still_splits_per_contig(hl) -> None:
    """``n_splits == n_contigs`` is the boundary of the per-contig branch."""
    contig_lengths = _reference_lengths(hl)

    tasks = VariantDatasetTaskIntervals.get_interval_tasks(
        reference_genome=REFERENCE_GENOME,
        interval_size=COARSE_INTERVAL_SIZE,
        n_splits=len(contig_lengths),
    )

    assert [task.name for task in tasks] == list(contig_lengths)


def test_large_n_splits_approximates_the_requested_count(hl, large_tasks) -> None:
    """With ``n_splits`` large, the genome is packed into roughly ``n_splits``."""
    contig_lengths = _reference_lengths(hl)

    assert len(large_tasks) > len(contig_lengths), (
        "a large n_splits should pack past the per-contig default"
    )
    assert abs(len(large_tasks) - LARGE_N_SPLITS) <= LARGE_N_SPLITS * 0.05
    names = [task.name for task in large_tasks]
    assert len(set(names)) == len(names), "task names are not unique"
    assert all(task.intervals for task in large_tasks)


# --- interval boundaries -----------------------------------------------------


def test_small_tasks_tile_each_contig_without_gaps(hl, small_tasks) -> None:
    """A per-contig task tiles its whole contig without gaps or overlap."""
    _assert_tiles_contigs(_flatten(small_tasks), _reference_lengths(hl))


def test_large_tasks_tile_the_genome_without_gaps(hl, large_tasks) -> None:
    """Packing the genome into splits never drops or double-counts a position."""
    _assert_tiles_contigs(_flatten(large_tasks), _reference_lengths(hl))


# --- metadata round-trip -----------------------------------------------------


def test_task_intervals_round_trip(hl) -> None:
    """A task's intervals survive ``to_dict``/``from_dict`` unchanged."""
    intervals = (
        hl.Interval(
            hl.Locus("chr1", 1, REFERENCE_GENOME),
            hl.Locus("chr1", 100, REFERENCE_GENOME),
            includes_start=True,
            includes_end=False,
        ),
        hl.Interval(
            hl.Locus("chr1", 100, REFERENCE_GENOME),
            hl.Locus("chr1", 200, REFERENCE_GENOME),
            includes_start=True,
            includes_end=True,
        ),
    )
    task = VariantDatasetTaskIntervals("chr1", intervals, REFERENCE_GENOME)

    restored = VariantDatasetTaskIntervals.from_dict(
        json.loads(json.dumps(task.to_dict()))
    )

    assert restored.to_dict() == task.to_dict()
    assert restored.to_dict() == {
        "name": "chr1",
        "reference_genome": REFERENCE_GENOME,
        "intervals": [
            {
                "start": {"contig": "chr1", "position": 1},
                "end": {"contig": "chr1", "position": 100},
                "includes_start": True,
                "includes_end": False,
            },
            {
                "start": {"contig": "chr1", "position": 100},
                "end": {"contig": "chr1", "position": 200},
                "includes_start": True,
                "includes_end": True,
            },
        ],
    }


def test_parameters_round_trip_preserves_configuration(tmp_path, hl) -> None:
    """Writing then reading the metadata loses none of the configuration."""
    task = VariantDatasetTaskIntervals(
        "chr1",
        (
            hl.Interval(
                hl.Locus("chr1", 1, REFERENCE_GENOME),
                hl.Locus("chr1", 500, REFERENCE_GENOME),
                includes_start=True,
                includes_end=True,
            ),
        ),
        REFERENCE_GENOME,
    )
    params = VariantDatasetParameters(
        base=tmp_path,
        name="vds",
        gvcfs=("a.g.vcf.gz", "b.g.vcf.gz"),
        tasks=(task,),
        reference_genome=REFERENCE_GENOME,
        overwrite=True,
        interval_size=PACKED_INTERVAL_SIZE,
        n_splits=LARGE_N_SPLITS,
        branch_factor=42,
        target_records=1234,
        task_status={"chr1": ("finished", ""), "chr2": ("error", "boom")},
    )
    metadata_path = tmp_path / "vds.metadata.json"

    params.to_file(metadata_path)
    restored = VariantDatasetParameters.from_dict(json.loads(metadata_path.read_text()))

    assert restored.base == params.base
    assert restored.name == params.name
    assert tuple(restored.gvcfs) == params.gvcfs
    assert restored.reference_genome == params.reference_genome
    assert restored.overwrite is True
    assert restored.interval_size == params.interval_size
    assert restored.n_splits == params.n_splits
    assert restored.branch_factor == params.branch_factor
    assert restored.target_records == params.target_records
    assert [t.to_dict() for t in restored.tasks] == [t.to_dict() for t in params.tasks]
    assert restored.task_status == {
        name: list(status) for name, status in params.task_status.items()
    }
