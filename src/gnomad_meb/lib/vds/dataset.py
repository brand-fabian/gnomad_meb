"""The authoritative VDS API.

A :class:`VariantDataset` is built from GVCFs by chunking the reference genome
into splits (see :class:`VariantDatasetTaskIntervals`) and importing each split
as its own VDS under a common dataset root. A metadata file next to the splits
records the parameters, the split intervals and a per-task status, so a build is
resumable: finished splits are skipped, errored splits retried, and
``overwrite`` forces every split to be rebuilt.

The handle is split-aware rather than merged: :meth:`VariantDataset.apply`,
:meth:`VariantDataset.map_datasets` and :meth:`VariantDataset.reduce_datasets`
transform and measure each split in place, so a step need not union everything
through :meth:`VariantDataset.to_hail` (ADR-0015). Materialization is explicit
and idempotent: :meth:`VariantDataset.write` persists the in-memory splits to a
chosen location, skipping an unchanged handle (ADR-0016).

Only the local filesystem is supported. The dataset root and every split path
are local paths; S3 input/output is deliberately not implemented (see
``docs/adr/0007-vds-datasets-are-local-only.md``). GVCF *discovery* may still
search S3 buckets, but the resulting VDS paths must be local.

Reference-block entries retain ``GT``, ``DP``, ``GQ`` and ``MIN_DP``, pinned
through ``gvcf_reference_entry_fields_to_keep`` in
:meth:`VariantDataset.process`. ``DP`` is required by the coverage aggregation
(``.scratch/annotation-coverage/spec.md``, D4), which reads depth from both the
reference and variant densified entries; without the explicit list the combiner
guesses the fields from the first gVCF's sampled reference rows and can drop a
declared-but-sparse ``DP``, which coverage would then silently treat as zero.
Variant entries pass every gVCF FORMAT field through, so their ``DP`` follows
the input gVCF. Because this changes the VDS reference-entry schema, an already
built dataset must be rebuilt with ``--overwrite`` to carry the new entries.
"""

from pathlib import Path
from typing import Iterable, Any, Literal, LiteralString, Callable, TypeVar
import json
from dataclasses import dataclass, field, fields
from time import time
from traceback import format_exc
import secrets

from gnomad_meb.constants import REFERENCE_GENOMES, CONTIGS
from gnomad_meb.hail import hail as hl, get_default_partitions
from gnomad_meb.logging import get_logger
from gnomad_meb.lib.checkpoint import get_checkpoint_path, CHECKPOINT_TYPE
from gnomad_meb.config import get_config

logger = get_logger()


VariantDatasetTaskStatus = Literal["init", "started", "finished", "error"]
Call = Callable[[hl.vds.VariantDataset], hl.vds.VariantDataset]
Result = TypeVar("Result")


class VariantDatasetMapError(Exception):
    """One or more per-split applications failed; no result was produced."""


def _get_chromosome_lengths(reference_genome: REFERENCE_GENOMES = "GRCh38", contigs: tuple[LiteralString, ...] = CONTIGS):
    hl_contig_lengths_d = hl.get_reference(reference_genome).lengths
    missing_contigs = [ c for c in contigs if c not in hl_contig_lengths_d ]
    if missing_contigs:
        raise ValueError(f"missing contig in reference genome: missing = {missing_contigs}, reference_genome = {reference_genome}, reference_contigs = {set(hl_contig_lengths_d.keys())}")

    return {
        c: hl_contig_lengths_d[c]
        for c in contigs
    }
        


@dataclass(frozen=True)
class VariantDatasetTaskIntervals:
    name: str
    intervals: tuple[hl.Interval]
    reference_genome: REFERENCE_GENOMES = "GRCh38"

    def to_dict(self):
        intervals = []
        for intv in self.intervals:
            intervals.append(
                {
                    "start": {
                        "contig": intv.start.contig,
                        "position": intv.start.position,
                    },
                    "end": {"contig": intv.end.contig, "position": intv.end.position},
                    "includes_start": intv.includes_start,
                    "includes_end": intv.includes_end,
                }
            )
        return {
            "name": self.name,
            "intervals": intervals,
            "reference_genome": self.reference_genome,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        default_reference_genome: REFERENCE_GENOMES = "GRCh38",
    ):
        name = data["name"]
        reference_genome = (
            data["reference_genome"]
            if "reference_genome" in data
            else default_reference_genome
        )

        intervals = []
        for intv in data["intervals"]:
            intervals.append(
                hl.Interval(
                    hl.Locus(
                        intv["start"]["contig"],
                        intv["start"]["position"],
                        reference_genome,
                    ),
                    hl.Locus(
                        intv["end"]["contig"], intv["end"]["position"], reference_genome
                    ),
                    includes_start=intv["includes_start"],
                    includes_end=intv["includes_end"],
                )
            )
        return cls(name, tuple(intervals), reference_genome)

    @classmethod
    def get_interval_tasks(
        cls,
        reference_genome: REFERENCE_GENOMES = "GRCh38",
        contigs: tuple[LiteralString, ...] = CONTIGS,
        interval_size: int = 1_200_000,
        n_splits: int = 0,
    ) -> list["VariantDatasetTaskIntervals"]:
        """Split the reference genome into multiple interval subsets.

        Returns
        -------
        list[VariantDatasetTaskIntervals]
            The list of tasks with their corresponding genomic intervals.
        """
        contig_lengths = _get_chromosome_lengths(reference_genome=reference_genome, contigs=contigs)
        tasks = []
        if n_splits <= len(contig_lengths):
            for contig, length in contig_lengths.items():
                starts = list(range(1, length + 1, interval_size))
                intervals = []
                for i, start in enumerate(starts):
                    end = length if i == len(starts) - 1 else starts[i + 1] - 1
                    intervals.append(
                        hl.Interval(
                            hl.Locus(contig, start, reference_genome),
                            hl.Locus(contig, end, reference_genome),
                            includes_start=True,
                            includes_end=True,
                        )
                    )
                tasks.append(cls(contig, tuple(intervals), reference_genome))
        else:
            genome_size = sum(contig_lengths.values())
            target_size = (genome_size + 1) // n_splits

            current_task_size = 0
            intervals = []
            for contig, length in contig_lengths.items():
                for start in range(1, length + 1, interval_size):
                    real_interval_size = min(interval_size, length - start + 1)
                    if (
                        intervals
                        and current_task_size + real_interval_size > target_size
                    ):
                        tasks.append(
                            cls(
                                f"task-{len(tasks)}", tuple(intervals), reference_genome
                            )
                        )
                        intervals = []
                        current_task_size = 0

                    intervals.append(
                        hl.Interval(
                            hl.Locus(contig, start, reference_genome),
                            hl.Locus(
                                contig, start + real_interval_size - 1, reference_genome
                            ),
                            includes_start=True,
                            includes_end=True,
                        )
                    )
                    current_task_size += real_interval_size

            if intervals:
                tasks.append(
                    cls(f"task-{len(tasks)}", tuple(intervals), reference_genome)
                )

        return tasks


@dataclass
class VariantDatasetParameters:
    base: Path
    name: str
    gvcfs: tuple[str, ...]
    tasks: tuple[VariantDatasetTaskIntervals, ...]
    reference_genome: REFERENCE_GENOMES = "GRCh38"
    overwrite: bool = False
    interval_size: int = 1_200_000
    n_splits: int = 0
    branch_factor: int = 100
    target_records: int = 24_000
    task_status: dict[str, tuple[VariantDatasetTaskStatus, str]] = field(
        default_factory=lambda: {}
    )

    def to_file(self, path: Path | str):
        with open(path, "w") as out_f:
            json.dump(self.to_dict(), out_f, indent=4)

    def to_dict(self):
        data = {}
        for f in fields(self):
            if f.name == "tasks":
                data[f.name] = [t.to_dict() for t in self.tasks]
            elif f.name == "base":
                data[f.name] = str(getattr(self, f.name))
            else:
                data[f.name] = getattr(self, f.name)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        kwargs = {}
        for key, value in data.items():
            if key == "tasks":
                kwargs[key] = tuple(
                    [VariantDatasetTaskIntervals.from_dict(t) for t in value]
                )
            elif key == "base":
                kwargs[key] = Path(value)
            else:
                kwargs[key] = value
        return cls(**kwargs)


class VariantDataset:
    def __init__(
        self,
        base: Path | str,
        name: str,
        params: VariantDatasetParameters | None = None,
        tmp_dir: Path | str | None = None,
    ) -> None:
        base = Path(base)
        base.mkdir(exist_ok=True, parents=True)

        self.base = base
        self.name = name
        self.tmp_dir = tmp_dir

        if params is None:
            self.params = self.load_metadata()
        else:
            self.params = params

        self._datasets: dict[str, hl.vds.VariantDataset | None] | None = None
        """Pointer to hail dataset instances by task name."""

        self._changed: bool = False
        """True when the in-memory splits differ from what is persisted at base/name."""

    def load_metadata(self):
        md = self.get_metadata_path()
        assert md.is_file(), (
            f"invalid variant dataset: {self.base} does not contain a valid metadata file"
        )

        with open(md, "r") as in_f:
            params = VariantDatasetParameters.from_dict(json.load(in_f))

        if len(params.tasks) == 0:
            raise ValueError(f"invalid variant dataset parameters: params = {params}")
        return params

    def get_metadata_path(self):
        return self.base / f"{self.name}.metadata.json"

    @property
    def is_changed(self) -> bool:
        """Whether the in-memory splits differ from what is persisted at base/name."""
        return self._changed

    def map_datasets(self, fn: Callable[[hl.vds.VariantDataset], Result]) -> dict[str, Result]:
        """Apply ``fn`` to every finished split VDS; results keyed by task name.

        Fails fast: if any split raises, the whole map raises rather than
        silently dropping a task, so a partial result can never masquerade as a
        complete one.
        """
        datasets = self._open_datasets()
        results: dict[str, Result] = {}
        failures: dict[str, Exception] = {}
        logger.info(f"mapping function over datasets: fn = {fn.__qualname__}, n_vds = {len(datasets)}")
        for name, vds in datasets.items():
            if vds is None:
                continue
            logger.debug(f"processing function for dataset: name = {name}, fn = {fn.__qualname__}")
            try:
                results[name] = fn(vds)
            except Exception as exc:
                logger.error(f"failed to process function for task: name = {name}, fn = {fn.__qualname__}, error = {exc}")
                failures[name] = exc
        if failures:
            raise VariantDatasetMapError(
                f"failed to map {fn.__qualname__} over {len(failures)} task(s): {sorted(failures)}"
            ) from next(iter(failures.values()))
        return results

    def reduce_datasets(
        self,
        fn: Callable[[hl.vds.VariantDataset], Result],
        combine: Callable[[Result, Result], Result],
    ) -> Result:
        """Map ``fn`` over each split, then left-fold the results with ``combine``.

        ``combine`` folds two per-split results into one; for site-level tables
        that is a plain union (see :meth:`union_tables`), for per-sample tables
        an algebraic re-aggregation.
        """
        parts = list(self.map_datasets(fn).values())
        if not parts:
            raise ValueError("got no finished splits to reduce")
        acc = parts[0]
        for part in parts[1:]:
            acc = combine(acc, part)
        return acc

    def union_tables(
        self, fn: Callable[[hl.vds.VariantDataset], hl.Table]
    ) -> hl.Table:
        """Reduce site-level tables whose rows are disjoint across splits."""
        return self.reduce_datasets(fn, combine=lambda a, b: a.union(b))

    def apply(self, fn: Call) -> "VariantDataset":
        """Apply a VDS-returning function to all split datasets contained herein.

        Returns a new, *changed* handle holding the transformed splits in memory.
        The storage location of the result is chosen later with :meth:`write`.
        """
        if not self.is_complete:
            logger.warning("applying function to incomplete VariantDataset")

        updated = self.map_datasets(fn)
        for name, result in updated.items():
            if result is not None and not isinstance(result, hl.vds.VariantDataset):
                raise TypeError(
                    f"apply expects a VDS-returning function: fn = {fn.__qualname__}, "
                    f"task = {name}, got {type(result).__name__}"
                )

        ret_val = self.__class__(
            self.base,
            self.name,
            VariantDatasetParameters.from_dict(self.params.to_dict()),
            self.tmp_dir,
        )
        ret_val._datasets = updated
        ret_val._changed = True
        return ret_val

    def filter_intervals(
        self,
        intervals: Iterable[hl.Interval],
        *,
        keep: bool = True,
        split_reference_blocks: bool = False,
    ) -> "VariantDataset":
        """Filter each split to ``intervals`` via ``hl.vds.filter_intervals``."""
        return self.apply(
            lambda vds: hl.vds.filter_intervals(
                vds,
                intervals,
                keep=keep,
                split_reference_blocks=split_reference_blocks,
            )
        )

    def _chromosomes_to_keep(
        self,
        *,
        keep: Iterable[str] | None = None,
        remove: Iterable[str] | None = None,
        keep_autosomes: bool = False,
    ) -> tuple[str, ...]:
        """The contigs ``hl.vds.filter_chromosomes`` would retain.

        Mirrors the three Hail filtering modes to answer the equivalent question
        for task pruning: which contigs can still carry data afterwards. With no
        arguments at all it returns every reference contig, so nothing is pruned
        and the Hail call reports the missing-argument error itself.
        """
        reference_genome = hl.get_reference(self.params.reference_genome)
        if keep is not None:
            return (keep,) if isinstance(keep, str) else tuple(keep)
        if remove is not None:
            removed = {remove} if isinstance(remove, str) else set(remove)
            return tuple(c for c in reference_genome.contigs if c not in removed)
        if keep_autosomes:
            excluded = (
                set(reference_genome.x_contigs)
                | set(reference_genome.y_contigs)
                | set(reference_genome.mt_contigs)
            )
            return tuple(c for c in reference_genome.contigs if c not in excluded)
        return tuple(reference_genome.contigs)

    def filter_chromosomes(
        self,
        *,
        keep: Iterable[str] | None = None,
        remove: Iterable[str] | None = None,
        keep_autosomes: bool = False,
    ) -> "VariantDataset":
        """Filter each *relevant* split to a chromosome subset.

        Splits that cannot contain any of the requested chromosomes are dropped
        first (see :meth:`filter_contigs`), so only the splits that can
        contribute are opened; the survivors are then filtered per split with
        ``hl.vds.filter_chromosomes``.
        """
        return self.filter_contigs(
            self._chromosomes_to_keep(
                keep=keep, remove=remove, keep_autosomes=keep_autosomes
            )
        ).apply(
            lambda vds: hl.vds.filter_chromosomes(
                vds, keep=keep, remove=remove, keep_autosomes=keep_autosomes
            )
        )

    def filter_variants(
        self, sites_ht: hl.Table, *, keep: bool = True
    ) -> "VariantDataset":
        """Semi-join each split to ``sites_ht``'s row keys."""
        return self.apply(lambda vds: hl.vds.filter_variants(vds, sites_ht, keep=keep))

    def filter_samples(
        self,
        samples: Iterable[str],
        *,
        keep: bool = True,
        remove_dead_alleles: bool = False,
    ) -> "VariantDataset":
        """Filter each split to a sample subset via ``hl.vds.filter_samples``."""
        return self.apply(
            lambda vds: hl.vds.filter_samples(
                vds, samples, keep=keep, remove_dead_alleles=remove_dead_alleles
            )
        )

    def filter_rows(
        self, predicate: Callable[[hl.MatrixTable], Any], *, keep: bool = True
    ) -> "VariantDataset":
        """Filter each split's variant-data rows by ``predicate(variant_data)``.

        ``hl.vds`` exposes no ``filter_rows``; this owns the variant-data row
        predicate (e.g. the LCR/segdup interval filter) as a per-split transform.
        """

        def _fn(vds: hl.vds.VariantDataset) -> hl.vds.VariantDataset:
            return hl.vds.VariantDataset(
                vds.reference_data,
                vds.variant_data.filter_rows(predicate(vds.variant_data), keep=keep),
            )

        return self.apply(_fn)

    def split_multi(self, *, filter_changed_loci: bool = False) -> "VariantDataset":
        """Split multiallelic sites per split via ``hl.vds.split_multi``."""
        return self.apply(
            lambda vds: hl.vds.split_multi(
                vds, filter_changed_loci=filter_changed_loci
            )
        )

    def select_tasks(
        self, predicate: Callable[[VariantDatasetTaskIntervals], bool]
    ) -> "VariantDataset":
        """A lazy, read-only handle over the tasks satisfying ``predicate(task)``.

        The result copies the parameters, keeps only the selected tasks and their
        matching ``task_status`` entries, and starts with no in-memory splits, so
        the first read opens only the selected ``.vds`` files. ``is_complete`` and
        ``_finished_tasks`` are scoped to the subset. With no in-memory splits,
        :meth:`write` refuses, so the parent dataset's metadata cannot be clobbered
        by a subset.
        """
        params = VariantDatasetParameters.from_dict(self.params.to_dict())
        params.tasks = tuple(t for t in params.tasks if predicate(t))
        names = {t.name for t in params.tasks}
        params.task_status = {
            k: v for k, v in params.task_status.items() if k in names
        }
        return self.__class__(self.base, self.name, params, self.tmp_dir)

    def filter_contigs(self, contigs: Iterable[str]) -> "VariantDataset":
        """A lazy, read-only handle over the tasks covering ``contigs``.

        Drops every task whose intervals do not touch any of ``contigs``, so a
        later read opens only the splits that can contribute. This is the
        task-pruning primitive :meth:`filter_chromosomes` builds on before it
        applies the Hail chromosome filter.
        """
        wanted = set(contigs)
        return self.select_tasks(
            lambda task: any(i.start.contig in wanted for i in task.intervals)
        )

    def process(self):
        params = self.params
        # Initialize all task status objects
        for task in params.tasks:
            if task.name not in params.task_status:
                params.task_status[task.name] = ("init", "")
        params.to_file(self.get_metadata_path())

        logger.info(
            f"processing variant dataset in genomic chunks: n_tasks = {len(params.tasks)}"
        )
        for task in params.tasks:
            logger.debug(f"processing variant dataset task: task = {task.name}")
            status, msg = params.task_status[task.name]
            if status in ("init", "started", "error") or params.overwrite:
                logger.info(
                    f"processing variant dataset creation task: task = {task.name}, status = {status}, overwrite = {params.overwrite}"
                )
                output_dir = self.base / f"{self.name}.{task.name}.vds"
                tmp_dir = (
                    Path(self.tmp_dir)
                    if self.tmp_dir is not None
                    else self.base / f"{self.name}.{task.name}.temp"
                )
                combiner = hl.vds.new_combiner(
                    output_path=str(output_dir),
                    temp_path=str(tmp_dir),
                    save_path=str(tmp_dir / "combiner-plan.json"),
                    gvcf_paths=list(params.gvcfs),
                    intervals=list(task.intervals),
                    reference_genome=params.reference_genome,
                    target_records=params.target_records,
                    branch_factor=params.branch_factor,
                    # The combiner's default (see the `defined_entry_fields`
                    # fallback in hail/vds/combiner/variant_dataset_combiner.py)
                    # derives the reference entry fields from those *defined* in
                    # the first 100k reference rows of the first gVCF, so a
                    # declared-but-sparse `DP` can be dropped. Coverage reads
                    # `DP` from both densified entry types (spec D4), so pin the
                    # fields explicitly.
                    gvcf_reference_entry_fields_to_keep=[
                        "GT",
                        "DP",
                        "GQ",
                        "MIN_DP",
                    ],
                )

                params.task_status[task.name] = ("started", "")

                start_time = time()
                try:
                    combiner.run()
                except Exception as e:
                    msg = f"error processing variant dataset creation task: task = {task.name}, elapsed = {time() - start_time:.02f}s, error = {e}\n{format_exc()}"
                    logger.error(msg)
                    params.task_status[task.name] = ("error", msg)
                    params.to_file(self.get_metadata_path())
                    continue

                params.task_status[task.name] = ("finished", "")
                logger.info(
                    f"finished creating variant dataset: task = {task.name}, elapsed = {time() - start_time:.02f}s"
                )
                params.to_file(self.get_metadata_path())

        tasks_by_status = {}
        for _, (status, _) in params.task_status.items():
            tasks_by_status[status] = (
                1 if status not in tasks_by_status else tasks_by_status[status] + 1
            )
        logger.info(
            f"finished all metadata creation tasks: status = {tasks_by_status}"
        )

        # Load the just-written splits into memory: disk is now authoritative, so
        # the handle is clean and a later write() won't duplicate the data.
        self._open_datasets(force=True)
        self._changed = False

    @property
    def is_complete(self):
        return all(
            t.name in self.params.task_status
            and self.params.task_status[t.name][0] == "finished"
            for t in self.params.tasks
        )

    def _finished_tasks(self) -> list[VariantDatasetTaskIntervals]:
        """The tasks whose split has finished and is safe to read."""
        return [
            task
            for task in self.params.tasks
            if self.params.task_status.get(task.name, ("init", ""))[0] == "finished"
        ]

    def _get_vds_path(
        self,
        task: VariantDatasetTaskIntervals,
        base: Path | str | None = None,
        name: str | None = None,
    ):
        base = Path(base) if base is not None else self.base
        name = name if name is not None else self.name
        return str(base / f"{name}.{task.name}.vds")

    def _open_datasets(self, force: bool = False):
        if self._datasets is None or force:
            if force and self._changed:
                raise ValueError(
                    "refusing to reopen from disk while in-memory changes are "
                    "pending; call write() or discard the handle first"
                )
            self._datasets = {
                task.name: hl.vds.read_vds(self._get_vds_path(task)) if self.params.task_status.get(task.name, ("init", ""))[0] == "finished" else None
                for task in self.params.tasks 
            }
        return self._datasets

    def write_metadata(
        self,
        base: Path | str | None = None,
        name: str | None = None,
    ) -> "VariantDataset":
        """Persist only the metadata file, without touching any split VDSs.

        Lets a bare handle (see :meth:`create_bare`) record its parameters and
        task list so that a later :meth:`open` can enumerate the tasks, before
        any data exists. ``base``/``name`` retarget the handle first, mirroring
        :meth:`write`.
        """
        if base is not None:
            self.base = Path(base)
        if name is not None:
            self.name = name
        self.params.base, self.params.name = self.base, self.name
        self.params.to_file(self.get_metadata_path())
        return self

    def write(
        self,
        base: Path | str | None = None,
        name: str | None = None,
        overwrite: bool = True,
    ) -> "VariantDataset":
        """Persist the current set of split VDSs and the metadata file.

        ``base``/``name`` choose where the new VDS lands and default to the
        handle's current location. Split writes are skipped only when the data is
        unchanged *and* the destination is the current location, so a checkpoint
        of an unchanged handle right after :meth:`process` writes metadata only,
        while an :meth:`apply` result (or a copy to a new location) is
        materialized once.

        A bare handle that holds no splits yet (see :meth:`create_bare`) has
        nothing to write here; record its parameters with
        :meth:`write_metadata` instead.
        """
        if self._datasets is None:
            raise ValueError(
                "no in-memory datasets to write; call process(), apply() or "
                "_open_datasets() first, or write_metadata() for a bare handle"
            )

        target_base = Path(base) if base is not None else self.base
        target_name = name if name is not None else self.name
        retarget = target_base != self.base or target_name != self.name

        if not self._changed and not retarget:
            logger.info("variant dataset unchanged at current location; skipping split writes")
        else:
            target_base.mkdir(exist_ok=True, parents=True)
            for task in self.params.tasks:
                vds = self._datasets.get(task.name)
                if vds is None:
                    continue
                dst = self._get_vds_path(task, base=target_base, name=target_name)
                logger.info(f"writing variant dataset split: task = {task.name}, path = {dst}")
                vds.write(dst, overwrite=overwrite)
                self.params.task_status[task.name] = ("finished", "")

        self.base, self.name = target_base, target_name
        self.params.base, self.params.name = target_base, target_name
        self._changed = False
        self.params.to_file(self.get_metadata_path())
        return self

    def to_hail(self,
        n_partitions: int | None = None,
        apply_repartition: bool = False,
        fix_haploid: bool = True,
        fix_lpl: bool = True,
        error_on_missing: bool = True,
    ) -> "hl.vds.VariantDataset":
        """Row-wise union over all finished split VDS instances.

        Unions the finished split VDSes, repartitions the reference and variant
        data to ``n_partitions`` and applies corrections for haploid genotypes and
        lpl values. It raises when the dataset is incomplete unless
        ``error_on_missing`` is false, and always raises when no split has
        finished.

        Returns
        -------
        hl.vds.VariantDataset
            A hail VariantDataset containing all data from this dataset.
        """
        finished = self._finished_tasks()
        self._open_datasets()
        logger.info(
            f"read vdses of finished tasks: n_tasks = {len(self.params.tasks)}, n_vds = {len(finished)}"
        )

        if error_on_missing and not self.is_complete:
            raise ValueError(
                f"can not merge incomplete variant dataset: n_tasks = {len(self.params.tasks)}, n_vds = {len(finished)}"
            )

        if len(finished) == 0:
            raise ValueError("got no valid variant datasets to merge")

        start_time = time()
        assert self._datasets is not None, "opening hail vds instances failed"
        vdses = [self._datasets[t.name] for t in finished]
        assert all(v is not None and isinstance(v, hl.vds.VariantDataset) for v in vdses), "could not read vds from some finished tasks"
        vds = hl.vds.VariantDataset.union_rows(*vdses)
        logger.info(
            f"finished row-wise union of vds: n_vds = {len(vdses)}, elapsed = {time() - start_time:.02f}s"
        )

        if n_partitions is None:
            n_partitions = get_default_partitions()

        logger.info(
            f"loading variant dataset with n_partitions = {n_partitions!s} (apply = {apply_repartition}), "
            f"fix_haploid = {fix_haploid}, fix_lpl = {fix_lpl}"
        )

        if apply_repartition:
            variant_mt = vds.variant_data.repartition(n_partitions, shuffle=True)
            reference_mt = vds.reference_data.repartition(n_partitions, shuffle=True)
        else:
            variant_mt = vds.variant_data
            reference_mt = vds.reference_data

        if fix_haploid:
            logger.debug("updating haploid genotypes")
            variant_mt = variant_mt.annotate_entries(
                LGT=hl.if_else(
                    variant_mt.LGT.ploidy == 1,
                    hl.call(variant_mt.LGT[0], variant_mt.LGT[0]),
                    variant_mt.LGT,
                )
            )

        if fix_lpl:
            # Make sure that the LPL field matches the number of possible local
            # genotypes, as defined by `ngt_expr` (i.e. the number of possible
            # combinations of local alleles). DRAGEN calls frequently contain
            # calls that violate these assumptions and feature more than 2 LA
            # values.
            logger.debug("extending local PL values")
            ngt_expr = (hl.len(variant_mt.LA) * (hl.len(variant_mt.LA) + 1)) // 2
            variant_mt = variant_mt.annotate_entries(
                LPL=hl.if_else(
                    hl.len(variant_mt.LPL) != ngt_expr,
                    hl.range(0, ngt_expr).map(
                        lambda i: hl.if_else(
                            i < hl.len(variant_mt.LPL), variant_mt.LPL[i], 1000
                        )
                    ),
                    variant_mt.LPL,
                )
            )

        return hl.vds.VariantDataset(
            reference_data=reference_mt, variant_data=variant_mt
        )

    def to_dense_mt(
        self,
        sites_ht: hl.Table | None = None,
        *,
        filter_to_adj: bool = False,
        fix_haploid: bool = True,
        fix_lpl: bool = False,
        apply_repartition: bool = False,
        n_partitions: int | None = None,
    ) -> hl.MatrixTable:
        """Densify the unioned callset, optionally pushing ``sites_ht`` per split.

        When ``sites_ht`` is given, every split is filtered to it *before* the
        one unavoidable union, so the merge only ever sees the requested sites.
        ``hl.vds.to_dense_mt`` fills reference-block entries as ``0/0``, and the
        :meth:`to_hail` haploid correction is kept.

        ``fix_lpl`` defaults to ``False``: densification consumers (PCA
        projection, relatedness, ancestry, contamination) read calls, not
        ``PL``, so :meth:`to_hail`'s LPL-extension expression is dead weight
        that only enlarges the task closure of the densify write (see
        ``docs/bugs/10-sample-qc-infer-sex-oom.md``). Pass ``fix_lpl=True`` when
        a caller genuinely indexes ``LPL``/``PL``.

        ``apply_repartition``/``n_partitions`` forward to :meth:`to_hail`, so
        the unioned callset can be evenly repartitioned once at the merge
        boundary before densification, instead of inheriting the uneven union of
        the per-split partitions.

        ``filter_to_adj`` keeps the adjusted-call behaviour of the ancestry step.
        """
        if sites_ht is not None:
            handle = self.filter_variants(sites_ht)
            settings = get_config()
            vds_checkpoint = Path(str(get_checkpoint_path(
                f"{settings.project}/intermediate/vds/densify.{secrets.token_hex(6)}/vds",
                CHECKPOINT_TYPE.NATIVE_VARIANT_DATASET,
            )))
            handle.write(
                base=vds_checkpoint.parent,
                name=vds_checkpoint.name.removesuffix(".metadata.json"),
            )
            handle = handle.open(vds_checkpoint)
        else:
            handle = self

        vds = handle.to_hail(
            fix_haploid=fix_haploid,
            fix_lpl=fix_lpl,
            apply_repartition=False,
        )
        mt = hl.vds.to_dense_mt(vds)
        if apply_repartition:
            mt = mt.repartition(n_partitions, shuffle=True)
        if filter_to_adj:
            from gnomad.utils.filtering import filter_to_adj as _filter_to_adj

            mt = _filter_to_adj(mt)
        return mt

    @property
    def samples(self) -> tuple[str, ...]:
        """The sorted sample set of the unioned finished variant data.

        A dataset with no finished split has no readable samples and yields the
        empty set, so a partial build can still be inspected.
        """
        if not self._finished_tasks():
            return tuple()
        vds = self.to_hail(error_on_missing=False)
        return tuple(sorted(vds.variant_data.s.collect()))

    def inspect(self) -> dict[str, Any]:
        """A serializable summary of the dataset: parameters plus sample set."""
        summary = self.params.to_dict()
        summary["samples"] = list(self.samples)
        return summary

    @classmethod
    def open(
        cls,
        path: Path | str,
        tmp_dir: Path | str | None = None,
    ):
        """Open a VariantDataset from its metadata.json file."""
        path = Path(path)
        if path.is_file():
            with open(path, "r") as in_f:
                params = VariantDatasetParameters.from_dict(json.load(in_f))
            # Update name and base in case the VariantDataset was relocated outside
            # of this tool
            expected_base = path.parent
            expected_name = path.name.removesuffix(".metadata.json")

            if expected_base != params.base:
                params.base = expected_base

            if expected_name != params.name:
                params.name = expected_name

            return cls(params.base, params.name, params, tmp_dir)
        else:
            raise FileNotFoundError(f"invalid vds path: path = {path}")

    @classmethod
    def create_bare(
        cls,
        base: Path | str,
        name: str,
        gvcfs: Iterable[str] = (),
        reference_genome: REFERENCE_GENOMES = "GRCh38",
        overwrite: bool = False,
        interval_size: int = 1_200_000,
        n_splits: int = 0,
        branch_factor: int = 100,
        target_records: int = 24_000,
        tmp_dir: Path | str | None = None,
    ) -> "VariantDataset":
        """Create a bare VariantDataset from the given parameters.
        
        Create a VariantDataset from the given parameters, but do not
        process the GVCF files supplied. This function records the calling
        intervals and sets up the general object structure only.

        Arguments
        ---------
        base : Path | str
               Path where the variant dataset instances and metadata will
               be stored.
        name : str
               Name of the variant dataset.
        gvcfs : Iterable[str]
                GVCFs to add to the variant datasets.
        reference_genome : REFERENCE_GENOMES
                           Name of the reference genome to use when importing
                           GVCF files.
        overwrite : bool
                    If true, every split is rebuilt, discarding prior
                    results; finished splits are otherwise skipped.
        interval_size : int
                        Size of each interval inside a split.
        n_splits : int
                   Number of splits to use when importing the GVCF files. If
                   less than or equal to the number of chromosomes, the dataset
                   will be partitioned by chromosome, otherwise exactly as many
                   partitions will be created.
        branch_factor : int
                        VariantDatasetCombiner branch_factor attribute.
        target_records : int
                         VariantDatasetCombiner target number of records.
        tmp_dir : Path | str
                  Temporary directory for intermediate combiner steps.
        Returns
        -------
        VariantDataset
            An empty VariantDataset instance.
        """
        gvcfs = tuple(sorted(gvcfs))
        
        base = Path(base)
        base.mkdir(exist_ok=True, parents=True)

        md_p = base / f"{name}.metadata.json"
        if md_p.is_file():
            logger.info(
                f"found existing variant dataset: path = {md_p}, base = {base}, name = {name}"
            )
            with open(md_p, "r") as in_f:
                params = VariantDatasetParameters.from_dict(json.load(in_f))
            params.overwrite = overwrite

            # The persisted parameters win over the arguments: the tasks below
            # are derived from ``params``, so a caller that asks for a different
            # split layout than the existing metadata records would otherwise be
            # silently ignored. Warn instead of changing the layout.
            if (
                reference_genome != params.reference_genome
                or interval_size != params.interval_size
                or n_splits != params.n_splits
            ):
                logger.warning(
                    "ignoring VDS build arguments that differ from the existing "
                    "metadata; the persisted parameters win: arguments = "
                    f"(reference_genome={reference_genome}, "
                    f"interval_size={interval_size}, n_splits={n_splits}), "
                    f"metadata = (reference_genome={params.reference_genome}, "
                    f"interval_size={params.interval_size}, "
                    f"n_splits={params.n_splits})"
                )
        else:
            params = VariantDatasetParameters(
                base,
                name,
                gvcfs,
                tuple(),
                reference_genome,
                overwrite=overwrite,
                interval_size=interval_size,
                n_splits=n_splits,
                branch_factor=branch_factor,
                target_records=target_records,
            )

        # If the first version of this instance has no gvcfs loaded, allow setting
        # the gvcfs parameter here:
        if set(gvcfs) != set(params.gvcfs):
            if len(gvcfs) > 0 and len(params.gvcfs) > 0 and not overwrite:
                raise ValueError("can not change gvcf input without overwriting the VariantDataset")

            if len(gvcfs) > 0:
                params.gvcfs = gvcfs

        tasks = VariantDatasetTaskIntervals.get_interval_tasks(
            reference_genome=params.reference_genome,
            interval_size=params.interval_size,
            n_splits=params.n_splits,
        )
        if len(params.tasks) > 0:
            new_tasks = json.dumps([t.to_dict() for t in tasks])
            old_tasks = json.dumps([t.to_dict() for t in params.tasks])
            if new_tasks != old_tasks:
                logger.warning(
                    f"overwriting previous tasks in the vds parameters: params = {params}, tasks = {tasks}"
                )
                params.task_status = {}
        params.tasks = tuple(tasks)

        logger.debug(
            f"finished creating variant dataset tasks: n_tasks = {len(tasks)}, interval_size = {params.interval_size}"
        )
        vds = cls(
            base,
            name,
            params,
            tmp_dir,
        )
        return vds

    @classmethod
    def create_from_gvcf(
        cls,
        base: Path | str,
        name: str,
        gvcfs: Iterable[str] = (),
        reference_genome: REFERENCE_GENOMES = "GRCh38",
        overwrite: bool = False,
        interval_size: int = 1_200_000,
        n_splits: int = 0,
        branch_factor: int = 100,
        target_records: int = 24_000,
        tmp_dir: Path | str | None = None,
    ) -> "VariantDataset":
        """Create a VariantDataset instance from the given parameters.

        Arguments
        ---------
        base : Path | str
               Path where the variant dataset instances and metadata will
               be stored.
        name : str
               Name of the variant dataset.
        gvcfs : Iterable[str]
                GVCFs to add to the variant datasets.
        reference_genome : REFERENCE_GENOMES
                           Name of the reference genome to use when importing
                           GVCF files.
        overwrite : bool
                    If true, every split is rebuilt, discarding prior
                    results; finished splits are otherwise skipped.
        interval_size : int
                        Size of each interval inside a split.
        n_splits : int
                   Number of splits to use when importing the GVCF files. If
                   less than or equal to the number of chromosomes, the dataset
                   will be partitioned by chromosome, otherwise exactly as many
                   partitions will be created.
        branch_factor : int
                        VariantDatasetCombiner branch_factor attribute.
        target_records : int
                         VariantDatasetCombiner target number of records.
        tmp_dir : Path | str
                  Temporary directory for intermediate combiner steps.
        Returns
        -------
        VariantDataset
            A VariantDataset instance containing the data from all GVCFs.
        """
        vds = cls.create_bare(base, name, gvcfs, reference_genome, overwrite,
                              interval_size, n_splits, branch_factor, target_records,
                              tmp_dir)
        vds.process()
        return vds
