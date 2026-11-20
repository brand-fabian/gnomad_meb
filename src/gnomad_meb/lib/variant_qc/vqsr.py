"""The VQSR stage: one Snakemake invocation whose scatter is the VDS task list.

The rewritten VQSR workflow trains the SNP and INDEL recalibration models once
globally and applies them per VDS split. This module is the pipeline entry
point: :class:`VqsrPipeline` registers a single Snakemake-backed step,
``run_vqsr_pipeline``, that declares every per-task
``{output_dir}/task-{task}/vqsr.vcf.bgz`` as an artifact (so resume checks them
all) and, before invoking Snakemake, computes each task's restricted calling
intervals in plain Python.

The Snakemake rule rewrite (global models, per-task applies, one
``concat_as_info``) is owned by a separate change; the step body delegates to
:func:`gnomad_meb.vqsr.vqsr` lazily, so importing this module never pulls in
Snakemake.
"""

from __future__ import annotations

from pathlib import Path

from gnomad_meb.cmd.task import Task
from gnomad_meb.lib.checkpoint import Artifact, declares_step
from gnomad_meb.lib.variant_qc.constants import GATK_VERSION_FILENAME
from gnomad_meb.pipeline import InputStep, Pipeline, RunInput

#: The stage name; also the step-group prefix.
GROUP = "vqsr"


def _read_interval_header(path: str | Path) -> list[str]:
    """The SAM-style header lines (``@...``) of a picard ``interval_list``."""
    header: list[str] = []
    with open(path) as handle:
        for line in handle:
            if line.startswith("@"):
                header.append(line.rstrip("\n"))
            elif line.strip():
                break
    return header


def _read_intervals(path: str | Path) -> list[tuple[str, int, int]]:
    """Parse a picard ``interval_list``'s ``(contig, start, end)`` data rows.

    Rows are 1-based inclusive; the header is ignored. The input is already an
    interval_list, so no reference-genome lookup is needed.
    """
    intervals: list[tuple[str, int, int]] = []
    with open(path) as handle:
        for line in handle:
            if not line.strip() or line.startswith("@"):
                continue
            fields = line.rstrip("\n").split("\t")
            intervals.append((fields[0], int(fields[1]), int(fields[2])))
    return intervals


def _contig_order(header: list[str]) -> list[str]:
    """The ``@SQ`` contigs in header order, used to sort the output rows."""
    order: list[str] = []
    for line in header:
        if not line.startswith("@SQ"):
            continue
        for field in line.split("\t")[1:]:
            if field.startswith("SN:"):
                order.append(field[3:])
    return order


def _merge(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or adjacent 1-based inclusive intervals."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _per_task_intervals(calling_intervals: str, task, target) -> str:
    """Write ``calling_intervals ∩ task.intervals`` as a picard ``interval_list``.

    The global calling intervals are indexed with :mod:`intervaltree` and
    queried with the task's hail intervals, so each per-task apply is restricted
    to that split's callable sites. Overlapping/adjacent sub-intervals are merged
    to keep the list compact, and the ``@SQ`` contig lengths are carried over
    from the input header. This is a plain function -- no Step and no Snakemake
    rule -- so nothing is scheduled for the cheap intersection.
    """
    from intervaltree import IntervalTree

    header = _read_interval_header(calling_intervals)
    tree = IntervalTree()
    for contig, start, end in _read_intervals(calling_intervals):
        tree[start - 1 : end] = contig  # 0-based half-open

    by_contig: dict[str, list[tuple[int, int]]] = {}
    for interval in task.intervals:
        contig = interval.start.contig
        start = interval.start.position + (0 if interval.includes_start else 1)
        end = interval.end.position - (0 if interval.includes_end else 1)
        if end < start:
            continue
        # hail intervals are 0-based half-open; convert to a query range
        query_begin, query_end = start - 1, end
        for hit in tree.overlap(query_begin, query_end):
            if hit.data != contig:
                continue
            begin = max(hit.begin, query_begin)
            finish = min(hit.end, query_end)
            if finish > begin:
                # back to 1-based inclusive for the interval_list
                by_contig.setdefault(contig, []).append((begin + 1, finish))

    order = _contig_order(header)
    contigs = [contig for contig in order if contig in by_contig]
    contigs.extend(contig for contig in by_contig if contig not in order)

    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w") as handle:
        for line in header:
            handle.write(f"{line}\n")
        for contig in contigs:
            for start, end in _merge(by_contig[contig]):
                handle.write(f"{contig}\t{start}\t{end}\t+\t{contig}:{start}-{end}\n")
    return str(target)


class VqsrPipeline(Pipeline):
    """Declare and run the VQSR stage.

    One Snakemake-backed step runs the whole callset's workflow; the VDS task
    list is the scatter (Snakemake wildcards are task names), so ~100 per-split
    apply jobs are submitted in a single DAG under the configured cluster
    profile instead of ~100 sequential invocations. The step declares every
    per-task output so resume checks them all.
    """

    @classmethod
    def name(cls) -> str:
        return GROUP

    @property
    def resources(self) -> tuple:
        return ()

    @property
    def snakemake_args(self) -> tuple[str, ...]:
        """The Snakemake pass-through: the command's override, else the config.

        Cluster arguments (``--profile`` / ``--executor``) are run configuration,
        so they live under ``[vqsr].snakemake``. The standalone ``vqsr pipeline``
        command may still pass them explicitly, which wins; when it does not —
        notably inside the composed ``gnomad-workflow``, which has no such flag —
        the setting supplies them.
        """
        override = getattr(self.command, "snakemake", ()) or ()
        if override:
            return tuple(override)
        vqsr = self.settings.vqsr
        return tuple(vqsr.snakemake) if vqsr is not None else ()

    def get_input(self) -> tuple[InputStep, ...]:
        """The run's VDS handle: the task list is read from its metadata."""
        from gnomad_meb.lib.vds import VariantDataset

        return (
            InputStep(
                name="vds",
                task=Task(settings=self.settings, command=self.command),
                group=type(self).name(),
                source="vds.create_variant_dataset",
                value=getattr(self.command, "vds", None),
                run=VariantDataset.open,
            ),
            InputStep(
                name="as_info",
                task=Task(settings=self.settings, command=self.command),
                group=type(self).name(),
                value=getattr(self.command, "as_info", None),
            ),
        )

    def prepare(self) -> None:
        """Read the task list from the VDS handle and finalize the step's artifacts.

        ``VariantDataset.open`` reads metadata only, so ``--describe`` and
        reconstruction rebuild the same per-task artifact set from the same
        input. A missing handle fails loudly, so the two invocation scenarios can
        never disagree about the steps.
        """
        from gnomad_meb.lib.variant_qc.tasks import vds_tasks

        super().prepare()

        tasks = vds_tasks(self.command)
        self._task_names = tuple(task.name for task in tasks)

    def _artifacts(self, name: str, method) -> tuple:
        """The per-task VQSR VCFs, gathered tranches and provenance the step declares.

        The gathered ``vqsr.{snp,indel}.tranches.txt`` are declared outputs in
        their own right (the workflow's ``rule all`` builds them); the import
        stage parses them for the applied VQSLOD cutoffs, and declaring them
        here makes the resume check cover them too. ``gatk.version.json`` is
        written by :func:`gnomad_meb.vqsr.vqsr.vqsr` once Snakemake succeeds,
        next to the VCFs, and read back by the import.
        """
        if name == "run_vqsr_pipeline":
            output_dir = self.command.output_dir
            return (
                tuple(
                    Artifact.from_path(f"{output_dir}/task-{task}/vqsr.vcf.bgz")
                    for task in self._task_names
                )
                + tuple(
                    Artifact.from_path(f"{output_dir}/vqsr.{mode}.tranches.txt")
                    for mode in ("snp", "indel")
                )
                + (Artifact.from_path(f"{output_dir}/{GATK_VERSION_FILENAME}"),)
            )
        return super()._artifacts(name, method)

    @declares_step
    def run_vqsr_pipeline(self, vds=RunInput("vds"), as_info=RunInput("as_info")):  # noqa: B008
        """Run the VQSR workflow once for the whole callset.

        The task list is the scatter: the per-task calling-interval lists are
        computed here, in Python, before Snakemake is invoked; the workflow
        itself is owned by :func:`gnomad_meb.vqsr.vqsr` and imported lazily.
        """
        command = self.command
        working_dir = Path(self.settings.scratch_dir) / "vqsr"

        task_inputs: dict[str, dict[str, str]] = {}
        for task in vds.params.tasks:
            target = (
                working_dir / f"task-{task.name}" / "calling_intervals.interval_list"
            )
            task_inputs[task.name] = {
                "as_info": f"{command.as_info}/task-{task.name}/as_info.vcf.bgz",
                "calling_intervals": _per_task_intervals(
                    command.calling_intervals, task, target
                ),
            }

        from gnomad_meb.vqsr import vqsr as run_workflow

        return run_workflow(task_inputs, command.output_dir, self.snakemake_args)
