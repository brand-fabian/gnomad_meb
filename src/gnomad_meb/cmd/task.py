"""The merged, serializable description of one pipeline invocation.

:class:`Task` couples the resolved global :class:`~gnomad_meb.config.Settings`
with the options of exactly one leaf command. The command part is a
discriminator-tagged union, so a dumped task reloads with
``Task.model_validate(...)`` from its JSON alone — without consulting the
environment or a config file. Its JSON form is the invocation's task
description.

Click is retained as the parser (see ``docs/adr/0005``). Each leaf command
declares its options as a :class:`CommandModel`, is wrapped with
:func:`task_command`, and receives the single merged ``Task``; the wrapper also
handles the global ``--describe`` flag and moves hail initialization into the
command.
"""

from __future__ import annotations

import functools
from typing import Annotated, Any, Literal

import click
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from gnomad_meb.config import Settings
from gnomad_meb.constants import REFERENCE_GENOMES


class CommandModel(BaseModel):
    """Base for a leaf command's option model."""

    model_config = ConfigDict(extra="ignore")


class VdsCreate(CommandModel):
    """``vds create`` options.

    ``overwrite`` rebuilds every split, discarding prior results; finished
    splits are otherwise skipped on a resume. The composed workflow inherits
    this flag, so ``flow --overwrite`` reaches the VDS build too.
    """

    kind: Literal["vds.create"] = "vds.create"
    gvcf: tuple[str, ...] = ()
    bucket: tuple[str, ...] = ()
    search: tuple[str, ...] = ()
    vds_name: str = "vds"
    output_base: str = "./output"
    reference_genome: REFERENCE_GENOMES = "GRCh38"
    s3_search_pattern: tuple[str, ...] = ("**/*gvcf.gz",)
    local_search_pattern: tuple[str, ...] = ("*gvcf.gz",)
    n_splits: int = 0
    interval_size: int = 1_200_000
    overwrite: bool = False


class VdsInspect(CommandModel):
    """``vds inspect`` options."""

    kind: Literal["vds.inspect"] = "vds.inspect"
    vds: str
    format: str = "json"


class VdsFind(CommandModel):
    """``vds find`` options."""

    kind: Literal["vds.find"] = "vds.find"
    gvcf: tuple[str, ...] = ()
    bucket: tuple[str, ...] = ()
    search: tuple[str, ...] = ()
    s3_search_pattern: tuple[str, ...] = ("**/*gvcf.gz",)
    local_search_pattern: tuple[str, ...] = ("*gvcf.gz",)
    format: str = "json"


class SampleQcRun(CommandModel):
    """``sample-qc run`` options."""

    kind: Literal["sample_qc.run"] = "sample_qc.run"
    vds: str


class VariantQcExport(CommandModel):
    """``variant-qc export`` options.

    ``as_info`` / ``quasi_info`` are output *directories*: the export stage
    writes one sites-only VCF per VDS split to
    ``<dir>/task-<task>/{as,quasi}_info.vcf.bgz``.
    """

    kind: Literal["variant_qc.export"] = "variant_qc.export"
    vds: str
    as_info: str | None = None
    quasi_info: str | None = None
    infer_as_annotations: bool = True
    cdf_k: int = 200
    retain_cdfs: bool = False


class VariantQcLoadVqsr(CommandModel):
    """``variant-qc load-vqsr`` options.

    ``vcf`` is the per-split VQSR VCF *directory* (one
    ``<dir>/task-<task>/vqsr.vcf.bgz`` per VDS split); ``as_info`` is the
    per-split ``as_info`` hail table directory the export stage writes
    (``<dir>/task-<task>/as_info.ht``). ``vds`` supplies the split task list.
    """

    kind: Literal["variant_qc.load_vqsr"] = "variant_qc.load_vqsr"
    vds: str
    vcf: str
    as_info: str


class VqsrRun(CommandModel):
    """``vqsr pipeline`` options, including the snakemake passthrough.

    ``kind`` stays ``"vqsr.pipeline"`` for serialized-run compatibility. The
    input locations are explicit invocation options: ``vds`` is the run's VDS
    handle (its task list is the scatter), ``as_info`` the per-split
    ``as_info`` VCF directory, ``output_dir`` the per-task VQSR VCF root, and
    ``calling_intervals`` the global picard ``interval_list`` the per-task
    intersections are derived from. ``snakemake`` is an optional override for
    the ``[vqsr].snakemake`` setting (where cluster arguments live), so the
    composed workflow can be configured without a VQSR-specific CLI flag.
    """

    kind: Literal["vqsr.pipeline"] = "vqsr.pipeline"
    vds: str
    as_info: str
    output_dir: str
    calling_intervals: str
    snakemake: tuple[str, ...] = ()


class AnnotationRun(CommandModel):
    """``annotation run`` options."""

    kind: Literal["annotation.run"] = "annotation.run"
    vds: str


class FilterRun(CommandModel):
    """``filter run`` options.

    ``mt`` is the annotated matrix table to filter. It is optional at the model
    level so the composed workflow can leave it unset and rely on the wired
    annotation edge; the ``filter run`` CLI enforces it as a required option.
    """

    kind: Literal["filter.run"] = "filter.run"
    mt: str | None = None


class BrowserExportRun(CommandModel):
    """``browser run`` options.

    ``mt`` is the MatrixTable to reshape into the browser schema. It is optional
    so a run with no ``--mt`` defaults to the configured annotation checkpoint.
    """

    kind: Literal["browser.run"] = "browser.run"
    mt: str | None = None


class GwasExport(CommandModel):
    """``gwas export`` options.

    ``mt`` is a single ``high_quality_gnomad`` MatrixTable path; ``manifest`` is
    the Filter stage's ``manifest.json.gz``. Exactly one is required.
    ``include``/``exclude`` select tables from a manifest (by label, filter name
    or region name) and are rejected with ``mt``. ``metadata_csv`` and ``base``
    optionally override the ``[gwas]`` config for this invocation. Tables below
    the ``[gwas].min_n_snps`` floor (or with no recorded count) are dropped.
    """

    kind: Literal["gwas.export"] = "gwas.export"
    mt: str | None = None
    manifest: str | None = None
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    metadata_csv: str | None = None
    base: str | None = None

    @model_validator(mode="after")
    def _exactly_one_input(self) -> GwasExport:
        if (self.mt is None) == (self.manifest is None):
            raise ValueError("exactly one of 'mt' or 'manifest' is required")
        if self.mt is not None and (self.include or self.exclude):
            raise ValueError("'include'/'exclude' are only valid with a manifest")
        return self


class GwasDatasetRef(BaseModel):
    """One exported dataset and where its regenie results land.

    ``label`` is the step-id-safe dataset suffix (``""`` for a raw ``--mt``,
    else ``{table}.{region}``); ``dataset`` is the exported region directory and
    ``gwas``/``burden`` are the regenie output directories under the run root.
    Small and JSON-serializable so a serialized task round-trips.
    """

    label: str = ""
    dataset: str
    gwas: str
    burden: str


class GwasPipeline(CommandModel):
    """``gwas pipeline`` options.

    ``dataset`` is the exported region directory (PLINK prefix + six TSVs) and
    ``output_dir`` the root the regenie ``gwas``/``burden`` outputs land under.
    ``datasets`` is the multi-dataset form the composed workflow uses; when it
    is empty the legacy single ``dataset``/``output_dir`` pair is used.
    ``snakemake`` is an optional override for the ``[gwas].snakemake`` setting,
    where cluster arguments (``--profile`` / ``--executor``) live.
    """

    kind: Literal["gwas.pipeline"] = "gwas.pipeline"
    dataset: str | None = None
    output_dir: str | None = None
    snakemake: tuple[str, ...] = ()
    datasets: list[GwasDatasetRef] = Field(default_factory=list)

    @model_validator(mode="after")
    def _legacy_or_datasets(self) -> GwasPipeline:
        if not self.datasets and (not self.dataset or not self.output_dir):
            raise ValueError(
                "provide 'dataset' and 'output_dir', or a non-empty 'datasets'"
            )
        return self


class GwasLoad(CommandModel):
    """``gwas load`` options.

    ``dataset`` is the exported region directory (for the phenotype table),
    ``gwas``/``burden`` the regenie output directories. The regenie provenance
    file is read from the parent of ``gwas``. ``datasets`` is the multi-dataset
    form the composed workflow uses; when it is empty the legacy single
    ``dataset``/``gwas``/``burden`` triple is used.
    """

    kind: Literal["gwas.load"] = "gwas.load"
    dataset: str | None = None
    gwas: str | None = None
    burden: str | None = None
    datasets: list[GwasDatasetRef] = Field(default_factory=list)

    @model_validator(mode="after")
    def _legacy_or_datasets(self) -> GwasLoad:
        if not self.datasets and not (self.dataset and self.gwas and self.burden):
            raise ValueError(
                "provide 'dataset', 'gwas' and 'burden', or a non-empty 'datasets'"
            )
        return self


class GwasWorkflowRun(CommandModel):
    """``gwas workflow`` options: export -> regenie -> load as one DAG.

    ``mt``/``manifest`` (exactly one) is the workflow's input; ``include`` and
    ``exclude`` narrow a manifest's tables (by label, filter name or region
    name). Tables below the ``[gwas].min_n_snps`` floor (or with no recorded
    count) are dropped from the whole workflow. ``output_base`` and
    ``output_dir`` override the export and run roots.
    """

    kind: Literal["gwas.workflow"] = "gwas.workflow"
    mt: str | None = None
    manifest: str | None = None
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    output_base: str | None = None
    output_dir: str | None = None
    snakemake: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _exactly_one_input(self) -> GwasWorkflowRun:
        if (self.mt is None) == (self.manifest is None):
            raise ValueError("exactly one of 'mt' or 'manifest' is required")
        if self.mt is not None and (self.include or self.exclude):
            raise ValueError("'include'/'exclude' are only valid with a manifest")
        return self


class GnomadWorkflowRun(VdsCreate):
    """``flow`` options: the whole GVCF -> VDS -> QC -> annotation -> filter run.

    The composed workflow's front door is the VDS build, so its command carries
    the ``vds create`` options and the inherited run-level ``overwrite`` flag
    (the composite builds the VDS it then runs QC and annotation over).
    ``kind`` distinguishes it from a standalone ``vds.create`` so a dumped run
    reloads as the workflow.

    The composed stages' own knobs are not duplicated here: they are config
    (``[vqsr].snakemake``, ``resources.calling_intervals``, derived paths), so a
    run is configured through the TOML file or ``GMEB_`` environment rather than
    a growing set of ``flow`` flags.
    """

    kind: Literal["gnomad_workflow.run"] = "gnomad_workflow.run"


class ResourcePrepare(CommandModel):
    """One mini-Step's worth of a Resource's preparation.

    A Resource is realised as an ordered chain of ``resource.prepare`` Steps: a
    load Step (``modification_kind`` is ``None``) followed by one Step per
    modification. The command carries enough to route each Step to its executor
    without importing anything hail-dependent.
    """

    kind: Literal["resource.prepare"] = "resource.prepare"
    loader: str
    modification_kind: str | None = None
    params: dict[str, Any] = {}
    source: str
    output: str
    output_kind: str
    source_reference_genome: REFERENCE_GENOMES
    target_reference_genome: REFERENCE_GENOMES


class PipelineStep(CommandModel):
    """One work Step of a ported pipeline.

    Ported pipelines (Sample QC, Variant QC export/import, Filter, Annotation,
    VDS build) declare their Steps with this single discriminated kind rather
    than one CommandModel per closure. ``name`` and ``group`` identify the Step
    (its id is ``<group>.<name>``); ``args`` carries its JSON-serializable
    inputs. The durable form stays declarative JSON.
    """

    kind: Literal["pipeline.step"] = "pipeline.step"
    name: str
    group: str | None = None
    args: dict[str, Any] = {}


#: The discriminated union of every leaf command's options.
Command = Annotated[
    VdsCreate
    | GnomadWorkflowRun
    | VdsInspect
    | VdsFind
    | SampleQcRun
    | VariantQcExport
    | VariantQcLoadVqsr
    | VqsrRun
    | AnnotationRun
    | FilterRun
    | BrowserExportRun
    | GwasExport
    | GwasPipeline
    | GwasLoad
    | GwasWorkflowRun
    | ResourcePrepare
    | PipelineStep,
    Field(discriminator="kind"),
]
CommandAdapter = TypeAdapter(Command)


class Task(BaseModel):
    """One CLI invocation: the resolved ``settings`` plus one ``command``.

    ``model_dump_json`` is the canonical task description; ``model_validate``
    reloads a dumped task from its JSON alone.
    """

    model_config = ConfigDict(extra="ignore")

    settings: Settings
    command: Command


def task_command(command_type: type[CommandModel], pipeline: str | None = None):
    """Wrap a leaf command so click parses options into a merged :class:`Task`.

    The wrapped function receives the ``Task`` and reads its inputs from it. The
    two global inspection flags are honoured here before hail is initialized:

    - ``--describe`` prints the run envelope (``pipeline``, ``task``, and no
      ``steps``) as JSON or YAML and returns immediately.
    - ``--dry-run`` calls the command body with ``dry_run=True`` so it can build
      its pipeline and print the run manifest; hail is not initialized and no
      step executes.

    ``pipeline`` names the top-level pipeline this command drives, matching the
    pipeline class its body constructs. It is ``None`` for a command that runs
    no pipeline (``vds inspect``/``vds find``), whose envelope is then not
    runnable. The name is decorator metadata because ``--describe`` is
    intercepted here, before the leaf body builds anything.
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(**params):
            ctx = click.get_current_context()
            root = ctx.find_root()
            task = Task(
                settings=root.obj["settings"],
                command=command_type(**params),
            )
            output_format = root.obj["output_format"]

            if root.obj["describe"]:
                from gnomad_meb.cmd.output import emit, manifest_payload

                emit(
                    manifest_payload(
                        pipeline,
                        task,
                        overwrite=params.get("overwrite", False),
                    ),
                    output_format,
                )
                return None

            from gnomad_meb.logging import get_logger

            get_logger().debug(f"task: {task.model_dump_json()}")

            # Boot Hail once for every ``task_command`` invocation that will touch
            # it, before the body builds anything. ``pipeline is None`` marks the
            # pipelineless commands (``vds inspect``/``vds find``) that do no Hail
            # work, so they stay boot-free. ``--describe`` returns above without
            # building a pipeline and is likewise Hail-free. Booting here, from
            # the one decorator every pipeline command passes through, keeps
            # ``hail_init``'s fd-1 guard over the whole build/prepare window.
            if pipeline is not None:
                from gnomad_meb.hail import hail_init

                hail_init()

            if root.obj["dry_run"]:
                return fn(task, dry_run=True, output_format=output_format)
            return fn(task)

        return wrapper

    return decorator
