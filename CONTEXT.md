# gnomad-meb

The gnomad-meb pipeline turns per-sample GVCFs into a quality-controlled,
annotated callset through a sequence of CLI-driven steps.

## Language

**VDS (Variant Dataset)**:
Hail's sparse representation of variant data, built from GVCFs and the input to
Quality Control and Annotation.
_Avoid_: raw dataset, combined dataset

**VariantDataset**:
The chunked, split-aware handle over a set of Split VDSes in `gnomad_meb.lib.vds`.
It maps transforms over the splits (`apply`, `map_datasets`, `reduce_datasets`,
ADR-0015) and persists them explicitly and idempotently through `write`
(ADR-0016). Distinguish it from the Hail data representation above.

Quality Control consumes the handle per split rather than merging. Sparse filters
are `filter_*` wrappers that run each split's `hl.vds` filter through `apply`
(`filter_intervals`, `filter_chromosomes`, `filter_variants`, `filter_samples`,
`filter_rows`, and `split_multi`); `select_tasks` and `filter_contigs` lazily
narrow the task set without reading the data. Cross-sample operations
(relatedness, ancestry) read **one shared checkpointed dense matrix** over the
PCA loadings sites, produced once by `densify_pca_sites` (`pca_sites`), instead
of each recomputing the densification. Per-sample metrics (`hl.vds.sample_qc`
and the adj call rates) run per split and are combined additively, with the
derived ratio fields recomputed from the folded counts. Only the genuinely
non-decomposable steps still merge: `get_infer_sex`'s sex-contig-scoped merge
and `get_contamination_estimate` (`hl.compute_charr`); the rest stay at split
granularity (ADR-0015).
_Avoid_: merged VDS, combined dataset

**Split VDS**:
One per-task hail `VariantDataset` — the unit `VariantDataset.process` builds and
`VariantDataset.apply`/`map_datasets` iterate. Transforms and metrics run per
split so a step need not merge (ADR-0015).
_Avoid_: chunk, shard, partition

**Materialization**:
Persisting the in-memory Split VDSes of a `VariantDataset` to disk. Done once by
`VariantDataset.write`, and skipped for an unchanged handle (ADR-0016).
_Avoid_: merge, flush, write-through

**Checkpoint**:
A cached intermediate artifact (`.vds`, `.mt`, `.ht`) stored under
`{checkpoint}/{project}/` and reused instead of recomputed. A checkpoint is
written and queried through a `CheckpointedCallable` (see **Checkpoint
status**).
_Avoid_: cache, temp output

**Checkpoint status**:
The verdict a `CheckpointedCallable` reports for its artifacts without invoking
the step: `absent` (no artifact, or data incomplete), `present` (artifact and
sidecar agree with the run), `stale` (artifact present, provenance or lineage
disagrees), `unknown` (artifact present, no sidecar to compare). The runner's
skip decision and the wrapper's cache hit read the same status, so they cannot
drift (ADR-0020).
_Avoid_: cache state, skip flag

**Staleness**:
An existing artifact whose provenance (generating source, git revision, software
version) or lineage (an upstream producer recomputed this run) disagrees with the
current run. Recorded as a `StalenessReason`; the callable's stale flag makes the
next call recompute rather than reuse.
_Avoid_: invalid, expired

**Staleness policy**:
`completeness` (default — data presence only, the historical behaviour),
`source` (also recompute when `source_md5` changed) or `source_or_git` (also
`git_revision`/`version`). Set by `Settings.staleness_policy` / env
`GMEB_STALENESS_POLICY`.
_Avoid_: cache policy, invalidation mode

**Artifact**:
A declared output of a Step: a `CHECKPOINT_TYPE` kind, a `{project}`-relative
`prefix`, and (optionally) a `suffix` overriding the kind's default extension
(`.vcf.bgz`, `.onnx`, `.metadata.json`). It is declared by a `@checkpoint`
decorator (which returns a `CheckpointedCallable`) and serialized by the kind's
serializer in `lib.checkpoint`; `Artifact.exists()` delegates completeness
(`_SUCCESS`, the metadata sidecar, S3) to that serializer.
_Avoid_: output, file, result

**Project**:
The experiment namespace that scopes checkpoint and output paths
(`{checkpoint}/{project}/…`), set by `--project-name`.
_Avoid_: experiment, run, cohort

**Quality Control**:
The pipeline stage producing metrics; split into Sample QC and Variant QC,
which run independently of each other.

**Sample QC**:
Per-sample quality control — inferred sex, relatedness/family structure,
ancestry assignment, and hard filters — written to the `sample_qc` table.
_Avoid_: per-sample QC, individual QC

**Sex karyotype**:
A sample's inferred sex-chromosome complement — `XX`/`XY` for a euploid call,
otherwise a sex-aneuploidy value or `ambiguous` — read from
`sample_qc.ploidy.sex_karyotype`. Distinct from *inferred sex*, which maps a
karyotype onto the pedigree `M`/`F` vocabulary by presence of a Y.
_Avoid_: inferred sex, gender

**Ambiguous sex**:
A sample whose sex karyotype cannot be decided because its chrX hom-alt
fraction falls inside the screened band between the two f-stat-group medians;
`sex_karyotype` is then the literal `"ambiguous"`.
_Avoid_: unknown sex, indeterminate sex

**Sex aneuploidy**:
A sex karyotype that is none of `XX`, `XY` or `ambiguous` (for example `X`,
`XXX`, `XXY`), flagged by the `sex_aneuploidy` hard filter. The sample stays in
the `sample_qc` table; exclusion is a downstream decision.
_Avoid_: aneuploidy (unqualified), autosomal aneuploidy

**Screening rule**:
The rule that restricts the samples used to derive sex-karyotype cutoffs to
those whose chrX hom-alt fraction is concordant with their f-stat group. Under
`between_group_medians` (`screening_rule`) a sample is kept when its fraction
lies outside the band spanned by the two group medians; the kept samples are the
*derivation set*, and the excluded — intermediately homozygous — samples still
receive a karyotype but do not set the cutoffs. The rule name and the excluded
band are exported as the `screening_rule` / `screening_band` globals.
_Avoid_: filtering, outlier removal

**Variant QC**:
Per-variant quality control, split-aware end to end. Export computes one
`info`/`as_info`/`quasi_info` checkpoint and one sites-only VCF **per split**;
VQSR concatenates the per-split VCFs into one site-only VCF, trains one global
SNP and one global INDEL model on it, then applies each model back per split as
a scatter-gather pass (`VariantRecalibrator --output-tranches-for-scatter` →
`GatherTranches` → `ApplyVQSR`); load imports and splits one VCF per split, and a
terminal `merge` unions the per-split tables once into the `variant_qc` table.
Only that final union is a merge — every earlier combination is a cheap
`bcftools concat` of site-only VCFs used for model training.
_Avoid_: global info table, per-split model

**VQSR**:
Variant Quality Score Recalibration — the GATK/snakemake stage that operates on
the exported per-split sites-only VCFs. Its scatter unit is the VDS task list,
not a separate `scatter` setting, and the SNP and INDEL models are trained once
on the whole callset (then applied per split with `GatherTranches`-collated
tranches) so `AS_VQSLOD` stays comparable across splits.
_Avoid_: variant recalibration, per-split model

**Annotation**:
The final stage joining Sample QC, Variant QC, and reference annotations onto
the VDS, written as the `annotation` matrix table.

**Reference genome**:
A reference build (GRCh38, GRCh37, CHM13) that scopes the fixture dataset and
the reference resources.
_Avoid_: build, assembly

**Pedigree**:
The family relations, sex, and affected status of the fixture samples, given in
`samples.tsv`.
_Avoid_: family data, metadata, samples

**Expected values**:
The asserted ground truth for the fixture, split by what it verifies and how it
is keyed: Sample QC results keyed by sample (`sample_qc.tsv`), VQSR results
keyed by site (`vqsr.tsv`), and annotation values keyed by site
(`annotation.tsv`). The Pedigree (`samples.tsv`) is separate; the former
single-file `expected.tsv` is retired.
_Avoid_: golden output, baseline, truth set

### GWAS

**GWAS table**:
One input MatrixTable the GWAS stages fan out over — the `high_quality` basis,
a filter's `mt`, or a filter-region's `mt`. A raw `--mt` is a one-table
inventory whose label is empty, so its paths reproduce the single-table ones.
_Avoid_: dataset, matrix

**Dataset**:
The exported regenie directory for one `(table, region)` pair — the unit
`gwas pipeline` consumes and `gwas load` keys.
_Avoid_: table, region

**Region**:
One `[gwas].regions` spec (`gene`, `window_10000`, …); the export's fan-out axis
within a table.
_Avoid_: interval, window

**Dataset key**:
A `(table, region)` identity and every name derived from it — the dataset label,
its path segment, its step name and its checkpoint prefix. The empty table label
reproduces the single-table names, so a raw `--mt` is the same object as the
namespaced one.
_Avoid_: label, suffix, name

**Response**:
One phenotype column after the `min_case_count` gate (case/control, a `set`
value's indicator, or a direct binary/quantitative column). The export writes
only the selected responses, so the exported phenotype table is the single
source of truth for which regenie outputs exist.
_Avoid_: phenotype, trait

**GWAS metadata**:
The study-metadata CSV and the exported dataset's metadata file, read through one
object: it parses the CSV, applies the `min_case_count` gate, and reports only
the selected responses (with their kind and display-name maps), and reads each
dataset's recorded provenance without hail. The study `study_md` projection is a
deferred-hail method on it.
_Avoid_: sidecar, metadata file, phenotype table

**GWAS inventory**:
The ordered `list[GwasTable]` a GWAS command's input (`--mt` or `--manifest`,
narrowed by `--include`/`--exclude`) resolves to, built during `prepare` without
hail — a `GwasInventory` object constructed from the command.
_Avoid_: manifest, table list

**GWAS manifest**:
The load stage's `manifest.json.gz`, indexing every loaded `(table, region)`
dataset, its `gwas`/`burden` tables and its provenance — the GWAS analog of the
Filter manifest.
_Avoid_: load manifest, output list

### Pipeline structure

**Pipeline**:
A stage declared as a subclass of the `Pipeline` base. It declares its `name`
(an abstract classmethod), its `resources` (a property returning a tuple of
`Resource` objects — possibly computed from the run's settings — or `()` when it
needs none), and its `get_input` endpoints (`InputStep`s, or `()`). Its work
is `@checkpoint`-decorated methods. Constructing a Pipeline registers its
resources and steps in the process-global `ResourceRegistry` and `StepRegistry`;
it is **not** a durable model. `PipelineFactory.make` finds the right subclass by
`name()` and re-instantiates it; `PipelineRunner.load` restores a dumped run
through the factory and the status model.
_Avoid_: adapter, wrapper, workflow

**Step**:
A single registered unit of work — a `@checkpoint`-decorated method on a
`Pipeline` subclass, projected into a `Step` (its `Artifact`s, its
`depends_on`/`resources` maps, an optional group, and the run callable). Its
dependencies, Resources, and caller-supplied inputs come from its parameter
defaults, the markers `FromStep`/`ResourceRef`/`RunInput`. It embeds exactly one
Task. Its lifecycle status
(`init`/`started`/`skipped`/`finished`/`error`, plus `dropped` for a de-selected
step and `pending` for a runtime-queued one) and its
`started_at`/`finished_at` timing are recorded by the runner in the run's status
model, keyed by step id.
_Avoid_: node, job, stage

**Run manifest** (the run envelope):
The declarative projection of one run — `RunManifest`, the single model
`--describe` and `--dry-run` both emit and `run` validates. It carries
`{pipeline, task, steps?, overwrite, runnable}`: `pipeline` names the top-level
pipeline (or `None` iff `runnable` is `False`), the root `task.settings` is the
one authoritative copy (each `steps[i].task.settings` is omitted and
re-injected on load), and `steps` is omitted by `--describe` and is the
dependency-ordered `Step` list by `--dry-run`. `run <path>` replays the same
envelope through `PipelineFactory` (ADR-0018).
_Avoid_: plan, spec, dump, superset

**Per-task Step**:
A `Step` generated per split task, producing one checkpointed artifact per split,
so each split is registered, checkpointed, and resumed individually. The step set
is finalized at construction by `Pipeline.prepare()`, from the inputs, before
discovery, so the direct and serialized scenarios agree (ADR-0017). It is not the
established in-body per-split fan-out: `map_datasets`/`union_tables`/
`reduce_datasets` iterate the splits *inside* one step's body (ADR-0015), whereas
a per-task Step lifts one split into its own `Step` in the DAG.
_Avoid_: split step, shard step, per-chunk task

**InputStep**:
The endpoint a `RunInput` resolves to. It uses the result of the step named by
its `source` when that step ran; otherwise it reads that step's declared
artifact, else loads its bound value with its own `run` loader (the identity
loader by default). A `source` that failed refuses the input, so the failure
propagates.
_Avoid_: binding, parameter

**Resource**:
A `(name, path, reference genome, loader)` value a Pipeline declares in its
`resources` property; the registry resolves it to a load Step that reads its
declared path in place with the resource's own loader (plus the registry's
modification Steps when it must be adapted). A resource whose path is already a
sibling Step's declared Artifact is *derived* and resolves to that producing
Step. Resources are de-duplicated by `(path, resource reference genome)`; the
run's reference genome is a property of the run (one registry per run) rather
than a key. A resource carries **no modification**: the **registry** owns
modifications (e.g. liftover) and applies them to make a resource conform to the
run's reference genome, and only then is the output a checkpoint. Its loader is
never serialized.
_Avoid_: input, dependency, asset

**Run report**:
A line-oriented description of a run — a plan before the walk, one line per step
lifecycle change, and a summary at the end — rendered purely from the `Step`s
the runner walks and their status-model records (`StepResult`); it holds no
state of its own (ADR-0014).
_Avoid_: progress bar, live display, dashboard

**Stage report**:
A per-stage marimo notebook (`playbooks/report/NN-<stage>.py`) that opens a
pipeline stage's primary artifact and describes it — its scale, shape and
outliers — through the collectors and plot builders of
`gnomad_meb.reporting`. Deliberately distinct from the *run report*: a stage
report opens what a stage produced, a run report tracks the run that produced
it. A stage's collectors and builders live together in one
`reporting/<stage>.py` module.
_Avoid_: run report, dashboard, export

**Run observer**:
The single seam a `PipelineRunner` speaks to about a run, in three calls:
`run_started` (the planned order), `step_changed` (one step's status change),
and `run_finished` (the order actually walked). `TerminalObserver` renders the
run report to the process log and is the default; a recording observer for tests
and a deferred JSON mirror are adapters at the same seam.
_Avoid_: logger, handler, callback, listener

### Browser export

**Browser export**:
The stage that turns a MatrixTable's rows into the standardized, sites-only
**Browser site table** and, through genohype, its per-contig parquet. It reads
whatever MatrixTable is bound (the annotation table by default) and never its
entries or columns. Its reshape synthesizes `joint` (the frequency table),
`genome.quality_metrics` (histograms plus the `site_quality_metrics` array) and
`coverage` from required source columns
(`freq`/`freq_meta`, `grpmax`, `fafmax`, `histograms`, `info`,
`inbreeding_coef`, `filters`, `coverage_stats`/`coverage_stats_meta`); a missing
source is a loud `BrowserSchemaError`, not a typed null. Its export step set is
one checkpointed step per reference contig, generated in `prepare()`.
_Avoid_: browser dump, web export

**Browser site table**:
The sites-only hail table in gnomad-browser-lite's variant schema — `locus`/
`alleles` keyed, with `variant_id`, a row-level `transcript_consequences`, the
frequency table under `joint` (the population breakdown in
`joint.freq.all.ancestry_groups`), a `genome.freq.all` callstats summary, the
`genome.quality_metrics` histograms and `site_quality_metrics`, the
`coverage` struct as `{genome: …}`, and
the optional detail fields — that the browser's `hail` backend reads directly and
the parquet is exported from. A checkpoint inside the pipeline.
_Avoid_: browser table, sites table

**Browser parquet dataset**:
The browser export's main outcome: the directory
`{base}/variants.parquet/` holding one `locus.contig=<contig>.parquet` per
reference contig (zero-row files included), each carrying a materialized `xpos`
and a per-contig provenance JSON. The directory is named `variants.parquet` so
the browser's `duckdb` backend (`<data_dir>/variants.parquet`) reads it directly;
completeness is per-contig checkpoint presence, and there is no `_SUCCESS`.
_Avoid_: browser parquet, export parquet

**`xpos`**:
The scalar global coordinate `contig_number * 1_000_000_000 + position` (Int64;
X=23, Y=24, M/MT=25; `chr` stripped) materialized on every browser-export row so
a parquet/duckdb range query prunes on one integer. Computed by the export, not
by genohype's sequential path.
_Avoid_: global position, packed position

### Configuration

**Configuration**:
The human-authored inputs that determine a run: the TOML config file(s),
`GMEB_`-prefixed environment variables, and command-line flags.
_Avoid_: config, settings, parameters, options

**Settings**:
The resolved, validated, serializable result of merging the Configuration
layers, read by the pipeline during a run.
_Avoid_: config object, resolved config, options

**Parameters**:
The resolved configuration values and cutoffs a run was built with; the
concrete carrier is Settings. Use it when recording what a run was configured
with.
_Avoid_: options, arguments, flags

**Task**:
A single CLI invocation — a command together with its resolved Settings. Its
serialized form is the task description, and it is embedded by exactly one Step.
_Avoid_: run, job, invocation
