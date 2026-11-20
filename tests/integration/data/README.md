# Integration-test fixture data

This directory holds the **reviewable fixture oracle** for the integration
tests: the small, hand-maintained files that describe the fixture dataset and
the values the pipeline is expected to produce from it. They are tracked in
git so changes to the expected results are reviewed deliberately.

Only these oracle files live here. The GVCF data itself is large and is *not*
tracked; it is provisioned separately under the repository's `data/` root
(gitignored, intended to move to git LFS later). See
`docs/adr/0002-integration-tests-run-in-a-minimal-container.md` and the
integration-test spec for the surrounding design.

## Layout

```
tests/integration/data/
├── README.md                  this file
└── <reference>/               one directory per reference genome, e.g. grch38/
    ├── samples.tsv            pedigree (input metadata)
    ├── sample_qc.tsv          Sample QC expected values, keyed by sample
    ├── vqsr.tsv               VQSR expected values, keyed by site
    └── annotation.tsv         annotation expected values, keyed by site
```

The GVCF dataset for a reference genome is discovered by the harness from the
`data/` root — one directory per reference genome, e.g. `data/grch38/`, holding
one subdirectory per sample with a `<sample>.gvcf.gz` (+ `.tbi`) inside. The
`<reference>` directory name here mirrors those dataset directory names so the
oracle and the GVCFs for a reference stay paired. Only GRCh38 has a runnable
dataset today; the CHM13 site list is prepared (see Provenance) but CHM13 is
recognised and skipped until the pipeline supports it (see the spec's Out of
Scope).

## Reference-root inputs

The rendered config also points at the **reference-resource root**
(`GMEB_TEST_REF_ROOT`, `$ref_root`), provisioned separately from the tracked
fixtures above. Alongside the reference bundle, the annotation tables and the
VEP cache, the Filter stage's fixed `annotate_constraints_table` step reads the
gnomAD LoF constraint metrics under:

```
$ref_root/annotation/hail/gnomad.v4.1.constraint_metrics.ht
```

`harness.validate_config_resources` treats that table as an input, so a missing
constraint table is a hard error naming the path — not a silently skipped Filter
step.

## Provenance

The fixture dataset is four families (14 samples total), spanning the autosomes
and the sex chromosomes so that every pipeline step has enough data to run
correctly (sex inference needs the sex chromosomes, relatedness needs the
families, and so on). The pedigree in `samples.tsv` describes those families.

Each sample's GVCF is **reduced to a common set of sites**, keeping the fixture
small enough to run the pipeline cheaply while still exercising variant QC and
annotation on real records. The retained intervals are defined by
`data/bed/gvcf_sites.grch38.bed` for GRCh38 and `data/bed/gvcf_sites.chm13.bed`
for CHM13 (lifted over from the GRCh38 intervals with
`grch38-chm13v2.chain`); they are ~200 bp windows spanning every chromosome
(chr1–chr22, chrX, chrY). These site lists live under the `data/` root — they
are dataset provisioning, not part of the tracked oracle.

The GVCFs are provided separately and are the input to `vds create`. The
tracked `samples.tsv` is the authoritative pedigree for the tests; the copy of
`samples.tsv` that may sit next to the GVCFs under `data/` is convenience only.

## The Expected values

The expected values are split by what they verify and how they are keyed: one
file per stage, loaded and compared by `tests/integration/oracle.py`. The former
single-file, record-type-tagged `expected.tsv` is retired. A blank cell means
"not filled in yet": it is simply not compared until it is filled from a
known-good run, so a blank is never silently compared as if it held a real
expectation. Only a column name absent from the oracle is a hard error.

Site-keyed files identify a site with four explicit columns — `chrom`, `pos`,
`ref`, `alt` — and the oracle builds the `chrom:pos:ref:alt` key from them.

## `samples.tsv` — pedigree

Tab-separated, one row per sample, no header comment. Columns:

| column       | meaning                                          |
| ------------ | ------------------------------------------------ |
| `family_id`  | family the sample belongs to                     |
| `sample_id`  | sample identifier (matches the GVCF sample name) |
| `father_id`  | father's `sample_id`, or `0` for founders        |
| `mother_id`  | mother's `sample_id`, or `0` for founders        |
| `sex`        | recorded sex, `M` or `F`                         |
| `is_affected`| affected status, `0`/`1`                         |

## `sample_qc.tsv` — Sample QC expected values

Keyed by `sample_id`. One row per sample; the expectation columns are filled in
as the Sample QC step is tested (intentionally blank until then):

| column               | meaning                                     |
| -------------------- | ------------------------------------------- |
| `inferred_sex`       | inferred sex, in the pedigree `M`/`F` vocabulary (the pipeline's inferred karyotype is mapped onto `M`/`F` by presence of a Y) |
| `sex_karyotype`      | the pipeline's raw inferred karyotype (`sample_qc.ploidy.sex_karyotype`): `XX`/`XY` for the euploid samples and `ambiguous` for the aneuploid, which has no confident Y call |
| `inferred_family_id` | family recovered from relatedness; the test asserts this induces the same *partition* as the pedigree families, so the label itself may be any consistent family id (the pipeline's own family id is not named) |
| `ancestry`           | assigned genetic ancestry group (`sample_qc.ancestry.gen_anc`) |
| `hard_filter`        | hard-filter outcome: `pass` when no filter fired, otherwise the active filter names sorted and joined with `;` (for example `low_coverage;relatedness`) |

The Sample QC values are read back from the produced `{project}/sample_qc`
hail table (see `tests/integration/test_gnomad_workflow.py`); the inferred family
structure is read back from the relatedness matrix
`{project}/intermediate/sample_qc/king.mt`. A related pair is one with
`phi >= 0.1` (`RELATEDNESS_KINSHIP_THRESHOLD`); samples in no such pair are
their own singleton family. The recovered partition is asserted against both the
pedigree's families and the `inferred_family_id` column below.

To fill (or refresh) this file from a run, use
`tests/integration/regen_oracle.py` inside the container:

```
python tests/integration/regen_oracle.py --reference GRCh38 \
    --sample-qc <run>/…/sample_qc.ht \
    --relatedness <run>/…/intermediate/sample_qc/king.mt
```

The `--relatedness` matrix fills `inferred_family_id` (union-find over the pairs
at `--relatedness-threshold`, default `0.1`, labelled by each component's
smallest member); without it that column is left blank.

## `vqsr.tsv` — VQSR expected values

Header-only until a known-good run exists. Keyed by site
(`chrom`,`pos`,`ref`,`alt`). One row per asserted site, carrying the Variant QC
values asserted by `tests/integration/test_variant_qc_load.py`:

| column              | meaning                                     |
| ------------------- | ------------------------------------------- |
| `a_index`           | split-allele index of the site in the imported VQSR table |
| `was_split`         | whether the site was split from a multiallelic site (`True`/`False`) |
| `vqslod`            | allele-specific recalibration score (`variant_qc.info.AS_VQSLOD`), compared with a small tolerance |
| `variant_qc_filter` | VQSR FILTER outcome: `pass` when no filter fired, otherwise the active filter names sorted and joined with `;` |

The Variant QC values are read back from the produced `{project}/variant_qc`
hail table, which is a terminal `merge` of one import per VDS split, itself
imported from that split's recalibrated VQSR VCF produced by the `vqsr pipeline`
step. Rows are added only once a known-good run exists; until then the blank
cells are simply not compared.

To fill (or refresh) this file from a run, use
`tests/integration/regen_oracle.py` inside the container:

```
python tests/integration/regen_oracle.py --reference GRCh38 \
    --variant-qc <run>/…/variant_qc.ht \
    --variant-subset-frac 0.05
```

The table can be large, so the written rows are a configurable subset:
`--variant-subset-n` (a count) or `--variant-subset-frac` (a fraction), never
both. Selection is reproducible given `--variant-subset-seed` (default `0`): it
takes one variant per chromosome first, so every contig present is represented,
then fills the remainder with a seeded shuffle. Without either flag every site
is written.

> **Scatter-gather VQSR (ticket 03).** The VQSR workflow trains the SNP and INDEL
> models once globally, then scatters the apply per VDS task
> (`VariantRecalibrator --input-model --output-tranches-for-scatter`) and
> collates the per-task tranches with `GatherTranches` before `ApplyVQSR`. This
> replaces the old single global `COMBINED` INDEL recalibration and fixes the
> INDEL cross-wiring (the INDEL path no longer reads the SNP tranches). It
> changes INDEL — and, through the scattered apply, SNP — `AS_VQSLOD` / filters
> relative to the old workflow, so the first oracle filled in the container
> **must be regenerated after that fix is present**.

## `annotation.tsv` — annotation expected values

Header-only until a known-good run exists. Keyed by site
(`chrom`,`pos`,`ref`,`alt`). One row per asserted site, carrying the values
asserted by `tests/integration/test_gnomad_workflow.py`:

| column                   | meaning                                     |
| ------------------------ | ------------------------------------------- |
| `annotation_filter`      | row `filters` of the final callset: `pass` when no filter fired, otherwise the active filter names sorted and joined with `;` |
| `region_lcr`             | whether the site falls in a low-complexity region (`region_flags.lcr`), `True`/`False` |
| `region_segdup`          | whether the site falls in a segmental duplication (`region_flags.segdup`), `True`/`False` |
| `region_non_par`         | whether the site falls in a non-PAR region of a sex chromosome (`region_flags.non_par`), `True`/`False` |
| `most_severe_consequence`| VEP most-severe consequence of the site (`vep.most_severe_consequence`) |
| `gnomad_af`              | gnomAD population allele frequency of the site, adj group (`gnomad_freq[...].AF`), compared with a small tolerance |
| `grpmax_af`              | highest genetic-ancestry-group allele frequency, adj group (`grpmax.AF`), compared with a small tolerance |
| `fafmax_faf95_max`       | maximum filtering allele frequency at the 95% threshold (`fafmax.faf95_max`), compared with a small tolerance |
| `cadd`                   | CADD PHRED score of the site (`in_silico_predictors.cadd.phred`), compared with a small tolerance |

The Annotation values are read back from the produced
`{project}/annotation` matrix table (see `tests/integration/test_gnomad_workflow.py`),
the final callset produced by `annotation run`. As with the Variant QC columns,
rows are filled only once a known-good run exists and the blank cells are simply
not compared until then.

To fill (or refresh) this file from a run, use
`tests/integration/regen_oracle.py` inside the container:

```
python tests/integration/regen_oracle.py --reference GRCh38 \
    --annotation <run>/…/annotation.mt \
    --variant-subset-frac 0.05
```

The written rows are the same configurable subset as `vqsr.tsv`
(`--variant-subset-n` or `--variant-subset-frac`, reproducible via
`--variant-subset-seed`).

## Extending the fixture

To add a reference genome, add `data/<new-reference>/` with the GVCFs and the
matching `tests/integration/data/<new-reference>/` oracle files
(`samples.tsv`, `sample_qc.tsv`, `vqsr.tsv`, `annotation.tsv`). If the new
reference needs its own reduced site set, add `data/bed/gvcf_sites.<reference>.bed`
(GRCh38 is `gvcf_sites.grch38.bed`; CHM13's is lifted over from GRCh38). Tests
are parameterized by reference genome, so a new dataset
directory adds coverage without changing the tests — provided the pipeline
supports the reference (see the spec's Out of Scope note on CHM13).

To add an asserted value for an existing step, add a column to that step's file;
later steps add their own columns to their own files. The oracle loads every
non-key column, so a new column is picked up without a code change.
