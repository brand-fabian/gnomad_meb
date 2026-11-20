"""Annotate the final callset with QC information.

This pipeline annotates the callset with additional information
from gnomAD, clinvar and potentially other sources, as well as
the QC statistics computed in `variant_qc` and `sample_qc`
respectively.

The output is a matrix table whose rows have a format akin to
the gnomad hail table, i.e. featuring:
    * freq: stratified allele frequencies
    * grpmax
    * faf
    * fafmax
    * (a_index, was_split): result from split_multi_hts
    * info: info fields from variant qc
    * vqsr_results: VQSR results from variant qc
    * region_flags: annotation of segmental duplication and low complexity regions
    * allele_info: gnomAD allele_info
    * vep: vep annotation from gnomAD ht
    * in_silico_predictors: cadd and other scores, from gnomAD ht
    * gnomad_freq: gnomad population allele frequency
While the column fields contain the results from the sample QC
pipeline, i.e.:
    * sex_karyotype
    * genetic ancestry
    * sample filters
Entry fields are kept as-is during the pipeline, but they are
filtered, removing low-quality variants and samples from the pipeline
as indicated by the hard-filter cutoffs.

The stage is a :class:`~gnomad_meb.pipeline.Pipeline` subclass: its work is
declared as ``@checkpoint``-decorated methods whose parameter *defaults* are the
wire. ``import_clinvar_ht`` / ``import_cadd_ht`` / ``import_alphamissense_ht``
import the annotation sources. :meth:`AnnotationPipeline.prepare` then reads the
run's VDS task list and generates one dense step and seven annotate substeps per
task, plus a terminal :meth:`AnnotationPipeline.merge`;
``variant_dataset_to_matrix_table`` densifies one split of the VDS (a
:class:`~gnomad_meb.pipeline.RunInput`) and the substeps annotate that split's
rows in order: ``annotate_resources`` (the source-keyed joins plus sample QC),
``annotate_variant_qc`` (the Variant QC joins, isolated so their lineage is the
only lineage in that write's closure), ``annotate_vep`` (cluster VEP; its own
checkpointed substep, idempotent for the same databases and VEP config),
``annotate_postprocess_gnomad_and_insilico_predictors`` (``process_consequences``
on the merged VEP field plus the cadd/alphamissense in-silico finalisation), and
``annotate_regions_and_freq`` (the regions and frequencies aggregation), then
``compute_coverage_stats`` (per-locus coverage over the dense rows minus the
telomere/centromere regions) and ``annotate_coverage_stats`` (the coverage join
onto the per-task matrix). The
internal annotators are plain helpers called by the substep bodies in the order
the pre-port ``build`` used; the in-silico finalisation runs after VEP inside
``annotate_postprocess_gnomad_and_insilico_predictors`` because VEP replaces that
field. See
``.scratch/pipeline-resources/design.md`` sections 4-6.
"""

from __future__ import annotations

import secrets
from collections.abc import Callable
from functools import cached_property, partial
from typing import TYPE_CHECKING

from gnomad_meb.cmd.task import Task
from gnomad_meb.constants import DEFAULT_REFERENCE_GENOME
from gnomad_meb.lib.checkpoint import CHECKPOINT_TYPE, Artifact, checkpoint
from gnomad_meb.pipeline import (
    FromStep,
    InputStep,
    Pipeline,
    Resource,
    ResourceRef,
    RunInput,
)

if TYPE_CHECKING:
    from gnomad_meb.hail import hail as hl


#: Intermediate artifact prefix for the per-task Annotation steps.
_INTERMEDIATE_PREFIX = "{project}/intermediate/annotation"


class AnnotationPipeline(Pipeline):
    """Declare and run the annotation stage.

    The VDS is a :class:`~gnomad_meb.pipeline.RunInput`; its task list finalizes
    the per-task steps in :meth:`prepare`, with ``merge`` the terminal step.
    Resources are declared as :class:`~gnomad_meb.pipeline.Resource`
    objects and resolved by the registry in place; ``sample_qc_ht`` /
    ``variant_qc_ht`` are derived resources (the producing stage's declared
    artifact) when the config does not point at an external table.
    """

    @classmethod
    def name(cls) -> str:
        return "annotation"

    @cached_property
    def resources(self) -> tuple[Resource, ...]:
        """The nine datasets the stage declares (design §6).

        ``gnomad_ht``, the interval tables and the two QC tables are read with
        ``hl.read_table``; ``sample_qc_ht`` / ``variant_qc_ht`` are the derived
        ``{project}/sample_qc`` / ``{project}/variant_qc`` artifacts when the
        config points at no external table, so the registry orders each ``annotate``
        step after the producing QC steps. The annotation sources carry their own
        readers: ``read_clinvar_vcf`` (path-only), the combined CADD reader built
        over both CADD files, and ``get_alphamissense_ht``.
        """
        from gnomad_meb.hail import hail as hl
        from gnomad_meb.lib.annotation.resources import (
            get_alphamissense_ht,
            get_cadd_ht_loader,
            read_clinvar_vcf,
        )

        cfg = self.settings
        annotation = cfg.annotation
        assert annotation is not None, "missing annotation configuration"
        annotation_resources = annotation.resources
        assert annotation_resources is not None, (
            "missing annotation resources configuration"
        )
        vep = annotation.vep
        assert vep is not None, "missing annotation vep configuration"

        declared: list[Resource] = [
            Resource(
                name="gnomad_ht",
                path=str(cfg.resources.gnomad_v4),
                reference_genome=DEFAULT_REFERENCE_GENOME,
                loader=hl.read_table,
            ),
            Resource(
                name="sample_qc_ht",
                path=self._quality_control_path(
                    annotation_resources.sample_qc, "{project}/sample_qc"
                ),
                reference_genome=DEFAULT_REFERENCE_GENOME,
                loader=hl.read_table,
            ),
            Resource(
                name="variant_qc_ht",
                path=self._quality_control_path(
                    annotation_resources.variant_qc, "{project}/variant_qc"
                ),
                reference_genome=DEFAULT_REFERENCE_GENOME,
                loader=hl.read_table,
            ),
            Resource(
                name="lcr_intervals_ht",
                path=str(cfg.resources.lcr_intervals),
                reference_genome=DEFAULT_REFERENCE_GENOME,
                loader=hl.read_table,
            ),
            Resource(
                name="seg_dup_intervals_ht",
                path=str(cfg.resources.seg_dup_intervals),
                reference_genome=DEFAULT_REFERENCE_GENOME,
                loader=hl.read_table,
            ),
            Resource(
                name="telomeres_and_centromeres",
                path=str(cfg.resources.telomeres_and_centromeres),
                reference_genome=DEFAULT_REFERENCE_GENOME,
                loader=hl.read_table,
            ),
            Resource(
                name="clinvar_vcf",
                path=str(annotation_resources.clinvar_vcf),
                reference_genome=DEFAULT_REFERENCE_GENOME,
                loader=read_clinvar_vcf,
            ),
        ]
        if vep.cadd is not None and vep.cadd_indel is not None:
            declared.append(
                Resource(
                    name="cadd",
                    path=str(vep.cadd),
                    reference_genome=DEFAULT_REFERENCE_GENOME,
                    loader=get_cadd_ht_loader(str(vep.cadd), str(vep.cadd_indel)),
                )
            )
            declared.append(
                Resource(
                    name="cadd_indel",
                    path=str(vep.cadd_indel),
                    reference_genome=DEFAULT_REFERENCE_GENOME,
                    loader=get_cadd_ht_loader(str(vep.cadd), str(vep.cadd_indel)),
                )
            )
        if vep.alpha_missense is not None:
            declared.append(
                Resource(
                    name="alpha_missense",
                    path=str(vep.alpha_missense),
                    reference_genome=DEFAULT_REFERENCE_GENOME,
                    loader=get_alphamissense_ht,
                )
            )
        return tuple(declared)

    def _quality_control_path(self, configured: str | None, prefix: str) -> str:
        """The QC table path: the configured external table, else the artifact.

        The derived default is the producing stage's declared artifact
        (``{project}/sample_qc`` / ``{project}/variant_qc``), so the registry's
        path index orders each ``annotate`` step after that producer.
        """
        if configured is not None:
            return configured
        return Artifact(prefix=prefix, kind=CHECKPOINT_TYPE.HAIL_TABLE).path(
            self.settings
        )

    def get_input(self) -> tuple[InputStep, ...]:
        """The run's VDS: the VDS-build step's result, else the bound handle."""
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
        )

    def prepare(self) -> None:
        """Finalize the per-task step set from the run's VDS handle.

        One dense step and one annotate step are generated per task, plus a
        single terminal merge; the whole-callset dense/annotate steps disappear
        (their bodies are now the plain methods the generated partials wrap). The
        task list is read from the handle's metadata only (``VariantDataset.open``
        does no data read), so ``--describe`` and reconstruction rebuild the same
        step set from the same input. A missing handle fails loudly, so the two
        invocation scenarios can never disagree about the steps.
        """
        from gnomad_meb.lib.vds import VariantDataset

        super().prepare()

        annotation = self.settings.annotation
        assert annotation is not None, "missing annotation configuration"
        annotate_info_fields = annotation.annotate_info_fields

        vds_path = getattr(self.command, "vds", None)
        if vds_path is None:
            raise ValueError(
                "annotation pipeline needs the run's VDS handle to finalize "
                "its per-task steps: command = "
                f"{type(self.command).__name__}"
            )
        tasks = VariantDataset.open(vds_path).params.tasks

        for task in tasks:
            if annotate_info_fields:
                self._methods.append(
                    (
                        f"compute_default_info_table.{task.name}",
                        checkpoint(
                            f"{_INTERMEDIATE_PREFIX}/task-{task.name}/default_info",
                            CHECKPOINT_TYPE.HAIL_TABLE,
                        )(
                            partial(
                                self.compute_default_info_table,
                                task=task.name,
                            )
                        ),
                        task.name,
                    )
                )

            self._methods.append(
                (
                    f"variant_dataset_to_matrix_table.{task.name}",
                    checkpoint(
                        f"{_INTERMEDIATE_PREFIX}/task-{task.name}/dense_callset",
                        CHECKPOINT_TYPE.MATRIX_TABLE,
                    )(
                        partial(
                            self.variant_dataset_to_matrix_table,
                            info_ht=(
                                FromStep(f"compute_default_info_table.{task.name}")
                                if annotate_info_fields
                                else None
                            ),
                            task=task.name,
                        )
                    ),
                    task.name,
                )
            )

            self._methods.extend(self._annotation_steps(task.name))

        self._methods.append(
            (
                "merge",
                checkpoint("{project}/annotation", CHECKPOINT_TYPE.MATRIX_TABLE)(
                    partial(
                        self.merge,
                        **{
                            task.name: FromStep(
                                f"annotate_coverage_stats.{task.name}"
                            )
                            for task in tasks
                        },
                    )
                ),
                None,
            )
        )

    def _annotation_steps(self, task: str) -> list[tuple[str, Callable, str]]:
        """The seven ordered per-task Annotation substeps.

        Each is an ordinary per-task ``Step`` (ADR-0017) with its own
        ``@checkpoint`` ``Artifact`` (ADR-0011): the source-keyed joins and
        sample QC, then the Variant QC joins, then VEP, then
        ``process_consequences`` on the merged VEP field plus the
        in-silico-predictor finalisation, then the regions-and-frequencies
        aggregation, then the per-locus coverage table, then the coverage join.
        The Variant QC block is its own substep so its heavy,
        schema-mutating lineage is the only lineage in that write's closure;
        the boundary returns a disk read (``checkpoint.py``) so no upstream
        lineage enters the next write. The chain stays linear -- each body
        consumes the matrix the previous substep produced -- so the merged
        schema is unchanged. The ``partial`` keyword is the per-task edge;
        ``_wire_step`` reads it from the partial's keywords and applies them
        after the body's signature defaults, so the per-task ``mt`` overrides
        the whole-stage default and no whole-stage name is ever wired.
        """
        return [
            (
                f"annotate_resources.{task}",
                checkpoint(
                    f"{_INTERMEDIATE_PREFIX}/task-{task}/annotate_resources",
                    CHECKPOINT_TYPE.MATRIX_TABLE,
                )(
                    partial(
                        self.annotate_resources,
                        mt=FromStep(f"variant_dataset_to_matrix_table.{task}"),
                    )
                ),
                task,
            ),
            (
                f"annotate_variant_qc.{task}",
                checkpoint(
                    f"{_INTERMEDIATE_PREFIX}/task-{task}/annotate_variant_qc",
                    CHECKPOINT_TYPE.MATRIX_TABLE,
                )(
                    partial(
                        self.annotate_variant_qc,
                        mt=FromStep(f"annotate_resources.{task}"),
                    )
                ),
                task,
            ),
            (
                f"annotate_vep.{task}",
                checkpoint(
                    f"{_INTERMEDIATE_PREFIX}/task-{task}/annotate_vep",
                    CHECKPOINT_TYPE.MATRIX_TABLE,
                )(
                    partial(
                        self.annotate_vep,
                        mt=FromStep(f"annotate_variant_qc.{task}"),
                    )
                ),
                task,
            ),
            (
                f"annotate_postprocess_gnomad_and_insilico_predictors.{task}",
                checkpoint(
                    f"{_INTERMEDIATE_PREFIX}/task-{task}/annotate_postprocess_gnomad_and_insilico_predictors",
                    CHECKPOINT_TYPE.MATRIX_TABLE,
                )(
                    partial(
                        self.annotate_postprocess_gnomad_and_insilico_predictors,
                        mt=FromStep(f"annotate_vep.{task}"),
                    )
                ),
                task,
            ),
            (
                f"annotate_regions_and_freq.{task}",
                checkpoint(
                    f"{_INTERMEDIATE_PREFIX}/task-{task}/annotate_regions_and_freq",
                    CHECKPOINT_TYPE.MATRIX_TABLE,
                )(
                    partial(
                        self.annotate_regions_and_freq,
                        mt=FromStep(
                            f"annotate_postprocess_gnomad_and_insilico_predictors.{task}"
                        ),
                    )
                ),
                task,
            ),
            (
                f"compute_coverage_stats.{task}",
                checkpoint(
                    f"{_INTERMEDIATE_PREFIX}/task-{task}/coverage",
                    CHECKPOINT_TYPE.HAIL_TABLE,
                )(
                    partial(
                        self.compute_coverage_stats,
                        mt=FromStep(
                            f"annotate_resources.{task}"
                        ),
                    )
                ),
                task,
            ),
            (
                f"annotate_coverage_stats.{task}",
                checkpoint(
                    f"{_INTERMEDIATE_PREFIX}/task-{task}/annotate_coverage_stats",
                    CHECKPOINT_TYPE.MATRIX_TABLE,
                )(
                    partial(
                        self.annotate_coverage_stats,
                        mt=FromStep(
                            f"annotate_regions_and_freq.{task}"
                        ),
                        coverage_ht=FromStep(f"compute_coverage_stats.{task}"),
                    )
                ),
                task,
            ),
        ]

    # --- work steps ----------------------------------------------------------

    @checkpoint("{project}/intermediate/annotation/clinvar", CHECKPOINT_TYPE.HAIL_TABLE)
    def import_clinvar_ht(
        self,
        clinvar_vcf=ResourceRef("clinvar_vcf"),  # noqa: B008
    ) -> hl.Table:
        """Import a clinvar VCF and convert it to a hail table.

        The source travels as the ``clinvar_vcf`` resource; its (path-only)
        loader hands back the VCF path, which is still needed for the release
        date.

        Note
        ----
        This function has been lifted from the gnomAD browser code:
        https://github.com/broadinstitute/gnomad-browser/blob/2f90e9f24f0ee0e05c91cf495e759018ccd886df/data-pipeline/src/data_pipeline/datasets/clinvar.py#L49-L65
        """
        from gnomad_meb.hail import get_default_partitions
        from gnomad_meb.hail import hail as hl
        from gnomad_meb.lib.annotation.utils import get_clinvar_filedate

        vcf = str(clinvar_vcf)

        recode = {f"{i}": f"chr{i}" for i in [*list(range(1, 23)), "X", "Y"]}
        recode["MT"] = "chrM"
        clinvar_ht = hl.import_vcf(
            vcf,
            call_fields=["GT"],
            force_bgz=True,
            contig_recoding=recode,
            min_partitions=get_default_partitions(),
            drop_samples=True,
            skip_invalid_loci=True,
            reference_genome=DEFAULT_REFERENCE_GENOME,
        ).rows()

        clinvar_ht = clinvar_ht.annotate_globals(
            clinvar_release=get_clinvar_filedate(vcf)
        )
        clinvar_ht = clinvar_ht.filter(hl.len(clinvar_ht.alleles) == 2)
        return clinvar_ht

    @checkpoint("{project}/intermediate/annotation/cadd", CHECKPOINT_TYPE.HAIL_TABLE)
    def import_cadd_ht(
        self,
        cadd=ResourceRef("cadd"),  # noqa: B008
    ) -> hl.Table:
        """Import the CADD SNV and indel annotation files into one table.

        The ``cadd`` resource's loader reads both CADD files (the resource's
        path is the SNV file) and returns the combined table.
        """
        return cadd

    @checkpoint(
        "{project}/intermediate/annotation/alphamissense",
        CHECKPOINT_TYPE.HAIL_TABLE,
    )
    def import_alphamissense_ht(
        self,
        alpha_missense=ResourceRef("alpha_missense"),  # noqa: B008
    ) -> hl.Table:
        """Import the AlphaMissense scores into a hail table.

        The ``alpha_missense`` resource's loader reads the source in place.
        """
        return alpha_missense

    def compute_default_info_table(
        self,
        vds=RunInput("vds"),  # noqa: B008
        task: str | None = None,
    ) -> hl.Table:
        """Build the default info table for the densified callset's info fields."""
        from gnomad_meb.hail import hail as hl
        from gnomad_meb.lib.variant_qc import (
            build_info_table,
            select_info_table,
            split_info,
        )

        vds = vds.select_tasks(lambda t: t.name == task)
        mt = vds.to_hail(fix_lpl=True).variant_data

        info_ht = build_info_table(mt, retain_cdfs=False)
        info_ht = select_info_table(info_ht, "AS_info")
        info_ht = info_ht.annotate(
            info=info_ht.info.annotate(
                AS_SB_TABLE=info_ht.info.AS_SB_TABLE.split("\\|").map(
                    lambda x: hl.or_missing(
                        x != "",
                        x.split(",").map(lambda y: hl.or_missing(y != "", hl.int(y))),
                    )
                ),
            )
        )
        info_ht = split_info(info_ht)
        return info_ht

    def variant_dataset_to_matrix_table(
        self,
        vds=RunInput("vds"),  # noqa: B008
        info_ht: FromStep | None = FromStep(  # noqa: B008
            "compute_default_info_table"
        ),
        task: str | None = None,
    ) -> hl.MatrixTable:
        """Convert one split of the VariantDataset to a dense MatrixTable.

        The handle is sliced to the bound ``task`` (:meth:`VariantDataset.select_tasks`)
        before densifying, so the existing ``to_hail``/``to_dense_mt`` call is already
        split-local. The generated per-task step binds ``task``; this method is
        plain, so it is never registered as a whole-callset step.

        Arguments
        ---------
        vds : VariantDataset
            The run's VariantDataset handle.
        task : str
            The split task name to densify.
        """
        from gnomad_meb.hail import get_default_partitions
        from gnomad_meb.hail import hail as hl
        from gnomad_meb.lib.annotation.utils import cohort_expr

        annotation = self.settings.annotation
        assert annotation is not None, "missing annotation configuration"
        cohort_table = annotation.cohort_table
        cohort = annotation.cohort
        annotate_info_fields = annotation.annotate_info_fields

        vds = vds.select_tasks(lambda t: t.name == task)
        dense_mt = vds.to_dense_mt(fix_lpl=True)

        full_mt = dense_mt.annotate_entries(
            GT=hl.vds.lgt_to_gt(dense_mt.LGT, dense_mt.LA),
            AD=hl.vds.local_to_global(
                dense_mt.LAD,
                dense_mt.LA,
                n_alleles=hl.len(dense_mt.alleles),
                fill_value=0,
                number="R",
            ),
            PGT=hl.vds.lgt_to_gt(dense_mt.LPGT, dense_mt.LA),
            PL=hl.vds.local_to_global(
                dense_mt.LPL,
                dense_mt.LA,
                n_alleles=hl.len(dense_mt.alleles),
                fill_value=1000,
                number="G",
            ),
        )

        bi_mt = full_mt.filter_rows(hl.len(full_mt.alleles) == 2)
        bi_mt = bi_mt.annotate_rows(a_index=1, was_split=False)
        multi_mt = full_mt.filter_rows(hl.len(full_mt.alleles) > 2)
        split_mt = hl.split_multi_hts(multi_mt)
        full_mt = split_mt.union_rows(bi_mt)
        full_mt = full_mt.repartition(get_default_partitions(), shuffle=True)
        full_mt = full_mt.checkpoint(
            hl.utils.new_temp_file(
                f"full-dense-t{task if task is not None else '0'}."
                f"{secrets.token_hex(4)}",
                extension="mt",
            ),
            overwrite=True,
        )

        if cohort is not None and "cohort" not in full_mt.col:
            full_mt = full_mt.annotate_cols(
                cohort=cohort_expr(full_mt, cohort_table, cohort)
            )

        # Annotate the default info fields for all variants:
        if annotate_info_fields:
            assert info_ht is not None, "info ht missing for info-field annotation"
            full_mt = full_mt.annotate_rows(
                info=info_ht[full_mt.row_key].info,
                allele_info=info_ht[full_mt.row_key].allele_info,
            )

        full_mt = full_mt.select_entries(
            *(set(full_mt.entry.keys()) - {"LGT", "LAD", "LPGT", "LPL", "LA"})
        )
        return full_mt

    def annotate_resources(
        self,
        mt=FromStep("variant_dataset_to_matrix_table"),  # noqa: B008
        clinvar_ht=FromStep("import_clinvar_ht"),  # noqa: B008
        gnomad_ht=ResourceRef("gnomad_ht"),  # noqa: B008
        sample_qc_ht=ResourceRef("sample_qc_ht"),  # noqa: B008
    ) -> hl.MatrixTable:
        """Annotate one split with the source-keyed joins and the sample QC table.

        Plain (no ``@checkpoint``): this body is the first of the five per-task
        substeps, wrapped by ``checkpoint`` in ``_annotation_steps``. The
        partial binds the per-task ``mt`` to the matching dense step and
        overrides the whole-stage default below; the import/resource markers
        stay on the signature, so wiring still finds them. The keyed joins run
        in the order the pre-port ``build`` used; the in-silico finalisation
        stays in :meth:`annotate_postprocess_gnomad_and_insilico_predictors`
        because VEP replaces that field. The Variant QC joins run in their own
        substep (:meth:`annotate_variant_qc`) so their lineage does not enter
        this write's closure.
        """
        mt = self._annotate_metadata(mt)
        mt = self._annotate_gnomad(mt, gnomad_ht)
        mt = self._annotate_clinvar(mt, clinvar_ht)
        mt = self._annotate_sample_qc(mt, sample_qc_ht)
        return mt

    def annotate_variant_qc(
        self,
        mt=FromStep("annotate_resources"),  # noqa: B008
        variant_qc_ht=ResourceRef("variant_qc_ht"),  # noqa: B008
    ) -> hl.MatrixTable:
        """Annotate the Variant QC ``filters`` and per-allele fields for one split.

        Plain (no ``@checkpoint``): this body is the second of the five
        per-task substeps, wrapped by ``checkpoint`` in ``_annotation_steps``,
        so the partial binds the per-task ``mt`` to the preceding
        ``annotate_resources`` substep; the ``variant_qc_ht`` resource marker
        stays on the signature for wiring.

        The row set is the input matrix either way, so this is one conditional
        ``annotate_rows`` over a single keyed lookup rather than the former two
        ``filter_rows`` passes and a ``union_rows``. The externally observable
        contract is preserved exactly: rows present in ``variant_qc_ht`` take
        its ``filters``/``info``/``allele_info``/``a_index``/``was_split``/
        ``vqsr_results``; rows absent get the ``vqc_uninformative`` filter and
        missing values, and the Variant QC provenance is carried as the
        ``variant_qc`` global.
        """
        from gnomad_meb.hail import hail as hl

        hit = variant_qc_ht[mt.row_key]
        in_vqc = hl.is_defined(hit)

        # Align the input ``info`` struct to ``variant_qc_ht.info`` once: build
        # a struct of the Variant QC info dtype, then overlay only the matrix
        # fields that are present with a matching dtype. Known mismatches
        # (``DB`` | ``SB``) and unknown fields stay missing. This is a single
        # struct-dtype alignment rather than a per-field conditional: the base
        # supplies the dtype, the overlay copies the shared fields.
        aligned_info = hl.Struct(
            **{
                key: hl.missing(variant_qc_ht.info[key].dtype)
                for key in variant_qc_ht.info
            }
        ).annotate(
            **{
                key: mt.info[key]
                for key in variant_qc_ht.info
                if key in mt.info
                and mt.info[key].dtype == variant_qc_ht.info[key].dtype
            }
        )
        vqsr_results_dtype = hl.tstruct(
            AS_VQSLOD=hit.info.AS_VQSLOD.dtype,
            AS_culprit=hit.info.AS_culprit.dtype,
            positive_train_site=hit.info.POSITIVE_TRAIN_SITE.dtype,
            negative_train_site=hit.info.NEGATIVE_TRAIN_SITE.dtype,
        )

        mt = mt.annotate_rows(
            filters=hl.if_else(
                in_vqc, hit.filters, hl.literal({"vqc_uninformative"})
            ),
            info=hl.if_else(in_vqc, hit.info, aligned_info),
            allele_info=hl.if_else(
                in_vqc,
                hit.allele_info,
                hl.missing(variant_qc_ht.allele_info.dtype),
            ),
            a_index=hl.if_else(in_vqc, hit.a_index, hl.missing(hl.tint32)),
            was_split=hl.if_else(in_vqc, hit.was_split, hl.missing(hl.tbool)),
            vqsr_results=hl.if_else(
                in_vqc,
                hl.Struct(
                    AS_VQSLOD=hit.info.AS_VQSLOD,
                    AS_culprit=hit.info.AS_culprit,
                    positive_train_site=hit.info.POSITIVE_TRAIN_SITE,
                    negative_train_site=hit.info.NEGATIVE_TRAIN_SITE,
                ),
                hl.missing(vqsr_results_dtype),
            ),
        )
        # Carry the Variant QC provenance (VQSR cutoffs/params) as a global, as
        # ``_annotate_sample_qc`` does for Sample QC, so a released callset
        # records which model produced its ``filters``/``AS_VQSLOD``.
        mt = mt.annotate_globals(variant_qc=variant_qc_ht.index_globals())
        return mt

    def annotate_vep(
        self,
        mt=FromStep("annotate_variant_qc"),  # noqa: B008
    ) -> hl.MatrixTable:
        """Annotate all variants that are not contained in the gnomad table.

        Plain (no ``@checkpoint``): this body is the second per-task substep,
        wrapped by ``checkpoint`` in ``_annotation_steps``, so the partial binds
        the per-task ``mt`` to the preceding ``annotate_variant_qc``
        substep. This function assumes that mt[vep_key] has already been filled
        with gnomAD data and annotates all missing variants using VEP on the
        cluster, replacing the whole ``vep``/``in_silico_predictors`` row fields
        from the VEP table; the postprocess substep runs after this one.
        """
        from gnomad_meb.lib.annotation.vep import vep_or_lookup_vep
        from gnomad_meb.lib.constants import IN_SILICO_PREDICTORS_KEY, VEP_KEY

        annotation = self.settings.annotation
        assert annotation is not None, "missing annotation configuration"

        rows_ht = mt.rows()
        vep_config_path = annotation.vep_config_path
        assert vep_config_path is not None, "missing vep configuration path"
        vep_ht = vep_or_lookup_vep(
            rows_ht,
            vep_config_path,
            vep_key=VEP_KEY,
            include_csq=False,
        )

        mt = mt.annotate_rows(
            **{
                VEP_KEY: vep_ht[mt.row_key][VEP_KEY],
                IN_SILICO_PREDICTORS_KEY: vep_ht[mt.row_key][IN_SILICO_PREDICTORS_KEY],
            }
        )
        mt = mt.annotate_globals(vep_globals=vep_ht.index_globals().vep_globals)
        return mt

    def annotate_postprocess_gnomad_and_insilico_predictors(
        self,
        mt=FromStep("annotate_vep"),  # noqa: B008
        cadd_ht=FromStep("import_cadd_ht"),  # noqa: B008
        alphamissense_ht=FromStep("import_alphamissense_ht"),  # noqa: B008
    ) -> hl.MatrixTable:
        """Postprocess the merged VEP field and finalise ``in_silico_predictors``.

        Plain (no ``@checkpoint``): this body is the third per-task substep,
        wrapped by ``checkpoint`` in ``_annotation_steps``, so the partial binds
        the per-task ``mt`` to the preceding ``annotate_vep`` substep. It runs
        ``process_consequences`` on the merged VEP field and then merges the
        cadd/alphamissense imports into ``in_silico_predictors``; both must run
        after VEP, which reassigns those row fields.
        """
        from gnomad.utils.vep import process_consequences

        from gnomad_meb.lib.constants import IN_SILICO_PREDICTORS_KEY

        mt = process_consequences(
            mt,
            has_polyphen=False,
        )
        mt = self._update_insilico_predictors(
            mt,
            cadd_ht,
            alphamissense_ht,
            in_silico_predictors_key=IN_SILICO_PREDICTORS_KEY,
        )
        return mt

    def annotate_regions_and_freq(
        self,
        mt=FromStep("annotate_postprocess_gnomad_and_insilico_predictors"),  # noqa: B008
        lcr_intervals_ht=ResourceRef("lcr_intervals_ht"),  # noqa: B008
        seg_dup_intervals_ht=ResourceRef("seg_dup_intervals_ht"),  # noqa: B008
    ) -> hl.MatrixTable:
        """Annotate the region flags, frequencies and histograms for one split.

        Plain (no ``@checkpoint``): this body is the fourth per-task substep,
        wrapped by ``checkpoint`` in ``_annotation_steps``, so the partial binds
        the per-task ``mt`` to the preceding
        ``annotate_postprocess_gnomad_and_insilico_predictors`` substep; the
        interval resources stay on the signature for wiring.
        """
        from gnomad.utils.annotations import (
            annotate_adj,
            annotate_freq,
            bi_allelic_site_inbreeding_expr,
            faf_expr,
            gen_anc_faf_max_expr,
            grpmax_expr,
            qual_hist_expr,
            region_flag_expr,
        )

        from gnomad_meb.hail import hail as hl

        sample_qc = self.settings.sample_qc
        assert sample_qc is not None, "missing sample qc configuration"
        annotation = self.settings.annotation
        assert annotation is not None, "missing annotation configuration"

        mt = annotate_adj(
            mt,
            adj_gq=sample_qc.adj_gq,
            adj_dp=sample_qc.adj_dp,
            adj_ab=sample_qc.adj_ab,
            haploid_adj_dp=sample_qc.haploid_adj_dp,
        )
        mt = mt.annotate_rows(
            region_flags=region_flag_expr(
                mt,
                non_par=True,
                prob_regions={
                    "lcr": lcr_intervals_ht,
                    "segdup": seg_dup_intervals_ht,
                },
            )
        )

        if annotation.group_by_cohort:
            additional_strata_expr = [{"cohort": mt.cohort}]

            if "ploidy" in mt.col_value:
                additional_strata_expr.append({
                    "cohort": mt.cohort,
                    "sex": mt.ploidy.sex_karyotype,
                })

            gen_anc_label = "cohort"
        else:
            additional_strata_expr = None
            gen_anc_label = "gen_anc"

        has_gen_anc = (
            "ancestry" in mt.row_value or "ancestry" in mt.col_value
        ) and "gen_anc" in mt.ancestry.keys()
        if has_gen_anc:
            gen_anc_expr = hl.coalesce(mt.ancestry.gen_anc, "<UNKNOWN>")
        else:
            gen_anc_expr = None

        mt = annotate_freq(
            mt,
            gen_anc_expr=gen_anc_expr,
            sex_expr=mt.ploidy.sex_karyotype if "ploidy" in mt.col_value else None,
            additional_strata_expr=additional_strata_expr,
        )

        if has_gen_anc:
            faf, faf_meta = faf_expr(
                mt.freq, mt.freq_meta, mt.locus, gen_anc_label=gen_anc_label
            )
            fafmax = gen_anc_faf_max_expr(faf, faf_meta, gen_anc_label=gen_anc_label)
            grpmax = grpmax_expr(mt.freq, mt.freq_meta, gen_anc_label=gen_anc_label)
        else:
            faf, faf_meta = (
                hl.missing(hl.tarray(hl.tfloat64)),
                hl.missing(hl.tarray(hl.tdict(hl.tstr, hl.tstr))),
            )
            fafmax = hl.missing(hl.tstruct(fafmax=hl.tfloat64, gen_anc=hl.tstr))
            grpmax = hl.missing(hl.tstruct(gnomad_grpmax=hl.tfloat64, gen_anc=hl.tstr))

        qual_hists = qual_hist_expr(
            mt.GT,
            mt.GQ,
            mt.DP,
            mt.AD,
            mt.adj,
            mt.AD[1] / mt.DP,
            split_adj_and_raw=True,
        )

        mt = mt.annotate_rows(
            inbreeding_coef=bi_allelic_site_inbreeding_expr(mt.GT),
            faf=faf,
            fafmax=fafmax,
            grpmax=grpmax,
            histograms=qual_hists,
        )
        mt = mt.annotate_globals(
            has_faf_data=has_gen_anc,
            faf_meta=faf_meta,
        )
        return mt

    def compute_coverage_stats(
        self,
        mt=FromStep("annotate_resources"),  # noqa: B008
        telomeres_and_centromeres=ResourceRef("telomeres_and_centromeres"),  # noqa: B008
    ) -> hl.Table:
        """Annotate per-locus coverage statistics for one split.

        Coverage is aggregated directly over the rows of the already-dense split
        matrix produced by ``annotate_resources.{task}``: the reference-block and
        variant entries the task's callset realizes are the coverage site set, so
        no VDS, ``ref_ht``, or second densification is involved. The site set is
        those dense rows minus the telomere-and-centromere intervals, which are
        filtered out before aggregation. The per-locus ``DP`` aggregation itself
        lives in :func:`gnomad_meb.lib.annotation.utils.compute_coverage_stats`.

        Strata are a list of dicts built from the dense columns: the sample-QC
        ``hard_filters`` column already joined onto the matrix (``pass`` when it
        is empty, else ``filtered``), plus the ``cohort`` column when
        ``annotation.group_by_cohort`` is configured -- the preceding
        :meth:`variant_dataset_to_matrix_table` step annotated ``cohort``, so no
        second :func:`~gnomad_meb.lib.annotation.utils.cohort_expr` call is made.
        The checkpoint stays on this step's ``_annotation_steps`` wiring (D7).
        """
        from gnomad_meb.hail import get_default_partitions
        from gnomad_meb.hail import hail as hl
        from gnomad_meb.lib.annotation.utils import compute_coverage_stats

        annotation = self.settings.annotation
        assert annotation is not None, "missing annotation configuration"

        mt = hl.filter_intervals(
            mt,
            telomeres_and_centromeres.interval.collect(),
            keep=False,
        )

        strata_expr: list[dict[str, object]] = [
            {"filtered": hl.if_else(hl.len(mt.hard_filters) == 0, "pass", "filtered")}
        ]
        if annotation.group_by_cohort:
            strata_expr.append({"cohort": mt.cohort})

        ht = compute_coverage_stats(mt, strata_expr=strata_expr)
        return ht.naive_coalesce(get_default_partitions())

    def annotate_coverage_stats(
        self,
        mt=FromStep("annotate_regions_and_freq"),  # noqa: B008
        coverage_ht=FromStep("compute_coverage_stats"),  # noqa: B008
    ) -> hl.MatrixTable:
        """Join per-locus coverage onto the matrix carrying regions/frequencies.

        The per-locus coverage table is keyed by ``locus``, so the join key is
        the matrix's ``locus`` row key: ``coverage_ht[mt.locus]`` adds the
        ``coverage_stats`` row field (the gnomAD element array, one entry per
        stratum). Only the ``coverage_stats_meta`` global is carried -- it is
        deterministic across tasks, so ``merge``'s first part wins like the
        other deterministic globals. The per-stratum sample counts
        (``coverage_stats_meta_sample_count``) are provenance for the standalone
        coverage table and are deliberately **not** propagated onto every
        annotation matrix or through ``merge``.
        """
        mt = mt.annotate_rows(
            coverage_stats=coverage_ht[mt.locus].coverage_stats,
        )
        mt = mt.annotate_globals(
            coverage_stats_meta=coverage_ht.index_globals().coverage_stats_meta,
        )
        return mt

    def merge(self, **parts: hl.MatrixTable) -> hl.MatrixTable:
        """Union the per-split matrices into the terminal annotation matrix.

        Row- and column-wise the splits from one GVCF set share their key and
        schemas, so ``union_rows`` is the one join. Globals come from the first
        part; all are deterministic across splits except VEP's per-split
        ``defined``/``missing`` counts, which are summed so the terminal matrix
        equals today's whole-callset output. One repartition here, not per split.
        """
        from gnomad_meb.hail import get_default_partitions
        from gnomad_meb.hail import hail as hl

        split_parts = tuple(parts.values())
        mt = hl.MatrixTable.union_rows(*split_parts)
        mt = _sum_vep_defined_missing(mt, split_parts)
        return mt.repartition(get_default_partitions(), shuffle=True)

    # --- internal annotators (plain helpers, called by the substep bodies) ---

    def _annotate_metadata(self, mt: hl.MatrixTable) -> hl.MatrixTable:
        from gnomad_meb.lib.metadata import get_metadata_expr

        annotation = self.settings.annotation
        assert annotation is not None, "missing annotation configuration"
        mt = mt.annotate_globals(
            project_meta=get_metadata_expr(
                project_name=self.settings.project,
                caller=annotation.caller_name,
                caller_version=annotation.caller_version,
            ),
        )
        return mt

    def _annotate_gnomad(
        self, mt: hl.MatrixTable, gnomad_ht: hl.Table
    ) -> hl.MatrixTable:
        """Use the gnomAD data to annotate frequencies and VEP data for each variant."""
        gnomad_row = gnomad_ht[mt.row_key]
        gnomad_mt = mt.annotate_rows(
            gnomad_freq=gnomad_row.freq,
            vep=gnomad_row.vep,
            in_silico_predictors=gnomad_row.in_silico_predictors,
        )

        gnomad_globals = gnomad_ht.index_globals()
        gnomad_mt = gnomad_mt.annotate_globals(
            gnomad_freq_index_dict=gnomad_globals.freq_index_dict,
            gnomad_freq_meta=gnomad_globals.freq_meta,
            gnomad_date=gnomad_globals.date,
            gnomad_version=gnomad_globals.version,
            gnomad_vep_globals=gnomad_globals.vep_globals,
            gnomad_tool_versions=gnomad_globals.tool_versions,
        )

        return gnomad_mt

    def _annotate_clinvar(
        self, mt: hl.MatrixTable, clinvar_ht: hl.Table
    ) -> hl.MatrixTable:
        from gnomad_meb.hail import hail as hl

        mt = mt.annotate_rows(
            clinvar=hl.Struct(
                **{
                    "clinical_significance": clinvar_ht[mt.row_key].info.CLNSIG,
                    "clinvar_variation_id": clinvar_ht[mt.row_key].rsid,
                    "review_status": clinvar_ht[mt.row_key].info.CLNREVSTAT,
                }
            )
        )

        mt = mt.annotate_rows(
            clinvar=hl.Struct(
                **{
                    "clinical_significance": hl.sorted(
                        mt.clinvar.clinical_significance,
                        key=lambda s: s.replace("^_", "z"),
                    ).map(lambda s: s.replace("^_", "")),
                    "clinvar_variation_id": mt.clinvar.clinvar_variation_id,
                    "review_status": hl.sorted(
                        mt.clinvar.review_status, key=lambda s: s.replace("^_", "z")
                    ).map(lambda s: s.replace("^_", "")),
                }
            )
        )
        return mt

    def _update_insilico_predictors(
        self,
        mt: hl.MatrixTable,
        cadd_ht: hl.Table,
        alphamissense_ht: hl.Table,
        in_silico_predictors_key: str = "in_silico_predictors",
    ) -> hl.MatrixTable:
        """Update the in_silico_predictors field with cadd and alphamissense data."""
        from gnomad_meb.hail import hail as hl

        mt = mt.annotate_rows(
            **{
                in_silico_predictors_key: hl.struct(
                    **{
                        **mt[in_silico_predictors_key],
                        "cadd": cadd_ht[mt.row_key].cadd,
                        "alphamissense": alphamissense_ht[mt.row_key].am_pathogenicity,
                    }
                )
            }
        )
        mt = mt.annotate_globals(
            **{
                f"{in_silico_predictors_key}_globals": hl.struct(
                    cadd=cadd_ht.index_globals(),
                    alphamissense=alphamissense_ht.index_globals(),
                )
            }
        )
        return mt

    def _annotate_sample_qc(
        self, mt: hl.MatrixTable, sample_qc_ht: hl.Table
    ) -> hl.MatrixTable:
        mt = mt.annotate_globals(
            sample_qc=sample_qc_ht.index_globals(),
        )
        mt = mt.annotate_cols(**sample_qc_ht[mt.col_key])
        return mt


def _sum_vep_defined_missing(
    mt: hl.MatrixTable, parts: tuple[hl.MatrixTable, ...]
) -> hl.MatrixTable:
    """Sum the per-split VEP ``defined``/``missing`` globals onto the union.

    ``union_rows`` keeps the first part's globals; every Annotation global is
    deterministic across splits except the two VEP counts, which are set per
    split in ``lib/annotation/vep.py`` from that split's row count. Sum them so
    the merged matrix's ``vep_globals`` equals the whole-callset counts.
    """
    from gnomad_meb.hail import hail as hl

    defined = sum(hl.eval(part.vep_globals.defined) for part in parts)
    missing = sum(hl.eval(part.vep_globals.missing) for part in parts)
    return mt.annotate_globals(
        vep_globals=mt.vep_globals.annotate(defined=defined, missing=missing)
    )
