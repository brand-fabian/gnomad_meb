"""gnomAD-like Sample QC pipeline as a :class:`Pipeline` subclass.

The stage lives here as :class:`SampleQcPipeline`, a
:class:`~gnomad_meb.pipeline.Pipeline` subclass. Its steps are
``@checkpoint``-decorated methods whose bodies are the original sample-QC
computations (recovered from git history): each method receives its data as
arguments — sibling outputs through ``FromStep`` parameter defaults, declared
datasets through ``ResourceRef`` defaults, and the run's VDS through the bound
``RunInput`` — reads scalar cutoffs from ``self.settings.sample_qc``, and
returns its result. The ``@checkpoint`` decorator owns the output artifacts
(their prefixes and kinds), so the method never reads or writes a checkpoint
path itself.

Computation covers coverage, hail builtin sample QC, ancestry classification,
sex inference and contamination estimates. Hail-bearing imports are made lazily
inside the method bodies, so declaring and inspecting the DAG does not run the
hail stack (see ``.scratch/pipeline-resources/design.md`` §4-§6 and issue 17).
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Any

from gnomad_meb.cmd.task import Task
from gnomad_meb.lib.checkpoint import CHECKPOINT_TYPE, checkpoint
from gnomad_meb.lib.sample_qc.utils import infer_sex_karyotype
from gnomad_meb.lib.vds import VariantDataset
from gnomad_meb.pipeline import (
    FromStep,
    InputStep,
    Pipeline,
    Resource,
    ResourceRef,
    RunInput,
)


def compute_sample_rankings(ht: Any, score_expr: Any = None) -> Any:
    """Rank samples by their coverage score, filtered samples first.

    A plain helper (not a step): :meth:`SampleQcPipeline.compute_hard_filters`
    calls it to order samples before deciding which relatives to drop. ``ht``
    is a hail Table carrying ``hard_filters`` and ``ploidy``.
    """
    from gnomad_meb.hail import hail as hl

    if score_expr is None:
        score_expr = ht.ploidy.autosomal_mean_dp
    ht = ht.select(
        score=score_expr,
        filtered=hl.len(ht.hard_filters) > 0,
    )
    ht = ht.order_by(ht.filtered, hl.desc(ht.score))
    ht = ht.add_index(name="rank")
    return ht.key_by("s").select("filtered", "rank")


def _pinned(cutoff: Any) -> Any:
    """Convert an optional pinned settings cutoff to the helper's tuple form.

    Pinned cutoffs are optional; ``None`` means "derive from the cohort" and is
    forwarded as ``None`` so :func:`infer_sex_karyotype` can screen instead.
    """
    return cutoff.as_tuple() if cutoff is not None else None


def _import_locus_intervals(path: Path | str) -> Any:
    """Read a BED locus-interval file on GRCh38, repartitioned to the default count.

    The calling-intervals resource is consumed by ``annotate_sex``; the old
    eager load repartitioned it to ``get_default_partitions`` with a shuffle, so
    keep that here. Hail is imported lazily, so declaring the resource does not
    import the hail stack.
    """
    from gnomad_meb.hail import get_default_partitions
    from gnomad_meb.hail import hail as hl

    intervals = hl.import_locus_intervals(str(path), reference_genome="GRCh38")
    return intervals.repartition(get_default_partitions(), shuffle=True)


def _read_table(path: Path | str) -> Any:
    """Read a hail table in place; hail is imported lazily."""
    from gnomad_meb.hail import hail as hl

    return hl.read_table(str(path))


def _read_table_default_partitions(path: Path | str) -> Any:
    """Read a hail table in place, repartitioned to the default partition count.

    The old eager load passed ``_n_partitions=get_default_partitions()`` for the
    LCR/seg-dup interval tables and the PCA loadings; preserve that.
    """
    from gnomad_meb.hail import get_default_partitions
    from gnomad_meb.hail import hail as hl

    return hl.read_table(str(path), _n_partitions=get_default_partitions())


def _load_onnx_model(path: Path | str) -> Any:
    """Load an ONNX model from ``path`` and return the model directly.

    The ``Resource``'s loader returns what the ancestry computation consumes, so
    the old ``_as_onnx_model`` bytes shim is gone.
    """
    import onnx

    return onnx.load(str(path))


def _recompute_sample_qc_ratios(ht: Any) -> Any:
    """Recompute the four derived ratio fields from the summed counts."""
    from hail.utils.misc import divide_null

    from gnomad_meb.hail import hail as hl

    return ht.annotate(
        r_ti_tv=divide_null(hl.float64(ht.n_transition), ht.n_transversion),
        r_ti_tv_singleton=divide_null(hl.float64(ht.n_singleton_ti), ht.n_singleton_tv),
        r_het_hom_var=divide_null(hl.float64(ht.n_het), ht.n_hom_var),
        r_insertion_deletion=divide_null(hl.float64(ht.n_insertion), ht.n_deletion),
    )


def _tuple_to_array(t: Any) -> Any:
    """A Hail tuple has a static length; build an array without bin counts."""
    from gnomad_meb.hail import hail as hl

    return hl.array([t[i] for i in range(len(t))])


class SampleQcPipeline(Pipeline):
    """The Sample QC stage: coverage, sex, contamination, relatedness, ancestry."""

    @classmethod
    def name(cls) -> str:
        return "sample_qc"

    @cached_property
    def resources(self) -> tuple[Resource, ...]:
        """The run's datasets, each constructed with its path from settings.

        Note
        ----
        Loaders import hail lazily, so declaring (and inspecting) the
        resources does not import the hail stack.

        Reference genome is set to GRCh38 for all resources.
        """
        cfg = self.settings.resources
        assert cfg is not None, "missing resources configuration"
        return (
            Resource(
                name="calling_intervals",
                path=str(cfg.calling_intervals),
                reference_genome="GRCh38",
                loader=_import_locus_intervals,
            ),
            Resource(
                name="telomeres_and_centromeres",
                path=str(cfg.telomeres_and_centromeres),
                reference_genome="GRCh38",
                loader=_read_table,
            ),
            Resource(
                name="lcr_intervals",
                path=str(cfg.lcr_intervals),
                reference_genome="GRCh38",
                loader=_read_table_default_partitions,
            ),
            Resource(
                name="seg_dup_intervals",
                path=str(cfg.seg_dup_intervals),
                reference_genome="GRCh38",
                loader=_read_table_default_partitions,
            ),
            Resource(
                name="gnomad_v4",
                path=str(cfg.gnomad_v4),
                reference_genome="GRCh38",
                loader=_read_table,
            ),
            Resource(
                name="gnomad_pca_model",
                path=str(cfg.gnomad_pca_model),
                reference_genome="GRCh38",
                loader=_load_onnx_model,
            ),
            Resource(
                name="gnomad_pca_loadings",
                path=str(cfg.gnomad_pca_loadings),
                reference_genome="GRCh38",
                loader=_read_table_default_partitions,
            ),
        )

    def get_input(self) -> tuple[InputStep, ...]:
        """The run's VDS: the VDS-build step's result, else the bound handle."""
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

    @checkpoint(
        "{project}/intermediate/sample_qc/filter_intervals",
        CHECKPOINT_TYPE.NATIVE_VARIANT_DATASET,
    )
    def filter_intervals(
        self,
        vds=RunInput("vds"),  # noqa: B008
        telomeres_and_centromeres=ResourceRef("telomeres_and_centromeres"),  # noqa: B008
        lcr_intervals=ResourceRef("lcr_intervals"),  # noqa: B008
        seg_dup_intervals=ResourceRef("seg_dup_intervals"),  # noqa: B008
    ):
        """Drop telomeres/centromeres (and optionally LCR/seg-dup) from the VDS.

        Returns the filtered :class:`VariantDataset` handle; the checkpoint
        serializer materializes its changed splits (ADR-0016).
        """
        from gnomad_meb.hail import hail as hl
        from gnomad_meb.logging import get_logger

        params = self.settings.sample_qc
        assert params is not None, "missing sample qc configuration"

        tel_centr = telomeres_and_centromeres.aggregate(
            hl.agg.collect(telomeres_and_centromeres.interval)
        )
        vds = vds.filter_intervals(tel_centr, keep=False)
        if params.filter_lcr_and_seg_dup:
            get_logger().debug("filtering lcr and seg_dup intervals from input vds")
            for interval_table in [lcr_intervals, seg_dup_intervals]:
                vds = vds.filter_rows(
                    lambda vmt, t=interval_table: hl.is_defined(t[vmt.locus]),
                    keep=False,
                )
        return vds

    @checkpoint(
        "{project}/intermediate/sample_qc/base_sample_qc",
        CHECKPOINT_TYPE.HAIL_TABLE,
    )
    def base_sample_qc(self, vds=FromStep("filter_intervals")):  # noqa: B008
        """Hail's builtin sample QC on the autosomes, summed per split."""
        from gnomad_meb.hail import hail as hl
        from gnomad_meb.lib.sample_qc.constants import SAMPLE_QC_SUM_FIELDS

        params = self.settings.sample_qc
        assert params is not None, "missing sample qc configuration"

        autosomes = vds.filter_chromosomes(keep_autosomes=True)  # per split, no merge

        def per_split(split):
            res = hl.vds.sample_qc(
                split, gq_bins=params.gq_bins, dp_bins=params.dp_bins
            )
            # ``bases_over_*`` come out as ``tuple<int64>``; ``array_sum`` needs
            # arrays, so convert once, generically from the static tuple length.
            return res.annotate(
                bases_over_gq_threshold=_tuple_to_array(res.bases_over_gq_threshold),
                bases_over_dp_threshold=_tuple_to_array(res.bases_over_dp_threshold),
            )

        sample_qcs = list(autosomes.map_datasets(per_split).values())
        sample_qc_ht = (
            sample_qcs[0]
            if len(sample_qcs) == 1
            else sample_qcs[0].union(*sample_qcs[1:], unify=True)
        )
        sample_qc_ht = sample_qc_ht.group_by(sample_qc_ht.s).aggregate(
            **{
                field: hl.agg.sum(sample_qc_ht[field]) for field in SAMPLE_QC_SUM_FIELDS
            },
            bases_over_gq_threshold=hl.agg.array_sum(
                sample_qc_ht.bases_over_gq_threshold
            ),
            bases_over_dp_threshold=hl.agg.array_sum(
                sample_qc_ht.bases_over_dp_threshold
            ),
        )
        sample_qc_ht = _recompute_sample_qc_ratios(sample_qc_ht)
        # ``group_by``/``aggregate`` does not carry per-split globals; restore the
        # bins ``hl.vds.sample_qc`` set.
        return sample_qc_ht.annotate_globals(
            gq_bins=hl.tuple(params.gq_bins),
            dp_bins=hl.tuple(params.dp_bins),
        )

    @checkpoint(
        "{project}/intermediate/sample_qc/annotate_adj",
        CHECKPOINT_TYPE.HAIL_TABLE,
    )
    def get_annotated_sample_qc(
        self,
        vds=FromStep("filter_intervals"),  # noqa: B008
        sample_qc_ht=FromStep("base_sample_qc"),  # noqa: B008
    ):
        """Annotate the base sample QC table with adj call rates, per split."""
        from gnomad.utils.annotations import annotate_adj

        from gnomad_meb.hail import hail as hl

        params = self.settings.sample_qc
        assert params is not None, "missing sample qc configuration"

        num_samples = sample_qc_ht.count()  # one row per sample; no merge

        def per_split(split):
            mt = annotate_adj(
                split.variant_data,
                adj_gq=params.adj_gq,
                adj_dp=params.adj_dp,
                adj_ab=params.adj_ab,
                haploid_adj_dp=params.haploid_adj_dp,
            )
            mt = mt.filter_rows(
                hl.agg.filter(
                    hl.is_defined(mt.LGT) & mt.adj,
                    ((hl.agg.count() / num_samples) > params.min_site_callrate)
                    & (
                        (hl.agg.sum(mt.LGT.n_alt_alleles()) / (hl.agg.count() * 2))
                        > params.callrate_min_af
                    ),
                )
            )
            n_rows = mt.count_rows()
            cols = mt.annotate_cols(
                n_called=hl.agg.count_where(hl.is_defined(mt.LGT)),
                n_called_adj=hl.agg.count_where(hl.is_defined(mt.LGT) & mt.adj),
            ).cols()
            return cols.annotate(n_rows=hl.int64(n_rows))

        adj_counts = list(vds.map_datasets(per_split).values())
        all_counts = (
            adj_counts[0]
            if len(adj_counts) == 1
            else adj_counts[0].union(*adj_counts[1:], unify=True)
        )

        counts_ht = all_counts.group_by(all_counts.s).aggregate(
            n_called=hl.agg.sum(all_counts.n_called),
            n_called_adj=hl.agg.sum(all_counts.n_called_adj),
            n_rows=hl.agg.sum(all_counts.n_rows),
        )
        counts_ht = counts_ht.annotate(
            callrate=counts_ht.n_called / counts_ht.n_rows,
            callrate_adj=counts_ht.n_called_adj / counts_ht.n_rows,
        )

        return sample_qc_ht.annotate(
            callrate=counts_ht[sample_qc_ht.s].callrate,
            callrate_adj=counts_ht[sample_qc_ht.s].callrate_adj,
        )

    @checkpoint(
        "{project}/intermediate/sample_qc/infer_sex",
        CHECKPOINT_TYPE.VARIANT_DATASET,
    )
    def compute_sex_inference_vds(
        self,
        vds=FromStep("filter_intervals"),  # noqa: B008
        lcr_intervals=ResourceRef("lcr_intervals"),  # noqa: B008
        seg_dup_intervals=ResourceRef("seg_dup_intervals"),  # noqa: B008
    ):
        """Checkpoint the VariantDataset used for sex inference."""
        from gnomad_meb.hail import get_default_partitions
        from gnomad_meb.hail import hail as hl
        from gnomad_meb.lib.checkpoint import get_checkpoint_path
        from gnomad_meb.logging import get_logger

        params = self.settings.sample_qc
        assert params is not None, "missing sample qc configuration"
        sex_contigs = params.sex_contigs

        sex_inference_contigs = (params.normalization_contig, *sex_contigs)
        get_logger().info(
            "checkpoint hail VariantDataset for sex inference: "
            f"contigs = {sex_inference_contigs}"
        )

        # Only open the splits covering the sex contigs, rather than every split.
        sex_vds = vds.filter_contigs(sex_inference_contigs)
        interval_ht = lcr_intervals.union(seg_dup_intervals)
        sex_vds = sex_vds.filter_rows(
            lambda vmt: hl.is_defined(interval_ht[vmt.locus]),
            keep=False,
        )

        intermediate_vds_checkpoint = Path(
            str(
                get_checkpoint_path(
                    f"{self.settings.project}/intermediate/sample_qc/compute_sex_inference_vds/intermediate.vds/vds",
                    CHECKPOINT_TYPE.NATIVE_VARIANT_DATASET,
                )
            )
        )
        sex_vds.write(
            base=intermediate_vds_checkpoint.parent,
            name=intermediate_vds_checkpoint.name.removesuffix(".metadata.json"),
        )
        sex_vds = sex_vds.open(intermediate_vds_checkpoint)

        return sex_vds.to_hail(
            apply_repartition=True,
            n_partitions=get_default_partitions(4),
        )

    @checkpoint(
        "{project}/intermediate/sample_qc/ploidy",
        CHECKPOINT_TYPE.HAIL_TABLE,
    )
    def get_infer_sex(
        self,
        vds=FromStep("compute_sex_inference_vds"),  # noqa: B008
        calling_intervals=ResourceRef("calling_intervals"),  # noqa: B008
    ):
        """Infer sample sex/ploidy from the sex chromosomes of the VDS."""
        from gnomad.sample_qc.pipeline import annotate_sex

        from gnomad_meb.hail import hail as hl
        from gnomad_meb.logging import get_logger

        params = self.settings.sample_qc
        assert params is not None, "missing sample qc configuration"
        min_cov = params.hard_filter.min_cov
        sex_contigs = params.sex_contigs

        contigs = vds.variant_data.aggregate_rows(
            hl.agg.collect_as_set(vds.variant_data.locus.contig)
        )
        if not all(x in contigs for x in [params.normalization_contig, *sex_contigs]):
            get_logger().warning(
                f"no sex chromosomes are contained in the variant dataset, "
                f"inferring default ploidy table: sex_contigs = {sex_contigs}, "
                f"defaults = {{ karyotype = {hl.missing(hl.tstr)}, "
                f"autosomal_mean_dp = {min_cov + 1} }}"
            )
            # ``vds`` is a native hail VariantDataset (the merged checkpoint from
            # ``compute_sex_inference_vds``), so the sample columns come straight
            # off its variant data — there is no split selection to do here.
            sample_ht = vds.variant_data.cols()
            return sample_ht.select(
                # Default fields per ``annotate_sex`` documentation
                chrX_ploidy=hl.missing(hl.tfloat32),
                chrY_ploidy=hl.missing(hl.tfloat32),
                chrX_mean_dp=hl.missing(hl.tfloat32),
                chrY_mean_dp=hl.missing(hl.tfloat32),
                autosomal_mean_dp=hl.float64(min_cov + 1),
                chrx_frac_hom_alt=hl.missing(hl.tfloat32),
                chrx_frac_hom_alt_adj=hl.missing(hl.tfloat32),
                # f-stat fields
                f_stat=hl.missing(hl.tfloat64),
                n_called=hl.missing(hl.tint64),
                expected_homs=hl.missing(hl.tfloat64),
                observed_homs=hl.missing(hl.tint64),
                # infer_karyotype fields
                X_karyotype=hl.missing(hl.tstr),
                Y_karyotype=hl.missing(hl.tstr),
                sex_karyotype=hl.missing(hl.tstr),
            )

        # annotate_sex uses the hl.agg.inbreeding aggregator, which checks for
        # ploidy == 2 and call.unphased_diploid_gt_index() <= 2, valid for
        # ["0/0", "0/1", "1/1"] genotypes, which might not hold for DRAGEN
        # VCFs. To rectify this, hl.vds.split_multi downcodes multiallelic
        # local alleles and translates LGT -> GT.
        #
        # Filtering for calls with ploidy == 2 and call.unphased_gt_index() <= 2
        # discards most variants on X/Y and would make the sex inference return
        # garbage values.
        #
        # to_hail runs before split_multi so the haploid fix still sees LGT.
        # merged = hl.vds.split_multi(sex_vds.to_hail(), filter_changed_loci=True)

        ploidy_ht = annotate_sex(
            vds,
            included_intervals=calling_intervals,
            normalization_contig=params.normalization_contig,
            # aaf_expr="AF",
            gt_expr="LGT",
            variants_only_x_ploidy=True,
            variants_only_y_ploidy=False,
            variants_filter_lcr=False,
            variants_filter_segdup=False,
            variants_filter_decoy=False,
            f_stat_cutoff=params.f_stat_cutoff,
            aaf_threshold=params.aaf_threshold,
            # Defer inferring the sex karyotype until we have removed samples
            # that look aneuploid, in order to not conflate the cutoffs
            compute_fstat=True,
            compute_x_frac_variants_hom_alt=True,
            infer_karyotype=False,
        )

        return infer_sex_karyotype(
            ploidy_ht,
            f_stat_cutoff=params.f_stat_cutoff,
            normal_ploidy_cutoff=params.normal_ploidy_cutoff,
            aneuploidy_cutoff=params.aneuploidy_cutoff,
            x_ploidy_cutoffs=_pinned(params.x_ploidy_cutoffs),
            y_ploidy_cutoffs=_pinned(params.y_ploidy_cutoffs),
            chr_x_frac_hom_alt_cutoffs=_pinned(params.chr_x_frac_hom_alt_cutoffs),
        )

    @checkpoint(
        "{project}/intermediate/sample_qc/pca_sites",
        CHECKPOINT_TYPE.MATRIX_TABLE,
    )
    def densify_pca_sites(
        self,
        vds=FromStep("filter_intervals"),  # noqa: B008
        pca_loadings_ht=ResourceRef("gnomad_pca_loadings"),  # noqa: B008
    ):
        """Densify the callset over the gnomAD PCA loadings sites, once.

        Relatedness and ancestry both project onto the same loadings sites, so
        the sites-filtered union + ``hl.vds.to_dense_mt`` is checkpointed once
        here and read by both consumers through ``FromStep`` (ADR-0015: push
        the site filter to split granularity; the one unavoidable union carries
        only the sites). ``filter_to_adj`` keeps today's adjusted-call
        behaviour for both consumers.

        ``fix_lpl`` is left at its ``False`` default: the PCA-sites densification
        is consumed through ``GT`` only, so carrying :meth:`to_hail`'s
        LPL-extension IR would only enlarge the task closure of this write
        (``docs/bugs/10-sample-qc-infer-sex-oom.md``). The union is repartitioned
        once at the merge boundary so the densify write does not inherit the
        uneven union of the per-split partitions.
        """
        from gnomad_meb.hail import get_default_partitions
        from gnomad_meb.hail import hail as hl

        mt = vds.to_dense_mt(
            sites_ht=pca_loadings_ht,
            filter_to_adj=True,
            apply_repartition=True,
            n_partitions=get_default_partitions(2),
        )
        # Normalize the call field once here: ``hl.vds.to_dense_mt`` coalesces
        # on ``LGT`` (the variant data has no global ``GT``), while relatedness
        # reads ``GT``. Doing it in the shared step keeps both consumers
        # agreeing on one field, and makes ancestry's ``GT``/``LGT`` branch
        # dead.
        return mt.annotate_entries(GT=hl.vds.lgt_to_gt(mt.LGT, mt.LA))

    @checkpoint(
        "{project}/intermediate/sample_qc/charr",
        CHECKPOINT_TYPE.HAIL_TABLE,
    )
    def get_contamination_estimate(
        self,
        mt=FromStep("densify_pca_sites"),  # noqa: B008
        gnomad_v4_sites=ResourceRef("gnomad_v4"),  # noqa: B008
    ):
        """Estimate sample contamination against gnomAD v4 allele frequencies.

        Note
        ----
        ``hl.compute_charr`` is invoked only with the variants known to
        ``pca_loadings_ht``, which might confound the contamination estimation
        in some cases.
        """
        from gnomad_meb.hail import hail as hl

        return hl.compute_charr(mt, ref_AF=(1 - gnomad_v4_sites[mt.row_key].freq[0].AF))

    @checkpoint(
        "{project}/intermediate/sample_qc/king",
        CHECKPOINT_TYPE.MATRIX_TABLE,
    )
    def get_king_estimate(
        self,
        mt=FromStep("densify_pca_sites"),  # noqa: B008
    ):
        """Estimate the relatedness phi using hl.king."""
        from gnomad_meb.hail import hail as hl

        return hl.king(mt.GT)

    @checkpoint(
        "{project}/intermediate/sample_qc/relatedness2",
        CHECKPOINT_TYPE.HAIL_TABLE,
    )
    def get_relatedness2(
        self,
        mt=FromStep("densify_pca_sites"),  # noqa: B008
        king_mt=FromStep("get_king_estimate"),  # noqa: B008
    ):
        """Compute the pairwise relatedness using hl.pc_relate."""
        from gnomad.sample_qc.ancestry import run_pca_with_relateds

        from gnomad_meb.hail import get_default_partitions
        from gnomad_meb.hail import hail as hl
        from gnomad_meb.lib.sample_qc.utils import get_related_samples
        from gnomad_meb.logging import get_logger

        params = self.settings.sample_qc
        assert params is not None, "missing sample qc configuration"

        related_ht = get_related_samples(king_mt, params.relatedness_king_kinship)
        n_samples = mt.count_cols()
        n_related = related_ht.count()

        if n_related >= n_samples:
            get_logger().warning(
                "KING marked every sample as related: king_threshold = %s",
                params.relatedness_king_kinship,
            )
            return hl.Table.parallelize(
                [],
                schema=hl.tstruct(
                    i=hl.tstr,
                    j=hl.tstr,
                    kin=hl.tfloat64,
                    ibd0=hl.tfloat64,
                    ibd1=hl.tfloat64,
                    ibd2=hl.tfloat64,
                ),
                key=["i", "j"],
            ).repartition(get_default_partitions())

        n_pcs = max(1, min((n_samples - n_related) // 3, params.relatedness_n_pcs))
        get_logger().debug(
            "computing relatedness based on king estimates: "
            f"n_samples = {n_samples}, n_relateds = {n_related}, n_pcs = {n_pcs}",
        )

        _, scores_ht, _ = run_pca_with_relateds(
            mt, related_samples_to_drop=related_ht, n_pcs=n_pcs
        )
        scores = scores_ht[mt.col_key].scores

        relatedness_ht = hl.pc_relate(
            mt.GT,
            min_individual_maf=params.min_individual_maf,
            scores_expr=scores,
            block_size=4096,
            min_kinship=params.min_kinship,
            statistics="all",
            include_self_kinship=True,
        )
        relatedness_ht = relatedness_ht.key_by(
            i=relatedness_ht.i.s, j=relatedness_ht.j.s
        )
        return relatedness_ht.repartition(get_default_partitions())

    @checkpoint(
        "{project}/intermediate/sample_qc/relatedness",
        CHECKPOINT_TYPE.HAIL_TABLE,
    )
    def get_relatedness(
        self,
        mt=FromStep("densify_pca_sites"),  # noqa: B008
        pca_loadings_ht=ResourceRef("gnomad_pca_loadings"),  # noqa: B008
    ):
        """Compute pairwise relatedness with hail's pc_relate.

        Compute the relatedness phi between all samples in the dataset using
        hail's pc_relate, instead of cuKING like more modern gnomAD pipeline
        versions. The principal-component scores PC-Relate consumes are
        projected from the gnomAD ancestry loadings rather than fit on the
        cohort itself.

        PC-Relate models each sample's individual-specific allele frequency as
        a linear function of its principal-component coordinates, so those
        coordinates must describe *ancestry*, not *relatedness*. Fitting the
        PCA on the cohort (the previous ``hwe_normalized_pca`` path) lets family
        structure leak into the leading components; for a cohort dominated by
        families the projection then absorbs the allele-sharing it is meant to
        measure and every kinship collapses below ``min_kinship`` (bug 08).
        Projecting onto the gnomAD reference loadings avoids that circularity:
        the reference model was fit on unrelated HGDP/1KG samples, so it carries
        ancestry only.

        This projection was chosen because the samples processed here are
        expected to be of Northern European (NFE) ancestry, which the gnomAD
        reference panel represents well. For large cohorts, or for ancestries
        the reference panel does not span, KING-based preprocessing (prune close
        relatives with ``hl.king``, then fit the ancestry PCA on the unrelated
        remainder) may be superior in accuracy, at the cost of an extra pairwise
        pass.

        The cross-sample computation densifies only the gnomAD PCA loadings
        sites: the shared ``densify_pca_sites`` step pushes the site filter to
        each split before the one unavoidable union (ADR-0015).
        """
        from gnomad_meb.hail import get_default_partitions
        from gnomad_meb.hail import hail as hl

        params = self.settings.sample_qc
        assert params is not None, "missing sample qc configuration"

        n_pcs = max(1, min(mt.count_cols() // 3, params.relatedness_n_pcs))

        pca_ht = hl.experimental.pc_project(
            mt.GT,
            pca_loadings_ht.loadings,
            pca_loadings_ht.pca_af,
        )
        scores = pca_ht[mt.col_key].scores[:n_pcs]

        relatedness_ht = hl.pc_relate(
            mt.GT,
            min_individual_maf=params.min_individual_maf,
            scores_expr=scores,
            block_size=4096,
            min_kinship=params.min_kinship,
            statistics="all",
            include_self_kinship=True,
        )
        relatedness_ht = relatedness_ht.key_by(
            i=relatedness_ht.i.s, j=relatedness_ht.j.s
        )
        return relatedness_ht.repartition(get_default_partitions())

    @checkpoint(
        "{project}/intermediate/sample_qc/ancestry_classification",
        CHECKPOINT_TYPE.HAIL_TABLE,
    )
    def apply_ancestry_classification(
        self,
        mt=FromStep("densify_pca_sites"),  # noqa: B008
        fit=ResourceRef("gnomad_pca_model"),  # noqa: B008
        pca_loadings_ht=ResourceRef("gnomad_pca_loadings"),  # noqa: B008
    ):
        """Apply the ancestry PCA components and assign a population per sample."""
        from gnomad.sample_qc.ancestry import (
            apply_onnx_classification_model,
            assign_genetic_ancestry_pcs,
        )

        from gnomad_meb.hail import get_default_partitions
        from gnomad_meb.hail import hail as hl
        from gnomad_meb.logging import get_logger

        params = self.settings.sample_qc
        assert params is not None, "missing sample qc configuration"

        # ``mt`` is already densified over the loadings sites, adj-filtered, and
        # normalized to a global ``GT`` by ``densify_pca_sites``.

        def _get_empty_ancestry_table(ht: hl.Table):
            ht = ht.select(
                pop=hl.missing(hl.tstr),
                prob_other=hl.missing(hl.tfloat64),
                prob_afr=hl.missing(hl.tfloat64),
                prob_ami=hl.missing(hl.tfloat64),
                prob_amr=hl.missing(hl.tfloat64),
                prob_eas=hl.missing(hl.tfloat64),
                prob_mid=hl.missing(hl.tfloat64),
                prob_nfe=hl.missing(hl.tfloat64),
                prob_sas=hl.missing(hl.tfloat64),
            )
            return ht.repartition(get_default_partitions())

        gt_expr = mt.GT

        if mt.count_rows() < 1:
            get_logger().warning(
                "got empty variant matrix table, can not compute ancestry pca"
            )
            return _get_empty_ancestry_table(mt.cols())

        pca_ht = hl.experimental.pc_project(
            gt_expr,
            pca_loadings_ht.loadings,
            pca_loadings_ht.pca_af,
        )

        if pca_ht.count() < 1:
            get_logger().warning(
                "got empty pca_ht table, loadings and data appear to be incompatible"
            )
            return _get_empty_ancestry_table(mt.cols())

        rows = pca_ht.aggregate(hl.agg.take(pca_ht.scores[: params.gnomad_num_pcs], 1))
        n_pcs = len(rows[0]) if rows and rows[0] is not None else 0
        if n_pcs != params.gnomad_num_pcs:
            get_logger().warning(
                "invalid number of principal components for the current dataset: "
                f"n_pcs_actual = {n_pcs}, expected = {params.gnomad_num_pcs}"
            )
            return _get_empty_ancestry_table(mt.cols())

        ht, _model = assign_genetic_ancestry_pcs(
            pca_ht,
            pc_cols=pca_ht.scores[: params.gnomad_num_pcs],
            fit=fit,
            min_prob=params.gnomad_pca_min_prob,
            apply_model_func=apply_onnx_classification_model,
        )
        return ht.repartition(get_default_partitions())

    @checkpoint(
        "{project}/intermediate/sample_qc/combined",
        CHECKPOINT_TYPE.HAIL_TABLE,
    )
    def get_combined_sample_qc_table(
        self,
        sample_qc_ht=FromStep("get_annotated_sample_qc"),  # noqa: B008
        ploidy_ht=FromStep("get_infer_sex"),  # noqa: B008
        charr_ht=FromStep("get_contamination_estimate"),  # noqa: B008
        ancestry_ht=FromStep("apply_ancestry_classification"),  # noqa: B008
    ):
        """Combine the sample QC, ploidy, contamination and ancestry tables."""
        from gnomad_meb.hail import get_default_partitions
        from gnomad_meb.hail import hail as hl

        ht = sample_qc_ht.annotate(
            charr=charr_ht[sample_qc_ht.s].charr,
            ploidy=hl.Struct(
                **{
                    key: ploidy_ht[sample_qc_ht.key][key]
                    for key in ploidy_ht.row.keys()
                    if key not in ploidy_ht.key.keys()
                }
            ),
            ancestry=hl.Struct(
                **{
                    key: ancestry_ht[sample_qc_ht.key][key]
                    for key in ancestry_ht.row.keys()
                    if key not in ancestry_ht.key.keys()
                }
            ),
        )
        ht = ht.annotate_globals(
            charr_params=charr_ht.index_globals(),
            ploidy_params=ploidy_ht.index_globals(),
            ancestry_params=ancestry_ht.index_globals(),
        )
        return ht.repartition(get_default_partitions())

    @checkpoint("{project}/sample_qc", CHECKPOINT_TYPE.HAIL_TABLE)
    def compute_hard_filters(
        self,
        sample_qc_ht=FromStep("get_combined_sample_qc_table"),  # noqa: B008
        king_mt=FromStep("get_king_estimate"),  # noqa: B008
    ):
        """Annotate the combined table with the hard filters and sample rankings."""
        from gnomad.sample_qc.relatedness import compute_related_samples_to_drop
        from gnomad.utils.filtering import add_filters_expr

        from gnomad_meb.hail import get_default_partitions
        from gnomad_meb.hail import hail as hl
        from gnomad_meb.lib.sample_qc.utils import get_king_pairs

        params = self.settings.sample_qc
        assert params is not None, "missing sample qc configuration"
        hard = params.hard_filter
        max_n_singleton = hard.max_n_singletons
        max_r_het_hom_var = hard.max_r_het_hom_var
        min_bases_dp_over_1 = hard.min_bases_dp_over_1
        min_bases_dp_over_20 = hard.min_bases_dp_over_20
        # Chimera rates are not checked, as long as we rely only on hail metrics
        # max_chimera: float = 0.05,
        max_contamination_estimate = hard.max_contamination_estimate
        min_cov = hard.min_cov
        min_qc_mt_adj_callrate = hard.min_qc_mt_adj_callrate
        max_kinship = hard.max_kinship

        ht = sample_qc_ht
        ht = ht.annotate_globals(
            sample_qc_filter_cutoffs=hl.struct(
                max_n_singleton=max_n_singleton,
                max_r_het_hom_var=max_r_het_hom_var,
                min_bases_dp_over_1=min_bases_dp_over_1,
                min_bases_dp_over_20=min_bases_dp_over_20,
                # max_chimera=max_chimera,
                max_contamination_estimate=max_contamination_estimate,
                max_kinship=max_kinship,
            ),
        )
        if min_cov is not None:
            ht = ht.annotate_globals(
                sample_qc_filter_cutoffs=ht.sample_qc_filter_cutoffs.annotate(
                    autosomal_min_cov=min_cov,
                ),
            )
        if min_qc_mt_adj_callrate is not None:
            ht = ht.annotate_globals(
                sample_qc_filter_cutoffs=ht.sample_qc_filter_cutoffs.annotate(
                    min_qc_mt_adj_callrate=min_qc_mt_adj_callrate,
                )
            )
        hard_filters = {}
        hard_filters["high_n_singleton"] = (
            sample_qc_ht[ht.key].n_singleton > max_n_singleton
        )
        hard_filters["high_r_het_hom_var"] = (
            sample_qc_ht[ht.key].r_het_hom_var > max_r_het_hom_var
        )
        hard_filters["low_bases_dp_over_1"] = (
            sample_qc_ht[ht.key].bases_over_dp_threshold[1] < min_bases_dp_over_1
        )
        hard_filters["low_bases_dp_over_20"] = (
            sample_qc_ht[ht.key].bases_over_dp_threshold[3] < min_bases_dp_over_20
        )
        hard_filters["high_contamination"] = (
            sample_qc_ht[ht.key].charr > max_contamination_estimate
        )
        if min_qc_mt_adj_callrate is not None:
            hard_filters["low_adj_callrate"] = (
                sample_qc_ht[ht.key].callrate_adj < min_qc_mt_adj_callrate
            )
        if min_cov is not None:
            hard_filters["low_coverage"] = (
                sample_qc_ht[ht.key].ploidy.autosomal_mean_dp < min_cov
            )
        if hard.include_sex_filter:
            sex = sample_qc_ht[ht.key].ploidy
            hard_filters["ambiguous_sex"] = sex.sex_karyotype == "ambiguous"
            hard_filters["sex_aneuploidy"] = hl.is_defined(sex.sex_karyotype) & ~hl.set(
                {"ambiguous", "XX", "XY"}
            ).contains(sex.sex_karyotype)

        ht = ht.annotate(
            hard_filters=add_filters_expr(filters=hard_filters),
        )

        king_pairs_ht = get_king_pairs(king_mt)
        ranking_ht = compute_sample_rankings(ht).repartition(
            get_default_partitions(), shuffle=True
        )
        relateds_to_drop_ht = compute_related_samples_to_drop(
            king_pairs_ht,
            ranking_ht,
            kin_threshold=max_kinship,
        )
        relateds_to_drop = relateds_to_drop_ht.aggregate(
            hl.agg.collect_as_set(relateds_to_drop_ht.s)
        )
        ht = ht.annotate(
            hard_filters=hl.if_else(
                ht.s in relateds_to_drop,
                ht.hard_filters.add("relatedness"),
                ht.hard_filters,
            ),
        )

        return ht.repartition(get_default_partitions())
