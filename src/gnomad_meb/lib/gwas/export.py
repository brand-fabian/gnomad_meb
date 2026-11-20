"""Export a matrix table into one regenie dataset per ``(table, region)``.

``gwas_export`` accepts either a single MatrixTable (``--mt``) or the Filter
stage's manifest (``--manifest``), optionally narrowed by ``--include`` /
``--exclude``. :meth:`GwasExportPipeline.prepare` resolves that input into the
table inventory and fans out over the inventory crossed with ``[gwas].regions``:
the region-independent base chain (``gwas_base`` … ``aaf``) is generated once per
table, while the annotation, variant-sets and effect-masks tables are computed
and exported once per region into a self-contained ``{base}/{table}/{region}/``
directory holding the PLINK1 trio, the six TSVs, the response-name map and the
PLINK prefix.

An empty table label (a raw ``--mt``) reproduces the single-table step ids,
paths and artifacts exactly.
"""

from __future__ import annotations

from functools import partial, reduce
from pathlib import Path

from gnomad_meb.cmd.task import Task
from gnomad_meb.lib.checkpoint import (
    CHECKPOINT_TYPE,
    Artifact,
    checkpoint,
    declares_step,
)
from gnomad_meb.lib.gwas.constants import (
    AAF,
    ANNOTATION,
    COVARIATES,
    EFFECT_MASKS,
    FILTER_GLOBAL_NAMES,
    INTERMEDIATE_PREFIX,
    METADATA_SUFFIX,
    N_VARIANTS_KEY,
    PHENOTYPE,
    PLINK_REFERENCE_NAME,
    TABLE_NAMES,
    VARIANT_SETS,
)
from gnomad_meb.lib.gwas.inventory import GwasInventory, GwasTable
from gnomad_meb.lib.gwas.metadata import GwasDatasetMetadata, GwasMetadata
from gnomad_meb.lib.gwas.names import DatasetKey
from gnomad_meb.lib.gwas.utils import (
    build_covariate_exprs,
    define_plink_reference,
    export_ht_as_regenie_table,
    get_aaf_ht,
    get_annotation_ht,
    get_covariate_ht,
    get_effects_ht,
    get_freq_expr,
    get_gene_region_expr,
    get_phenotype_ht,
    get_region_name,
    get_variant_sets_ht,
    get_weight_expr,
    region_label,
    resolve_iid_fid,
    variant_id_expr,
)
from gnomad_meb.logging import get_logger
from gnomad_meb.pipeline import FromStep, InputStep, Pipeline, Resource, RunInput

#: The region-independent base chain, in order: ``(kind, kind_type)``. The
#: checkpoint prefix is ``DatasetKey.prefix(kind)`` (see
#: :data:`~gnomad_meb.lib.gwas.constants.BASE_CHAIN_PREFIXES`).
BASE_CHAIN: tuple[tuple[str, CHECKPOINT_TYPE], ...] = (
    ("gwas_base", CHECKPOINT_TYPE.MATRIX_TABLE),
    ("normalize_regenie", CHECKPOINT_TYPE.MATRIX_TABLE),
    ("filter_regenie_chromosomes", CHECKPOINT_TYPE.MATRIX_TABLE),
    ("phenotype", CHECKPOINT_TYPE.HAIL_TABLE),
    ("covariates", CHECKPOINT_TYPE.HAIL_TABLE),
    ("aaf", CHECKPOINT_TYPE.HAIL_TABLE),
)


class GwasExportException(Exception):
    """A required GWAS export input or configuration is missing."""


class GwasExportPipeline(Pipeline):
    """Turn each selected input MatrixTable into a regenie dataset."""

    @classmethod
    def name(cls) -> str:
        return "gwas_export"

    @property
    def resources(self) -> tuple[Resource, ...]:
        return ()

    def get_input(self) -> tuple[InputStep, ...]:
        """One ``mt`` endpoint per table (composition deferred to the workflow)."""
        from gnomad_meb.hail import hail as hl

        cmd = self.command
        return tuple(
            InputStep(
                name=DatasetKey.of(table.label, "").step("mt"),
                task=Task(settings=self.settings, command=cmd),
                group=type(self).name(),
                value=table.mt,
                run=hl.read_matrix_table,
            )
            for table in self._tables
        )

    # --- region fan-out ------------------------------------------------------

    def _gwas(self):
        gwas = self.settings.gwas
        if gwas is None:
            raise GwasExportException("missing gwas configuration section")
        return gwas

    def _metadata_csv(self) -> str:
        """The study metadata CSV: the command override, else the config path."""
        return getattr(self.command, "metadata_csv", None) or self._gwas().metadata_csv

    def _region_specs(self) -> list[tuple[str, str, int | None]]:
        """``(label, kind, size)`` for each configured region, in order."""
        return [
            (region_label(region.kind, region.size), region.kind, region.size)
            for region in self._gwas().regions
        ]

    def _base_chain(self, table: GwasTable) -> list[tuple[str, object, None]]:
        """The base-chain ``(name, callable, task)`` entries for one table."""
        key = DatasetKey.of(table.label, "")

        def named(base: str) -> str:
            return key.step(base)

        edges = {
            "gwas_base": {"mt": RunInput(key.step("mt"))},
            "normalize_regenie": {"mt": FromStep(named("gwas_base"))},
            "filter_regenie_chromosomes": {
                "mt": FromStep(named("normalize_regenie")),
                "n_variants": table.n_variants,
            },
            "phenotype": {"mt": FromStep(named("filter_regenie_chromosomes"))},
            "covariates": {"mt": FromStep(named("filter_regenie_chromosomes"))},
            "aaf": {"mt": FromStep(named("filter_regenie_chromosomes"))},
        }

        entries: list[tuple[str, object, None]] = []
        for base, kind in BASE_CHAIN:
            method = checkpoint(key.prefix(base), kind)(
                partial(getattr(self, base), **edges[base])
            )
            entries.append((named(base), method, None))
        return entries

    def prepare(self) -> None:
        """Derive the per-table, per-region step set from config and the input.

        Resolves the inventory once (pure Python: a raw ``--mt`` or the read
        Filter manifest), then generates each table's base chain and each
        ``(table, region)``'s region steps. Reads only config and the input
        declaration, so ``--describe`` and reconstruction yield the same step
        set (ADR-0017).
        """
        super().prepare()

        self._tables = GwasInventory.from_command(self.settings, self.command).tables
        self._metadata = GwasMetadata.from_settings(
            self.settings, metadata_csv=self._metadata_csv()
        )

        for table in self._tables:
            self._methods.extend(self._base_chain(table))

        for table in self._tables:
            table_key = DatasetKey.of(table.label, "")
            for label, kind, size in self._region_specs():
                key = DatasetKey.of(table.label, label)
                input_prefix = key.prefix("regenie_input")
                self._methods.append(
                    (
                        key.step("regenie_input"),
                        checkpoint(
                            (
                                f"{input_prefix}/{ANNOTATION}",
                                f"{input_prefix}/{VARIANT_SETS}",
                                f"{input_prefix}/{EFFECT_MASKS}",
                            ),
                            (
                                CHECKPOINT_TYPE.HAIL_TABLE,
                                CHECKPOINT_TYPE.HAIL_TABLE,
                                CHECKPOINT_TYPE.HAIL_TABLE,
                            ),
                        )(
                            partial(
                                self.regenie_input,
                                region_kind=kind,
                                region_size=size,
                                mt=FromStep(
                                    table_key.step("filter_regenie_chromosomes")
                                ),
                            )
                        ),
                        label,
                    )
                )
                self._methods.append(
                    (
                        key.step("export_regenie"),
                        declares_step(
                            partial(
                                self.export_regenie,
                                table=table,
                                region_label=label,
                                mt=FromStep(
                                    table_key.step("filter_regenie_chromosomes")
                                ),
                                phenotype=FromStep(table_key.step(PHENOTYPE)),
                                covariates=FromStep(table_key.step(COVARIATES)),
                                aaf=FromStep(table_key.step(AAF)),
                                annotation=FromStep(
                                    key.step("regenie_input"), output=0
                                ),
                                variant_sets=FromStep(
                                    key.step("regenie_input"), output=1
                                ),
                                effect_masks=FromStep(
                                    key.step("regenie_input"), output=2
                                ),
                            )
                        ),
                        label,
                    )
                )

    def _basename(self) -> str:
        return self.settings.project

    def _export_base(self) -> Path:
        override = getattr(self.command, "base", None)
        if override:
            return Path(override)
        if self.settings.checkpoint is None:
            raise ValueError(
                "can not export plink and regenie files without checkpoint path"
            )
        return (
            Path(self.settings.checkpoint)
            / INTERMEDIATE_PREFIX.format(project=self._basename())
            / "dataset"
        )

    def _base_path(self, table_label: str = "") -> Path:
        """The per-table dataset base directory (empty label ⇒ today's path)."""
        base = self._export_base()
        return base / table_label if table_label else base

    def _artifacts(self, name: str, method):
        """The self-contained dataset files the terminal export writes."""
        stem, _, _ = name.partition(".")
        if stem != "export_regenie":
            return super()._artifacts(name, method)

        segments = name.split(".")
        if len(segments) == 2:
            table_label, region = "", segments[1]
        elif len(segments) == 3:
            table_label, region = segments[1], segments[2]
        else:
            raise GwasExportException(f"malformed export step name: {name}")

        basename = self._basename()
        target = self._base_path(table_label) / region
        prefix = target / basename
        artifacts = [
            Artifact.from_path(f"{prefix}.{ext}") for ext in ("bed", "bim", "fam")
        ]
        artifacts += [
            Artifact.from_path(f"{target}/{basename}.{table}.tsv.bgz")
            for table in TABLE_NAMES
        ]
        artifacts.append(
            Artifact.from_path(
                f"{target}/{basename}{METADATA_SUFFIX}",
                kind=CHECKPOINT_TYPE.FILE,
            )
        )
        return tuple(artifacts)

    def _fix_sex_ploidy(self, mt):
        """Repair non-PAR sex-chromosome genotype calls for the PLINK export."""
        import hail as hl

        # Missing sex must repair nothing, hence the coalesces.
        is_female = hl.coalesce(mt._is_female, False)
        is_par = mt.locus.in_autosome_or_par()
        on_sex_contig = (mt.locus.contig == "X") | (mt.locus.contig == "Y")
        get_logger().debug(
            "repairing sex chromosome genotypes for the plink export: "
            f"reference = {mt.locus.dtype.reference_genome.name}"
        )
        return mt.annotate_entries(
            GT=hl.case()
            .when(
                ~is_par & (mt.locus.contig == "Y") & is_female,
                hl.null(hl.tcall),
            )
            .when(~is_par & ~is_female & on_sex_contig & mt.GT.is_het(), hl.call(1, 1))
            .default(mt.GT),
        )

    # --- work steps ----------------------------------------------------------

    def gwas_base(self, mt=RunInput("mt")):  # noqa: B008
        """Load study metadata, rebuild ``study_md``, drop cohorts, split multi.

        The export does not assume ``study_md`` exists: it is rebuilt from the
        configured metadata CSV, joined on the MT's column key.
        """
        import hail as hl

        gwas = self._gwas()
        md_ht = self._metadata.study_md()

        mt = mt.annotate_cols(study_md=md_ht[mt.col_key])
        if gwas.drop_cohorts:
            drop = hl.literal(set(gwas.drop_cohorts))
            mt = mt.filter_cols(~drop.contains(mt.study_md.cohort))

        return hl.split_multi_hts(mt)

    def normalize_regenie(self, mt=FromStep("gwas_base")):  # noqa: B008
        """Re-key rows to the un-prefixed ``plink`` reference regenie expects."""
        import hail as hl

        reference, contig_d = define_plink_reference(
            self.settings.reference_genome, PLINK_REFERENCE_NAME
        )
        mapping = hl.literal(contig_d)
        mt = mt.filter_rows(mapping.contains(mt.locus.contig))
        mt = mt.annotate_rows(
            plink_locus=hl.locus(
                contig=mapping[mt.locus.contig],
                pos=mt.locus.position,
                reference_genome=reference,
            )
        )
        return (
            mt.key_rows_by("plink_locus", "alleles")
            .drop("locus")
            .rename({"plink_locus": "locus"})
        )

    def filter_regenie_chromosomes(
        self,
        mt=FromStep("normalize_regenie"),  # noqa: B008
        n_variants: int | None = None,
    ):
        """Keep 1-22, X, Y and record the table's variant count as a global.

        The count is resolved once per table: the inventory's manifest summary
        when present, else ``mt.count_rows()`` on the filtered chromosome MT.
        Storing it as a global lets the per-region export write the GWAS
        metadata file without recounting.
        """
        import hail as hl

        contigs = hl.literal({*map(str, range(1, 23)), "X", "Y"})
        mt = mt.filter_rows(contigs.contains(mt.locus.contig))
        if n_variants is None:
            n_variants = mt.count_rows()
        return mt.annotate_globals(**{N_VARIANTS_KEY: int(n_variants)})

    def phenotype(self, mt=FromStep("filter_regenie_chromosomes")):  # noqa: B008
        """The ``phenotype`` table: only the gated, selected response columns.

        The shared :class:`GwasMetadata` frames are the single source of truth:
        the gated :attr:`~GwasMetadata.response_df` is projected into a hail
        table and joined on the MT's column key, so every written column passed
        the ``min_case_count`` gate at construction and ``regenie/meta.smk``'s
        gate is idempotent over it. With ``case_only`` set responses the
        non-cases are missing, so regenie sees NA. The
        ``response -> display name`` and ``response -> kind`` maps ride along as
        globals so the terminal export can persist them.
        """
        import hail as hl

        metadata = self._metadata
        response_ht = hl.Table.from_pandas(
            metadata.response_df.reset_index(drop=False, names=["sample_id"]),
            key=["sample_id"],
        )
        responses = response_ht[mt.col_key]
        y_exprs = {
            name: hl.float64(responses[name]) for name in metadata.responses
        }

        iid_expr, fid_expr = resolve_iid_fid(mt)
        ht = get_phenotype_ht(mt, iid_expr, fid_expr, y_exprs)
        return ht.annotate_globals(
            response_name_map=hl.dict(metadata.names),
            response_kind_map=hl.dict(metadata.kinds),
        )

    def covariates(self, mt=FromStep("filter_regenie_chromosomes")):  # noqa: B008
        """The ``covariates`` table: intercept plus the configured covariates."""
        gwas = self._gwas()
        exprs = build_covariate_exprs(
            mt,
            gwas.covariates,
            intercept=gwas.intercept,
        )
        iid_expr, fid_expr = resolve_iid_fid(mt)
        return get_covariate_ht(mt, iid_expr, fid_expr, exprs)

    def aaf(self, mt=FromStep("filter_regenie_chromosomes")):  # noqa: B008
        """The ``aaf`` table: ``variant_id,aaf`` (NFE/adj, default 0)."""
        return get_aaf_ht(mt, get_freq_expr(mt))

    def regenie_input(
        self,
        region_kind: str = "gene",
        region_size: int | None = None,
        mt=FromStep("filter_regenie_chromosomes"),  # noqa: B008
    ):
        """The three region-dependent tables for one region.

        Returns ``(annotation, variant_sets, effect_masks)``; the callers are
        the per-region checkpointed steps ``prepare`` generates.
        """
        gwas = self._gwas()
        if region_kind == "window":
            region = get_region_name(mt, region_size)
        else:
            region = get_gene_region_expr(mt)

        freq_expr = get_freq_expr(mt)
        weight_expr = get_weight_expr(freq_expr)
        annotation = get_annotation_ht(
            mt,
            weight_expr,
            region,
            rare_variants_maf=gwas.rare_variants_maf,
            min_cadd=gwas.min_cadd,
        )
        variant_sets = get_variant_sets_ht(mt, region)
        effect_masks = get_effects_ht(mt)
        return annotation, variant_sets, effect_masks

    def export_regenie(
        self,
        region_label: str,
        table: GwasTable | None = None,
        mt=FromStep("filter_regenie_chromosomes"),  # noqa: B008
        phenotype=FromStep(PHENOTYPE),  # noqa: B008
        covariates=FromStep(COVARIATES),  # noqa: B008
        aaf=FromStep(AAF),  # noqa: B008
        annotation=FromStep(ANNOTATION),  # noqa: B008
        variant_sets=FromStep(VARIANT_SETS),  # noqa: B008
        effect_masks=FromStep(EFFECT_MASKS),  # noqa: B008
    ):
        """Write one region's PLINK trio, six TSVs and GWAS metadata file.

        The trio's sex-chromosome genotypes are repaired to PLINK's
        diploid-male / no-female-Y convention before writing; see
        :meth:`_fix_sex_ploidy`.
        """
        import hail as hl

        gwas = self._gwas()
        basename = self._basename()
        table_label = table.label if table is not None else ""
        target = self._base_path(table_label) / region_label
        target.mkdir(parents=True, exist_ok=True)

        iid_expr, fid_expr = resolve_iid_fid(mt)
        mt = mt.annotate_cols(
            _iid=iid_expr,
            _fid=fid_expr,
            _pheno=mt.study_md.cohort == gwas.case_cohort,
            _is_female=mt.study_md.sex == "F",
        )
        mt = self._fix_sex_ploidy(mt)
        get_logger().debug(
            "writing plink and regenie output files: "
            f"path = {(target / basename).resolve()!s}"
        )
        hl.export_plink(
            mt,
            str((target / basename).resolve()),
            fam_id=mt._fid,
            ind_id=mt._iid,
            is_female=mt._is_female,
            pheno=mt._pheno,
            varid=variant_id_expr(mt),
        )

        tables = {
            PHENOTYPE: phenotype,
            COVARIATES: covariates,
            AAF: aaf,
            ANNOTATION: annotation,
            VARIANT_SETS: variant_sets,
            EFFECT_MASKS: effect_masks,
        }
        for name in TABLE_NAMES:
            export_ht_as_regenie_table(tables[name], name, target, basename)

        names = hl.eval(phenotype.globals.response_name_map)
        kinds = hl.eval(phenotype.globals.response_kind_map)
        n_variants = (
            hl.eval(mt.globals[N_VARIANTS_KEY])
            if N_VARIANTS_KEY in mt.globals.dtype.fields
            else None
        )
        # Carry the filter metadata globals, when the exported MT has them. The
        # writer records only the present, non-empty ones.
        filter_globals = {
            name: hl.eval(mt.globals[name])
            for name in FILTER_GLOBAL_NAMES
            if name in mt.globals.dtype.fields
        }
        # Carry the effect-masks provenance (the variant_info get_effects_ht
        # aggregated for this region's Mask-<class> rows), when present.
        effect_mask_info = {}
        if "variant_info" in effect_masks.globals.dtype.fields:
            from gnomad_meb.lib.filter.filter_checkpoint import hail_struct_to_dict

            effect_mask_info = hail_struct_to_dict(
                hl.eval(effect_masks.globals.variant_info)
            )
        GwasDatasetMetadata(
            names=names,
            kinds=kinds,
            n_variants=n_variants,
            source=table.source if table is not None else None,
            filter_name=table.filter_name if table is not None else None,
            region_name=table.region_name if table is not None else None,
            filter_globals=filter_globals,
            effect_mask_info=effect_mask_info,
        ).write(target, basename=basename)
        return str(target)
