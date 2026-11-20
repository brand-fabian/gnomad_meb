"""Export the VDS data for VQSR.

This module exports all variant data from the current hail.vds.VariantDataset
instance to a VCF containing allele-specific row-level data for use with the
VQSR pipeline. The work is declared as ``@checkpoint``-decorated methods: the
info tables are checkpointed hail tables, and the two VCF exports are terminal
side effects (:meth:`hl.export_vcf` writes the file) whose declared artifact is
the bound export target.
"""

from __future__ import annotations

from functools import partial

from gnomad_meb.cmd.task import Task
from gnomad_meb.lib.checkpoint import (
    CHECKPOINT_TYPE,
    Artifact,
    checkpoint,
    declares_step,
)
from gnomad_meb.pipeline import FromStep, InputStep, Pipeline, Resource, RunInput

# NOTE: the export stage has no single terminal step: it derives one
# ``export_as_info_vcf.<task>`` / ``export_quasi_info_vcf.<task>`` terminal per
# VDS split, so no ``TERMINAL`` constant names a registered step.

_INTERMEDIATE_PREFIX = "{project}/intermediate/variant_qc"


class VariantQcExportException(Exception):
    pass


def _compute_info_for_split(
    vds,
    *,
    infer_as_annotations: bool,
    retain_cdfs: bool,
    cdf_k: int,
):
    """Compute the site-only info table for a single split VDS.

    The computation only touches ``vds.variant_data`` and is row-disjoint
    across splits; export keeps each split's result separate rather than
    unioning them, so no union or repartition happens here.
    """
    import gnomad_qc.v4.annotations.generate_variant_qc_annotations as v4ag
    from gnomad.utils.sparse_mt import default_compute_info

    from gnomad_meb.hail import hail as hl
    from gnomad_meb.lib.variant_qc.constants import (
        REQUIRED_ALLELE_SPECIFIC_ANNOTATIONS,
        REQUIRED_ANNOTATIONS,
    )
    from gnomad_meb.logging import get_logger

    logger = get_logger()

    mt = vds.variant_data
    mt = mt.annotate_rows(alt_alleles_range_array=hl.range(1, hl.len(mt.alleles)))
    mt = mt.filter_rows(
        (hl.len(mt.alleles) > 1) & (hl.agg.any(mt.LGT.is_non_ref()))
    )
    mt = mt.annotate_rows(site_dp=hl.agg.sum(mt.DP))
    if all(
        x in mt.gvcf_info
        for x in [*REQUIRED_ANNOTATIONS, *REQUIRED_ALLELE_SPECIFIC_ANNOTATIONS]
    ):
        pass
    elif infer_as_annotations:
        logger.warning(
            "missing some gvcf annotations, overriding values with inferred "
            "annotations"
        )
        mt = mt.annotate_entries(
            gvcf_info=mt.gvcf_info.annotate(
                QUALapprox=mt.LPL[0],
                VarDP=mt.DP,
                AS_QUALapprox=hl.range(hl.len(mt.LA)).map(
                    lambda x: hl.if_else(x == 0, mt.LPL[x], 0)
                ),
                AS_SB_TABLE=hl.range(hl.len(mt.LA)).map(
                    lambda x: hl.if_else(
                        2 * x + 1 < hl.len(mt.SB),
                        [mt.SB[2 * x], mt.SB[2 * x + 1]],
                        [0, 0],
                    )
                ),
                AS_RAW_MQ=hl.range(hl.len(mt.LA)).map(
                    lambda x: hl.if_else(
                        x == 1,
                        hl.if_else(
                            mt.gvcf_info.RAW_MQandDP[1] > 0,
                            hl.sqrt(
                                mt.gvcf_info.RAW_MQandDP[0]
                                / mt.gvcf_info.RAW_MQandDP[1]
                            ),
                            0,
                        ),
                        0,
                    )
                ),
                AS_RAW_ReadPosRankSum=hl.range(hl.len(mt.LA)).map(
                    lambda x: hl.if_else(
                        x == mt.LA[1], [mt.gvcf_info.ReadPosRankSum], [0]
                    )
                ),
                AS_RAW_MQRankSum=hl.range(hl.len(mt.LA)).map(
                    lambda x: hl.if_else(
                        x == mt.LA[1], [mt.gvcf_info.MQRankSum], [0]
                    )
                ),
                AS_VarDP=mt.LAD,
            ),
        )
    else:
        missing_annotations = {
            *REQUIRED_ANNOTATIONS,
            *REQUIRED_ALLELE_SPECIFIC_ANNOTATIONS,
        } - set(mt.gvcf_info.keys())
        logger.error(
            f"missing gvcf annotations {missing_annotations}, and "
            "infer_as_annotations = False"
        )
        raise VariantQcExportException("missing annotations")

    correct_mt = mt.annotate_entries(
        gvcf_info=mt.gvcf_info.annotate(
            AS_QUALapprox=v4ag.recompute_as_qualapprox_from_lpl(mt),
            **v4ag.correct_as_annotations(mt),
        )
    )
    quasi_info_ht = default_compute_info(
        correct_mt,
        as_annotations=True,
        site_annotations=True,
        n_partitions=None,
        retain_cdfs=retain_cdfs,
        cdf_k=cdf_k,
    )
    quasi_info_ht = quasi_info_ht.annotate(
        info=quasi_info_ht.info.annotate(DB=mt.rows()[quasi_info_ht.key].site_dp)
    )
    as_info = quasi_info_ht.info
    as_keys = hl.eval(
        hl.array(list(as_info.keys())).group_by(
            lambda x: (
                hl.switch(x[:2])
                .when("AS", "AS_info")
                .when("AC", "AC_info")
                .default("site_info")
            )
        )
    )
    quasi_info = {k: as_info.select(*v) for k, v in as_keys.items()}
    return quasi_info_ht.select(
        AS_info=quasi_info["AS_info"],
        AC_info=quasi_info["AC_info"],
        site_info=quasi_info["site_info"],
        quasi_info=quasi_info_ht["quasi_info"],
        lowqual=quasi_info_ht.lowqual,
        AS_lowqual=quasi_info_ht.AS_lowqual,
    )


class VariantQcExportPipeline(Pipeline):
    """Compute the VQSR info tables and export the two sites-only VCFs."""

    @classmethod
    def name(cls) -> str:
        return "variant_qc_export"

    @property
    def resources(self) -> tuple[Resource, ...]:
        """The export stage declares no external resources."""
        return ()

    def get_input(self) -> tuple[InputStep, ...]:
        """The run's VDS, export targets and export options."""
        from gnomad_meb.lib.vds import VariantDataset

        cmd = self.command
        group = type(self).name()

        def _input(name: str, value, run=None) -> InputStep:
            return InputStep(
                name=name,
                task=Task(settings=self.settings, command=cmd),
                group=group,
                value=value,
                **({"run": run} if run is not None else {}),
            )

        return (
            InputStep(
                name="vds",
                task=Task(settings=self.settings, command=cmd),
                group=group,
                source="vds.create_variant_dataset",
                value=getattr(cmd, "vds", None),
                run=VariantDataset.open,
            ),
            _input("as_info", getattr(cmd, "as_info", None)),
            _input("quasi_info", getattr(cmd, "quasi_info", None)),
            _input("infer_as_annotations", getattr(cmd, "infer_as_annotations", True)),
            _input("retain_cdfs", getattr(cmd, "retain_cdfs", False)),
            _input("cdf_k", getattr(cmd, "cdf_k", 200)),
        )

    def _artifacts(self, name, method):
        """The per-task VCF export target, else the decorator's artifacts.

        The VCF exports are terminal side effects written by the method itself;
        their declared artifact is the bound per-task target
        ``{target_dir}/task-{task}/as_info.vcf.bgz`` (the quasi equivalent for
        the quasi export), so the reconstruction of a skipped step can still see
        the file it is expected to produce.
        """
        stem, _, task = name.partition(".")
        if not task:
            return super()._artifacts(name, method)
        if stem == "export_as_info_vcf":
            target, filename = self.command.as_info, "as_info.vcf.bgz"
        elif stem == "export_quasi_info_vcf":
            target, filename = self.command.quasi_info, "quasi_info.vcf.bgz"
        else:
            return super()._artifacts(name, method)
        if not target:
            return ()
        return (Artifact.from_path(f"{target}/task-{task}/{filename}"),)

    def prepare(self) -> None:
        """Derive the per-split step set from the run's VDS handle.

        One info step, one ``as_info`` step, one ``quasi_info`` step and the two
        terminal VCF exports are generated per split. The task list is read from
        the handle's metadata only (``VariantDataset.open`` does no data read),
        so ``--describe`` and reconstruction rebuild the same step set from the
        same input. A missing handle fails loudly, so the two invocation
        scenarios can never disagree about the steps.
        """
        from gnomad_meb.lib.variant_qc.tasks import vds_tasks

        super().prepare()

        tasks = vds_tasks(self.command)

        for task in tasks:
            name = task.name
            self._methods.append(
                (
                    f"compute_info_table.{name}",
                    checkpoint(
                        f"{_INTERMEDIATE_PREFIX}/task-{name}/info",
                        CHECKPOINT_TYPE.HAIL_TABLE,
                    )(partial(self.compute_info_table, task=name)),
                    name,
                )
            )
            self._methods.append(
                (
                    f"compute_as_info_table.{name}",
                    checkpoint(
                        f"{_INTERMEDIATE_PREFIX}/task-{name}/as_info",
                        CHECKPOINT_TYPE.HAIL_TABLE,
                    )(
                        partial(
                            self.compute_as_info_table,
                            info_ht=FromStep(f"compute_info_table.{name}"),
                        )
                    ),
                    name,
                )
            )
            self._methods.append(
                (
                    f"compute_quasi_info_table.{name}",
                    checkpoint(
                        f"{_INTERMEDIATE_PREFIX}/task-{name}/quasi_info",
                        CHECKPOINT_TYPE.HAIL_TABLE,
                    )(
                        partial(
                            self.compute_quasi_info_table,
                            info_ht=FromStep(f"compute_info_table.{name}"),
                        )
                    ),
                    name,
                )
            )
            self._methods.append(
                (
                    f"export_as_info_vcf.{name}",
                    declares_step(
                        partial(
                            self.export_as_info_vcf,
                            task=name,
                            as_info_ht=FromStep(f"compute_as_info_table.{name}"),
                        )
                    ),
                    name,
                )
            )
            self._methods.append(
                (
                    f"export_quasi_info_vcf.{name}",
                    declares_step(
                        partial(
                            self.export_quasi_info_vcf,
                            task=name,
                            quasi_info_ht=FromStep(
                                f"compute_quasi_info_table.{name}"
                            ),
                        )
                    ),
                    name,
                )
            )

    # --- work steps ----------------------------------------------------------

    def compute_info_table(
        self,
        task,
        vds=RunInput("vds"),  # noqa: B008
        infer_as_annotations=RunInput("infer_as_annotations"),  # noqa: B008
        retain_cdfs=RunInput("retain_cdfs"),  # noqa: B008
        cdf_k=RunInput("cdf_k"),  # noqa: B008
    ):
        """Convert one split's variant data to an info table.

        Performs similar filtering steps to generate a site-only VCF as proposed
        by the two reference implementations, skipping the three different
        methods separation from the gnomAD pipeline:

            https://github.com/broadinstitute/gnomad_methods/blob/main/gnomad/utils/sparse_mt.py
            https://github.com/populationgenomics/production-pipelines/blob/94e4d06cf51425141e7955213e1ef807957a232d/cpg_workflows/large_cohort/site_only_vcf.py

        The split boundary is the VDS task: the handle is sliced to the one
        task, so no global union is formed in export.
        """
        split = vds.select_tasks(lambda t: t.name == task).to_hail()
        return _compute_info_for_split(
            split,
            infer_as_annotations=infer_as_annotations,
            retain_cdfs=retain_cdfs,
            cdf_k=cdf_k,
        )

    def compute_as_info_table(
        self,
        info_ht=FromStep("compute_info_table"),  # noqa: B008
    ):
        """Select the allele-specific info field from the meta info table."""
        from gnomad_meb.lib.variant_qc.utils import select_info_table

        return select_info_table(info_ht, "AS_info")

    def compute_quasi_info_table(
        self,
        info_ht=FromStep("compute_info_table"),  # noqa: B008
    ):
        """Select the allele-agnostic (quasi) info field from the meta table."""
        from gnomad_meb.lib.variant_qc.utils import select_info_table

        return select_info_table(info_ht, "quasi_info")

    def export_as_info_vcf(
        self,
        task,
        as_info_ht=FromStep("compute_as_info_table"),  # noqa: B008
        target=RunInput("as_info"),  # noqa: B008
    ):
        """Export one split's allele-specific info table as a tabix VCF."""
        if not target:
            return None
        from gnomad_meb.hail import hail as hl

        hl.export_vcf(
            as_info_ht, f"{target}/task-{task}/as_info.vcf.bgz", tabix=True
        )

    def export_quasi_info_vcf(
        self,
        task,
        quasi_info_ht=FromStep("compute_quasi_info_table"),  # noqa: B008
        target=RunInput("quasi_info"),  # noqa: B008
    ):
        """Export one split's quasi info table as a tabix sites-only VCF."""
        if not target:
            return None
        from gnomad_meb.hail import hail as hl

        hl.export_vcf(
            quasi_info_ht, f"{target}/task-{task}/quasi_info.vcf.bgz", tabix=True
        )
