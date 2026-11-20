"""Import the VQSR data into a hail table.

This pipeline imports the per-split site-only VCFs produced by the
allele-specific VQSR run (cf. ``gnomad_meb.vqsr``) into hail tables that can be
used to filter variant sites.

The work is declared as ``@checkpoint``-decorated methods whose parameter
*defaults* are the wire. ``VariantQcImportPipeline.prepare`` reads the run's VDS
task list and generates one :meth:`load_vqsr` step per split, plus a single
terminal :meth:`merge`. Each per-task load imports that split's own
``vqsr.vcf.bgz`` and attaches ``AS_lowqual`` from that split's own ``as_info``
table; ``merge`` unions the row-disjoint per-split tables once, attaches the
VQSR provenance globals (the settings cutoffs/params plus the applied VQSLOD
cutoffs parsed from the gathered tranches), and stays the producer of
``{project}/variant_qc``. Both inputs are
:class:`~gnomad_meb.pipeline.RunInput`s from the command object -- there is no
``pre_vqsr_ht`` :class:`~gnomad_meb.pipeline.Resource`.
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

from gnomad_meb.cmd.task import Task
from gnomad_meb.lib.checkpoint import CHECKPOINT_TYPE, checkpoint
from gnomad_meb.pipeline import FromStep, InputStep, Pipeline, Resource, RunInput

#: Intermediate artifact prefix for the per-task load steps.
_INTERMEDIATE_PREFIX = "{project}/intermediate/variant_qc"


class VariantQcImportException(Exception):
    pass


class VariantQcImportPipeline(Pipeline):
    """Import the per-split allele-specific VQSR VCFs into ``variant_qc``."""

    @classmethod
    def name(cls) -> str:
        return "variant_qc"

    @property
    def resources(self) -> tuple[Resource, ...]:
        """The import stage declares no external resources.

        The per-split ``as_info`` table and the VQSR VCF directory travel as
        :class:`~gnomad_meb.pipeline.RunInput`s from the command object.
        """
        return ()

    def get_input(self) -> tuple[InputStep, ...]:
        """The per-split VQSR VCF directory and ``as_info`` table directory.

        A composed workflow orders both endpoints after their producers through
        ordering-only ``depends_on`` edges; each resolves to its own command
        value.
        """
        cmd = self.command
        group = type(self).name()

        def _input(name: str, value) -> InputStep:
            return InputStep(
                name=name,
                task=Task(settings=self.settings, command=cmd),
                group=group,
                value=value,
            )

        return (
            _input("vcf", getattr(cmd, "vcf", None)),
            _input("as_info", getattr(cmd, "as_info", None)),
        )

    def prepare(self) -> None:
        """Derive the per-split step set from the run's VDS handle.

        One ``load_vqsr`` step is generated per split, plus a single terminal
        ``merge``. The task list is read from the handle's metadata only
        (``VariantDataset.open`` does no data read), so ``--describe`` and
        reconstruction rebuild the same step set from the same input. A missing
        handle fails loudly, so the two invocation scenarios can never disagree
        about the steps.
        """
        from gnomad_meb.lib.variant_qc.tasks import vds_tasks

        super().prepare()

        tasks = vds_tasks(self.command)

        for task in tasks:
            name = task.name
            self._methods.append(
                (
                    f"load_vqsr.{name}",
                    checkpoint(
                        f"{_INTERMEDIATE_PREFIX}/task-{name}/load_vqsr",
                        CHECKPOINT_TYPE.HAIL_TABLE,
                    )(partial(self.load_vqsr, task=name)),
                    name,
                )
            )

        self._methods.append(
            (
                "merge",
                checkpoint("{project}/variant_qc", CHECKPOINT_TYPE.HAIL_TABLE)(
                    partial(
                        self.merge,
                        **{
                            task.name: FromStep(f"load_vqsr.{task.name}")
                            for task in tasks
                        },
                    )
                ),
                None,
            )
        )

    # --- work steps ----------------------------------------------------------

    def load_vqsr(
        self,
        task,
        vcf=RunInput("vcf"),  # noqa: B008
        as_info=RunInput("as_info"),  # noqa: B008
    ):
        """Import one split's as-VQSR VCF, attaching split-local ``AS_lowqual``.

        The VCF is ``{vcf}/task-{task}/vqsr.vcf.bgz`` and ``AS_lowqual`` comes
        from that same split's ``{as_info}/task-{task}/as_info.ht``; both are
        supplied through the command object. No cross-split join or union is
        formed here.
        """
        from gnomad_meb.hail import hail as hl
        from gnomad_meb.lib.variant_qc import split_info
        from gnomad_meb.logging import get_logger

        logger = get_logger()
        variant_qc = self.settings.variant_qc
        assert variant_qc is not None, "missing variant qc import configuration"
        header_file = str(variant_qc.import_header_fixed)

        as_info_ht = hl.read_table(f"{as_info}/task-{task}/as_info.ht")
        vcf_path = f"{vcf}/task-{task}/vqsr.vcf.bgz"

        logger.info(f"importing as-vqsr vcf {vcf_path}")
        unsplit_ht = hl.import_vcf(
            vcf_path,
            reference_genome="GRCh38",
            force_bgz=True,
            array_elements_required=False,
            header_file=header_file,
        ).rows()

        unsplit_ht = unsplit_ht.annotate(
            info=unsplit_ht.info.annotate(
                AS_VQSLOD=unsplit_ht.info.AS_VQSLOD.map(lambda x: hl.float64(x)),
                AS_SB_TABLE=unsplit_ht.info.AS_SB_TABLE.split("\\|").map(
                    lambda x: hl.or_missing(
                        x != "", x.split(",").map(lambda y: hl.int(y))
                    )
                ),
            ),
            AS_lowqual=as_info_ht[unsplit_ht.key].AS_lowqual,
        )

        split_ht = hl.split_multi_hts(unsplit_ht)
        split_ht = split_info(split_ht)

        logger.debug(
            f"finished importing {unsplit_ht.count()} unsplit sites with"
            f" {split_ht.count()} split allele calls with vqsr annotations"
        )
        return split_ht

    def merge(self, **parts):
        """Union the per-split tables, then attach the VQSR provenance globals.

        The per-split ``variant_qc`` tables share their key and schema, so
        ``union`` is the one join (one repartition here, not per split). The
        terminal table stays the producer of ``{project}/variant_qc``, so
        downstream ``variant_qc_ht`` consumers resolve unchanged.

        The globals record how VQSR produced the rows: the static settings
        (``variant_qc_cutoffs.target_truth_sensitivity`` and ``vqsr_params``),
        the GATK provenance (``gatk_version``, read back from the
        ``gatk.version.json`` the VQSR step wrote next to the per-split VCFs),
        and the ``minVQSLod`` cutoffs GATK selected at those targets, parsed
        from the gathered tranches in the command's VQSR output directory (the
        same ``--vcf`` directory the per-split loads read). A missing tranches
        or provenance file (a legacy VQSR output) leaves its value missing
        rather than failing the import.
        """
        from gnomad_meb.hail import get_default_partitions
        from gnomad_meb.hail import hail as hl
        from gnomad_meb.lib.variant_qc.constants import GATK_VERSION_FILENAME
        from gnomad_meb.lib.variant_qc.tranches import select_vqslod_cutoff
        from gnomad_meb.logging import get_logger

        logger = get_logger()
        variant_qc = self.settings.variant_qc
        assert variant_qc is not None, "missing variant qc configuration"
        vcf = getattr(self.command, "vcf", None)
        assert vcf is not None, "missing vqsr output directory"
        targets = variant_qc.target_truth_sensitivity
        params = variant_qc.vqsr_params

        split_parts = tuple(parts.values())
        ht = split_parts[0].union(*split_parts[1:])
        ht = ht.repartition(get_default_partitions(), shuffle=True)

        def cutoff(mode: str, target: float):
            """The applied VQSLOD cutoff for ``mode``, missing if unsourced."""
            path = f"{vcf}/vqsr.{mode}.tranches.txt"
            if not Path(path).exists():
                logger.warning(
                    f"missing vqsr tranches file, no {mode} vqslod cutoff: {path}"
                )
                return hl.missing(hl.tfloat64)
            return select_vqslod_cutoff(path, target)

        provenance = Path(vcf) / GATK_VERSION_FILENAME
        if provenance.exists():
            gatk_version = hl.struct(**json.loads(provenance.read_text()))
        else:
            logger.warning(
                f"missing vqsr gatk provenance file, no gatk version: {provenance}"
            )
            gatk_version = hl.missing(
                hl.tstruct(version=hl.tstr, build=hl.tstr, url=hl.tstr)
            )

        return ht.annotate_globals(
            variant_qc_cutoffs=hl.struct(
                snp=hl.struct(
                    target_truth_sensitivity=targets.snp,
                    vqslod_cutoff=cutoff("snp", targets.snp),
                ),
                indel=hl.struct(
                    target_truth_sensitivity=targets.indel,
                    vqslod_cutoff=cutoff("indel", targets.indel),
                ),
            ),
            vqsr_params=hl.struct(
                allele_specific=params.allele_specific,
                max_gaussians=params.max_gaussians,
                variants_to_sample=params.variants_to_sample,
                snp_tranche_values=hl.tuple(params.snp_tranche_values),
                indel_tranche_values=hl.tuple(params.indel_tranche_values),
            ),
            gatk_version=gatk_version,
        )
