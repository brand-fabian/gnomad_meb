"""Filter a Variant Callset with parameters based on a given configuration.

The stage is a :class:`~gnomad_meb.pipeline.Pipeline` subclass. Its two fixed
computations — filtering to high-quality gnomAD variants and annotating the
constraints table — are ``@checkpoint``-decorated methods. Each configured
filter becomes one further Step off the constraints-annotated callset; its
declaration (the cutoffs) is captured in the method's closure, since the
configuration file is read once when the pipeline is constructed.

A terminal ``get_manifest`` step gathers the whole stage into one artifact:
every output path and each filter's runtime result. It is the index a
downstream stage reads to bind a filtered matrix and record what was applied;
see :mod:`gnomad_meb.lib.filter.manifest`.
"""

from __future__ import annotations

from functools import partial
from itertools import product
from typing import Any

from gnomad_meb.lib.checkpoint import (
    CHECKPOINT_TYPE,
    checkpoint,
    get_checkpoint_path,
)
from gnomad_meb.lib.filter.constants import (
    CHECKPOINT_PREFIX,
    CONSTRAINT_PREFIX,
    HIGH_QUALITY_PREFIX,
    MANIFEST_PREFIX,
)
from gnomad_meb.lib.filter.manifest import (
    FilterEntry,
    FilterManifest,
    FilterRegion,
    HighQuality,
)
from gnomad_meb.pipeline import (
    FromStep,
    InputStep,
    Pipeline,
    Resource,
    ResourceRef,
    RunInput,
)

from .config import FilterCheckpoint, RegionCheckpoint


class FilterPipeline(Pipeline):
    """The filter DAG: two fixed steps plus one Step per configured filter."""

    @classmethod
    def name(cls) -> str:
        return "filter"

    @property
    def resources(self) -> tuple[Resource, ...]:
        """The constraints table, when the run configures one."""
        from gnomad_meb.hail import hail as hl

        cfg = self.settings.filter
        if (
            cfg is not None
            and cfg.resources is not None
            and cfg.resources.constraints is not None
        ):
            return (
                Resource(
                    name="constraints",
                    path=str(cfg.resources.constraints),
                    reference_genome=self.settings.reference_genome,
                    loader=hl.read_table,
                ),
            )
        return ()

    def get_input(self) -> tuple[InputStep, ...]:
        """The matrix table to filter, when one is bound.

        The endpoint loads a bound path with ``hl.read_matrix_table`` so a
        standalone ``filter run`` reads the checkpoint; in the composed
        workflow the annotation stage's artifact is read the same way.
        """
        from gnomad_meb.hail import hail as hl

        return (
            InputStep(
                name="mt",
                task=self.task,
                group=type(self).name(),
                value=getattr(self.command, "mt", None),
                run=hl.read_matrix_table,
            ),
        )

    def prepare(self) -> None:
        from gnomad_meb.lib.filter.filter_checkpoint import apply_filter_checkpoint
        from gnomad_meb.lib.filter.region_checkpoint import apply_region_checkpoint
        super().prepare()

        filters = self._load_filters()
        regions = self._load_regions()

        # Compute all filters against the global dataset (no region filter)
        for flt in filters:
            if hasattr(type(self), flt.name):
                raise ValueError(
                    f"filter name {flt.name!r} collides with an existing attribute"
                )
            prefix = self._filter_prefix(flt.name)
            self._methods.append(
                (
                    flt.name,
                    checkpoint(
                        [prefix, f"{prefix}.variant_info"],
                        [CHECKPOINT_TYPE.MATRIX_TABLE, CHECKPOINT_TYPE.DICTIONARY],
                    )(
                        partial(
                            apply_filter_checkpoint,
                            mt=FromStep("annotate_constraints_table"),
                            flt=flt,
                        )
                    ),
                    None,
                ),
            )

        # Compute all filters against all region checkpoints
        for flt, reg in product(filters, regions):
            prefix = self._region_prefix(reg.name, flt.name)
            self._methods.append(
                (
                    f"filter_r{reg.name}_f{flt.name}",
                    checkpoint(prefix, CHECKPOINT_TYPE.MATRIX_TABLE)(
                        partial(
                            apply_region_checkpoint,
                            mt=FromStep(flt.name, output=0),
                            region=reg,
                        )
                    ),
                    None,
                ),
            )

        # The stage's central index: every output path plus each filter's
        # returned variant/sample info, inlined so a consumer need not read the
        # matrices back. It depends on the filters and every region leaf, so a
        # present manifest implies those steps ran.
        deps: dict[str, Any] = {
            "high_quality_variant_info": FromStep(
                "filter_high_quality_variants_gnomad", output=1
            ),
            "high_quality_sample_info": FromStep(
                "filter_high_quality_variants_gnomad", output=2
            ),
        }
        for flt in filters:
            deps[f"variant_info_{flt.name}"] = FromStep(flt.name, output=1)
        for flt, reg in product(filters, regions):
            deps[f"region_{reg.name}_{flt.name}"] = FromStep(
                f"filter_r{reg.name}_f{flt.name}", output=0
            )
        self._methods.append(
            (
                "get_manifest",
                checkpoint(MANIFEST_PREFIX, CHECKPOINT_TYPE.DICTIONARY)(
                    partial(self.get_manifest, **deps)
                ),
                None,
            )
        )

    @checkpoint(
        [
            HIGH_QUALITY_PREFIX,
            f"{HIGH_QUALITY_PREFIX}.variant_info",
            f"{HIGH_QUALITY_PREFIX}.sample_info",
        ],
        [
            CHECKPOINT_TYPE.MATRIX_TABLE,
            CHECKPOINT_TYPE.DICTIONARY,
            CHECKPOINT_TYPE.DICTIONARY,
        ],
    )
    def filter_high_quality_variants_gnomad(self, mt=RunInput("mt")):  # noqa: B008
        """Keep high-quality variants (passing VQSR, enough calls, inbreeding)."""
        from gnomad_meb.lib.filter.gnomad import filter_variants_gnomad
        filter = self.settings.filter
        if filter is None:
            return filter_variants_gnomad(mt)
        return filter_variants_gnomad(
            mt,
            min_inbreeding_coef=filter.min_inbreeding_coef,
            min_high_quality_calls=filter.min_high_quality_calls,
            apply_sample_filters=filter.apply_sample_filters,
            sample_filters_ignore=filter.sample_filters_ignore,
        )

    @checkpoint(CONSTRAINT_PREFIX, CHECKPOINT_TYPE.MATRIX_TABLE)
    def annotate_constraints_table(
        self,
        mt=FromStep("filter_high_quality_variants_gnomad", output=0),  # noqa: B008
        constraints_ht=ResourceRef("constraints"),  # noqa: B008
    ):
        """Annotate the callset with the gnomAD constraint metrics."""
        from gnomad_meb.lib.filter.gnomad import annotate_lof_constraint

        return annotate_lof_constraint(mt, constraints_ht)

    # --- the manifest --------------------------------------------------------

    def get_manifest(self, **outputs: Any) -> dict[str, Any]:
        """The stage's manifest: every output path and each filter's result.

        ``**outputs`` are the dependencies ``prepare`` wired in; their values are
        the runtime results inlined under each filter, so a downstream stage
        reads one artifact instead of the matrices. Paths come from the same
        prefixes the checkpoints declare, resolved through
        :func:`get_checkpoint_path`.
        """
        from gnomad_meb.hail import hail as hl

        settings = self.settings
        regions = self._load_regions()

        high_quality_prefix = HIGH_QUALITY_PREFIX.format(project=settings.project)
        high_quality = HighQuality(
            mt=get_checkpoint_path(high_quality_prefix, CHECKPOINT_TYPE.MATRIX_TABLE),
            variant_info=get_checkpoint_path(
                f"{high_quality_prefix}.variant_info", CHECKPOINT_TYPE.DICTIONARY
            ),
            sample_info=get_checkpoint_path(
                f"{high_quality_prefix}.sample_info", CHECKPOINT_TYPE.DICTIONARY
            ),
            variant_info_summary=outputs["high_quality_variant_info"],
            sample_info_summary=outputs["high_quality_sample_info"],
        )

        filter_entries: list[FilterEntry] = []
        for flt in self._load_filters():
            prefix = self._filter_prefix(flt.name).format(project=settings.project)
            filter_regions: list[FilterRegion] = []
            for reg in regions:
                # The deps are wired under their kwarg name (the checkpoint
                # prefix), not the region step's id.
                region_key = f"region_{reg.name}_{flt.name}"
                assert (
                    region_key in outputs
                ), f"missing output for filtered region: name = {region_key}"
                region_mt = outputs[region_key]
                region_info_summary = (
                    hl.eval(region_mt.filter_region_info)
                    if "filter_region_info" in region_mt.globals.dtype.fields
                    else {}
                )
                region_prefix = self._region_prefix(reg.name, flt.name).format(
                    project=settings.project
                )
                filter_regions.append(
                    FilterRegion(
                        name=reg.name,
                        mt=get_checkpoint_path(
                            region_prefix, CHECKPOINT_TYPE.MATRIX_TABLE
                        ),
                        region_info_summary=region_info_summary,
                    )
                )
            filter_entries.append(
                FilterEntry(
                    name=flt.name,
                    mt=get_checkpoint_path(prefix, CHECKPOINT_TYPE.MATRIX_TABLE),
                    variant_info=get_checkpoint_path(
                        f"{prefix}.variant_info", CHECKPOINT_TYPE.DICTIONARY
                    ),
                    variant_info_summary=outputs[f"variant_info_{flt.name}"],
                    regions=filter_regions,
                )
            )

        manifest = FilterManifest(
            project=settings.project,
            reference_genome=settings.reference_genome,
            checkpoint_root=settings.checkpoint,
            input_mt=getattr(self.command, "mt", None),
            constraint_mt=get_checkpoint_path(
                CONSTRAINT_PREFIX.format(project=settings.project),
                CHECKPOINT_TYPE.MATRIX_TABLE,
            ),
            high_quality=high_quality,
            filters=filter_entries,
        )
        return manifest.model_dump(mode="json")

    # --- the configured filters ---------------------------------------------

    def _filter_prefix(self, name: str) -> str:
        """The ``{project}``-templated prefix of one filter's matrix table."""
        return f"{CHECKPOINT_PREFIX}/{name}"

    def _region_prefix(self, region: str, name: str) -> str:
        """The ``{project}``-templated prefix of one ``(region, filter)`` matrix."""
        return f"{CHECKPOINT_PREFIX}/r{region}/f{name}"

    def _load_filters(self) -> list[FilterCheckpoint]:
        cfg = self.settings.filter
        return [] if cfg is None else cfg.load_filters()

    def _load_regions(self) -> list[RegionCheckpoint]:
        cfg = self.settings.filter
        return [] if cfg is None else cfg.load_regions()