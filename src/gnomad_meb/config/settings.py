"""Pydantic settings models for gnomad-meb.

Every former ``@dataclass`` settings class is expressed here as a pydantic
model with the same public field names and nesting, so consumers keep reading
``cfg.<section>.<field>``. The top-level :class:`Settings` is the
``pydantic_settings.BaseSettings`` root; the sections below it are plain
``BaseModel`` types.

Environment names are derived from field paths under a single fixed
``GMEB_`` prefix, with ``__`` separating nested sections::

    GMEB_S3__ACCESS_KEY
    GMEB_VQSR__SCATTER
    GMEB_RESOURCES__REFERENCE

The resolution entry points live in :mod:`gnomad_meb.config`.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    computed_field,
    field_validator,
    model_validator,
)
from pydantic_core import InitErrorDetails, PydanticCustomError
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from gnomad_meb.constants import (
    DEFAULT_FIELDS,
    DEFAULT_REFERENCE_GENOME,
    REFERENCE_GENOMES,
)
from gnomad_meb.lib.checkpoint.status import StalenessPolicy
from gnomad_meb.logging import get_log_level

if TYPE_CHECKING:
    from gnomad_meb.s3 import S3Provider

#: The sentinel used for S3 credentials that were never configured.
INVALID_TOKEN = "<INVALID>"

#: The single environment-variable prefix every setting is read under.
ENV_PREFIX = "GMEB_"

#: The delimiter between nested sections in environment-variable names.
ENV_NESTED_DELIMITER = "__"

#: ``<repo>/src/static``; the historical location of the fixed VQSR header.
_SRC_STATIC = Path(__file__).resolve().parents[3] / "static"

#: The package directory, used to locate the shipped VQSR/GWAS wrappers.
_PACKAGE_DIR = Path(__file__).resolve().parents[1]

#: The config file the next :class:`Settings` construction should read.
#:
#: ``settings_customise_sources`` is a classmethod, so it cannot receive the
#: per-call config path as an argument; this context variable carries it from
#: :func:`gnomad_meb.config.resolve` into the settings source list.
current_config_file: ContextVar[Path | None] = ContextVar(
    "gnomad_meb_config_file", default=None
)

_s3_provider: S3Provider | None = None

#: The three files the loftee plugin expects inside ``loftee_data``.
_LOFTEE_DATA_FILES = (
    "gerp_conservation_scores.homo_sapiens.GRCh38.bw",
    "human_ancestor.fa.gz",
    "loftee.sql",
)


def _field_error(loc: tuple[str, ...], message: str) -> InitErrorDetails:
    """A validation error bound to a field path, so callers see which failed."""
    return InitErrorDetails(
        type=PydanticCustomError("value_error", message),
        loc=loc,
        input=None,
    )


def _is_file(path: Path) -> bool:
    return path.is_file()


def _vep_path_errors(vep: VepAnnotationResources) -> list[InitErrorDetails]:
    """Filesystem probes for a ``[annotation.vep]`` section, naming each field."""
    errors: list[InitErrorDetails] = []
    prefix = ("annotation", "vep")

    def check_file(name: str, path: Path | None) -> None:
        if path is not None and not _is_file(path):
            errors.append(
                _field_error(
                    (*prefix, name),
                    f"missing file for vep resource {name}: {path}",
                )
            )

    check_file("alpha_missense", vep.alpha_missense)

    if vep.loftee_base is not None and not (
        vep.loftee_base.exists() and vep.loftee_base.is_dir()
    ):
        errors.append(
            _field_error(
                (*prefix, "loftee_base"),
                f"invalid loftee base directory: {vep.loftee_base}",
            )
        )
    if vep.loftee_data is not None:
        missing = [
            name for name in _LOFTEE_DATA_FILES if not (vep.loftee_data / name).exists()
        ]
        if missing:
            errors.append(
                _field_error(
                    (*prefix, "loftee_data"),
                    f"invalid loftee data path, missing {missing}: {vep.loftee_data}",
                )
            )

    check_file("splice_ai", vep.splice_ai)
    if vep.splice_ai is not None and not _is_file(vep.splice_ai_indel):
        errors.append(
            _field_error(
                (*prefix, "splice_ai"),
                f"missing splice ai indel vcf: {vep.splice_ai_indel}",
            )
        )

    check_file("cadd", vep.cadd)
    check_file("cadd_indel", vep.cadd_indel)
    check_file("utr_annotator", vep.utr_annotator)
    check_file("revel", vep.revel)
    check_file("phylop", vep.phylop)

    return errors


class _Section(BaseModel):
    """Base for a settings section nested under :class:`Settings`."""

    model_config = ConfigDict(extra="ignore")


class S3Config(_Section):
    access_key: str = INVALID_TOKEN
    secret_key: str = INVALID_TOKEN
    endpoint_url: str = "https://s3.intra.imbie"
    verify: bool = True
    storage_class: str = "standard_ia"
    s3_uri_schemes: list[str] = Field(default_factory=lambda: ["s3", "s3a"])

    @field_validator("access_key", "secret_key")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        if value.strip() == "":
            raise ValueError("must not be empty or whitespace")
        return value


class Resources(_Section):
    calling_intervals: Path
    telomeres_and_centromeres: Path
    lcr_intervals: Path
    seg_dup_intervals: Path
    gnomad_pca_loadings: Path
    gnomad_pca_model: Path
    gnomad_v4: Path
    reference: Path

    def to_dict(self) -> dict[str, str]:
        return {k: str(v) for k, v in self.model_dump().items()}


class VqsrResources(_Section):
    hapmap: Path
    omni: Path
    thousand_genome: Path
    dbsnp: Path
    mills: Path
    axiom_poly: Path

    def to_dict(self) -> dict[str, str]:
        return {k: str(v) for k, v in self.model_dump().items()}


class VqsrConfig(_Section):
    wrapper_dir: Path = Field(
        default_factory=lambda: (_PACKAGE_DIR / "vqsr" / "wrapper").resolve()
    )
    log_dir: Path = Path("./logs")
    resources: VqsrResources | None = None
    output_dir: Path = Path("./output")
    #: Extra arguments forwarded verbatim to the VQSR Snakemake invocation, e.g.
    #: a cluster profile and executor. Array in TOML
    #: (``[vqsr] snakemake = ["--profile", "slurm"]``) or JSON in the environment
    #: (``GMEB_VQSR__SNAKEMAKE='["--profile", "slurm"]'``). Read by the composed
    #: ``gnomad-workflow`` pipeline; the standalone ``vqsr pipeline`` command may
    #: still pass the same arguments explicitly, which take precedence.
    snakemake: list[str] = Field(default_factory=list)


class GwasResponseColumn(_Section):
    """One study-metadata CSV column used directly as a response.

    ``kind`` selects the derivation:

    - ``binary`` -- the column's values as-is (``0``/``1``; excluded rows stay
      missing).
    - ``quantitative`` -- the column's values as-is (a continuous trait; missing
      values are allowed).
    - ``set`` -- a set-valued cell; every distinct value in
      ``(column - exclude)`` becomes a case-only indicator response named
      ``snake_case(value)``.

    ``metadata_field`` names the source column in the external metadata CSV and
    defaults to ``name`` (which is also the source ``study_md`` column for a
    ``set`` column).
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        description=(
            "The response name for ``binary``/``quantitative`` columns, or the "
            "source column label for a ``set`` column."
        )
    )
    kind: Literal["binary", "quantitative", "set"] = Field(
        default="binary",
        description="How the metadata column is turned into response variable(s).",
    )
    metadata_field: str | None = Field(
        default=None,
        description="The source column in the metadata CSV; defaults to ``name``.",
    )
    exclude: list[str] = Field(
        default_factory=list,
        description="Values removed from a ``set`` column's membership.",
    )

    @model_validator(mode="after")
    def _exclude_is_set_only(self) -> GwasResponseColumn:
        if self.kind != "set" and self.exclude:
            raise ValueError("'exclude' is only valid for a 'set' response column")
        return self

    @property
    def field(self) -> str:
        """The source column in the study-metadata CSV.

        ``metadata_field`` when set, else ``name`` -- the single definition of
        which CSV/``study_md`` column a response column reads.
        """
        return self.metadata_field or self.name


class GwasResponses(_Section):
    """The phenotype table's response variables."""

    model_config = ConfigDict(extra="forbid")

    case: str = Field(
        default="case",
        description="The name of the always-emitted case/control response.",
    )
    case_metadata_field: str = Field(
        default="cohort",
        description="Name of the metadata column containing case / control status.",
    )
    case_only: bool = Field(
        default=True,
        description=(
            "When true (the default), controls are missing for ``set`` responses; "
            "this documents the filter the export actually applies."
        ),
    )
    columns: list[GwasResponseColumn] = Field(
        default_factory=list,
        description=(
            "The metadata-sourced responses: binary/quantitative columns used "
            "directly, and set-valued columns whose (column - exclude) union is "
            "the case-only membership expression."
        ),
    )


class GwasCovariate(_Section):
    """One covariate regenie is adjusted for.

    ``name`` is the field the covariate is resolved from: for a
    metadata-sourced covariate it is the (rebuilt) ``study_md`` column name, and
    otherwise it is a (possibly dotted) MT column expression. ``metadata_field``
    names the source column in the external metadata CSV; the one exception is
    ``name="age"``, declared as ``metadata_field="birthdate"`` and rewritten on
    import. A ``sex`` covariate is encoded ``1``/``2`` when the covariates are
    built. ``indices`` selects subindices of an array field, fanning out into one
    output column per index.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    """Name of the field. Used by hail to retrieve it from the annotated table."""
    metadata_field: str | None = None
    """Name of the field in the metadata table."""
    indices: list[int] | None = None
    """Subindices of the field to use if it is an array (default = all)."""


class GwasRegion(_Section):
    """One region definition the regenie burden tables fan out over."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["gene", "window"] = Field(
        description="The region kind: per-gene groups or fixed-size windows."
    )
    size: int | None = Field(
        default=None,
        description="The window size in base pairs; required for ``window``.",
    )

    @model_validator(mode="after")
    def _window_requires_size(self) -> GwasRegion:
        if self.kind == "window" and self.size is None:
            raise ValueError("a window region requires a 'size'")
        return self


class GwasConfig(_Section):
    wrapper_dir: Path = Field(
        default_factory=lambda: (_PACKAGE_DIR / "regenie" / "wrapper").resolve()
    )
    log_dir: Path = Path("./logs")
    output_dir: Path = Path("./output")
    #: Extra arguments forwarded verbatim to the GWAS Snakemake invocation, e.g.
    #: a cluster profile and executor. See :attr:`VqsrConfig.snakemake` for the
    #: accepted forms; explicit ``gwas pipeline`` arguments take precedence.
    snakemake: list[str] = Field(default_factory=list)
    #: The target number of variants per regenie chunk. The run stage derives the
    #: chunk count from each dataset's recorded variant count at run time
    #: (``ceil(n_variants / variants_per_split)``), so it tracks dataset size.
    variants_per_split: int = 100_000
    #: The minimum variant count a table needs to be exported and tested. Tables
    #: below this floor (or with no recorded count) are dropped at inventory
    #: selection, before any stage runs: a SNP-filtered dataset this small cannot
    #: be chunked for level-0 regenie (``--split-l0`` yields fewer non-empty
    #: chunks than ``n_chunks``), which otherwise fails the whole DAG. The default
    #: targets WGS, where any real filter yields far more than this; small, array
    #: or test inputs lower it.
    min_n_snps: int = 1000
    min_case_count: int = 10
    min_mac: int = 3
    min_geno_frac: float = 0.01
    min_maf: float = 0.01
    hwe: float = 1e-15
    snps_only: bool = True

    #: The external study metadata CSV; the one genuinely new external input.
    #: Required when a ``[gwas]`` section is configured.
    metadata_csv: Path = Field(
        description="Path to the external study metadata CSV, keyed by sample_id."
    )
    case_cohort: str = Field(
        default="",
        description="The ``cohort`` value treated as the case.",
    )
    control_cohorts: list[str] = Field(
        default_factory=list,
        description="The ``cohort`` values treated as controls.",
    )
    drop_cohorts: list[str] = Field(
        default_factory=list,
        description="The ``cohort`` values dropped from the MT columns.",
    )
    rare_variants_maf: float = Field(
        default=0.01,
        description="The NFE/adj AF threshold below which a variant is rare.",
    )
    min_cadd: float = Field(
        default=20.0,
        description="The CADD phred threshold for deleterious missense.",
    )
    responses: GwasResponses = Field(
        default_factory=GwasResponses,
        description="The response-variable model.",
    )
    intercept: bool = Field(
        default=True,
        description="Whether to emit a constant intercept covariate (``y0``).",
    )
    covariates: list[GwasCovariate] = Field(
        default_factory=list,
        description="The covariates regenie is adjusted for, in output order.",
    )
    regions: list[GwasRegion] = Field(
        default_factory=lambda: [GwasRegion(kind="gene")],
        description="The region specs the export fans out over; per-gene default.",
    )


class BrowserExportConfig(_Section):
    """The ``[browser_export]`` section: the browser-export stage's knobs.

    ``fields`` selects the optional passthrough annotations populated when
    present on the input. The browser's mandatory columns are always written
    regardless, and a listed field absent on the input is dropped.
    """

    fields: list[str] = Field(
        default_factory=lambda: list(DEFAULT_FIELDS),
        description="Optional passthrough top-level fields to populate when present.",
    )


class SampleQcHardFilterCutoffs(_Section):
    max_n_singletons: float = 50_000.0
    max_r_het_hom_var: float = 10.0
    min_bases_dp_over_1: float = 5e7
    min_bases_dp_over_20: float = 4e7
    max_contamination_estimate: float = 0.05
    min_cov: int = 15
    min_qc_mt_adj_callrate: float = 0.95
    max_kinship: float = 0.2
    #: Flag ``ambiguous_sex``/``sex_aneuploidy`` in the hard filters.
    include_sex_filter: bool = True


class XPloidyCutoffs(_Section):
    """Pinned chromosome X ploidy cutoffs.

    Field names mirror the ``x_ploidy_cutoffs`` globals struct exported by
    :func:`gnomad_meb.lib.sample_qc.utils.infer_sex_karyotype`.
    """

    upper_cutoff_X: float
    lower_cutoff_XX: float
    upper_cutoff_XX: float
    lower_cutoff_XXX: float

    def as_tuple(self) -> tuple[float, tuple[float, float], float]:
        """The positional shape ``get_ploidy_cutoffs``/``get_sex_expr`` expect."""
        return (
            self.upper_cutoff_X,
            (self.lower_cutoff_XX, self.upper_cutoff_XX),
            self.lower_cutoff_XXX,
        )


class YPloidyCutoffs(_Section):
    """Pinned chromosome Y ploidy cutoffs (cf. :class:`XPloidyCutoffs`)."""

    lower_cutoff_Y: float
    upper_cutoff_Y: float
    lower_cutoff_YY: float

    def as_tuple(self) -> tuple[tuple[float, float], float]:
        """The positional shape ``get_ploidy_cutoffs``/``get_sex_expr`` expect."""
        return ((self.lower_cutoff_Y, self.upper_cutoff_Y), self.lower_cutoff_YY)


class ChrXFracHomAltCutoffs(_Section):
    """Pinned chrX hom-alt fraction cutoffs (cf. :class:`XPloidyCutoffs`)."""

    lower_cutoff_more_than_one_X: float
    upper_cutoff_more_than_one_X: float
    lower_cutoff_single_X: float

    def as_tuple(self) -> tuple[tuple[float, float], float]:
        """The positional shape the hom-alt cutoff helpers expect."""
        return (
            (self.lower_cutoff_more_than_one_X, self.upper_cutoff_more_than_one_X),
            self.lower_cutoff_single_X,
        )


class SampleQcConfig(_Section):
    gq_bins: list[int] = Field(default_factory=lambda: [0, 20, 60])
    dp_bins: list[int] = Field(default_factory=lambda: [0, 1, 10, 20, 30])
    adj_gq: int = 20
    adj_dp: int = 10
    adj_ab: float = 0.2
    haploid_adj_dp: int = 5
    min_site_callrate: float = 0.95
    callrate_min_af: float = 0.0001
    min_individual_maf: float = 0.01
    #: Hail's ``min_kinship`` pruning floor. ``None`` disables pruning (Hail's
    #: default is ``-inf``), writing every pair including negatives/self-pairs.
    #: O(n^2) rows: keep ``None`` for small cohorts, set a floor in production.
    min_kinship: float | None = None
    relatedness_king_kinship: float = 0.1
    gnomad_num_pcs: int = 20
    gnomad_pca_min_prob: float = 0.50
    f_stat_cutoff: float = 0.5
    aaf_threshold: float = 0.001
    relatedness_n_pcs: int = 10
    hard_filter: SampleQcHardFilterCutoffs = Field(
        default_factory=SampleQcHardFilterCutoffs
    )
    filter_lcr_and_seg_dup: bool = False
    normalization_contig: str = "chr20"
    sex_contigs: tuple[str, ...] = ("chrX", "chrY")
    #: Number of standard deviations for XX/XY ploidy cutoffs.
    normal_ploidy_cutoff: int = 5
    #: Number of standard deviations for sex-aneuploidy ploidy cutoffs.
    aneuploidy_cutoff: int = 6
    #: Optional pinned reference cutoffs. When all three are set,
    #: ``infer_sex_karyotype`` uses them instead of deriving cutoffs from the
    #: screened cohort (the f-stat grouping/reporting is unchanged).
    x_ploidy_cutoffs: XPloidyCutoffs | None = None
    y_ploidy_cutoffs: YPloidyCutoffs | None = None
    chr_x_frac_hom_alt_cutoffs: ChrXFracHomAltCutoffs | None = None

    @model_validator(mode="after")
    def _pinned_cutoffs_are_all_or_none(self) -> SampleQcConfig:
        pinned = (
            self.x_ploidy_cutoffs,
            self.y_ploidy_cutoffs,
            self.chr_x_frac_hom_alt_cutoffs,
        )
        if any(cutoff is not None for cutoff in pinned) and not all(
            cutoff is not None for cutoff in pinned
        ):
            raise ValueError(
                "x_ploidy_cutoffs, y_ploidy_cutoffs and chr_x_frac_hom_alt_cutoffs "
                "must be set together (all or none)"
            )
        return self


class VariantQcSensitivityCutoffs(_Section):
    """Per-mode GATK truth-sensitivity targets for ``ApplyVQSR``.

    Field names mirror the ``variant_qc_cutoffs`` global's per-mode
    ``target_truth_sensitivity``: the sensitivity level each mode is filtered
    at (SNP ``99.7``, INDEL ``99.0``).

    Note
    ----
    SNP and Indel default sensitivity filter values from populationgenomics pipeline:
        https://github.com/populationgenomics/production-pipelines/blob/1132f9c133b8f734b482dbdaf5c6206c4f7d3af0/configs/defaults/rd_combiner.toml#L77
    """

    snp: float = 99.7
    indel: float = 99.0


class VariantQcVqsrParams(_Section):
    """The ``VariantRecalibrator`` tunables annotated as the ``vqsr_params`` global.

    The training resources are deliberately excluded: they are resources, not
    parameters, and their paths are host-specific.
    """

    allele_specific: bool = True
    max_gaussians: int = 4
    variants_to_sample: int = 10
    snp_tranche_values: list[float] = Field(
        default_factory=lambda: [
            100.0,
            99.95,
            99.9,
            99.8,
            99.6,
            99.5,
            99.4,
            99.3,
            99.0,
            98.0,
            97.0,
            90.0,
        ]
    )
    indel_tranche_values: list[float] = Field(
        default_factory=lambda: [
            100.0,
            99.95,
            99.9,
            99.5,
            99.0,
            97.0,
            96.0,
            95.0,
            94.0,
            93.5,
            93.0,
            92.0,
            91.0,
            90.0,
        ]
    )


class VariantQcConfig(_Section):
    import_header_fixed: Path = _SRC_STATIC / "vqsr.header.fixed"
    #: The truth-sensitivity targets the VQSR step filters each mode at; lifted
    #: out of the ApplyVQSR wrapper and annotated as ``variant_qc_cutoffs``.
    target_truth_sensitivity: VariantQcSensitivityCutoffs = Field(
        default_factory=VariantQcSensitivityCutoffs
    )
    #: The VariantRecalibrator tunables (excluding its reference resources),
    #: passed to the VQSR Snakemake config and annotated as ``vqsr_params``.
    vqsr_params: VariantQcVqsrParams = Field(default_factory=VariantQcVqsrParams)


class AnnotationResources(_Section):
    clinvar_vcf: Path
    sample_qc: str | None = None
    variant_qc: str | None = None


class VepAnnotationResources(_Section):
    vep: Path
    version: int = 110
    cache_dir: Path
    alpha_missense: Path | None = None
    loftee_data: Path | None = None
    loftee_base: Path | None = None
    cadd: Path | None = None
    cadd_indel: Path | None = None
    splice_ai: Path | None = None
    utr_annotator: Path | None = None
    revel: Path | None = None
    phylop: Path | None = None

    @model_validator(mode="after")
    def _validate_pairs(self) -> VepAnnotationResources:
        errors: list[InitErrorDetails] = []
        if (self.loftee_base is None) != (self.loftee_data is None):
            missing = "loftee_data" if self.loftee_data is None else "loftee_base"
            errors.append(
                _field_error(
                    (missing,),
                    "loftee requires both 'loftee_base' and 'loftee_data' "
                    "(both or neither)",
                )
            )
        if (self.cadd is None) != (self.cadd_indel is None):
            missing = "cadd_indel" if self.cadd_indel is None else "cadd"
            errors.append(
                _field_error(
                    (missing,),
                    "CADD requires both 'cadd' and 'cadd_indel' (both or neither)",
                )
            )
        if errors:
            raise ValidationError.from_exception_data(self.__class__.__name__, errors)
        return self

    @property
    def splice_ai_indel(self) -> Path:
        return self.splice_ai.with_name(self.splice_ai.name.replace("snv", "indel"))

    @property
    def loftee_files(self) -> dict[str, Path]:
        if self.loftee_data is None:
            return {}

        return {
            "gerp_bigwig": self.loftee_data / _LOFTEE_DATA_FILES[0],
            "human_ancestor_fa": self.loftee_data / _LOFTEE_DATA_FILES[1],
            "conservation_file": self.loftee_data / _LOFTEE_DATA_FILES[2],
        }


class AnnotationConfig(_Section):
    cohort: str | None = None
    cohort_table: str | None = None
    caller_name: str = "parabricks"
    caller_version: str = "4.4.0"
    resources: AnnotationResources | None = None
    vep: VepAnnotationResources | None = None
    vep_config_path: str | None = None
    group_by_cohort: bool = True
    annotate_info_fields: bool = True

    def todict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in type(self).model_fields
            if name != "resources"
        }


class FilterResources(_Section):
    constraints: str | None = None

    def todict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in type(self).model_fields}


class FilterCheckpoint(_Section):
    """One declared filter's knobs, validated at config resolution."""

    model_config = ConfigDict(extra="forbid")

    name: str
    min_gnomad_af: float | None = None
    max_gnomad_af: float | None = None
    filter_snps: bool = False
    #: The gnomAD ancestry group the frequency sub-filter reads; ``None`` means
    #: the sub-filter only runs when an AF bound or ``filter_snps`` drives it.
    ancestry_group: str | None = None
    min_z_score: float | None = None
    min_pLI: float | None = None
    min_cadd: float | None = None
    impact: set[str] | None = None
    filter_clinvar_pathogenic: bool = False
    filter_splice_variants: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FilterCheckpoint:
        """Build one declaration from a mapping (env override or JSON)."""
        return cls.model_validate(data)

    @classmethod
    def load_filters(cls, entries: Iterable[Any]) -> list[FilterCheckpoint]:
        """Normalize declarations (models or mappings) to checkpoints."""
        return [
            entry if isinstance(entry, cls) else cls.from_dict(dict(entry))
            for entry in entries
        ]

    def gnomad_af_conditions(self):
        conditions = set()
        if self.min_gnomad_af is not None:
            conditions.add("min_af")
        if self.max_gnomad_af is not None:
            conditions.add("max_af")
        if self.filter_snps:
            conditions.add("snps")
        return frozenset(conditions)


class RegionCheckpoint(_Section):
    """One declared region's knobs, validated at config resolution."""

    model_config = ConfigDict(extra="forbid")

    name: str
    intervals: str | None = None
    reference_genome: REFERENCE_GENOMES = DEFAULT_REFERENCE_GENOME
    keep: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RegionCheckpoint:
        """Build one declaration from a mapping (env override or JSON)."""
        return cls.model_validate(data)

    @classmethod
    def load_regions(cls, entries: Iterable[Any]) -> list[RegionCheckpoint]:
        """Normalize declarations (models or mappings) to checkpoints."""
        return [
            entry if isinstance(entry, cls) else cls.from_dict(dict(entry))
            for entry in entries
        ]


class FilterConfig(_Section):
    filters: list[FilterCheckpoint] = Field(default_factory=list)
    regions: list[RegionCheckpoint] = Field(default_factory=list)
    resources: FilterResources | None = None
    min_inbreeding_coef: float = -0.3
    min_high_quality_calls: int = 1
    apply_sample_filters: bool = True
    sample_filters_ignore: set[str] | None = None

    def load_filters(self) -> list[FilterCheckpoint]:
        return FilterCheckpoint.load_filters(self.filters)

    def load_regions(self) -> list[RegionCheckpoint]:
        return RegionCheckpoint.load_regions(self.regions)

    def todict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in type(self).model_fields}


class HailConfig(_Section):
    s3_timeout_ms: int = 30_000
    s3_request_timeout_ms: int = 120_000
    s3_max_connections: int = 20_000
    memory_overhead: int = 2_048
    max_failures: int = 3
    additional_packages: list[str] = Field(
        default_factory=lambda: [
            "org.apache.hadoop:hadoop-client:3.3.4",
            "org.apache.hadoop:hadoop-aws:3.3.4",
        ]
    )
    tmp_dir: str | None = None
    partitions_per_core: int = 10
    driver_memory_gb: int | None = None
    driver_memory_fraction: float = 1.0
    driver_max_memory_gb: int = 1_024
    driver_memory_overhead: int = 4_096
    driver_max_result_size: int = 4_096
    #: Verbosity of hail's java logs (log4j + the file bridge). Defaults to
    #: ``warning``: hail INFO is extremely chatty and opt-in via this setting or
    #: ``GMEB_HAIL__LOG_LEVEL``.
    log_level: str = "warning"

    @field_validator("log_level", mode="before")
    @classmethod
    def _validate_log_level(cls, value: Any) -> str:
        """Normalise and check against the known levels (mirrors ``Settings.verbose``)."""
        level = str(value).lower()
        if level not in get_log_level():
            raise ValueError(f"must be one of {sorted(get_log_level())}")
        return level


class Settings(BaseSettings):
    """The fully-resolved, serializable pipeline settings."""

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_nested_delimiter=ENV_NESTED_DELIMITER,
        extra="ignore",
    )

    project: str = Field(default_factory=lambda: f"gnomad_meb-{uuid4()!s}")
    scratch_dir: str = Field(default_factory=lambda: os.getenv("TMPDIR", "/tmp"))
    checkpoint: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def run_id(self) -> str:
        """A stable, collision-free id for one logical run.

        Composed of ``project``, the Slurm job id (when set) and the Slurm array
        task id (when set). The array task id is appended only when present, so
        an array job's tasks each get a distinct id -- every array task shares
        the parent ``SLURM_JOB_ID``, so without it they could clobber each
        other's temp directories. A non-array job simply omits it.

        This is a ``computed_field``: it is written into a serialized manifest
        for provenance, but is *ignored* when settings are re-loaded, so
        ``--manifest`` recomputes it from the current environment -- giving each
        array job its own temp directories.
        """
        parts = [self.project]
        job_id = os.getenv("SLURM_JOB_ID")
        if job_id:
            parts.append(job_id)
        array_task_id = os.getenv("SLURM_ARRAY_TASK_ID")
        if array_task_id:
            parts.append(f"array-{array_task_id}")
        return "-".join(parts)

    #: The reference genome the run is built against. Every resource's target
    #: genome comes from here, so a run on a genome that does not match an
    #: external source is surfaced at build. Defaults to GRCh38, so existing
    #: configs are unaffected.
    reference_genome: REFERENCE_GENOMES = DEFAULT_REFERENCE_GENOME
    n_cores: int | None = None
    #: The log level, recorded with the task so verbosity is part of the
    #: serialized description. Set from ``-v/--verbose``.
    verbose: str = "info"

    #: Whether the filesystem probes (path existence, file-vs-directory) run.
    #: Type coercion, list/tuple coercion and the cross-field constraints always
    #: run; only the probes are skipped when this is ``False``.
    validate_paths: bool = True

    #: How a reused checkpoint's provenance is verified. ``completeness``
    #: reproduces the historical data-presence-only behaviour; ``source``
    #: additionally recomputes a step whose generating source changed; git/version
    #: are opt-in via ``source_or_git``.
    staleness_policy: StalenessPolicy = StalenessPolicy.COMPLETENESS

    s3: S3Config = Field(default_factory=S3Config)
    vqsr: VqsrConfig | None = None
    resources: Resources | None = None
    sample_qc: SampleQcConfig | None = None
    variant_qc: VariantQcConfig | None = None
    hail: HailConfig = Field(default_factory=HailConfig)
    annotation: AnnotationConfig | None = None
    gwas: GwasConfig | None = None
    filter: FilterConfig | None = None
    #: The browser-export stage's knobs. Non-optional so ``browser run`` works
    #: with no ``[browser_export]`` section and defaults to ``DEFAULT_FIELDS``.
    browser_export: BrowserExportConfig = Field(default_factory=BrowserExportConfig)

    @model_validator(mode="after")
    def _validate_resource_paths(self) -> Settings:
        """Probe every configured resource path, naming the field on failure.

        Only the filesystem-dependent checks live here, so ``validate_paths=False``
        skips them while the section-level cross-field constraints still fire.
        """
        if not self.validate_paths:
            return self

        errors: list[InitErrorDetails] = []

        if self.resources is not None:
            for name in Resources.model_fields:
                path = getattr(self.resources, name)
                if not path.exists():
                    errors.append(
                        _field_error(
                            ("resources", name),
                            f"missing path for resources {name}: {path}",
                        )
                    )

        if self.vqsr is not None and self.vqsr.resources is not None:
            for name in VqsrResources.model_fields:
                path = getattr(self.vqsr.resources, name)
                if not _is_file(path):
                    errors.append(
                        _field_error(
                            ("vqsr", "resources", name),
                            f"missing file for vqsr resources {name}: {path}",
                        )
                    )

        if self.variant_qc is not None and (
            "import_header_fixed" in self.variant_qc.model_fields_set
        ):
            path = self.variant_qc.import_header_fixed
            if not _is_file(path):
                errors.append(
                    _field_error(
                        ("variant_qc", "import_header_fixed"),
                        f"invalid vqsr import header path: {path}",
                    )
                )

        if self.annotation is not None and self.annotation.resources is not None:
            path = self.annotation.resources.clinvar_vcf
            if not _is_file(path):
                errors.append(
                    _field_error(
                        ("annotation", "resources", "clinvar_vcf"),
                        f"missing file for annotation resources clinvar_vcf: {path}",
                    )
                )

        if self.annotation is not None and self.annotation.vep is not None:
            errors.extend(_vep_path_errors(self.annotation.vep))

        if errors:
            raise ValidationError.from_exception_data("Settings", errors)
        return self

    @field_validator("verbose", mode="before")
    @classmethod
    def _validate_verbose(cls, value: Any) -> str:
        """Normalise and check the log level against the known levels."""
        level = str(value).lower()
        if level not in get_log_level():
            raise ValueError(f"must be one of {sorted(get_log_level())}")
        return level

    @property
    def s3fs(self) -> S3Provider:
        global _s3_provider
        from gnomad_meb.s3 import S3ConnectionOptions, S3Provider

        if _s3_provider is None:
            _s3_provider = S3Provider(
                S3ConnectionOptions(
                    self.s3.access_key,
                    self.s3.secret_key,
                    self.s3.endpoint_url,
                    self.s3.verify,
                )
            )
        return _s3_provider

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        """Priority order: overrides > environment > config file > defaults."""
        sources = [init_settings, env_settings]
        config_file = current_config_file.get()
        if config_file is not None:
            sources.append(
                TomlConfigSettingsSource(settings_cls, toml_file=config_file)
            )
        return tuple(sources)
