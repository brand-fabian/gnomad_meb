"""Shared hail expressions and helpers for the GWAS stages.

Everything study-specific the notebook hardcoded is parameterized here and
built from the ``[gwas]`` config: variant classification, PLINK reference
normalization, response derivations, region helpers and study-metadata loading.
Hail is imported lazily inside the function bodies, so importing this module
never initializes the hail stack.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas

if TYPE_CHECKING:
    import hail as hl


# --- variant identity, impact and classification ----------------------------

def resolve_field_from_string(
    expr: hl.Table | hl.MatrixTable | hl.expr.Expression,
    name: str | None = None,
) -> hl.expr.Expression:
    if name is None:
        return expr

    if "." in name:
        next, _, rest = name.partition(".")
        return resolve_field_from_string(expr[next], rest)
    else:
        return expr[name]


def variant_id_expr(t: hl.Table | hl.MatrixTable) -> hl.expr.Expression:
    """``contig:pos:ref:alt`` with any ``chr`` prefix stripped.

    The load stage keys GWAS rows by exactly this shape (``id_cols =
    ["variant_id"]``), so it is a cross-stage contract.
    """
    import hail as hl

    return hl.format(
        "%s:%d:%s:%s",
        hl.if_else(
            t.locus.contig.startswith("chr"),
            hl.str(t.locus.contig[3:]),
            hl.str(t.locus.contig),
        ),
        t.locus.position,
        t.alleles[0],
        t.alleles[1],
    )


def str_to_expr(mt: hl.MatrixTable, expr: str) -> hl.expr.Expression:
    """Resolve a dotted config string (``study_md.sex``) against a matrix table."""
    return mt[expr]


def get_impact_expr(mt: hl.MatrixTable) -> hl.expr.Expression:
    """The canonical VEP impact, lowercased, ``"<none>"`` when undefined."""
    import hail as hl

    impact_base_expr = mt.vep.worst_csq_for_variant_canonical.impact
    return (
        hl.case()
        .when(~hl.is_defined(impact_base_expr), "<none>")
        .default(impact_base_expr.lower())
    )


def get_freq_expr(mt: hl.MatrixTable) -> hl.expr.Expression:
    """The NFE/adj gnomAD frequency, defaulting to ``0.0`` when undefined."""
    import hail as hl

    gnomad_af_idx = hl.eval(mt.gnomad_freq_meta).index(
        {"gen_anc": "nfe", "group": "adj"}
    )
    return hl.if_else(
        hl.is_defined(mt.gnomad_freq[gnomad_af_idx].AF),
        mt.gnomad_freq[gnomad_af_idx].AF,
        hl.float64(0.0),
    )


def get_weight_expr(freq_expr: hl.expr.Expression) -> hl.expr.Expression:
    """The burden weight ``dbeta(aaf, 1.0, 25.0) ** 2``."""
    import hail as hl

    return hl.dbeta(freq_expr, 1.0, 25.0) ** 2


def get_variant_classification(
    mt: hl.MatrixTable,
    clinvar_expr: hl.expr.Expression | None = None,
    rare_variants_maf: float = 0.01,
    lof_expr: hl.expr.Expression | None = None,
    missense_expr: hl.expr.Expression | None = None,
    min_cadd: float = 20.0,
    freq_expr: hl.expr.Expression | None = None,
):
    """Classify each variant (first match wins) and report the class counters.

    Precedence: ``clinvar`` → ``lof`` (high impact ∧ NFE-AF ≤ MAF) →
    ``deleterious_missense`` (moderate impact ∧ NFE-AF ≤ MAF ∧ CADD ≥ min_cadd)
    → ``rare`` (NFE-AF ≤ MAF) → ``common``.
    """
    import hail as hl

    if freq_expr is None:
        freq_expr = get_freq_expr(mt)

    if clinvar_expr is None:
        clinvar_classifications = hl.literal(
            {
                "Likely_pathogenic",
                "Likely_pathogenic|protective",
                "Pathogenic",
                "Pathogenic/Likely_pathogenic",
                "Pathogenic/Pathogenic",
                "Pathogenic|risk_factor",
            }
        )
        clinvar_expr = hl.is_defined(mt.clinvar.clinical_significance) & (
            hl.len(
                hl.set(mt.clinvar.clinical_significance).intersection(
                    clinvar_classifications
                )
            )
            > 0
        )

    if lof_expr is None:
        lof_expr = hl.literal({"high"}).contains(get_impact_expr(mt))

    if missense_expr is None:
        missense_expr = hl.literal({"moderate"}).contains(get_impact_expr(mt))

    gnomad_af_expr = freq_expr <= rare_variants_maf
    cadd_expr = mt.in_silico_predictors.cadd.phred >= min_cadd

    variant_classification_expr = (
        hl.case(missing_false=True)
        .when(clinvar_expr, "clinvar")
        .when(lof_expr & gnomad_af_expr, "lof")
        .when(missense_expr & gnomad_af_expr & cadd_expr, "deleterious_missense")
        .when(gnomad_af_expr, "rare")
        .default("common")
    )

    mt = mt.annotate_rows(
        __variant_class=variant_classification_expr,
        __clinvar=clinvar_expr,
        __lof=lof_expr,
        __missense=missense_expr,
        __rare=gnomad_af_expr,
        __cadd=cadd_expr,
    )
    variant_info = mt.aggregate_rows(
        hl.struct(
            variant_class=hl.agg.counter(mt.__variant_class),
            clinvar=hl.agg.counter(mt.__clinvar),
            lof=hl.agg.counter(mt.__lof),
            missense=hl.agg.counter(mt.__missense),
            rare=hl.agg.counter(mt.__rare),
            cadd=hl.agg.counter(mt.__cadd),
        )
    )
    return variant_classification_expr, variant_info


# --- filter metadata --------------------------------------------------------


def to_hail_literal_value(value: Any) -> Any:
    """Reshape parsed filter metadata so ``hl.literal`` can type it.

    Filter metadata read from a dataset's ``.gwas.md.json`` is a plain Python
    tree of dicts, lists and scalars, and ``hl.literal`` cannot infer a type for
    two shapes it produces:

    * a ``None`` left by a parameter's JSON ``null`` default (the export records
      every parameter, set or not); and
    * a nested dict whose values only *partially* unify -- Hail first types a
      dict as ``dict<K, V>``, and when the values unify to a hole (``V = None``)
      it raises rather than falling back to a struct, even with no null in sight.

    Nulls are dropped and every dict is wrapped as an :class:`hail.Struct` so
    Hail takes the struct branch directly instead of the ambiguous dict one (a
    ``Struct()`` is also a valid ``struct{}``, unlike an empty plain dict). A
    list left empty by dropping nulls goes too: an empty array has no element
    type. The tree is otherwise returned unchanged (a provenance blob: dropping
    a null field loses no consumer-visible data).
    """
    import hail as hl

    if isinstance(value, dict):
        return hl.Struct(
            **{
                key: to_hail_literal_value(val)
                for key, val in value.items()
                if val is not None
            }
        )
    if isinstance(value, list):
        # An empty array has no element type, so a null-only (or empty) list is
        # dropped rather than handed to hail. Wrapped dicts are never empty-so-
        # far as hail is concerned (``Struct()`` is a valid ``struct{}``).
        items: list = []
        for val in value:
            if val is None:
                continue
            clean = to_hail_literal_value(val)
            if isinstance(clean, list) and not clean:
                continue
            items.append(clean)
        return items
    return value


# --- regenie annotation tables ----------------------------------------------


def get_effects_ht(
    mt: hl.MatrixTable,
    variant_classification_expr: hl.expr.Expression | None = None,
    key: Callable[[str], int] | None = None,
    variant_class_order: Iterable[str] = (),
) -> hl.Table:
    """The ``effect_masks`` table: ``Mask-<class>`` and ``-extended`` rows.

    The class order is fixed by :data:`constants.VARIANT_CLASS_ORDER` unless a
    caller overrides it, so the mask ids are stable across runs.
    """
    import hail as hl

    from gnomad_meb.lib.gwas.constants import VARIANT_CLASS_ORDER

    if variant_classification_expr is None:
        # Default branch: get_variant_classification already aggregates the full
        # variant_info (class counter plus the gate booleans), so reuse its
        # output directly.
        variant_classification_expr, variant_info = get_variant_classification(mt)
    else:
        # Caller-supplied class expression: only the class is known (the gate
        # booleans live in get_variant_classification's body), so variant_info
        # carries the class counter alone. Shapes therefore differ by design.
        mt = mt.annotate_rows(__class=variant_classification_expr)
        variant_info = hl.struct(
            variant_class=mt.aggregate_rows(hl.agg.counter(mt.__class)),
        )

    order = tuple(variant_class_order) or VARIANT_CLASS_ORDER
    if key is None:
        key = order.index

    mt = mt.annotate_rows(__class=variant_classification_expr)
    effects = sorted(
        mt.aggregate_rows(hl.agg.collect_as_set(mt.__class)),
        key=key,
    )

    effects_d = []
    for i, current in enumerate(effects):
        effects_d.append({"mask_id": f"Mask-{current}", "effects": current})
        effects_d.append(
            {
                "mask_id": f"Mask-{current}-extended",
                "effects": ",".join([*effects[:i], current]),
            }
        )

    ht = hl.Table.from_pandas(pandas.DataFrame(effects_d))
    return ht.annotate_globals(variant_info=variant_info)


def get_variant_sets_ht(mt: hl.MatrixTable, region: hl.expr.Expression) -> hl.Table:
    """The ``variant_sets`` table: ``region,contig,position,variants`` per region."""
    import hail as hl

    variant_sets_ht = mt.select_rows(
        contig=mt.locus.contig,
        position=mt.locus.position,
        region=region,
        variant=variant_id_expr(mt),
    ).rows()
    variant_sets_ht = variant_sets_ht.group_by(
        variant_sets_ht.contig, variant_sets_ht.region
    ).aggregate(
        position=hl.agg.min(variant_sets_ht.position),
        variants=hl.agg.collect(variant_sets_ht.variant),
    )
    variant_sets_ht = variant_sets_ht.key_by()
    variant_sets_ht = variant_sets_ht.filter(
        hl.len(variant_sets_ht.contig) > 0
    )
    variant_sets_ht = variant_sets_ht.annotate(
        contig=hl.if_else(
            variant_sets_ht.contig.startswith("chr"),
            hl.str(variant_sets_ht.contig[3:]),
            hl.str(variant_sets_ht.contig),
        )
    )
    variant_sets_ht.select("region", "contig", "position", "variants")
    return variant_sets_ht


def get_annotation_ht(
    mt: hl.MatrixTable,
    weight_expr: hl.expr.Expression,
    region: hl.expr.Expression,
    rare_variants_maf: float = 0.01,
    min_cadd: float = 20.0,
) -> hl.Table:
    """The ``annotation`` table: ``variant_id,region,effect,weight``."""
    variant_classification_expr, _ = get_variant_classification(
        mt, rare_variants_maf=rare_variants_maf, min_cadd=min_cadd
    )
    return mt.select_rows(
        variant_id=variant_id_expr(mt),
        region=region,
        effect=variant_classification_expr,
        weight=weight_expr,
    ).rows()


def get_aaf_ht(mt: hl.MatrixTable, freq_expr: hl.expr.Expression) -> hl.Table:
    """The ``aaf`` table: ``variant_id,aaf``."""
    return mt.select_rows(
        variant_id=variant_id_expr(mt),
        aaf=freq_expr,
    ).rows()


def get_phenotype_ht(
    mt: hl.MatrixTable,
    iid_expr: hl.expr.Expression,
    fid_expr: hl.expr.Expression,
    y_exprs: dict[str, hl.expr.Expression],
) -> hl.Table:
    """The ``phenotype`` table: ``FID,IID`` then one column per response."""
    return mt.select_cols(FID=fid_expr, IID=iid_expr, **y_exprs).cols()


def get_covariate_ht(
    mt: hl.MatrixTable,
    iid_expr: hl.expr.Expression,
    fid_expr: hl.expr.Expression,
    covariate_exprs: dict[str, hl.expr.Expression],
) -> hl.Table:
    """The ``covariates`` table: ``FID,IID`` then one column per covariate."""
    return mt.select_cols(FID=fid_expr, IID=iid_expr, **covariate_exprs).cols()


def resolve_iid_fid(
    mt: hl.MatrixTable,
    iid_expr: hl.expr.Expression | str = "s",
    fid_expr: hl.expr.Expression | str | None = None,
) -> tuple[hl.expr.Expression, hl.expr.Expression]:
    """Resolve the IID/FID expressions, defaulting FID to ``F<IID>``."""
    import hail as hl

    if isinstance(iid_expr, str):
        iid_expr = str_to_expr(mt, iid_expr)
    if fid_expr is None:
        fid_expr = hl.format("F%s", iid_expr)
    elif isinstance(fid_expr, str):
        fid_expr = str_to_expr(mt, fid_expr)
    return iid_expr, fid_expr


def get_regenie_input(
    mt: hl.MatrixTable,
    region: hl.expr.Expression | None = None,
    iid_expr: hl.expr.Expression | str = "s",
    fid_expr: hl.expr.Expression | str | None = None,
    y_exprs: dict[str, hl.expr.Expression | str] | None = None,
    covariate_exprs: dict[str, hl.expr.Expression | str] | None = None,
    rare_variants_maf: float = 0.01,
    min_cadd: float = 20.0,
    case_expr: hl.expr.Expression | None = None,
) -> tuple[hl.Table, hl.Table, hl.Table, hl.Table, hl.Table, hl.Table]:
    """Build the six regenie input tables for one region.

    Returns ``(phenotype, covariates, aaf, annotation, variant_sets,
    effect_masks)`` in the order the export writes them.
    """
    freq_expr = get_freq_expr(mt)
    weight_expr = get_weight_expr(freq_expr)

    if region is None:
        region = get_gene_region_expr(mt)

    iid_expr, fid_expr = resolve_iid_fid(mt, iid_expr, fid_expr)

    if y_exprs is None:
        if case_expr is None:
            raise ValueError("missing y_exprs and case_expr")
        y_exprs = {"case": case_expr}
    else:
        y_exprs = {
            k: str_to_expr(mt, v) if isinstance(v, str) else v
            for k, v in y_exprs.items()
        }

    if covariate_exprs is None:
        covariate_exprs = {"y0": 1.0}
    else:
        covariate_exprs = {
            k: str_to_expr(mt, v) if isinstance(v, str) else v
            for k, v in covariate_exprs.items()
        }

    import hail as hl

    covariate_exprs = {
        k: hl.float64(v) if isinstance(v, float | int) else v
        for k, v in covariate_exprs.items()
    }

    return (
        get_phenotype_ht(mt, iid_expr, fid_expr, y_exprs),
        get_covariate_ht(mt, iid_expr, fid_expr, covariate_exprs),
        get_aaf_ht(mt, freq_expr),
        get_annotation_ht(
            mt,
            weight_expr,
            region,
            rare_variants_maf=rare_variants_maf,
            min_cadd=min_cadd,
        ),
        get_variant_sets_ht(mt, region),
        get_effects_ht(mt),
    )


# --- regions ----------------------------------------------------------------


def get_gene_region_expr(mt: hl.MatrixTable) -> hl.expr.Expression:
    """The per-gene region expression (gene symbol, else gene id, else ``<none>``)."""
    import hail as hl

    return (
        hl.case()
        .when(
            hl.is_defined(mt.vep.worst_csq_for_variant_canonical.gene_symbol),
            mt.vep.worst_csq_for_variant_canonical.gene_symbol,
        )
        .when(
            hl.is_defined(mt.vep.worst_csq_for_variant_canonical.gene_id),
            mt.vep.worst_csq_for_variant_canonical.gene_id,
        )
        .default("<none>")
    )


def get_region_name(mt: hl.MatrixTable, n: int | float) -> hl.expr.Expression:
    """The windowed region expression ``contig:pos // n + 1``."""
    import hail as hl

    return hl.format("%s:%d", mt.locus.contig, mt.locus.position // n + 1)


def region_label(kind: str, size: int | None = None) -> str:
    """A stable, filesystem-safe region directory label for the config."""
    return kind if kind != "window" else f"window_{size}"


# --- responses --------------------------------------------------------------


def get_sex_cat_expr(sex_expr: hl.expr.Expression) -> hl.expr.Expression:
    """Encode sex as ``1`` (M) / ``2`` (F), else missing."""
    import hail as hl

    return (
        hl.case()
        .when(sex_expr == "M", 1)
        .when(sex_expr == "F", 2)
        .default(hl.missing(hl.tfloat64))
    )


def snake_case(name: str) -> str:
    """Normalize a response name to snake_case (safe for wildcards/filenames)."""
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()
    return normalized or "response"


def build_covariate_exprs(
    mt: hl.MatrixTable,
    covariates: Iterable[Any],
    intercept: bool = True,
) -> dict[str, hl.expr.Expression]:
    """The covariate expression dict for ``[gwas.covariates]``.

    Each configured covariate is resolved to an MT expression: one with a
    ``metadata_field`` reads the rebuilt ``study_md`` column named by ``name``;
    otherwise ``name`` is resolved against the MT (dotted paths are allowed, e.g.
    ``ancestry.pca_scores``). A ``sex`` covariate is encoded ``1``/``2``. An
    array covariate's ``indices`` fan out into one column per index, named
    ``f"{name}{index}"``.
    """
    import hail as hl

    covariates_expr: dict[str, hl.expr.Expression] = {}
    if intercept:
        covariates_expr["y0"] = hl.float64(1.0)
    for covariate in covariates:
        if covariate.metadata_field is not None:
            expr = mt.study_md[covariate.name]
        else:
            expr = resolve_field_from_string(mt, covariate.name)
        if covariate.name == "sex":
            expr = get_sex_cat_expr(expr)
        if covariate.indices is None:
            covariates_expr[covariate.name] = expr
        else:
            for index in covariate.indices:
                covariates_expr[f"{covariate.name}{index}"] = expr[index]
    return covariates_expr


# --- PLINK reference --------------------------------------------------------


def define_plink_reference(
    default: str = "GRCh38", name: str = "plink"
) -> tuple[hl.ReferenceGenome, dict[str, str]]:
    """Idempotently get-or-create the regenie-compatible ``plink`` reference.

    Maps ``chr1..chr22,X,Y`` to un-prefixed names and ``chrM`` to ``MT``, taking
    the contig lengths from the global ``default`` reference. A pre-existing
    ``plink`` reference with different contigs or lengths fails loudly rather
    than being silently reused. Must be called inside a step body, never at
    import time.
    """
    import hail as hl

    contig_d = {
        **{f"chr{i}": str(i) for i in range(1, 23)},
        "chrX": "X",
        "chrY": "Y",
        "chrM": "MT",
    }
    default_ref = hl.get_reference(default)
    expected_lengths = {contig_d[k]: default_ref.lengths[k] for k in contig_d}

    try:
        plink_ref = hl.get_reference(name)
    except KeyError:
        plink_ref = hl.ReferenceGenome(
            name=name,
            contigs=list(contig_d.values()),
            lengths=expected_lengths,
            x_contigs=[contig_d[k] for k in default_ref.x_contigs],
            y_contigs=[contig_d[k] for k in default_ref.y_contigs],
            par=[
                (contig_d[intv.start.contig], intv.start.position, intv.end.position)
                for intv in default_ref.par
            ],
        )
        return plink_ref, contig_d

    mismatched = {
        contig: (plink_ref.lengths.get(contig), length)
        for contig, length in expected_lengths.items()
        if plink_ref.lengths.get(contig) != length
    }
    if set(plink_ref.contigs) != set(expected_lengths) or mismatched:
        raise ValueError(
            f"pre-existing {name!r} reference does not match the expected GRCh38 "
            f"contigs/lengths (mismatched={mismatched})"
        )
    return plink_ref, contig_d


def export_ht_as_regenie_table(
    table: hl.Table,
    name: str,
    target: Path,
    basename: str,
) -> None:
    """Write one regenie input table as a TSV, with ``NA`` for missing values.

    Hail's ``Table.export`` serializes missing floats as ``NaN``, which regenie
    does not treat as missing -- it reads them as a value and miscounts those
    samples as cases. The phenotype table's float responses are therefore cast
    to strings, mapping missing or ``NaN`` values to regenie's ``NA`` sentinel;
    every other table and column is exported unchanged.
    """
    import hail as hl

    from gnomad_meb.lib.gwas.constants import EFFECT_MASKS, PHENOTYPE, VARIANT_SETS

    table_out = table.key_by().drop(*table.key)
    if name == VARIANT_SETS:
        table_out = table_out.annotate(variants=hl.delimit(table_out.variants, ","))
    if name == EFFECT_MASKS:
        table_out = table_out.filter(hl.len(table_out.effects) > 0)
    if name == PHENOTYPE:
        fields = {}
        for field in table_out.row.dtype.fields:
            col = table_out[field]
            if field in ("FID", "IID"):
                fields[field] = col
            elif col.dtype == hl.tfloat64:
                fields[field] = hl.if_else(
                    hl.is_missing(col) | hl.is_nan(col), "NA", hl.str(col)
                )
            else:
                fields[field] = hl.if_else(hl.is_missing(col), "NA", hl.str(col))
        table_out = table_out.select(**fields)
    table_out.export(str(target / f"{basename}.{name}.tsv.bgz"))


# --- the exported dataset manifest ------------------------------------------


@dataclass(frozen=True, eq=True)
class RegenieDataset:
    """The self-contained regenie dataset a region directory holds.

    The manifest is derived from the base path, never persisted. ``plink_files``
    maps each contig to its PLINK prefix (all contigs share one prefix when the
    export is not split).
    """

    base_path: Path | str
    basename: str
    plink_files: dict[str, str] = field(default_factory=dict)

    PHENOTYPE_FILES: set[str] = field(
        default_factory=lambda: {
            "phenotype",
            "covariates",
            "aaf",
            "annotation",
            "variant_sets",
            "effect_masks",
        },
        init=False,
        repr=False,
        compare=False,
    )

    def __getitem__(self, key: str):
        if key in {"base_path", "basename", "plink_files"}:
            return getattr(self, key)
        if key in self.PHENOTYPE_FILES:
            return getattr(self, key)
        if key in self.contigs:
            return Path(self.plink_files[key])
        raise KeyError(key)

    @property
    def contigs(self):
        return self["plink_files"].keys()

    def _get_phenotype_file_path(self, name: str) -> Path:
        return Path(self.base_path) / f"{self.basename}.{name}.tsv.bgz"

    @property
    def phenotype(self) -> Path:
        return self._get_phenotype_file_path("phenotype")

    @property
    def covariates(self) -> Path:
        return self._get_phenotype_file_path("covariates")

    @property
    def aaf(self) -> Path:
        return self._get_phenotype_file_path("aaf")

    @property
    def annotation(self) -> Path:
        return self._get_phenotype_file_path("annotation")

    @property
    def variant_sets(self) -> Path:
        return self._get_phenotype_file_path("variant_sets")

    @property
    def effect_masks(self) -> Path:
        return self._get_phenotype_file_path("effect_masks")

    def __bool__(self) -> bool:
        return all(
            self[x].exists() and self[x].is_file() for x in self.PHENOTYPE_FILES
        ) and all(
            Path(self.plink_files[c]).with_suffix(".bed").exists() for c in self.contigs
        )

def read_study_md(
    metadata_csv: str | Path,
    set_columns: Iterable[str] = (),
    required_columns: Iterable[str] = (),
    response_columns: Iterable[str] = (),
    sample_id_col: str = "sample_id",
    cohort_col: str = "cohort",
) -> pandas.DataFrame:
    """Read the study metadata CSV into the frame ``study_md`` projects.

    Pure pandas (no hail). Only ``sample_id``, ``cohort_col`` (the case/control
    status column, renamed to ``cohort``), ``required_columns`` (a covariate's
    ``metadata_field``), ``response_columns`` and the set-valued
    ``set_columns`` are read; the ``response_columns`` are coerced to numeric
    and each set column's cell is parsed with ``ast.literal_eval``. The one
    derived column is the explicit ``birthdate`` → ``age`` edge case,
    rewritten on import. ``sample_id`` is always read as a string: the hail
    joins key on the MT's string column key, so a numeric-looking id must not
    parse as ``int64`` or every join would miss.
    """
    read_columns = [
        sample_id_col,
        cohort_col,
        *required_columns,
        *response_columns,
        *set_columns,
    ]
    md_df = pandas.read_csv(metadata_csv, usecols=list(dict.fromkeys(read_columns)))
    md_df = md_df.rename(columns={sample_id_col: "sample_id", cohort_col: "cohort"})
    md_df["sample_id"] = md_df["sample_id"].astype(str)
    md_df = md_df.set_index("sample_id")

    if "birthdate" in md_df.columns:
        md_df["birthdate"] = pandas.to_datetime(
            md_df["birthdate"], format="%Y-%m-%d"
        )
        md_df["age"] = datetime.now().year - md_df["birthdate"].dt.year
        md_df = md_df.drop(columns=["birthdate"])
    for rcol in dict.fromkeys(response_columns):
        md_df[rcol] = pandas.to_numeric(md_df[rcol])
    for scol in set_columns:
        md_df[scol] = md_df[scol].apply(
            lambda x: ast.literal_eval(x) if pandas.notna(x) else set()
        )

    return md_df