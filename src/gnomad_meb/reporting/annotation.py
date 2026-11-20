"""The Annotation stage: collectors over the ``annotation`` matrix table and the
``plotnine`` builders over their frames.

The stage's declared primary output is the ``annotation`` matrix table written by
``AnnotationPipeline.merge``. It is the final callset: every row carries the
reference and in-silico annotations (``vep``, ``in_silico_predictors``,
``clinvar``), the gnomAD reference frequencies (``gnomad_freq``), the produced
cohort's own frequencies (``freq``, with its ``freq_meta`` population strata) and
the QC columns from Sample QC.

The report's job is first to **validate that the annotations are actually
present** — VEP on every row, the in-silico scores where they exist, ClinVar for
the rows ClinVar knows — and only then to summarise the callset (variant classes,
the pass/filtered split, the per-population allele frequencies). Two known
failure modes get their own collectors:

- VEP is validated against the SOCS1 interval (``data/bed/socs1_sites.bed``): a
  region where the reference lookup is known to have missed variants, so the
  report must show what annotation those sites *did* get rather than trusting the
  gnomAD join.
- every **potentially protein-changing** variant must be annotated even when its
  gnomAD allele frequency is zero or it is absent from gnomAD entirely, because
  those are exactly the rows the reference ``vep`` lookup cannot supply.

Each collector is stateless, returns a tidy ``pandas.DataFrame`` (or a plain dict
for scalars) with stable columns, and does not write to disk. Following
:mod:`gnomad_meb.reporting.variant_qc`, each Hail collector is a thin
``hl.MatrixTable`` read plus a **pure reshape helper**, so the shaping is
unit-testable without a Spark session. No collector unions the callset: the
matrix is already the terminal merge (ADR-0015), so a report is a read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotnine as pn

from gnomad_meb.hail import hail as hl
from gnomad_meb.reporting.sample_qc import _as_plain
from gnomad_meb.reporting.theme import THEME, ordered_splits

__all__ = [
    "PROTEIN_CHANGING_CONSEQUENCES",
    "annotation_overview",
    "annotation_presence",
    "annotation_presence_bar",
    "annotation_provenance",
    "bed_intervals",
    "clinvar_counts",
    "clinvar_significance_bar",
    "coverage_by_contig",
    "coverage_by_group",
    "coverage_contig_bar",
    "freq_by_stratum",
    "freq_stratum_bar",
    "gnomad_freq_by_population",
    "gnomad_freq_population_bar",
    "in_silico_presence",
    "in_silico_presence_bar",
    "protein_changing_check",
    "region_variants",
    "unannotated_protein_changing",
    "vep_consequence_bar",
    "vep_consequence_counts",
]

#: The ``in_silico_predictors`` struct fields the report checks for presence.
IN_SILICO_FIELDS = (
    "cadd",
    "revel_max",
    "spliceai_ds_max",
    "pangolin_largest_ds",
    "phylop",
    "sift_max",
    "polyphen_max",
    "alphamissense",
)

#: The gnomAD reference populations every frequency report shows (the
#: ``gen_anc``-stratified ``_adj`` entries of ``gnomad_freq_meta``).
GNOMAD_POPULATIONS = (
    "adj",
    "afr",
    "amr",
    "eas",
    "nfe",
    "sas",
    "fin",
    "mid",
    "ami",
    "asj",
    "remaining",
)

#: VEP consequence terms that can change the protein, in the sense the report
#: validates: every row whose *most severe* consequence is one of these must
#: carry a VEP annotation regardless of its gnomAD allele frequency.
PROTEIN_CHANGING_CONSEQUENCES = frozenset(
    {
        "missense_variant",
        "stop_gained",
        "stop_lost",
        "start_lost",
        "frameshift_variant",
        "inframe_insertion",
        "inframe_deletion",
        "disruptive_inframe_insertion",
        "disruptive_inframe_deletion",
        "splice_acceptor_variant",
        "splice_donor_variant",
        "protein_altering_variant",
        "transcript_amplification",
    }
)

#: Columns of the ``annotation_presence`` frame, in order.
_PRESENCE_COLUMNS = ("annotation", "n_present", "fraction")

#: Columns of the ``freq_by_stratum`` frame, in order.
_FREQ_STRATUM_COLUMNS = (
    "stratum",
    "group",
    "gen_anc",
    "sex",
    "cohort",
    "n_samples",
    "n_defined",
    "n_af_gt0",
    "mean_af",
)

#: Columns of the ``gnomad_freq_by_population`` frame, in order.
_GNOMAD_FREQ_COLUMNS = (
    "population",
    "index",
    "n_defined",
    "n_af_gt0",
    "mean_af",
)

#: Columns of the ``region_variants`` frame, in order.
_REGION_COLUMNS = (
    "locus",
    "alleles",
    "filters",
    "variant_class",
    "most_severe_consequence",
    "impact",
    "n_transcript_consequences",
    "insilico_cadd",
    "gnomad_af",
)

#: Columns of the ``unannotated_protein_changing`` frame, in order.
_UNANNOTATED_COLUMNS = (
    "locus",
    "alleles",
    "filters",
    "gnomad_af",
)

#: Columns of the ``coverage_by_group`` frame, in order.
_COVERAGE_GROUP_COLUMNS = (
    "stratum",
    "filtered",
    "cohort",
    "n_sites",
    "mean_coverage",
    "median_coverage",
    "fraction_over_10",
    "fraction_over_30",
)

#: Columns of the ``coverage_by_contig`` frame, in order.
_COVERAGE_CONTIG_COLUMNS = (
    "contig",
    "stratum",
    "n_sites",
    "mean_coverage",
    "fraction_over_10",
    "fraction_over_30",
)


# --- pure reshape helpers ---------------------------------------------------
# Each collector is a thin hail read; the shaping below is ordinary Python so it
# is unit-testable without a Spark session (the ``variant_qc.py`` precedent).


def _presence_frame(counts: dict[str, int], n_variants: int) -> pd.DataFrame:
    """A ``annotation``/``n_present``/``fraction`` frame from a defined counter."""
    rows = []
    for name, n_present in counts.items():
        rows.append(
            {
                "annotation": name,
                "n_present": int(n_present),
                "fraction": (n_present / n_variants) if n_variants else 0.0,
            }
        )
    return pd.DataFrame(rows, columns=_PRESENCE_COLUMNS)


def _consequence_frame(counts: dict[str, int]) -> pd.DataFrame:
    """One row per ``most_severe_consequence`` with its callset fraction."""
    total = sum(counts.values())
    rows = [
        {
            "most_severe_consequence": name,
            "n_variants": int(n),
            "fraction": (n / total) if total else 0.0,
        }
        for name, n in counts.items()
    ]
    frame = pd.DataFrame(
        rows, columns=["most_severe_consequence", "n_variants", "fraction"]
    )
    return frame.sort_values("n_variants", ascending=False, ignore_index=True)


def _insilico_frame(counts: dict[str, int], n_variants: int) -> pd.DataFrame:
    """One row per in-silico score with the fraction of rows carrying it."""
    rows = [
        {
            "predictor": name,
            "n_present": int(counts.get(name, 0)),
            "fraction": (counts.get(name, 0) / n_variants) if n_variants else 0.0,
        }
        for name in IN_SILICO_FIELDS
    ]
    return pd.DataFrame(
        rows, columns=["predictor", "n_present", "fraction"]
    ).sort_values("n_present", ascending=False, ignore_index=True)


def _clinvar_frame(counts: dict[str, Any], n_variants: int) -> pd.DataFrame:
    """One row per ClinVar clinical significance and its callset fraction."""
    if not counts:
        return pd.DataFrame(
            columns=["clinical_significance", "n_variants", "fraction"]
        )
    frame = pd.DataFrame(
        {
            "clinical_significance": [str(k) for k in counts],
            "n_variants": [int(v) for v in counts.values()],
        }
    )
    frame = frame.sort_values("n_variants", ascending=False, ignore_index=True)
    frame["fraction"] = frame["n_variants"] / n_variants if n_variants else 0.0
    return frame[["clinical_significance", "n_variants", "fraction"]]


def _freq_stratum_frame(
    freq_meta: list[dict[str, str]],
    sample_counts: list[int],
    stats: dict[str, Any],
) -> pd.DataFrame:
    """One row per frequency stratum from the per-index aggregate structs."""
    rows = []
    for index, fields in enumerate(freq_meta):
        label = ";".join(f"{k}={v}" for k, v in fields.items())
        stat = _as_plain(stats.get(str(index))) or {}
        rows.append(
            {
                "stratum": label,
                "group": fields.get("group"),
                "gen_anc": fields.get("gen_anc"),
                "sex": fields.get("sex"),
                "cohort": fields.get("cohort"),
                "n_samples": (
                    sample_counts[index] if index < len(sample_counts) else None
                ),
                "n_defined": int(stat.get("n_defined") or 0),
                "n_af_gt0": int(stat.get("n_af_gt0") or 0),
                "mean_af": stat.get("mean_af"),
            }
        )
    return pd.DataFrame(rows, columns=_FREQ_STRATUM_COLUMNS)


def _gnomad_freq_frame(
    index_dict: dict[str, int],
    stats: dict[str, Any],
) -> pd.DataFrame:
    """One row per gnomAD reference population from the per-index structs."""
    rows = []
    for population in GNOMAD_POPULATIONS:
        key = population if population in ("adj", "raw") else f"{population}_adj"
        stat = _as_plain(stats.get(key)) or {}
        rows.append(
            {
                "population": population,
                "index": index_dict.get(key),
                "n_defined": int(stat.get("n_defined") or 0),
                "n_af_gt0": int(stat.get("n_af_gt0") or 0),
                "mean_af": stat.get("mean_af"),
            }
        )
    return pd.DataFrame(rows, columns=_GNOMAD_FREQ_COLUMNS)


def _region_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """Format a bounded region pull: locus/alleles/filters as strings."""
    frame = rows.copy()
    if "filters" in frame.columns:
        frame["filters"] = frame["filters"].map(
            lambda value: ",".join(sorted(value)) if value else ""
        )
    return frame.reindex(columns=[c for c in _REGION_COLUMNS if c in frame])


def _unannotated_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """Format the bounded unannotated protein-changing pull."""
    frame = rows.copy()
    if "filters" in frame.columns:
        frame["filters"] = frame["filters"].map(
            lambda value: ",".join(sorted(value)) if value else ""
        )
    return frame.reindex(columns=[c for c in _UNANNOTATED_COLUMNS if c in frame])


def _coverage_stratum_label(fields: dict[str, str]) -> str:
    """The display label for one ``coverage_stats_meta`` entry (``all`` if bare)."""
    if not fields:
        return "all"
    return ";".join(f"{key}={fields[key]}" for key in sorted(fields))


def _coverage_stat(value: Any) -> dict[str, Any]:
    """One aggregated coverage stat: the site count and the summary means."""
    stat = _as_plain(value) or {}
    return {
        "n_sites": int(stat.get("n_sites") or 0),
        "mean_coverage": stat.get("mean_coverage"),
        "median_coverage": stat.get("median_coverage"),
        "fraction_over_10": stat.get("fraction_over_10"),
        "fraction_over_30": stat.get("fraction_over_30"),
    }


def _coverage_group_frame(
    meta: list[dict[str, str]], stats: dict[str, Any]
) -> pd.DataFrame:
    """One row per coverage stratum: the whole-callset depth summary."""
    rows = []
    for index, fields in enumerate(meta):
        fields = dict(fields)
        rows.append(
            {
                "stratum": _coverage_stratum_label(fields),
                "filtered": fields.get("filtered"),
                "cohort": fields.get("cohort"),
                **_coverage_stat(stats.get(str(index))),
            }
        )
    return pd.DataFrame(rows, columns=_COVERAGE_GROUP_COLUMNS)


def _coverage_contig_frame(
    meta: list[dict[str, str]], stats: dict[str, Any]
) -> pd.DataFrame:
    """One row per ``(contig, coverage stratum)``: depth summarised per contig."""
    rows = []
    for index, fields in enumerate(meta):
        label = _coverage_stratum_label(dict(fields))
        by_contig = _as_plain(stats.get(str(index))) or {}
        for contig, value in by_contig.items():
            stat = _coverage_stat(value)
            rows.append(
                {
                    "contig": contig,
                    "stratum": label,
                    "n_sites": stat["n_sites"],
                    "mean_coverage": stat["mean_coverage"],
                    "fraction_over_10": stat["fraction_over_10"],
                    "fraction_over_30": stat["fraction_over_30"],
                }
            )
    return pd.DataFrame(rows, columns=_COVERAGE_CONTIG_COLUMNS)


# --- collectors -------------------------------------------------------------


def annotation_provenance(mt: hl.MatrixTable) -> dict[str, Any]:
    """The matrix's globals (versions, tool versions, VEP counts) plus counts.

    ``gnomad_vep_globals`` describes the reference lookup, ``vep_globals`` the
    local VEP run (``defined``/``missing`` rows and the VEP version), and
    ``in_silico_predictors_globals`` / ``gnomad_tool_versions`` the score
    provenance. ``n_variants`` / ``n_samples`` size the callset.
    """
    globals_plain = _as_plain(hl.eval(mt.index_globals()))
    return {
        "n_variants": mt.count_rows(),
        "n_samples": mt.count_cols(),
        "project_meta": globals_plain.get("project_meta") or {},
        "gnomad_version": globals_plain.get("gnomad_version"),
        "gnomad_date": globals_plain.get("gnomad_date"),
        "gnomad_vep_globals": globals_plain.get("gnomad_vep_globals") or {},
        "gnomad_tool_versions": globals_plain.get("gnomad_tool_versions") or {},
        "vep_globals": globals_plain.get("vep_globals") or {},
        "in_silico_predictors_globals": (
            globals_plain.get("in_silico_predictors_globals") or {}
        ),
        "variant_qc": globals_plain.get("variant_qc") or {},
    }


def _is_filtered(mt: hl.MatrixTable) -> hl.expr.BooleanExpression:
    """Whether a row failed a filter: a non-empty ``filters`` set."""
    return hl.len(mt.filters) > 0


def _overview(stats: Any) -> dict[str, Any]:
    """The overview struct with the derived transition/transversion ratio."""
    plain = _as_plain(stats)
    n_ti = plain.get("n_ti") or 0
    n_tv = plain.get("n_tv") or 0
    plain["ti_tv"] = (n_ti / n_tv) if n_tv else float("nan")
    return plain


def annotation_overview(mt: hl.MatrixTable) -> dict[str, Any]:
    """Scalar tallies: the pass/filtered split, class and the uninformative rows.

    ``n_uninformative`` counts the rows absent from the Variant QC table (the
    ``vqc_uninformative`` filter): they are in the callset and VEP-annotated but
    carry no VQSR provenance.
    """
    stats = mt.aggregate_rows(
        hl.struct(
            n_variants=hl.agg.count(),
            n_pass=hl.agg.count_where(~_is_filtered(mt)),
            n_filtered=hl.agg.count_where(_is_filtered(mt)),
            n_snv=hl.agg.count_where(mt.allele_info.allele_type == "snv"),
            n_indel=hl.agg.count_where(
                hl.is_defined(mt.allele_info.allele_type)
                & (mt.allele_info.allele_type != "snv")
            ),
            n_uninformative=hl.agg.count_where(
                mt.filters.contains("vqc_uninformative")
            ),
            n_split_rows=hl.agg.count_where(mt.was_split),
            n_ti=hl.agg.count_where(hl.is_transition(mt.alleles[0], mt.alleles[1])),
            n_tv=hl.agg.count_where(hl.is_transversion(mt.alleles[0], mt.alleles[1])),
        )
    )
    return _overview(stats)


def annotation_presence(mt: hl.MatrixTable) -> pd.DataFrame:
    """One row per annotation source: how many rows carry it and the fraction.

    This is the "are the annotations actually there" table: VEP, the VEP
    most-severe consequence, the in-silico predictor struct, a ClinVar
    classification, the gnomAD reference frequencies and the cohort's own
    frequencies.
    """
    row = mt.row
    counts = mt.aggregate_rows(
        hl.struct(
            vep=hl.agg.count_where(hl.is_defined(row.vep)),
            most_severe_consequence=hl.agg.count_where(
                hl.is_defined(row.vep.most_severe_consequence)
            ),
            in_silico_predictors=hl.agg.count_where(
                hl.is_defined(row.in_silico_predictors)
            ),
            clinvar=hl.agg.count_where(
                hl.is_defined(row.clinvar.clinical_significance)
                & (hl.len(row.clinvar.clinical_significance) > 0)
            ),
            gnomad_freq=hl.agg.count_where(hl.is_defined(row.gnomad_freq)),
            freq=hl.agg.count_where(hl.is_defined(row.freq)),
        )
    )
    return _presence_frame(dict(_as_plain(counts) or {}), mt.count_rows())


def vep_consequence_counts(mt: hl.MatrixTable) -> pd.DataFrame:
    """One row per ``vep.most_severe_consequence`` and its callset fraction."""
    counts = mt.aggregate_rows(hl.agg.counter(mt.vep.most_severe_consequence))
    return _consequence_frame(dict(_as_plain(counts) or {}))


def in_silico_presence(mt: hl.MatrixTable) -> pd.DataFrame:
    """One row per in-silico score: how many rows carry a defined value.

    The scores are consequence-dependent (CADD for coding SNVs/indels, REVEL /
    SIFT / PolyPhen for missense, AlphaMissense for missense, SpliceAI /
    Pangolin for splice-adjacent), so a sub-100% presence is expected and the
    fraction is the point.
    """
    prefix = mt.in_silico_predictors
    counts = mt.aggregate_rows(
        hl.struct(
            **{
                field: hl.agg.count_where(hl.is_defined(prefix[field]))
                for field in IN_SILICO_FIELDS
            }
        )
    )
    return _insilico_frame(dict(_as_plain(counts) or {}), mt.count_rows())


def clinvar_counts(mt: hl.MatrixTable) -> pd.DataFrame:
    """One row per ClinVar clinical significance, among the annotated rows."""
    counts = mt.aggregate_rows(
        hl.agg.explode(
            lambda name: hl.agg.counter(name), mt.clinvar.clinical_significance
        )
    )
    return _clinvar_frame(dict(_as_plain(counts) or {}), mt.count_rows())


def _stratum_stats(mt: hl.MatrixTable, index: int) -> hl.expr.StructExpression:
    """The defined / AF>0 / mean-AF struct for one frequency stratum index."""
    entry = mt.freq[index]
    return hl.struct(
        n_defined=hl.agg.count_where(hl.is_defined(entry)),
        n_af_gt0=hl.agg.count_where(hl.is_defined(entry) & (entry.AF > 0)),
        mean_af=hl.agg.mean(entry.AF),
    )


def freq_by_stratum(mt: hl.MatrixTable) -> pd.DataFrame:
    """One row per cohort frequency stratum from the ``freq_meta`` globals.

    The produced callset's own allele frequencies, one row per stratum of
    ``freq_meta`` (overall / by genetic ancestry / by sex / by cohort), with the
    sample count that stratum covers.
    """
    globals_plain = _as_plain(hl.eval(mt.index_globals()))
    freq_meta = globals_plain.get("freq_meta") or []
    sample_counts = globals_plain.get("freq_meta_sample_count") or []
    stats = mt.aggregate_rows(
        hl.struct(
            **{
                str(index): _stratum_stats(mt, index)
                for index in range(len(freq_meta))
            }
        )
    )
    return _freq_stratum_frame(
        freq_meta, list(sample_counts), dict(_as_plain(stats) or {})
    )


def _gnomad_stratum_stats(
    mt: hl.MatrixTable, index: int
) -> hl.expr.StructExpression:
    """The defined / AF>0 / mean-AF struct for one gnomAD frequency index."""
    entry = mt.gnomad_freq[index]
    return hl.struct(
        n_defined=hl.agg.count_where(hl.is_defined(entry)),
        n_af_gt0=hl.agg.count_where(hl.is_defined(entry) & (entry.AF > 0)),
        mean_af=hl.agg.mean(entry.AF),
    )


def gnomad_freq_by_population(mt: hl.MatrixTable) -> pd.DataFrame:
    """One row per gnomAD reference population, from ``gnomad_freq_meta``.

    The reference allele frequencies the annotation join supplies, per
    ``gen_anc`` (the ``*_adj`` entries), so the report can compare the callset's
    frequencies against the reference populations.
    """
    globals_plain = _as_plain(hl.eval(mt.index_globals()))
    # ``gnomad_freq_index_dict`` is a ``Dict[String,Int32]`` global; ``hl.eval``
    # returns it as a plain ``{label: index}`` mapping (not the on-disk
    # ``Array[Struct{key,value}]`` encoding), so it is used directly.
    index_dict = globals_plain.get("gnomad_freq_index_dict") or {}
    keys = {
        population: (
            population if population in ("adj", "raw") else f"{population}_adj"
        )
        for population in GNOMAD_POPULATIONS
    }
    stats = mt.aggregate_rows(
        hl.struct(
            **{
                key: _gnomad_stratum_stats(mt, index_dict[key])
                for key in keys.values()
                if key in index_dict
            }
        )
    )
    return _gnomad_freq_frame(index_dict, dict(_as_plain(stats) or {}))


def _has_coverage(mt: hl.MatrixTable) -> bool:
    """Whether the joined coverage row field and its meta global are present."""
    return (
        "coverage_stats" in mt.row.dtype.fields
        and "coverage_stats_meta" in mt.globals.dtype.fields
    )


def coverage_by_group(mt: hl.MatrixTable) -> pd.DataFrame:
    """One row per coverage stratum, aggregated over the whole callset.

    The per-locus ``coverage_stats`` array (aligned to ``coverage_stats_meta``)
    summarised per sample group: the sample-QC pass/filtered split, plus the
    cohort strata when ``annotation.group_by_cohort`` is configured. An empty
    frame means the matrix predates the coverage join.
    """
    if not _has_coverage(mt):
        return pd.DataFrame(columns=_COVERAGE_GROUP_COLUMNS)
    meta = _as_plain(hl.eval(mt.coverage_stats_meta)) or []
    if not meta:
        return pd.DataFrame(columns=_COVERAGE_GROUP_COLUMNS)

    def _summarize_coverage_stats(stat):
        return hl.struct(
            n_sites=hl.agg.count_where(hl.is_defined(stat)),
            mean_coverage=hl.agg.mean(stat.mean),
            median_coverage=hl.agg.mean(stat.median_approx),
            fraction_over_10=hl.agg.mean(stat.over_10),
            fraction_over_30=hl.agg.mean(stat.over_30),
        )

    stats_array = mt.aggregate_rows(
        hl.agg.array_agg(
            _summarize_coverage_stats,
            mt.coverage_stats,
        )
    )
    stats = {
        str(i): stat for i, stat in enumerate(stats_array)
    }
    return _coverage_group_frame(meta, dict(_as_plain(stats) or {}))


def coverage_by_contig(mt: hl.MatrixTable) -> pd.DataFrame:
    """One row per ``(contig, coverage stratum)``, aggregated across the loci.

    The per-contig read of the ``coverage_by_group`` summary, so a lopsided or
    poorly-covered contig is visible within each sample group. An empty frame
    means the matrix predates the coverage join.
    """
    if not _has_coverage(mt):
        return pd.DataFrame(columns=_COVERAGE_CONTIG_COLUMNS)
    meta = _as_plain(hl.eval(mt.coverage_stats_meta)) or []
    if not meta:
        return pd.DataFrame(columns=_COVERAGE_CONTIG_COLUMNS)

    stats_array = mt.aggregate_rows(
        hl.agg.array_agg(
            lambda stat: hl.agg.group_by(
                mt.locus.contig,
                hl.struct(
                    n_sites=hl.agg.count_where(
                        hl.is_defined(stat)
                    ),
                    mean_coverage=hl.agg.mean(stat.mean),
                    fraction_over_10=hl.agg.mean(stat.over_10),
                    fraction_over_30=hl.agg.mean(stat.over_30),
                ),
            ),
            mt.coverage_stats
        )
    )

    stats = {
        str(i): stat for i, stat in enumerate(stats_array)
    }
    return _coverage_contig_frame(meta, dict(_as_plain(stats) or {}))


def region_variants(mt: hl.MatrixTable, intervals, n: int = 100) -> pd.DataFrame:
    """The variants overlapping ``intervals``, with their VEP and frequency data.

    A bounded driver pull (``n`` rows) so a dense region never lands on the
    driver; it feeds the SOCS1 validation, where the point is to see the VEP
    consequence, CADD score and gnomAD AF of the sites that *are* present.
    """
    sub = hl.filter_intervals(mt, intervals)
    rows = sub.rows().key_by()
    rows = rows.select(
        locus=hl.str(rows.locus),
        alleles=hl.delimit(rows.alleles, ","),
        filters=rows.filters,
        variant_class=rows.vep.variant_class,
        most_severe_consequence=rows.vep.most_severe_consequence,
        impact=rows.vep.worst_csq_for_variant.impact,
        n_transcript_consequences=hl.len(rows.vep.transcript_consequences),
        insilico_cadd=rows.in_silico_predictors.cadd.phred,
        gnomad_af=rows.gnomad_freq[0].AF,
    )
    return _region_frame(rows.head(n).to_pandas())


def _gnomad_adj_index(globals_plain: dict[str, Any]) -> int | None:
    """The index of the overall ``adj`` entry in ``gnomad_freq``, or ``None``."""
    freq_index_dict = globals_plain.get("gnomad_freq_index_dict") or {}
    return freq_index_dict.get("adj")


def _protein_changing_rows(mt: hl.MatrixTable) -> hl.MatrixTable:
    """The rows whose most severe VEP consequence can change the protein."""
    return mt.filter_rows(
        hl.is_defined(mt.vep.most_severe_consequence)
        & hl.literal(PROTEIN_CHANGING_CONSEQUENCES).contains(mt.vep.most_severe_consequence)
    )


def protein_changing_check(mt: hl.MatrixTable) -> dict[str, Any]:
    """Validate that every potentially protein-changing row is VEP-annotated.

    Counts the protein-changing rows and how many carry a VEP consequence, then
    the subset absent from gnomAD or with a gnomAD allele frequency of zero —
    the rows the reference lookup cannot annotate and the local VEP run must.
    A non-zero ``n_af0_unannotated`` / ``n_absent_unannotated`` is the failure.
    """
    globals_plain = _as_plain(hl.eval(mt.index_globals()))
    adj_index = _gnomad_adj_index(globals_plain)
    prot = _protein_changing_rows(mt)

    n_protein_changing = prot.count_rows()
    annotated = hl.is_defined(prot.vep.most_severe_consequence)

    if adj_index is None:
        return {
            "n_protein_changing": n_protein_changing,
            "n_annotated": prot.filter_rows(annotated).count_rows(),
            "n_unannotated": n_protein_changing
            - prot.filter_rows(annotated).count_rows(),
            "n_absent_gnomad": 0,
            "n_absent_unannotated": 0,
            "n_af0": 0,
            "n_af0_unannotated": 0,
        }

    gnomad_af = prot.gnomad_freq[adj_index].AF
    absent = hl.is_missing(prot.gnomad_freq[adj_index])
    af0 = hl.is_defined(prot.gnomad_freq[adj_index]) & (gnomad_af == 0)

    n_annotated = prot.filter_rows(annotated).count_rows()
    return {
        "n_protein_changing": n_protein_changing,
        "n_annotated": n_annotated,
        "n_unannotated": n_protein_changing - n_annotated,
        "n_absent_gnomad": prot.filter_rows(absent).count_rows(),
        "n_absent_unannotated": prot.filter_rows(absent & ~annotated).count_rows(),
        "n_af0": prot.filter_rows(af0).count_rows(),
        "n_af0_unannotated": prot.filter_rows(af0 & ~annotated).count_rows(),
    }


def unannotated_protein_changing(
    mt: hl.MatrixTable, n: int = 100
) -> pd.DataFrame:
    """Bounded examples of protein-changing rows that carry no VEP consequence.

    Should be empty; when it is not, this is the evidence table for the report's
    protein-changing validation.
    """
    globals_plain = _as_plain(hl.eval(mt.index_globals()))
    adj_index = _gnomad_adj_index(globals_plain)
    prot = _protein_changing_rows(mt)
    unannotated = prot.filter_rows(hl.is_missing(prot.vep.most_severe_consequence))
    rows = unannotated.rows().key_by()
    gnomad_af = (
        rows.gnomad_freq[adj_index].AF
        if adj_index is not None
        else hl.missing(hl.tfloat64)
    )
    rows = rows.select(
        locus=hl.str(rows.locus),
        alleles=hl.delimit(rows.alleles, ","),
        filters=rows.filters,
        gnomad_af=gnomad_af,
    )
    return _unannotated_frame(rows.head(n).to_pandas())


def _bed_records(path: str | Path) -> list[tuple[str, int, int]]:
    """Parse a BED file into ``(contig, start, end)`` records (still 0-based).

    Skips comment/track/browser lines and blank lines.
    """
    records = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "track", "browser")):
            continue
        fields = line.split()
        records.append((fields[0], int(fields[1]), int(fields[2])))
    return records


def bed_intervals(path: str | Path, reference_genome: str = "GRCh38") -> list:
    """Parse a BED file into hail locus intervals (0-based half-open → 1-based).

    BED's zero-based, half-open ``[start, end)`` becomes hail's inclusive
    ``start+1-end``; the parsing is :func:`_bed_records`.
    """
    return [
        hl.parse_locus_interval(
            f"{contig}:{start + 1}-{end}", reference_genome=reference_genome
        )
        for contig, start, end in _bed_records(path)
    ]


# --- plot builders ----------------------------------------------------------
# Each takes the matching collector frame; nothing reads config or files.


def annotation_presence_bar(df: pd.DataFrame) -> pn.ggplot:
    """The annotated fraction per annotation source."""
    return (
        pn.ggplot(df, pn.aes(x="reorder(annotation, fraction)", y="fraction"))
        + pn.geom_col(fill="#4c72b0")
        + pn.coord_flip()
        + pn.labs(x="", y="Fraction of variants", title="Annotation presence")
        + THEME
    )


def vep_consequence_bar(df: pd.DataFrame, n: int = 20) -> pn.ggplot:
    """The most severe VEP consequence counts (the top ``n``)."""
    top = df.head(n)
    return (
        pn.ggplot(
            top,
            pn.aes(
                x="reorder(most_severe_consequence, n_variants)", y="n_variants"
            ),
        )
        + pn.geom_col(fill="#4c72b0")
        + pn.coord_flip()
        + pn.labs(x="", y="Variants", title="VEP most severe consequence")
        + THEME
    )


def in_silico_presence_bar(df: pd.DataFrame) -> pn.ggplot:
    """The present fraction per in-silico score."""
    return (
        pn.ggplot(df, pn.aes(x="reorder(predictor, fraction)", y="fraction"))
        + pn.geom_col(fill="#55a868")
        + pn.coord_flip()
        + pn.labs(x="", y="Fraction of variants", title="In-silico predictor presence")
        + THEME
    )


def clinvar_significance_bar(df: pd.DataFrame) -> pn.ggplot:
    """Variants per ClinVar clinical significance."""
    return (
        pn.ggplot(
            df,
            pn.aes(
                x="reorder(clinical_significance, n_variants)", y="n_variants"
            ),
        )
        + pn.geom_col(fill="#c44e52")
        + pn.coord_flip()
        + pn.labs(x="", y="Variants", title="ClinVar clinical significance")
        + THEME
    )


def freq_stratum_bar(df: pd.DataFrame) -> pn.ggplot:
    """The fraction of variants polymorphic (AF > 0) per cohort stratum."""
    frame = df.copy()
    frame["polymorphic_fraction"] = (
        frame["n_af_gt0"].div(frame["n_defined"]).fillna(0.0)
    )
    return (
        pn.ggplot(
            frame,
            pn.aes(
                x="reorder(stratum, polymorphic_fraction)",
                y="polymorphic_fraction",
            ),
        )
        + pn.geom_col(fill="#4c72b0")
        + pn.coord_flip()
        + pn.labs(
            x="", y="Fraction with AF > 0", title="Polymorphic fraction per stratum"
        )
        + THEME
    )


def gnomad_freq_population_bar(df: pd.DataFrame) -> pn.ggplot:
    """The fraction of variants polymorphic (AF > 0) per gnomAD population."""
    frame = df.copy()
    frame["polymorphic_fraction"] = (
        frame["n_af_gt0"].div(frame["n_defined"]).fillna(0.0)
    )
    return (
        pn.ggplot(
            frame,
            pn.aes(
                x="reorder(population, polymorphic_fraction)",
                y="polymorphic_fraction",
            ),
        )
        + pn.geom_col(fill="#8172b3")
        + pn.coord_flip()
        + pn.labs(
            x="",
            y="Fraction with AF > 0",
            title="gnomAD polymorphic fraction per population",
        )
        + THEME
    )


def coverage_contig_bar(df: pd.DataFrame) -> pn.ggplot:
    """Mean depth per contig, one bar per coverage group, in contig order."""
    frame = df.copy()
    frame["contig"] = ordered_splits(frame["contig"])
    return (
        pn.ggplot(frame, pn.aes(x="contig", y="mean_coverage", fill="stratum"))
        + pn.geom_col(position="dodge")
        + pn.labs(
            x="",
            y="Mean depth",
            fill="Coverage group",
            title="Mean coverage by contig and group",
        )
        + THEME
    )
