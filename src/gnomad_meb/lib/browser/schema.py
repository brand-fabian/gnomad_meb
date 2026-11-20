"""The MatrixTable -> browser Site table reshape.

The browser's DuckDB backend reads a **fixed physical schema**, so the reshape is
a set of pure Hail expressions over ``mt.rows()`` that produce exactly those
columns: ``locus``/``alleles`` stay the key, ``variant_id``/``xpos`` are
synthesized, VEP is lifted from the nested ``vep`` struct to a row-level
``transcript_consequences``, and the cohort's positional
``freq``/``freq_meta``/``grpmax``/``fafmax``/``histograms``/``coverage_stats``
are consumed into the synthesized ``joint``/``genome``/``coverage`` columns. The
mandatory columns are always materialized with a fixed, input-independent dtype;
the optional ``fields`` select passthrough annotations, and a listed-but-absent
field is dropped.

Required source columns are **preconditions**, not best-effort inputs: a missing
or malformed one raises :class:`BrowserSchemaError` rather than degrading to a
typed null. Only ``rsids``, ``caid``, ``exome`` and ``in_silico_predictors``
keep an absent-column typed-null path.

Nothing here annotates export provenance (the ``browser_sites`` step owns that)
and nothing imports Hail at module import time: every function imports it lazily
inside its body, so importing :mod:`gnomad_meb.lib.browser.schema` never boots
the Hail stack.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, NamedTuple

from gnomad_meb.lib.browser.constants import CONTIG_ORDER

if TYPE_CHECKING:
    import hail as hl

#: Bare contig name -> the ``xpos`` contig number, derived from the stage's one
#: canonical order (:data:`~gnomad_meb.lib.browser.constants.CONTIG_ORDER`).
#: ``chr`` is stripped first (the order is chr-prefixed); ``M`` and ``MT`` both
#: map to 25. Anything else fails loudly.
CONTIG_NUMBERS: dict[str, int] = {
    contig.removeprefix("chr"): number
    for number, contig in enumerate(CONTIG_ORDER, start=1)
}
CONTIG_NUMBERS["MT"] = CONTIG_NUMBERS["M"]

#: The ``over_{x}`` coverage bins, in the struct's field order (matching the
#: frontend's ``coverage.genome``): the highest threshold first.
COVERAGE_BINS: tuple[int, ...] = (100, 50, 30, 25, 20, 15, 10, 5, 1)

#: Row fields (plus the ``freq_meta`` global) the reshape consumes rather than
#: exports: ``freq``/``grpmax``/``fafmax``/``histograms`` are synthesized into
#: ``joint``/``genome``, so listing one must not write a duplicate top-level
#: column.
CONSUMED_FIELDS = frozenset({"freq", "freq_meta", "grpmax", "fafmax", "histograms"})

__all__ = [
    "CONTIG_NUMBERS",
    "COVERAGE_BINS",
    "BrowserSchemaError",
    "OverallFrequency",
    "ancestry_stratum_positions",
    "callstats_expr",
    "coverage_expr",
    "coverage_stats_type",
    "finite_floats",
    "flags_expr",
    "genome_expr",
    "in_silico_predictors_type",
    "joint_expr",
    "overall_frequency",
    "population_expr",
    "quality_metrics_expr",
    "require_members",
    "required_global",
    "required_row_field",
    "site_quality_metrics_expr",
    "stratum_index",
    "stratum_key",
    "to_browser_sites",
    "transcript_consequences_expr",
    "unique_stratum_index",
    "validate_preconditions",
    "variant_id_expr",
    "xpos_expr",
]


class BrowserSchemaError(ValueError):
    """The input MatrixTable does not have the browser reshape's required shape.

    Raised for the documented preconditions (a ``(locus, alleles)`` row key, an
    ``array<str>`` biallelic ``alleles``, the run's reference genome) and for a
    required source column or member the synthesized columns consume. It is a
    data-validation error, never a bare :class:`AssertionError`.
    """


# --- required inputs ---------------------------------------------------------


def required_row_field(mt: hl.MatrixTable, name: str) -> hl.expr.Expression:
    """The named row field, or a loud error when the input lacks it.

    The reshape's required source columns are preconditions: they are consumed
    into synthesized columns, so a missing one is never degraded to a typed null.
    """
    if name not in mt.row.dtype.fields:
        raise BrowserSchemaError(f"the browser reshape requires the {name!r} row field")
    return mt[name]


def required_global(mt: hl.MatrixTable, name: str) -> hl.expr.Expression:
    """The named table global, or a loud error when the input lacks it."""
    if name not in mt.globals.dtype.fields:
        raise BrowserSchemaError(f"the browser reshape requires the {name!r} global")
    return mt[name]


def require_members(dtype: hl.tstruct, names: Iterable[str], *, where: str) -> None:
    """Raise a loud error naming every member ``dtype`` is missing.

    ``where`` describes the struct (e.g. ``"the 'grpmax' row field"``) so the
    message reads ``the browser reshape requires <where> to have [...] member(s)``.
    """
    missing = [name for name in names if name not in dtype.fields]
    if missing:
        raise BrowserSchemaError(
            f"the browser reshape requires {where} to have {missing} member(s)"
        )

# --- identity ----------------------------------------------------------------


def variant_id_expr(obj: hl.Table | hl.MatrixTable) -> hl.expr.StringExpression:
    """``contig-pos-ref-alt`` with any ``chr`` prefix stripped.

    The browser synthesizes this form; unlike the GWAS ``variant_id`` it is
    hyphen-separated. ``obj`` is any row-bearing object with ``locus`` and
    ``alleles``.
    """
    import hail as hl

    contig = obj.locus.contig
    stripped = hl.if_else(contig.startswith("chr"), contig[3:], contig)
    return hl.format(
        "%s-%d-%s-%s",
        stripped,
        obj.locus.position,
        obj.alleles[0],
        obj.alleles[1],
    )


def xpos_expr(obj: hl.Table | hl.MatrixTable) -> hl.expr.Int64Expression:
    """The browser's always-materialized ``xpos`` as an ``int64``.

    ``contig_number * 1_000_000_000 + position`` with the ``chr`` prefix
    stripped, ``X``/``Y``/``M``/``MT`` mapping to 23/24/25/25. An unsupported
    contig fails loudly when the expression is evaluated.
    """
    import hail as hl

    contig = obj.locus.contig
    stripped = hl.if_else(contig.startswith("chr"), contig[3:], contig)
    number = hl.literal(CONTIG_NUMBERS).get(stripped)
    return (
        hl.case(missing_false=True)
        .when(
            hl.is_defined(number),
            hl.int64(number) * hl.int64(1_000_000_000) + hl.int64(obj.locus.position),
        )
        .or_error(hl.format("xpos: unsupported contig %s", contig))
    )


# --- VEP lift ----------------------------------------------------------------


def transcript_consequences_expr(
    mt: hl.MatrixTable,
) -> hl.expr.ArrayExpression:
    """The row-level ``transcript_consequences`` lifted from ``vep``.

    Elements carry the browser's mapped schema (``major_consequence`` from
    ``most_severe_consequence``, ``is_canonical`` from ``canonical == 1``,
    ``is_mane_select`` from a defined, non-empty ``mane_select``, the
    HGVS/gene/consequence/lof/biotype pass-throughs) and ``domains`` is omitted.
    A source element missing any of the required ``gene_id``/``gene_symbol``/
    ``transcript_id``/``most_severe_consequence`` (``major_consequence``) fields
    is dropped, matching the browser's reference Hail reader
    (``hail.rs:extract_transcript_consequences``): the pinned backend types those
    four as mandatory ``String``s, so emitting a null there makes the whole
    ``transcript_consequences`` column unparseable on the detail path. The array
    is ordered canonical-first, then MANE-select, then original order, because
    the DuckDB backend reads ``transcript_consequences[1]``. When the input has no
    ``vep.transcript_consequences`` the result is an empty array of the mapped
    element type.
    """
    import hail as hl

    if (
        "vep" not in mt.row.dtype.fields
        or "transcript_consequences" not in mt.vep.dtype.fields
    ):
        return hl.empty_array(transcript_consequence_type())

    transcript_consequences = mt.vep.transcript_consequences

    # Match the browser's reference Hail reader: an element whose required
    # fields are undefined is omitted, not emitted as a null (which the pinned
    # DuckDB backend's strict `Vec<TranscriptConsequence>` would reject).
    transcript_consequences = transcript_consequences.filter(
        lambda consequence: hl.is_defined(consequence.gene_id)
        & hl.is_defined(consequence.gene_symbol)
        & hl.is_defined(consequence.transcript_id)
        & hl.is_defined(consequence.most_severe_consequence)
    )

    def element(consequence: hl.expr.StructExpression) -> hl.expr.StructExpression:
        return hl.struct(
            major_consequence=consequence.most_severe_consequence,
            is_canonical=consequence.canonical == 1,
            is_mane_select=hl.is_defined(consequence.mane_select)
            & (consequence.mane_select != ""),
            hgvsc=consequence.hgvsc,
            hgvsp=consequence.hgvsp,
            gene_id=consequence.gene_id,
            gene_symbol=consequence.gene_symbol,
            transcript_id=consequence.transcript_id,
            consequence_terms=consequence.consequence_terms,
            lof=consequence.lof,
            lof_filter=consequence.lof_filter,
            lof_flags=consequence.lof_flags,
            biotype=consequence.biotype,
        )

    indexed = hl.range(hl.len(transcript_consequences)).map(
        lambda i: hl.struct(
            rank=(
                hl.case(missing_false=True)
                .when(transcript_consequences[i].canonical == 1, 0)
                .when(
                    hl.is_defined(transcript_consequences[i].mane_select)
                    & (transcript_consequences[i].mane_select != ""),
                    1,
                )
                .default(2)
            ),
            original=i,
            element=element(transcript_consequences[i]),
        )
    )
    ordered = hl.sorted(indexed, key=lambda record: (record.rank, record.original))
    return ordered.map(lambda record: record.element)


def transcript_consequence_type() -> hl.tstruct:
    """The mapped element type used for the empty array when ``vep`` is absent."""
    import hail as hl

    return hl.tstruct(
        major_consequence=hl.tstr,
        is_canonical=hl.tbool,
        is_mane_select=hl.tbool,
        hgvsc=hl.tstr,
        hgvsp=hl.tstr,
        gene_id=hl.tstr,
        gene_symbol=hl.tstr,
        transcript_id=hl.tstr,
        consequence_terms=hl.tarray(hl.tstr),
        lof=hl.tstr,
        lof_filter=hl.tstr,
        lof_flags=hl.tstr,
        biotype=hl.tstr,
    )


# --- frequency pivot ---------------------------------------------------------


def stratum_key(entry: Mapping[str, str]) -> str:
    """The deterministic browser stratum key for one ``freq_meta`` entry.

    The grammar (after dropping ``group``): overall ``adj`` -> ``all``, overall
    ``raw`` -> ``all_raw``, ``{gen_anc: g}`` -> ``g``, ``{sex: s}`` -> ``s``,
    ``{gen_anc: g, sex: s}`` -> ``g_s``, ``{cohort: c}`` -> ``cohort_c``, and
    anything else -> ``k1=v1_k2=v2`` over its remaining metadata sorted by key,
    with a ``_raw`` suffix for ``group=raw``. Empty metadata with no
    ``adj``/``raw`` group -- the group-less overall coverage stratum -- also
    returns ``all``.
    """
    meta = {key: value for key, value in dict(entry).items() if value is not None}
    group = meta.pop("group", None)

    if not meta:
        if group == "raw":
            return "all_raw"
        # The overall stratum. An ``adj`` frequency entry and group-less coverage
        # metadata (whose ``group`` key ``compute_coverage_stats`` strips) both map
        # to the same clear key.
        return "all"

    if group == "adj":
        keys = set(meta)
        if keys == {"gen_anc"}:
            return meta["gen_anc"]
        if keys == {"sex"}:
            return meta["sex"]
        if keys == {"gen_anc", "sex"}:
            return f"{meta['gen_anc']}_{meta['sex']}"
        if keys == {"cohort"}:
            return meta["cohort"].upper()

    key = "_".join(f"{name}={meta[name]}" for name in sorted(meta))
    if group == "raw":
        key = f"{key}_raw" if key else "raw"
    return key


def ancestry_stratum_positions(meta: Sequence[Mapping[str, str]]) -> list[int]:
    """Positions of the ``adj`` ``{gen_anc}`` / ``{gen_anc, sex}`` strata.

    These are the browser's ``joint.freq.all.ancestry_groups`` rows. All frequency
    stratums with group ``adj`` are exported.
    """
    positions: list[int] = []
    for position, entry in enumerate(meta):
        values = {key: value for key, value in dict(entry).items() if value is not None}
        if values.get("group") != "adj":
            continue
        positions.append(position)
    return positions


def unique_stratum_index(
    meta: Sequence[Mapping[str, str]], *, label: str
) -> dict[str, int]:
    """Map each generated stratum key to its position, rejecting collisions.

    ``label`` names the stratum family so the loud :class:`BrowserSchemaError`
    reads ``duplicate browser <label> stratum key ...``. A second entry that
    generates an already-seen key fails loudly. This is the scan behind
    :func:`stratum_index`; it does not check the overall ``adj`` entry, which
    only the frequency grammar requires.
    """
    index: dict[str, int] = {}
    for position, entry in enumerate(meta):
        key = stratum_key(entry)
        if key in index:
            raise BrowserSchemaError(
                f"duplicate browser {label} stratum key {key!r} at "
                f"positions {index[key]} and {position}"
            )
        index[key] = position
    return index


def stratum_index(freq_meta: Sequence[Mapping[str, str]]) -> dict[str, int]:
    """Map each generated stratum key to its positional ``freq`` index.

    Exactly one overall ``adj`` entry is required (it becomes ``all``); a second
    entry that generates an already-seen key fails loudly.
    """
    overall_adj = 0
    for entry in freq_meta:
        meta = {k: v for k, v in dict(entry).items() if v is not None}
        if set(meta) == {"group"} and meta["group"] == "adj":
            overall_adj += 1
    if overall_adj != 1:
        raise BrowserSchemaError(
            "freq_meta must contain exactly one overall adj entry "
            f"(found {overall_adj})"
        )

    return unique_stratum_index(freq_meta, label="frequency")


class OverallFrequency(NamedTuple):
    """The overall ``adj`` callstats and the metadata ``joint`` also needs.

    Returned by :func:`overall_frequency`, which both :func:`genome_expr` (the
    ``all`` callstats) and :func:`joint_expr` (the callstats plus the ancestry
    ``Population`` rows) consume.
    """

    freq: hl.expr.ArrayExpression
    meta: list[Mapping[str, str]]
    index: dict[str, int]
    all_stats: hl.expr.StructExpression


def overall_frequency(mt: hl.MatrixTable) -> OverallFrequency:
    """The overall ``adj`` callstats plus the evaluated ``freq_meta``.

    Validates the required ``freq``/``freq_meta`` pair and the ``freq_meta``
    grammar (exactly one overall ``adj`` entry, no duplicate stratum keys).
    """
    import hail as hl

    freq = required_row_field(mt, "freq")
    freq_meta = required_global(mt, "freq_meta")
    meta = [dict(entry) for entry in hl.eval(freq_meta)]
    index = stratum_index(meta)
    return OverallFrequency(
        freq=freq,
        meta=meta,
        index=index,
        all_stats=callstats_expr(freq[index["all"]]),
    )


def callstats_expr(entry: hl.expr.StructExpression) -> hl.expr.StructExpression:
    """A browser callstats struct built from one source ``freq`` element.

    ``hemizygote_count`` is always emitted: the source element's value when it
    defines one, else ``0``. ``annotate_freq`` never computes hemizygotes, so the
    browser's required field stays present (showing zero) rather than being
    dropped or emitted as a typed null.
    """
    import hail as hl

    return hl.struct(
        ac=entry.AC,
        an=entry.AN,
        af=entry.AF,
        homozygote_count=entry.homozygote_count,
        hemizygote_count=(
            entry.hemizygote_count
            if "hemizygote_count" in entry.dtype.fields
            else hl.int32(0)
        ),
    )


def population_expr(
    entry: hl.expr.StructExpression,
    population_id: hl.expr.StringExpression,
) -> hl.expr.StructExpression:
    """One ``joint.freq.all.ancestry_groups`` element (the browser ``Population``)."""
    import hail as hl

    stats = callstats_expr(entry)
    return hl.struct(
        id=population_id,
        ac=stats.ac,
        an=stats.an,
        af=stats.af,
        homozygote_count=stats.homozygote_count,
        hemizygote_count=stats.hemizygote_count,
    )


def genome_expr(mt: hl.MatrixTable) -> hl.expr.StructExpression:
    """The ``genome`` struct: the overall callstats, quality metrics and counts.

    ``genome.freq.all`` is the one overall ``adj`` entry of the required
    ``freq``/``freq_meta`` (the DuckDB list/detail summary path). Every other
    stratum is dropped because the browser reads populations from
    ``joint.freq.all.ancestry_groups``. ``quality_metrics`` is reshaped from the
    required ``histograms``, and the top-level ``ac``/``an``/``homozygote_count``/
    ``hemizygote_count`` mirror ``freq.all`` (the frontend's ``SequencingTypeData``
    counts). A missing ``freq``/``freq_meta``/``histograms``, or a malformed
    ``freq_meta``, raises :class:`BrowserSchemaError`.
    """
    import hail as hl

    all_stats = overall_frequency(mt).all_stats
    return hl.struct(
        freq=hl.struct(all=all_stats),
        quality_metrics=quality_metrics_expr(mt),
        ac=all_stats.ac,
        an=all_stats.an,
        homozygote_count=all_stats.homozygote_count,
        hemizygote_count=all_stats.hemizygote_count,
    )


# --- joint: the frequency table ----------------------------------------------


def flags_expr(filters: hl.expr.Expression) -> hl.expr.ArrayExpression:
    """The variant's ``filters`` as a sorted ``array<string>``.

    The source is the Variant QC ``filters`` set (``set<str>``); a per-row null
    value becomes the empty array, and a defined empty set already is ``[]``.
    """
    import hail as hl

    if isinstance(filters.dtype, hl.tset):
        values = hl.or_else(filters, hl.empty_set(hl.tstr))
    else:
        values = hl.or_else(filters, hl.empty_array(hl.tstr))
    return hl.sorted(hl.array(values))


def site_quality_metrics_expr(mt: hl.MatrixTable) -> hl.expr.ArrayExpression:
    """``genome.quality_metrics.site_quality_metrics``: the browser's 12 metrics.

    The frontend's fixed ``METRIC_ORDER`` (``VariantSiteQualityMetrics.tsx``),
    as ``{metric, value}`` pairs. ``mt.info`` carries the site/AS INFO fields
    (``QUALapprox`` and the ``AS_*`` set) and ``mt.inbreeding_coef`` the site
    inbreeding coefficient; a missing field raises :class:`BrowserSchemaError`.
    Every value is cast to ``float64`` so the array has one input-independent
    dtype. The browser's ``SiteQuality`` is the site-level ``QUALapprox``, and
    its lowercase ``inbreeding_coeff`` is this repo's ``inbreeding_coef``.
    """
    import hail as hl

    info = required_row_field(mt, "info")
    require_members(
        info.dtype,
        (
            "QUALapprox",
            "AS_FS",
            "AS_MQ",
            "AS_MQRankSum",
            "AS_pab_max",
            "AS_QUALapprox",
            "AS_QD",
            "AS_ReadPosRankSum",
            "AS_SOR",
            "AS_VarDP",
            "AS_VQSLOD",
        ),
        where="the 'info' row field",
    )
    inbreeding = required_row_field(mt, "inbreeding_coef")

    return hl.array(
        [
            hl.struct(metric=hl.literal(metric), value=hl.float64(value))
            for metric, value in (
                ("SiteQuality", info.QUALapprox),
                ("inbreeding_coeff", inbreeding),
                ("AS_FS", info.AS_FS),
                ("AS_MQ", info.AS_MQ),
                ("AS_MQRankSum", info.AS_MQRankSum),
                ("AS_pab_max", info.AS_pab_max),
                ("AS_QUALapprox", info.AS_QUALapprox),
                ("AS_QD", info.AS_QD),
                ("AS_ReadPosRankSum", info.AS_ReadPosRankSum),
                ("AS_SOR", info.AS_SOR),
                ("AS_VarDP", info.AS_VarDP),
                ("AS_VQSLOD", info.AS_VQSLOD),
            )
        ]
    )


def quality_metrics_expr(mt: hl.MatrixTable) -> hl.expr.StructExpression:
    """``genome.quality_metrics`` reshaped from the required ``histograms``.

    The annotation's ``histograms`` is ``{raw_qual_hists, qual_hists}``, each
    holding ``gq_hist_all``/``gq_hist_alt``/``dp_hist_all``/``dp_hist_alt``/
    ``ab_hist_alt`` (``hl.agg.hist`` structs -- the browser's ``HistogramData``).
    They map onto the browser's ``genotype_quality``/``genotype_depth``/
    ``allele_balance`` ``HistogramSet`` keys with ``all_adj``/``alt_adj`` from
    ``qual_hists`` and ``all_raw``/``alt_raw`` from ``raw_qual_hists``. A missing
    ``histograms``, member or named histogram raises. ``site_quality_metrics`` is
    built from the ``info``/``inbreeding_coef`` sources.
    """
    import hail as hl

    histograms = required_row_field(mt, "histograms")
    require_members(
        histograms.dtype,
        ("qual_hists", "raw_qual_hists"),
        where="the 'histograms' row field",
    )

    qual_hists = histograms.qual_hists
    raw_qual_hists = histograms.raw_qual_hists
    histogram_names = (
        "gq_hist_all",
        "gq_hist_alt",
        "dp_hist_all",
        "dp_hist_alt",
        "ab_hist_alt",
    )
    for member, histograms_struct in (
        ("qual_hists", qual_hists),
        ("raw_qual_hists", raw_qual_hists),
    ):
        require_members(
            histograms_struct.dtype,
            histogram_names,
            where=f"the 'histograms' row field's {member!r}",
        )

    return hl.struct(
        genotype_quality=hl.struct(
            all_adj=qual_hists.gq_hist_all,
            alt_adj=qual_hists.gq_hist_alt,
            all_raw=raw_qual_hists.gq_hist_all,
            alt_raw=raw_qual_hists.gq_hist_alt,
        ),
        genotype_depth=hl.struct(
            all_adj=qual_hists.dp_hist_all,
            alt_adj=qual_hists.dp_hist_alt,
            all_raw=raw_qual_hists.dp_hist_all,
            alt_raw=raw_qual_hists.dp_hist_alt,
        ),
        allele_balance=hl.struct(
            alt_adj=qual_hists.ab_hist_alt,
            alt_raw=raw_qual_hists.ab_hist_alt,
        ),
        site_quality_metrics=site_quality_metrics_expr(mt),
    )


def joint_expr(mt: hl.MatrixTable) -> hl.expr.StructExpression:
    """The mandatory ``joint`` column: the browser's frequency table.

    Synthesized from the required ``freq``/``freq_meta`` (the overall callstats
    plus one ``Population`` per ``adj`` ancestry stratum), ``grpmax``, ``fafmax``
    and ``filters``. The dtype is fixed and input-independent: a missing column,
    a struct without a required member, or a ``grpmax``/``fafmax`` without an
    acceptable ancestry label raises :class:`BrowserSchemaError`, never a typed
    null.
    """
    import hail as hl

    frequency = overall_frequency(mt)
    freq = frequency.freq
    meta = frequency.meta
    index = frequency.index
    all_stats = frequency.all_stats

    positions = ancestry_stratum_positions(meta)
    if positions:
        ancestry_groups = hl.array(
            [
                population_expr(
                    freq[position],
                    hl.literal(stratum_key(meta[position])),
                )
                for position in positions
            ]
        )
    else:
        # No ancestry strata: still a fixed ``Population`` element type (the
        # no-ancestry input's ``grpmax``/``fafmax`` shape is rejected below).
        population_type = population_expr(freq[index["all"]], hl.missing(hl.tstr)).dtype
        ancestry_groups = hl.empty_array(population_type)

    grpmax = required_row_field(mt, "grpmax")
    require_members(grpmax.dtype, ("AC", "AN", "AF"), where="the 'grpmax' row field")
    grpmax_members = set(grpmax.dtype.fields)
    labels = [label for label in ("gen_anc", "cohort") if label in grpmax_members]
    if len(labels) != 1:
        raise BrowserSchemaError(
            "the browser reshape requires the 'grpmax' row field to have exactly "
            f"one ancestry label ('gen_anc' or 'cohort'), found {labels}"
        )

    fafmax = required_row_field(mt, "fafmax")
    require_members(
        fafmax.dtype,
        ("faf95_max", "faf95_max_gen_anc"),
        where="the 'fafmax' row field",
    )

    filters = required_row_field(mt, "filters")

    return hl.struct(
        freq=hl.struct(all=all_stats.annotate(ancestry_groups=ancestry_groups)),
        grpmax=hl.struct(
            ac=grpmax.AC,
            an=grpmax.AN,
            af=grpmax.AF,
            gen_anc=grpmax[labels[0]],
        ),
        fafmax=hl.struct(
            faf95_max=fafmax.faf95_max,
            faf95_max_gen_anc=fafmax.faf95_max_gen_anc,
        ),
        flags=flags_expr(filters),
    )


# --- coverage pivot ----------------------------------------------------------


def coverage_stats_type() -> hl.tstruct:
    """The gnomAD coverage element struct (the source ``coverage_stats`` shape).

    The leaves match ``gnomad.utils.sparse_mt.get_coverage_agg_func`` finalized by
    the default ``coverage_over_x_bins``: ``mean``, ``median_approx``,
    ``total_DP``, then one ``over_{x}`` fraction per bin in reversed bin order
    (``over_100`` ... ``over_1``, the struct's field order). This is the *source*
    element; :func:`coverage_expr` selects the frontend's ``coverage.genome``
    subset (dropping ``total_DP``) from it.
    """
    import hail as hl

    return hl.tstruct(
        mean=hl.tfloat64,
        median_approx=hl.tint32,
        total_DP=hl.tint64,
        **{f"over_{x}": hl.tfloat64 for x in COVERAGE_BINS},
    )


def coverage_expr(mt: hl.MatrixTable) -> hl.expr.StructExpression:
    """The ``coverage`` struct: ``{genome: <overall element>}``.

    Requires ``coverage_stats``/``coverage_stats_meta``. The single element whose
    metadata is empty (the overall stratum) is selected and rebuilt with the
    frontend's ``coverage.genome`` members (``mean``, ``median_approx`` and the
    ``over_{x}`` fractions); the source ``total_DP`` is excluded and the
    filtered/cohort strata are dropped because the browser reads none of them.
    Zero or more than one overall element, or a missing member, raises
    :class:`BrowserSchemaError`; the column is never a typed null.
    """
    import hail as hl

    stats = required_row_field(mt, "coverage_stats")
    meta = required_global(mt, "coverage_stats_meta")
    entries = [dict(entry) for entry in hl.eval(meta)]
    overall = [
        position
        for position, entry in enumerate(entries)
        if not {key: value for key, value in entry.items() if value is not None}
    ]
    if len(overall) != 1:
        raise BrowserSchemaError(
            "coverage_stats_meta must contain exactly one overall element "
            f"(empty metadata; found {len(overall)})"
        )

    element = stats[overall[0]]
    members = ("mean", "median_approx", *(f"over_{x}" for x in COVERAGE_BINS))
    require_members(element.dtype, members, where="the 'coverage_stats' element")

    return hl.struct(
        genome=hl.struct(
            mean=element.mean,
            median_approx=element.median_approx,
            **{f"over_{x}": element[f"over_{x}"] for x in COVERAGE_BINS},
        )
    )


# --- preconditions -----------------------------------------------------------


def validate_preconditions(mt: hl.MatrixTable, reference_genome: str) -> None:
    """Validate the browser reshape's documented input preconditions.

    The row key must be exactly ``(locus, alleles)``; ``alleles`` must be an
    ``array<str>`` with exactly two defined elements per row; and ``locus``'s
    reference genome must equal ``reference_genome``. Every violation raises
    :class:`BrowserSchemaError`; ``was_split`` is deliberately never consulted.
    """
    import hail as hl

    key_fields = tuple(mt.row_key.dtype.fields)
    if key_fields != ("locus", "alleles"):
        raise BrowserSchemaError(
            f"the browser reshape requires a (locus, alleles) row key, got {key_fields}"
        )
    if mt.alleles.dtype != hl.tarray(hl.tstr):
        raise BrowserSchemaError(
            "the browser reshape requires alleles to be array<string>, got "
            f"{mt.alleles.dtype}"
        )
    locus_reference = mt.locus.dtype.reference_genome
    if locus_reference is None or locus_reference.name != reference_genome:
        raise BrowserSchemaError(
            "the browser reshape requires the locus reference genome to equal "
            f"the run's reference genome {reference_genome!r}, got "
            f"{getattr(locus_reference, 'name', None)!r}"
        )

    not_biallelic = hl.len(mt.alleles) != 2
    with_nulls = hl.any(lambda allele: hl.is_missing(allele), mt.alleles)
    n_bad = mt.aggregate_rows(hl.agg.count_where(not_biallelic | with_nulls))
    if n_bad:
        raise BrowserSchemaError(
            f"the browser reshape requires biallelic rows; {n_bad} row(s) do not "
            "have exactly two defined alleles"
        )

    for name in (
        "freq",
        "grpmax",
        "fafmax",
        "histograms",
        "filters",
        "coverage_stats",
        "info",
        "inbreeding_coef",
    ):
        required_row_field(mt, name)
    for name in ("freq_meta", "coverage_stats_meta"):
        required_global(mt, name)


# --- the entry point ---------------------------------------------------------


def finite_floats(expr: hl.expr.Expression) -> hl.expr.Expression:
    """Replace every non-finite float leaf with missing, recursively.

    DuckDB's ``to_json`` renders ``NaN`` and ``Infinity`` as bare tokens, which
    are not valid JSON; the browser backend parses the exported nested columns
    with ``serde_json`` and rejects a column carrying one (``Failed to parse
    <column> JSON``). The annotation can legitimately hold non-finite floats --
    the site inbreeding coefficient and the allele-specific ``AS_VQSLOD`` are
    ``NaN`` where undefined -- so the export maps each non-finite float leaf to
    null (the browser's number fields are nullable) to keep every nested column
    parseable. Structs and arrays are rebuilt element-wise with their definedness
    preserved; non-float leaves pass through unchanged.
    """
    import hail as hl

    dtype = expr.dtype
    if isinstance(dtype, hl.tstruct):
        rebuilt = hl.struct(
            **{name: finite_floats(expr[name]) for name in dtype.fields}
        )
        return hl.or_missing(hl.is_defined(expr), rebuilt)
    if isinstance(dtype, hl.tarray) or isinstance(dtype, hl.tset):
        return expr.map(lambda element: finite_floats(element))
    if dtype in (hl.tfloat64, hl.tfloat32):
        return hl.or_missing(hl.is_finite(expr), expr)
    return expr


def to_browser_sites(
    mt: hl.MatrixTable, *, fields: Iterable[str], reference_genome: str
) -> hl.Table:
    """Reshape an input MatrixTable into the sites-only browser Site table.

    ``fields`` is the optional passthrough field list (``[browser_export].fields``
    / :data:`DEFAULT_FIELDS`): each listed field present on the input is carried
    through, and a listed-but-absent field is dropped. ``reference_genome`` is the
    run's reference genome (:data:`Settings.reference_genome`), validated against
    the locus key. The required source columns are validated up front
    (:func:`validate_required_sources`); entries and columns are dropped by the
    ``rows()`` projection, and the source MatrixTable's globals
    (``freq_meta``/``coverage_stats_meta``) are kept on the returned table. Every
    exported float leaf is passed through :func:`finite_floats`, so a non-finite
    source value becomes null rather than an unparseable duckdb ``NaN`` token.
    Export provenance is annotated by the ``browser_sites`` step, not here.
    """
    import hail as hl

    validate_preconditions(mt, reference_genome)
    enabled = tuple(fields)
    row_fields = set(mt.row.dtype.fields)

    annotations: dict[str, hl.expr.Expression] = {
        "variant_id": variant_id_expr(mt),
        "xpos": xpos_expr(mt),
        "rsids": (
            mt.rsids if "rsids" in row_fields else hl.missing(hl.tarray(hl.tstr))
        ),
        "caid": mt.caid if "caid" in row_fields else hl.missing(hl.tstr),
        "exome": structural_field(mt, "exome", enabled, null_exome_struct_expr()),
        "genome": genome_expr(mt),
        "joint": joint_expr(mt),
        "transcript_consequences": transcript_consequences_expr(mt),
        "in_silico_predictors": structural_field(
            mt,
            "in_silico_predictors",
            enabled,
            hl.missing(in_silico_predictors_type()),
        ),
        "coverage": coverage_expr(mt),
    }

    for name in enabled:
        if name in annotations or name in CONSUMED_FIELDS:
            continue
        if name in row_fields:
            annotations[name] = mt[name]

    # No exported float may be non-finite: DuckDB's to_json would emit an invalid
    # ``NaN``/``Infinity`` token and the backend's JSON parse would reject the
    # whole column. Applied last, over every synthesized and passthrough column.
    annotations = {name: finite_floats(expr) for name, expr in annotations.items()}

    return mt.select_rows(**annotations).rows()


def structural_field(
    mt: hl.MatrixTable,
    name: str,
    enabled: tuple[str, ...],
    null_expr: hl.expr.Expression,
) -> hl.expr.Expression:
    """An optional structural column's source value, else its typed null.

    The optional structural columns (``exome``, ``in_silico_predictors``) take
    their source value only when listed in ``fields`` and present; otherwise they
    are materialized as the supplied typed null so the browser's fixed SQL still
    prepares. ``genome``, ``joint`` and ``coverage`` are synthesized
    unconditionally instead.
    """
    if name in enabled and name in mt.row.dtype.fields:
        return mt[name]
    return null_expr


def null_exome_struct_expr() -> hl.expr.StructExpression:
    """A typed null struct with the ``freq.all.ac/an`` shape the browser reads.

    The absent-column fallback for the optional ``exome`` column, whose elements
    the DuckDB list path reads as ``exome.freq."all".{ac,an}``.
    """
    import hail as hl

    return hl.missing(
        hl.tstruct(freq=hl.tstruct(all=hl.tstruct(ac=hl.tint32, an=hl.tint32)))
    )


def in_silico_predictors_type() -> hl.tstruct:
    """The ``InSilicoPredictors`` shape used for the absent-column typed null.

    Non-empty on purpose: Parquet cannot encode an empty struct, and the DuckDB
    detail path only needs ``to_json(in_silico_predictors)`` to resolve.
    """
    import hail as hl

    return hl.tstruct(
        cadd=hl.tstruct(phred=hl.tfloat64, raw_score=hl.tfloat64),
        revel_max=hl.tfloat64,
        spliceai_ds_max=hl.tfloat64,
        pangolin_largest_ds=hl.tfloat64,
        phylop=hl.tfloat64,
        sift_max=hl.tfloat64,
        polyphen_max=hl.tfloat64,
    )
