"""Tests for the MatrixTable -> browser Site table reshape (``browser.schema``).

Everything here runs against small in-memory Hail fixtures, so the reshape's
expression functions are exercised directly: the VEP lift and its canonical-first
ordering, the frequency pivot's deterministic stratum grammar and callstats
shape, the synthesized ``joint`` / ``genome.quality_metrics`` / ``coverage``
columns, ``variant_id``/``xpos`` for chr-prefixed and bare contigs, the mandatory
typed nulls and optional passthrough/drop policy, the fail-fast required-source
preconditions, and the loud input precondition failures. The conftest ``hl``
fixture supplies the local Hail session (it skips only when the host has no
working JVM, which ``AGENTS.md`` says is not this host).
"""

from __future__ import annotations

import pytest

from gnomad_meb.constants import DEFAULT_FIELDS, MANDATORY_FIELDS
from gnomad_meb.lib.browser.schema import (
    COVERAGE_BINS,
    BrowserSchemaError,
    ancestry_stratum_positions,
    coverage_expr,
    coverage_stats_type,
    finite_floats,
    genome_expr,
    joint_expr,
    quality_metrics_expr,
    site_quality_metrics_expr,
    stratum_index,
    stratum_key,
    to_browser_sites,
    transcript_consequences_expr,
    variant_id_expr,
    xpos_expr,
)

# --- fixtures ---------------------------------------------------------------


def _meta_expr(hl, meta: list[dict[str, str]]):
    """An ``Array[Struct]`` ``freq_meta`` literal; absent keys become null."""
    return hl.array(
        [
            hl.struct(
                group=m.get("group", hl.missing(hl.tstr)),
                gen_anc=m.get("gen_anc", hl.missing(hl.tstr)),
                sex=m.get("sex", hl.missing(hl.tstr)),
                cohort=m.get("cohort", hl.missing(hl.tstr)),
            )
            for m in meta
        ]
    )


def _freq_expr(hl, freq: list[dict[str, object]], *, hemizygote: bool = False):
    """An ``Array[Struct]`` ``freq`` literal in the source callstats shape."""
    fields = {"AC", "AN", "AF", "homozygote_count"}
    if hemizygote:
        fields.add("hemizygote_count")
    return hl.array(
        [
            hl.struct(**{field: entry[field] for field in entry if field in fields})
            for entry in freq
        ]
    )


def _hist(hl, marker: int):
    """One source ``hl.agg.hist`` struct carrying a distinguishable ``marker``."""
    return hl.struct(
        bin_edges=hl.array([0.0, 1.0]),
        bin_freq=hl.array([marker]),
        n_larger=marker,
        n_smaller=marker,
    )


#: Distinct marker per ``histograms`` slot, so a test can prove the mapping.
_HIST_MARKERS = {
    ("qual_hists", "gq_hist_all"): 10,
    ("qual_hists", "gq_hist_alt"): 11,
    ("qual_hists", "dp_hist_all"): 12,
    ("qual_hists", "dp_hist_alt"): 13,
    ("qual_hists", "ab_hist_alt"): 14,
    ("raw_qual_hists", "gq_hist_all"): 20,
    ("raw_qual_hists", "gq_hist_alt"): 21,
    ("raw_qual_hists", "dp_hist_all"): 22,
    ("raw_qual_hists", "dp_hist_alt"): 23,
    ("raw_qual_hists", "ab_hist_alt"): 24,
}


def _hists_expr(
    hl,
    *,
    members: tuple[str, ...] = (
        "gq_hist_all",
        "gq_hist_alt",
        "dp_hist_all",
        "dp_hist_alt",
        "ab_hist_alt",
    ),
):
    """The top-level ``histograms`` struct: ``{raw_qual_hists, qual_hists}``."""

    def group(name: str):
        return hl.struct(
            **{member: _hist(hl, _HIST_MARKERS[(name, member)]) for member in members}
        )

    return hl.struct(
        qual_hists=group("qual_hists"), raw_qual_hists=group("raw_qual_hists")
    )


def _info_expr(hl):
    """The site/AS INFO struct the site-quality reshape consumes."""
    return hl.struct(
        QUALapprox=hl.int64(100),
        AS_FS=0.5,
        AS_MQ=60.0,
        AS_MQRankSum=0.1,
        AS_pab_max=1.0,
        AS_QUALapprox=hl.int64(90),
        AS_QD=10.0,
        AS_ReadPosRankSum=0.0,
        AS_SOR=0.7,
        AS_VarDP=hl.int32(10),
        AS_VQSLOD=1.0,
    )


def _tc(
    hl,
    *,
    canonical: int = 0,
    mane: str | None = None,
    gene: str = "GENE",
    csq: str = "missense_variant",
):
    """One source ``vep.transcript_consequences`` element."""
    return hl.struct(
        most_severe_consequence=csq,
        canonical=canonical,
        mane_select=mane if mane is not None else hl.missing(hl.tstr),
        hgvsc="c.1A>T",
        hgvsp="p.Lys1Asn",
        gene_id=f"ENSG_{gene}",
        gene_symbol=gene,
        transcript_id=f"ENST_{gene}",
        consequence_terms=[csq],
        lof="LC",
        lof_filter="",
        lof_flags="",
        biotype="protein_coding",
        domains=["PF00001"],
    )


def _mt(
    hl,
    *,
    contig: str = "chr1",
    position: int = 100,
    alleles: list[str] | None = None,
    reference_genome: str = "GRCh38",
    rows: dict | None = None,
):
    """A one-row MatrixTable keyed ``(locus, alleles)``."""
    row_columns = {
        "locus": [hl.locus(contig, position, reference_genome=reference_genome)],
        "alleles": [alleles if alleles is not None else ["A", "T"]],
    }
    if rows:
        row_columns.update({name: [value] for name, value in rows.items()})
    mt = hl.MatrixTable.from_parts(
        rows=row_columns,
        cols={"s": ["S1"]},
        entries={"GT": [[hl.call(0, 1)]]},
    )
    return mt.key_rows_by("locus", "alleles")


def _row(hl, table):
    """The single collected row of a one-row table."""
    return table.collect()[0]


#: ``freq_meta``: overall adj + overall raw + an ancestry stratum, a sex-only
#: stratum, an ancestry x sex stratum and a cohort stratum.
META = [
    {"group": "adj"},
    {"group": "raw"},
    {"gen_anc": "afr", "group": "adj"},
    {"sex": "XX", "group": "adj"},
    {"gen_anc": "afr", "sex": "XX", "group": "adj"},
    {"cohort": "c1", "group": "adj"},
]

FREQ = [
    {"AC": 10, "AN": 100, "AF": 0.1, "homozygote_count": 2},
    {"AC": 9, "AN": 90, "AF": 0.1, "homozygote_count": 1},
    {"AC": 3, "AN": 30, "AF": 0.1, "homozygote_count": 1},
    {"AC": 4, "AN": 40, "AF": 0.1, "homozygote_count": 1},
    {"AC": 2, "AN": 20, "AF": 0.1, "homozygote_count": 1},
    {"AC": 1, "AN": 10, "AF": 0.1, "homozygote_count": 1},
]

COVERAGE_BIN_LIST = (100, 50, 30, 25, 20, 15, 10, 5, 1)

#: The overall stratum (empty metadata) plus two named strata. Coverage metadata
#: carries no ``group`` key, so ``{}`` is the overall stratum.
COVERAGE_META = [
    {},
    {"filtered": "pass"},
    {"cohort": "c1"},
]

COVERAGE_STATS = [
    {"mean": 1.0, "median_approx": 1, "total_DP": 10, "over": {100: 0.5, 1: 0.8}},
    {"mean": 2.0, "median_approx": 2, "total_DP": 20, "over": {100: 0.4, 1: 0.7}},
    {"mean": 3.0, "median_approx": 3, "total_DP": 30, "over": {100: 0.3, 1: 0.6}},
]


def _coverage_element(hl, *, mean=1.0, median_approx=1, total_DP=1, over=None):
    """One source ``coverage_stats`` element in the gnomAD finalized shape."""
    over = over or {}
    return hl.struct(
        mean=mean,
        median_approx=median_approx,
        total_DP=hl.int64(total_DP),
        **{f"over_{x}": over.get(x, 0.0) for x in COVERAGE_BIN_LIST},
    )


def _coverage_meta_expr(hl, meta: list[dict[str, str]]):
    """An ``Array[Dict[str, str]]`` ``coverage_stats_meta`` literal."""
    return hl.array([hl.literal(entry, hl.tdict(hl.tstr, hl.tstr)) for entry in meta])


def _coverage_stats_expr(hl, stats: list[dict]):
    """An ``Array[Struct]`` ``coverage_stats`` literal."""
    return hl.array([_coverage_element(hl, **stat) for stat in stats])


def _grpmax_expr(hl, *, label: str = "gen_anc", label_value: str = "afr"):
    """The gnomAD ``grpmax`` row struct; the ancestry field is named ``label``."""
    return hl.struct(AC=10, AN=100, AF=0.1, homozygote_count=2, **{label: label_value})


def _fafmax_expr(hl):
    """The gnomAD ``fafmax`` row struct (the ``faf99_*`` pair is also present)."""
    return hl.struct(
        faf95_max=0.05,
        faf95_max_gen_anc="nfe",
        faf99_max=0.02,
        faf99_max_gen_anc="nfe",
    )


def _source_mt(
    hl,
    *,
    meta=META,
    freq=FREQ,
    hemizygote: bool = False,
    grpmax_label: str = "gen_anc",
    grpmax_value: str = "afr",
    filters=None,
    histograms=None,
    coverage_meta=COVERAGE_META,
    coverage_stats=COVERAGE_STATS,
    rows: dict | None = None,
    globals: dict | None = None,
):
    """A one-row MT carrying every required source the reshape consumes.

    ``grpmax_label`` selects the ``grpmax`` ancestry field name (``gen_anc``
    normally, ``cohort`` under ``group_by_cohort``). ``filters`` defaults to a
    ``{"PASS"}`` set and may be overridden with a raw expression (e.g. a typed
    null). Any ``rows``/``globals`` entry is merged last, so a test can drop or
    replace a single source column.
    """
    row_columns = {
        "freq": _freq_expr(hl, freq, hemizygote=hemizygote),
        "grpmax": _grpmax_expr(hl, label=grpmax_label, label_value=grpmax_value),
        "fafmax": _fafmax_expr(hl),
        "histograms": histograms if histograms is not None else _hists_expr(hl),
        "coverage_stats": _coverage_stats_expr(hl, coverage_stats),
        "info": _info_expr(hl),
        "inbreeding_coef": hl.float64(0.01),
        "filters": (
            filters if filters is not None else hl.literal({"PASS"}, hl.tset(hl.tstr))
        ),
    }
    if rows:
        row_columns.update(rows)
    mt = _mt(hl, rows=row_columns)
    global_columns = {
        "freq_meta": _meta_expr(hl, meta),
        "coverage_stats_meta": _coverage_meta_expr(hl, coverage_meta),
    }
    if globals:
        global_columns.update(globals)
    return mt.annotate_globals(**global_columns)


# --- variant_id -------------------------------------------------------------


def test_variant_id_strips_chr(hl):
    table = _mt(hl, contig="chr1", position=100).rows()
    table = table.annotate(variant_id=variant_id_expr(table))
    assert _row(hl, table).variant_id == "1-100-A-T"


def test_variant_id_bare_contig(hl):
    table = _mt(hl, contig="12", position=34, reference_genome="GRCh37").rows()
    table = table.annotate(variant_id=variant_id_expr(table))
    assert _row(hl, table).variant_id == "12-34-A-T"


# --- xpos -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("contig", "position", "reference_genome", "expected"),
    [
        ("chr1", 123, "GRCh38", 1_000_000_123),
        ("1", 123, "GRCh37", 1_000_000_123),
        ("chrX", 5, "GRCh38", 23_000_000_005),
        ("chrY", 7, "GRCh38", 24_000_000_007),
    ],
)
def test_xpos_canonical_contigs(hl, contig, position, reference_genome, expected):
    table = _mt(
        hl, contig=contig, position=position, reference_genome=reference_genome
    ).rows()
    table = table.annotate(xpos=xpos_expr(table))
    value = _row(hl, table).xpos
    assert value == expected
    assert table.xpos.dtype == hl.tint64


def test_xpos_chr_m_is_25(hl):
    table = _mt(hl, contig="chrM", position=9).rows()
    table = table.annotate(xpos=xpos_expr(table))
    assert _row(hl, table).xpos == 25_000_000_009


def test_xpos_bare_mt_is_25(hl):
    table = _mt(hl, contig="MT", position=9, reference_genome="GRCh37").rows()
    table = table.annotate(xpos=xpos_expr(table))
    assert _row(hl, table).xpos == 25_000_000_009


def test_xpos_unsupported_contig_fails_loudly(hl):
    reference = hl.ReferenceGenome(
        "browser_test_unsupported", contigs=["chrUn"], lengths={"chrUn": 1000}
    )
    table = hl.Table.parallelize(
        [{"locus": hl.locus("chrUn", 3, reference_genome=reference)}],
        hl.tstruct(locus=hl.tlocus(reference)),
    )
    table = table.annotate(xpos=xpos_expr(table))
    with pytest.raises(Exception, match="unsupported contig"):
        table.xpos.collect()


# --- VEP lift ---------------------------------------------------------------


def test_transcript_consequences_mapping(hl):
    mt = _mt(
        hl,
        rows={
            "vep": hl.struct(
                most_severe_consequence="missense_variant",
                transcript_consequences=hl.array([_tc(hl, canonical=1, gene="CANON")]),
            )
        },
    )
    table = mt.annotate_rows(
        transcript_consequences=transcript_consequences_expr(mt)
    ).rows()
    element = _row(hl, table).transcript_consequences[0]
    assert element.major_consequence == "missense_variant"
    assert element.is_canonical is True
    assert element.is_mane_select is False
    assert element.hgvsc == "c.1A>T"
    assert element.hgvsp == "p.Lys1Asn"
    assert element.gene_id == "ENSG_CANON"
    assert element.gene_symbol == "CANON"
    assert element.transcript_id == "ENST_CANON"
    assert element.consequence_terms == ["missense_variant"]
    assert element.lof == "LC"
    assert element.lof_filter == ""
    assert element.lof_flags == ""
    assert element.biotype == "protein_coding"
    assert "domains" not in table.transcript_consequences.dtype.element_type.fields


def test_transcript_consequences_canonical_first_then_mane(hl):
    mt = _mt(
        hl,
        rows={
            "vep": hl.struct(
                most_severe_consequence="missense_variant",
                transcript_consequences=hl.array(
                    [
                        _tc(hl, gene="PLAIN"),
                        _tc(hl, gene="MANE", mane="NM_1"),
                        _tc(hl, canonical=1, gene="CANON"),
                        _tc(hl, gene="PLAIN2"),
                    ]
                ),
            )
        },
    )
    table = mt.annotate_rows(
        transcript_consequences=transcript_consequences_expr(mt)
    ).rows()
    genes = [tc.gene_symbol for tc in _row(hl, table).transcript_consequences]
    assert genes == ["CANON", "MANE", "PLAIN", "PLAIN2"]
    assert _row(hl, table).transcript_consequences[0].is_canonical is True
    assert _row(hl, table).transcript_consequences[1].is_mane_select is True


def test_transcript_consequences_drops_undefined_required_fields(hl):
    """A source element with a null required field is dropped, not passed through.

    Mirrors the browser's reference Hail reader; emitting a null ``gene_symbol``
    (or ``gene_id``/``transcript_id``/``major_consequence``) would make the
    column unparseable on the pinned DuckDB detail path.
    """
    missing_symbol = _tc(hl, gene="LNCRNA").annotate(gene_symbol=hl.missing(hl.tstr))
    mt = _mt(
        hl,
        rows={
            "vep": hl.struct(
                most_severe_consequence="missense_variant",
                transcript_consequences=hl.array(
                    [_tc(hl, canonical=1, gene="CANON"), missing_symbol]
                ),
            )
        },
    )
    table = mt.annotate_rows(
        transcript_consequences=transcript_consequences_expr(mt)
    ).rows()
    genes = [tc.gene_symbol for tc in _row(hl, table).transcript_consequences]
    assert genes == ["CANON"]


def test_transcript_consequences_empty_when_vep_absent(hl):
    mt = _mt(hl)
    table = mt.annotate_rows(
        transcript_consequences=transcript_consequences_expr(mt)
    ).rows()
    assert _row(hl, table).transcript_consequences == []
    element = table.transcript_consequences.dtype.element_type
    assert "major_consequence" in element.fields
    assert "domains" not in element.fields


# --- non-finite floats -------------------------------------------------------


def test_finite_floats_maps_non_finite_to_missing(hl):
    """NaN/Infinity become null at every depth; definedness and other types hold."""
    nan = hl.float64(float("nan"))
    source = hl.struct(
        finite=hl.float64(1.5),
        pos_inf=hl.float64(float("inf")),
        neg_inf=hl.float64(float("-inf")),
        nan=nan,
        missing=hl.missing(hl.tfloat64),
        nested=hl.struct(x=nan),
        arr=hl.array([nan, hl.float64(2.0)]),
        text="keep",
    )
    table = _mt(hl).annotate_rows(coerced=finite_floats(source)).rows()

    row = _row(hl, table).coerced
    assert row.finite == 1.5
    assert row.pos_inf is None
    assert row.neg_inf is None
    assert row.nan is None
    assert row.missing is None
    assert row.nested.x is None
    assert list(row.arr) == [None, 2.0]
    assert row.text == "keep"


def test_to_browser_sites_coerces_non_finite_floats(hl):
    """A non-finite source float is exported as null, not an invalid JSON token."""
    nan = hl.float64(float("nan"))
    mt = _source_mt(
        hl,
        rows={
            "grpmax": hl.struct(AC=10, AN=100, AF=nan, gen_anc="afr"),
            "inbreeding_coef": nan,
            "in_silico_predictors": hl.struct(
                cadd=hl.struct(phred=nan, raw_score=nan), revel_max=nan
            ),
        },
    )
    table = to_browser_sites(
        mt, fields=("in_silico_predictors",), reference_genome="GRCh38"
    )
    row = _row(hl, table)

    assert row.joint.grpmax.ac == 10
    assert row.joint.grpmax.af is None
    site_quality = {
        metric.metric: metric.value
        for metric in row.genome.quality_metrics.site_quality_metrics
    }
    assert site_quality["inbreeding_coeff"] is None
    assert row.in_silico_predictors.cadd.phred is None
    assert row.in_silico_predictors.cadd.raw_score is None
    assert row.in_silico_predictors.revel_max is None
    # A finite float is untouched, and an absent struct stays null.
    assert row.genome.freq.all.af == pytest.approx(0.1)
    assert row.exome is None


# --- stratum grammar --------------------------------------------------------


def test_stratum_key_grammar():
    assert stratum_key({"group": "adj"}) == "all"
    assert stratum_key({"group": "raw"}) == "all_raw"
    assert stratum_key({"gen_anc": "afr", "group": "adj"}) == "afr"
    assert stratum_key({"sex": "XX", "group": "adj"}) == "XX"
    assert stratum_key({"gen_anc": "afr", "sex": "XX", "group": "adj"}) == "afr_XX"
    assert stratum_key({"cohort": "c1", "group": "adj"}) == "cohort_c1"
    assert stratum_key({"gen_anc": "afr", "group": "raw"}) == "gen_anc=afr_raw"
    assert (
        stratum_key({"subset": "non_ukb", "cohort": "c1", "group": "adj"})
        == "cohort=c1_subset=non_ukb"
    )


def test_stratum_key_empty_metadata_is_all():
    """The overall stratum (no ``group`` key, as coverage meta) keys as ``all``."""
    assert stratum_key({}) == "all"


def test_stratum_index_requires_exactly_one_overall_adj():
    with pytest.raises(BrowserSchemaError, match="exactly one overall"):
        stratum_index([{"gen_anc": "afr", "group": "adj"}])
    with pytest.raises(BrowserSchemaError, match="exactly one overall"):
        stratum_index([{"group": "adj"}, {"group": "adj"}])


def test_stratum_index_duplicate_collision_fails_loudly():
    with pytest.raises(BrowserSchemaError, match="duplicate"):
        stratum_index(
            [
                {"group": "adj"},
                {"gen_anc": "afr", "group": "adj"},
                {"gen_anc": "afr", "group": "adj"},
            ]
        )


def test_stratum_index_maps_keys_to_positions():
    assert stratum_index(
        [
            {"group": "adj"},
            {"gen_anc": "afr", "group": "adj"},
            {"sex": "XX", "group": "adj"},
        ]
    ) == {"all": 0, "afr": 1, "XX": 2}


def test_ancestry_stratum_positions_selects_only_ancestry_strata():
    """The overall, raw, sex-only and cohort strata are not Populations."""
    assert ancestry_stratum_positions(META) == [2, 4]


def test_ancestry_stratum_positions_carries_custom_ids_verbatim():
    meta = [
        {"group": "adj"},
        {"gen_anc": "<UNKNOWN>", "group": "adj"},
        {"gen_anc": "custom_pop", "sex": "XY", "group": "adj"},
    ]
    positions = ancestry_stratum_positions(meta)
    assert [stratum_key(meta[position]) for position in positions] == [
        "<UNKNOWN>",
        "custom_pop_XY",
    ]


# --- callstats --------------------------------------------------------------


def _pivot_table(hl, mt):
    return mt.annotate_rows(genome=genome_expr(mt)).rows()


def test_genome_freq_has_only_the_all_stratum(hl):
    """The pivot drops raw/ancestry/sex/cohort strata; only ``all`` remains."""
    table = _pivot_table(hl, _source_mt(hl))
    assert tuple(table.genome.dtype["freq"].fields) == ("all",)
    freq = _row(hl, table).genome.freq
    assert freq.all.ac == 10
    assert freq.all.an == 100
    assert freq.all.af == pytest.approx(0.1)
    assert freq.all.homozygote_count == 2


def test_callstats_always_emits_hemizygote_count_zero_when_absent(hl):
    """``annotate_freq`` never computes hemizygotes, so the reshape emits 0."""
    table = _pivot_table(hl, _source_mt(hl))
    fields = set(table.genome.dtype["freq"]["all"].fields)
    assert fields == {"ac", "an", "af", "homozygote_count", "hemizygote_count"}
    assert _row(hl, table).genome.freq.all.hemizygote_count == 0


def test_callstats_uses_the_source_hemizygote_count_when_defined(hl):
    freq = [dict(entry, hemizygote_count=1) for entry in FREQ]
    table = _pivot_table(hl, _source_mt(hl, freq=freq, hemizygote=True))
    assert _row(hl, table).genome.freq.all.hemizygote_count == 1


def test_genome_counts_mirror_freq_all(hl):
    table = _pivot_table(hl, _source_mt(hl))
    genome = _row(hl, table).genome
    assert genome.ac == 10
    assert genome.an == 100
    assert genome.homozygote_count == 2
    assert genome.hemizygote_count == 0


def test_genome_freq_duplicate_key_fails_loudly(hl):
    mt = _source_mt(
        hl,
        meta=[
            {"group": "adj"},
            {"gen_anc": "afr", "group": "adj"},
            {"gen_anc": "afr", "group": "adj"},
        ],
        freq=FREQ[:3],
    )
    with pytest.raises(BrowserSchemaError, match="duplicate"):
        genome_expr(mt)


def _mt_annotated_without(hl, *drop: str):
    """A fully annotated MT with the named non-key row columns removed."""
    mt = _source_mt(hl)
    key_fields = set(mt.row_key.dtype.fields)
    return mt.select_rows(
        **{
            name: mt[name]
            for name in mt.row.dtype.fields
            if name not in drop and name not in key_fields
        }
    )


def test_genome_expr_requires_freq(hl):
    mt = _mt_annotated_without(hl, "freq")
    with pytest.raises(BrowserSchemaError, match="'freq' row field"):
        genome_expr(mt)


def test_genome_expr_requires_freq_meta(hl):
    mt = _source_mt(hl)
    mt = mt.select_globals(
        **{name: mt[name] for name in mt.globals.dtype.fields if name != "freq_meta"}
    )
    with pytest.raises(BrowserSchemaError, match="'freq_meta' global"):
        genome_expr(mt)


# --- joint ------------------------------------------------------------------


def test_joint_freq_all_counts(hl):
    mt = _source_mt(hl)
    table = mt.annotate_rows(joint=joint_expr(mt)).rows()
    joint = _row(hl, table).joint
    assert tuple(table.joint.dtype["freq"]["all"].fields) == (
        "ac",
        "an",
        "af",
        "homozygote_count",
        "hemizygote_count",
        "ancestry_groups",
    )
    assert joint.freq.all.ac == 10
    assert joint.freq.all.an == 100
    assert joint.freq.all.hemizygote_count == 0


def test_joint_ancestry_groups_are_the_ancestry_strata(hl):
    mt = _source_mt(hl)
    table = mt.annotate_rows(joint=joint_expr(mt)).rows()
    groups = {
        group.id: group for group in _row(hl, table).joint.freq.all.ancestry_groups
    }
    assert set(groups) == {"afr", "afr_XX"}
    assert groups["afr"].ac == 3
    assert groups["afr"].an == 30
    assert groups["afr_XX"].ac == 2
    element_fields = table.joint.dtype["freq"]["all"]["ancestry_groups"].element_type
    assert tuple(element_fields.fields) == (
        "id",
        "ac",
        "an",
        "af",
        "homozygote_count",
        "hemizygote_count",
    )
    assert groups["afr"].hemizygote_count == 0


def test_joint_ancestry_groups_carry_custom_ids_verbatim(hl):
    meta = [
        {"group": "adj"},
        {"gen_anc": "<UNKNOWN>", "group": "adj"},
        {"gen_anc": "custom_pop", "group": "adj"},
    ]
    mt = _source_mt(hl, meta=meta, freq=FREQ[:3])
    table = mt.annotate_rows(joint=joint_expr(mt)).rows()
    ids = [group.id for group in _row(hl, table).joint.freq.all.ancestry_groups]
    assert ids == ["<UNKNOWN>", "custom_pop"]


def test_joint_ancestry_groups_present_in_cohort_mode(hl):
    """A cohort additional stratum is additive; the ancestry strata remain."""
    meta = [
        {"group": "adj"},
        {"gen_anc": "afr", "group": "adj"},
        {"cohort": "c1", "group": "adj"},
    ]
    mt = _source_mt(
        hl, meta=meta, freq=FREQ[:3], grpmax_label="cohort", grpmax_value="c1"
    )
    table = mt.annotate_rows(joint=joint_expr(mt)).rows()
    joint = _row(hl, table).joint
    assert [group.id for group in joint.freq.all.ancestry_groups] == ["afr"]
    assert joint.grpmax.gen_anc == "c1"


def test_joint_grpmax_maps_counts_and_gen_anc(hl):
    mt = _source_mt(hl)
    table = mt.annotate_rows(joint=joint_expr(mt)).rows()
    grpmax = _row(hl, table).joint.grpmax
    assert grpmax.ac == 10
    assert grpmax.an == 100
    assert grpmax.af == pytest.approx(0.1)
    assert grpmax.gen_anc == "afr"
    assert tuple(table.joint.dtype["grpmax"].fields) == ("ac", "an", "af", "gen_anc")


def test_joint_fafmax_selects_the_faf95_pair(hl):
    mt = _source_mt(hl)
    table = mt.annotate_rows(joint=joint_expr(mt)).rows()
    fafmax = _row(hl, table).joint.fafmax
    assert fafmax.faf95_max == pytest.approx(0.05)
    assert fafmax.faf95_max_gen_anc == "nfe"
    assert tuple(table.joint.dtype["fafmax"].fields) == (
        "faf95_max",
        "faf95_max_gen_anc",
    )


def test_joint_flags_are_sorted(hl):
    mt = _source_mt(hl, filters=hl.literal({"b", "a"}, hl.tset(hl.tstr)))
    table = mt.annotate_rows(joint=joint_expr(mt)).rows()
    assert _row(hl, table).joint.flags == ["a", "b"]


def test_joint_flags_null_filters_is_empty(hl):
    mt = _source_mt(hl, filters=hl.missing(hl.tset(hl.tstr)))
    table = mt.annotate_rows(joint=joint_expr(mt)).rows()
    assert _row(hl, table).joint.flags == []


def test_joint_requires_the_grpmax_label(hl):
    mt = _source_mt(hl, rows={"grpmax": hl.struct(AC=1, AN=2, AF=0.5)})
    with pytest.raises(BrowserSchemaError, match="ancestry label"):
        joint_expr(mt)


def test_joint_rejects_the_no_ancestry_grpmax_shape(hl):
    mt = _source_mt(hl, rows={"grpmax": hl.struct(gnomad_grpmax=0.5, gen_anc="afr")})
    with pytest.raises(BrowserSchemaError, match="'AC', 'AN', 'AF'|member"):
        joint_expr(mt)


def test_joint_requires_grpmax_members(hl):
    mt = _source_mt(hl, rows={"grpmax": hl.struct(AC=1, gen_anc="afr")})
    with pytest.raises(BrowserSchemaError, match="member"):
        joint_expr(mt)


def test_joint_rejects_the_no_ancestry_fafmax_shape(hl):
    mt = _source_mt(hl, rows={"fafmax": hl.struct(fafmax=0.5, gen_anc="afr")})
    with pytest.raises(BrowserSchemaError, match="'faf95_max'"):
        joint_expr(mt)


# --- quality metrics --------------------------------------------------------


def test_quality_metrics_mapping(hl):
    mt = _source_mt(hl)
    table = mt.annotate_rows(quality_metrics=quality_metrics_expr(mt)).rows()
    metrics = _row(hl, table).quality_metrics
    assert metrics.genotype_quality.all_adj.n_larger == 10
    assert metrics.genotype_quality.alt_adj.n_larger == 11
    assert metrics.genotype_quality.all_raw.n_larger == 20
    assert metrics.genotype_quality.alt_raw.n_larger == 21
    assert metrics.genotype_depth.all_adj.n_larger == 12
    assert metrics.genotype_depth.alt_adj.n_larger == 13
    assert metrics.genotype_depth.all_raw.n_larger == 22
    assert metrics.genotype_depth.alt_raw.n_larger == 23
    assert metrics.allele_balance.alt_adj.n_larger == 14
    assert metrics.allele_balance.alt_raw.n_larger == 24
    assert tuple(table.quality_metrics.dtype.fields) == (
        "genotype_quality",
        "genotype_depth",
        "allele_balance",
        "site_quality_metrics",
    )


def test_site_quality_metrics_mapping(hl):
    mt = _source_mt(hl)
    table = mt.annotate_rows(
        site_quality_metrics=site_quality_metrics_expr(mt)
    ).rows()
    metrics = _row(hl, table).site_quality_metrics
    assert [metric.metric for metric in metrics] == [
        "SiteQuality",
        "inbreeding_coeff",
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
    ]
    values = {metric.metric: metric.value for metric in metrics}
    assert values["SiteQuality"] == pytest.approx(100.0)
    assert values["inbreeding_coeff"] == pytest.approx(0.01)
    assert values["AS_QUALapprox"] == pytest.approx(90.0)
    assert values["AS_VQSLOD"] == pytest.approx(1.0)


def test_site_quality_requires_info(hl):
    mt = _mt_annotated_without(hl, "info")
    with pytest.raises(BrowserSchemaError, match="'info' row field"):
        quality_metrics_expr(mt)


def test_site_quality_requires_info_member(hl):
    mt = _source_mt(hl, rows={"info": hl.struct(QUALapprox=hl.int64(1))})
    with pytest.raises(BrowserSchemaError, match="'AS_FS'"):
        quality_metrics_expr(mt)


def test_site_quality_requires_inbreeding_coef(hl):
    mt = _mt_annotated_without(hl, "inbreeding_coef")
    with pytest.raises(BrowserSchemaError, match="'inbreeding_coef' row field"):
        quality_metrics_expr(mt)


def test_quality_metrics_requires_histograms(hl):
    mt = _mt_annotated_without(hl, "histograms")
    with pytest.raises(BrowserSchemaError, match="'histograms' row field"):
        quality_metrics_expr(mt)


def test_quality_metrics_requires_qual_hists(hl):
    mt = _source_mt(hl, rows={"histograms": hl.struct(raw_qual_hists=_raw_group(hl))})
    with pytest.raises(BrowserSchemaError, match="'qual_hists'"):
        quality_metrics_expr(mt)


def _raw_group(hl):
    return hl.struct(
        **{
            member: _hist(hl, _HIST_MARKERS[("raw_qual_hists", member)])
            for member in (
                "gq_hist_all",
                "gq_hist_alt",
                "dp_hist_all",
                "dp_hist_alt",
                "ab_hist_alt",
            )
        }
    )


def test_quality_metrics_requires_a_named_histogram(hl):
    mt = _source_mt(
        hl, histograms=_hists_expr(hl, members=("gq_hist_all", "gq_hist_alt"))
    )
    with pytest.raises(BrowserSchemaError, match="'dp_hist_all'"):
        quality_metrics_expr(mt)


# --- coverage ---------------------------------------------------------------


def test_coverage_is_the_overall_element_without_total_dp(hl):
    mt = _source_mt(hl)
    table = mt.annotate_rows(coverage=coverage_expr(mt)).rows()
    coverage = _row(hl, table).coverage
    assert tuple(table.coverage.dtype.fields) == ("genome",)
    assert coverage.genome.mean == pytest.approx(1.0)
    assert coverage.genome.median_approx == 1
    assert coverage.genome.over_100 == pytest.approx(0.5)
    assert coverage.genome.over_1 == pytest.approx(0.8)
    assert "total_DP" not in table.coverage.dtype["genome"].fields
    assert tuple(table.coverage.dtype["genome"].fields) == (
        "mean",
        "median_approx",
        *(f"over_{x}" for x in COVERAGE_BINS),
    )


def test_coverage_drops_named_strata(hl):
    """Filtered/cohort strata are dropped; only the overall element is kept."""
    mt = _source_mt(hl)
    table = mt.annotate_rows(coverage=coverage_expr(mt)).rows()
    assert _row(hl, table).coverage.genome.mean == pytest.approx(1.0)


def test_coverage_requires_exactly_one_overall_element(hl):
    mt = _source_mt(
        hl, coverage_meta=[{"filtered": "pass"}], coverage_stats=COVERAGE_STATS[:1]
    )
    with pytest.raises(BrowserSchemaError, match="exactly one overall"):
        coverage_expr(mt)


def test_coverage_duplicate_overall_element_fails_loudly(hl):
    mt = _source_mt(hl, coverage_meta=[{}, {}], coverage_stats=COVERAGE_STATS[:2])
    with pytest.raises(BrowserSchemaError, match="exactly one overall"):
        coverage_expr(mt)


def test_coverage_requires_its_members(hl):
    mt = _source_mt(hl)
    mt = mt.annotate_rows(
        coverage_stats=hl.array([hl.struct(mean=1.0, median_approx=1)])
    )
    with pytest.raises(BrowserSchemaError, match="member"):
        coverage_expr(mt)


def test_coverage_stats_type_is_the_source_element_type(hl):
    assert coverage_stats_type() == _coverage_element(hl).dtype


# --- required-source preconditions ------------------------------------------


REQUIRED_ROW_FIELDS = (
    "freq",
    "grpmax",
    "fafmax",
    "histograms",
    "filters",
    "coverage_stats",
)
REQUIRED_GLOBALS = ("freq_meta", "coverage_stats_meta")


@pytest.mark.parametrize("name", REQUIRED_ROW_FIELDS)
def test_missing_required_row_field_fails_loudly(hl, name):
    mt = _mt_annotated_without(hl, name)
    with pytest.raises(BrowserSchemaError, match=repr(name)):
        to_browser_sites(mt, fields=(), reference_genome="GRCh38")


@pytest.mark.parametrize("name", REQUIRED_GLOBALS)
def test_missing_required_global_fails_loudly(hl, name):
    mt = _source_mt(hl)
    mt = mt.select_globals(
        **{key: mt[key] for key in mt.globals.dtype.fields if key != name}
    )
    with pytest.raises(BrowserSchemaError, match=repr(name)):
        to_browser_sites(mt, fields=(), reference_genome="GRCh38")


def test_single_sided_freq_pair_fails_loudly(hl):
    """``freq`` without ``freq_meta`` is the same loud error as both absent."""
    mt = _mt_annotated_without(hl, "freq")
    with pytest.raises(BrowserSchemaError, match="'freq' row field"):
        genome_expr(mt)


def test_single_sided_coverage_pair_fails_loudly(hl):
    mt = _source_mt(hl)
    mt = mt.select_globals(
        **{
            key: mt[key]
            for key in mt.globals.dtype.fields
            if key != "coverage_stats_meta"
        }
    )
    with pytest.raises(BrowserSchemaError, match="'coverage_stats_meta' global"):
        coverage_expr(mt)


# --- field policy and the entry point ---------------------------------------


def test_default_fields_is_in_silico_predictors_only():
    assert DEFAULT_FIELDS == ("in_silico_predictors",)


def test_to_browser_sites_mandatory_columns(hl):
    table = to_browser_sites(_source_mt(hl), fields=(), reference_genome="GRCh38")
    assert tuple(table.row.dtype.fields)[: len(MANDATORY_FIELDS)] == MANDATORY_FIELDS
    row = _row(hl, table)
    assert row.rsids is None
    assert row.caid is None
    assert row.exome is None
    assert row.in_silico_predictors is None
    assert row.transcript_consequences == []
    assert table.rsids.dtype == hl.tarray(hl.tstr)
    assert table.caid.dtype == hl.tstr
    assert table.exome.dtype["freq"]["all"].fields == ("ac", "an")
    assert table.joint.dtype["freq"]["all"]["ac"] == hl.tint32
    assert tuple(table.coverage.dtype.fields) == ("genome",)


def test_to_browser_sites_joint_is_always_populated(hl):
    table = to_browser_sites(_source_mt(hl), fields=(), reference_genome="GRCh38")
    joint = _row(hl, table).joint
    assert joint.freq.all.ac == 10
    assert joint.grpmax.gen_anc == "afr"
    assert joint.fafmax.faf95_max == pytest.approx(0.05)
    assert joint.flags == ["PASS"]


def test_to_browser_sites_genome_has_quality_metrics_and_counts(hl):
    table = to_browser_sites(_source_mt(hl), fields=(), reference_genome="GRCh38")
    genome = _row(hl, table).genome
    assert tuple(table.genome.dtype.fields) == (
        "freq",
        "quality_metrics",
        "ac",
        "an",
        "homozygote_count",
        "hemizygote_count",
    )
    assert genome.quality_metrics.genotype_quality.all_adj.n_larger == 10
    assert genome.ac == 10


def test_to_browser_sites_consumed_fields_are_never_top_level(hl):
    """Listing a consumed field does not produce a duplicate top-level column."""
    mt = _source_mt(hl).annotate_rows(gnomad_freq=hl.array([hl.struct(AF=0.2)]))
    table = to_browser_sites(
        mt,
        fields=("grpmax", "fafmax", "histograms", "freq", "gnomad_freq"),
        reference_genome="GRCh38",
    )
    fields = table.row.dtype.fields
    assert "grpmax" not in fields
    assert "fafmax" not in fields
    assert "histograms" not in fields
    assert "freq" not in fields
    assert "gnomad_freq" in fields
    assert _row(hl, table).gnomad_freq[0].AF == pytest.approx(0.2)


def test_to_browser_sites_optional_fields_dropped_when_absent(hl):
    table = to_browser_sites(
        _source_mt(hl), fields=("clinvar", "gnomad_freq"), reference_genome="GRCh38"
    )
    assert "clinvar" not in table.row.dtype.fields
    assert "gnomad_freq" not in table.row.dtype.fields
    assert "freq_meta" in table.globals.dtype.fields
    assert "coverage_stats_meta" in table.globals.dtype.fields


def test_to_browser_sites_optional_structural_fields_take_source(hl):
    mt = _source_mt(hl)
    mt = mt.annotate_rows(
        exome=hl.struct(freq=hl.struct(all=hl.struct(ac=1, an=2))),
        in_silico_predictors=hl.struct(cadd=hl.struct(phred=2.0)),
    )
    table = to_browser_sites(
        mt, fields=("exome", "in_silico_predictors"), reference_genome="GRCh38"
    )
    row = _row(hl, table)
    assert row.exome.freq.all.ac == 1
    assert row.in_silico_predictors.cadd.phred == pytest.approx(2.0)


def test_to_browser_sites_schema_has_no_empty_struct_columns(hl):
    """Parquet/genohype cannot encode an empty struct, so no column may be one."""
    mt = _source_mt(hl).annotate_rows(
        in_silico_predictors=hl.struct(cadd=hl.struct(phred=1.0)),
    )
    table = to_browser_sites(
        mt, fields=("in_silico_predictors",), reference_genome="GRCh38"
    )
    empty_structs = [
        name for name, dtype in table.row.dtype.items() if dtype == hl.tstruct()
    ]
    assert empty_structs == []


def test_to_browser_sites_in_silico_predictors_typed_null_is_non_empty(hl):
    """Parquet cannot encode an empty struct, so the absent typed null is shaped."""
    table = to_browser_sites(_source_mt(hl), fields=(), reference_genome="GRCh38")
    fields = tuple(table.in_silico_predictors.dtype.fields)
    assert fields != ()
    assert "cadd" in fields
    assert _row(hl, table).in_silico_predictors is None


# --- preconditions ----------------------------------------------------------


def test_wrong_row_key_fails_loudly(hl):
    mt = hl.MatrixTable.from_parts(
        rows={
            "locus": [hl.locus("chr1", 100, reference_genome="GRCh38")],
            "alleles": [["A", "T"]],
        },
        cols={"s": ["S1"]},
        entries={"GT": [[hl.call(0, 1)]]},
    ).key_rows_by("locus")
    with pytest.raises(BrowserSchemaError, match="row key"):
        to_browser_sites(mt, fields=(), reference_genome="GRCh38")


def test_multiallelic_row_fails_loudly(hl):
    mt = hl.MatrixTable.from_parts(
        rows={
            "locus": [hl.locus("chr1", 100, reference_genome="GRCh38")],
            "alleles": [["A", "T", "C"]],
        },
        cols={"s": ["S1"]},
        entries={"GT": [[hl.call(0, 1)]]},
    ).key_rows_by("locus", "alleles")
    mt = mt.annotate_globals(
        freq_meta=_meta_expr(hl, META),
        coverage_stats_meta=_coverage_meta_expr(hl, COVERAGE_META),
    )
    mt = mt.annotate_rows(
        freq=_freq_expr(hl, FREQ),
        grpmax=_grpmax_expr(hl),
        fafmax=_fafmax_expr(hl),
        histograms=_hists_expr(hl),
        filters=hl.literal({"PASS"}, hl.tset(hl.tstr)),
        coverage_stats=_coverage_stats_expr(hl, COVERAGE_STATS),
    )
    with pytest.raises(BrowserSchemaError, match="biallelic"):
        to_browser_sites(mt, fields=(), reference_genome="GRCh38")


def test_reference_mismatch_fails_loudly(hl):
    mt = _mt(
        hl,
        contig="1",
        reference_genome="GRCh37",
        rows={
            "freq": _freq_expr(hl, FREQ),
            "grpmax": _grpmax_expr(hl),
            "fafmax": _fafmax_expr(hl),
            "histograms": _hists_expr(hl),
            "filters": hl.literal({"PASS"}, hl.tset(hl.tstr)),
            "coverage_stats": _coverage_stats_expr(hl, COVERAGE_STATS),
        },
    ).annotate_globals(
        freq_meta=_meta_expr(hl, META),
        coverage_stats_meta=_coverage_meta_expr(hl, COVERAGE_META),
    )
    with pytest.raises(BrowserSchemaError, match="reference genome"):
        to_browser_sites(mt, fields=(), reference_genome="GRCh38")


def test_alleles_not_array_of_string_fails_loudly(hl):
    mt = hl.MatrixTable.from_parts(
        rows={
            "locus": [hl.locus("chr1", 100, reference_genome="GRCh38")],
            "alleles": [[1, 2]],
        },
        cols={"s": ["S1"]},
        entries={"GT": [[hl.call(0, 1)]]},
    ).key_rows_by("locus", "alleles")
    with pytest.raises(BrowserSchemaError, match="array<string>"):
        to_browser_sites(mt, fields=(), reference_genome="GRCh38")
