"""In-process tests for ``annotation.utils.compute_coverage_stats`` (ticket 01).

The util aggregates coverage over the rows of an already-dense MatrixTable and
must reproduce ``gnomad.utils.sparse_mt.compute_coverage_stats`` element for
element. The parity fixture builds a tiny real ``hl.vds.VariantDataset`` with
``DP`` on both the reference and variant entries, densifies it with the upstream
``densify_all_reference_sites`` helper (the dense matrix the pipeline derives
from a VDS), and compares the two implementations for the ``filtered`` and
``filtered`` + ``cohort`` strata.

See ``.scratch/annotation-coverage/spec.md`` (D1-D3, D6, D8) and ticket 01.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("hail")

from gnomad.utils.sparse_mt import (
    compute_coverage_stats as upstream_compute_coverage_stats,
)
from gnomad.utils.sparse_mt import (
    densify_all_reference_sites,
)

from gnomad_meb.lib.annotation.utils import compute_coverage_stats

REFERENCE_GENOME = "GRCh38"
COVERAGE_BINS = (1, 5, 10, 15, 20, 25, 30, 50, 100)

# s1: pass / cohort A, s2: filtered / cohort B, s3: pass / cohort A.
SAMPLE_ANNOTATIONS = {
    "s1": {"hard_filters": [], "cohort": "A"},
    "s2": {"hard_filters": ["FAIL"], "cohort": "B"},
    "s3": {"hard_filters": [], "cohort": "A"},
}

# Reference blocks at chr1:1, chr1:2 and chr1:4, and a variant at chr1:3.
REFERENCE_BLOCKS = {
    1: {"s1": 10, "s2": 20, "s3": 30},
    2: {"s1": 11, "s2": 21, "s3": 31},
    4: {"s1": 13, "s2": 23, "s3": 33},
}
VARIANT_POSITION = 3
VARIANT_ALLELES = ("A", "C")
VARIANT_DP = {"s1": 5, "s2": 15, "s3": 25}

BROAD_URL = (
    "https://github.com/broadinstitute/gnomad_qc/blob/main/gnomad_qc/v4/"
    "annotations/compute_coverage.py#L470"
)
CPG_URL = (
    "https://github.com/populationgenomics/production-pipelines/blob/main/"
    "cpg_workflows/large_cohort/compute_stats_for_all_sites.py#L740"
)


def _locus(hl, position: int):
    return hl.Locus("chr1", position, REFERENCE_GENOME)


def _build_reference_data(hl):
    """A ``locus``/``s`` coordinate table of single-base reference blocks."""
    schema = hl.tstruct(
        locus=hl.tlocus(REFERENCE_GENOME),
        s=hl.tstr,
        LEN=hl.tint32,
        END=hl.tint32,
        DP=hl.tint32,
        LGT=hl.tcall,
        hard_filters=hl.tarray(hl.tstr),
        cohort=hl.tstr,
    )
    rows = [
        {
            "locus": _locus(hl, position),
            "s": sample,
            "LEN": 1,
            "END": position,
            "DP": dps[sample],
            "LGT": hl.Call([0, 0]),
            "hard_filters": annotations["hard_filters"],
            "cohort": annotations["cohort"],
        }
        for position, dps in REFERENCE_BLOCKS.items()
        for sample, annotations in SAMPLE_ANNOTATIONS.items()
    ]
    return hl.Table.parallelize(rows, schema=schema).to_matrix_table(
        row_key=["locus"], col_key=["s"], col_fields=["hard_filters", "cohort"]
    )


def _build_variant_data(hl):
    """A ``locus``/``alleles``/``s`` coordinate table with one variant."""
    schema = hl.tstruct(
        locus=hl.tlocus(REFERENCE_GENOME),
        alleles=hl.tarray(hl.tstr),
        s=hl.tstr,
        DP=hl.tint32,
        LGT=hl.tcall,
        hard_filters=hl.tarray(hl.tstr),
        cohort=hl.tstr,
    )
    rows = [
        {
            "locus": _locus(hl, VARIANT_POSITION),
            "alleles": list(VARIANT_ALLELES),
            "s": sample,
            "DP": dp,
            "LGT": hl.Call([0, 1]),
            "hard_filters": SAMPLE_ANNOTATIONS[sample]["hard_filters"],
            "cohort": SAMPLE_ANNOTATIONS[sample]["cohort"],
        }
        for sample, dp in VARIANT_DP.items()
    ]
    return hl.Table.parallelize(rows, schema=schema).to_matrix_table(
        row_key=["locus", "alleles"],
        col_key=["s"],
        col_fields=["hard_filters", "cohort"],
    )


@pytest.fixture(scope="module")
def tiny_callset(hl):
    """A tiny VDS plus its upstream-densified dense matrix and a reference table."""
    reference_data = _build_reference_data(hl)
    variant_data = _build_variant_data(hl)
    vds = hl.vds.VariantDataset(reference_data, variant_data)

    all_positions = [*REFERENCE_BLOCKS, VARIANT_POSITION]
    reference_ht = hl.Table.parallelize(
        [{"locus": _locus(hl, position)} for position in all_positions],
        schema=hl.tstruct(locus=hl.tlocus(REFERENCE_GENOME)),
    ).key_by("locus")

    dense = densify_all_reference_sites(
        vds, reference_ht, entry_keep_fields=["LGT", "DP"]
    )
    samples = variant_data.cols().select("hard_filters", "cohort")
    dense = dense.annotate_cols(**samples[dense.s])

    return SimpleNamespace(
        vds=vds,
        variant_data=variant_data,
        reference_ht=reference_ht,
        dense=dense,
    )


def _strata(hl, source, *, with_cohort: bool):
    strata = [
        {
            "filtered": hl.if_else(
                hl.len(source.hard_filters) == 0, "pass", "filtered"
            )
        }
    ]
    if with_cohort:
        strata.append({"cohort": source.cohort})
    return strata


def _meta_index(meta):
    return {
        tuple(sorted(dict(entry).items())): index
        for index, entry in enumerate(meta)
    }


@pytest.mark.parametrize("with_cohort", [False, True])
def test_parity_with_upstream_coverage_stats(hl, tiny_callset, with_cohort):
    """Element-for-element parity with the upstream sparse-mt implementation."""
    ours = compute_coverage_stats(
        tiny_callset.dense,
        strata_expr=_strata(hl, tiny_callset.dense, with_cohort=with_cohort),
    )
    upstream = upstream_compute_coverage_stats(
        tiny_callset.vds,
        tiny_callset.reference_ht,
        strata_expr=_strata(hl, tiny_callset.variant_data, with_cohort=with_cohort),
    )

    assert hl.eval(ours.coverage_stats_meta) == hl.eval(upstream.coverage_stats_meta)
    assert hl.eval(ours.coverage_stats_meta_sample_count) == hl.eval(
        upstream.coverage_stats_meta_sample_count
    )

    joined = ours.annotate(_upstream=upstream[ours.locus].coverage_stats)
    rows = joined.collect()
    assert len(rows) == len(REFERENCE_BLOCKS) + 1

    for row in rows:
        assert len(row.coverage_stats) == len(row._upstream)
        for ours_element, upstream_element in zip(
            row.coverage_stats, row._upstream, strict=False
        ):
            assert ours_element.mean == pytest.approx(upstream_element.mean)
            assert ours_element.median_approx == pytest.approx(
                upstream_element.median_approx
            )
            assert ours_element.total_DP == upstream_element.total_DP
            for bin_value in COVERAGE_BINS:
                over = f"over_{bin_value}"
                assert getattr(ours_element, over) == pytest.approx(
                    getattr(upstream_element, over)
                )


def test_strata_construction_filtered(hl, tiny_callset):
    """The ``filtered`` strata is materialized with ``group`` stripped."""
    ht = compute_coverage_stats(
        tiny_callset.dense,
        strata_expr=_strata(hl, tiny_callset.dense, with_cohort=False),
    )
    meta = hl.eval(ht.coverage_stats_meta)
    counts = hl.eval(ht.coverage_stats_meta_sample_count)
    index = _meta_index(meta)

    assert all("group" not in entry for entry in meta)
    assert index[()] == 0  # overall stratum is first
    assert counts[index[()]] == 3
    assert counts[index[(("filtered", "pass"),)]] == 2
    assert counts[index[(("filtered", "filtered"),)]] == 1
    assert tuple(ht.key) == ("locus",)


def test_strata_construction_filtered_and_cohort(hl, tiny_callset):
    """``cohort`` is added as a second strata group alongside ``filtered``."""
    ht = compute_coverage_stats(
        tiny_callset.dense,
        strata_expr=_strata(hl, tiny_callset.dense, with_cohort=True),
    )
    meta = hl.eval(ht.coverage_stats_meta)
    counts = hl.eval(ht.coverage_stats_meta_sample_count)
    index = _meta_index(meta)

    assert index[()] == 0
    assert counts[index[()]] == 3
    assert counts[index[(("filtered", "pass"),)]] == 2
    assert counts[index[(("filtered", "filtered"),)]] == 1
    assert counts[index[(("cohort", "A"),)]] == 2
    assert counts[index[(("cohort", "B"),)]] == 1


def test_missing_dp_raises_naming_the_rebuild(hl):
    """A matrix without ``dp_field`` fails loudly, naming DP and the rebuild."""
    mt = hl.utils.range_matrix_table(n_rows=2, n_cols=2)
    mt = mt.annotate_entries(depth=mt.row_idx + mt.col_idx)

    with pytest.raises(ValueError) as excinfo:
        compute_coverage_stats(
            mt, strata_expr=[{"filtered": hl.str(mt.row_idx)}]
        )

    message = str(excinfo.value)
    assert "DP" in message
    assert "--overwrite" in message


def test_docstring_cites_both_reference_urls():
    """The derived-from ``Note`` carries both reference URLs verbatim."""
    docstring = compute_coverage_stats.__doc__ or ""
    assert "Note" in docstring
    assert BROAD_URL in docstring
    assert CPG_URL in docstring


def test_coverage_element_type_matches_browser_schema(hl):
    """The util's element type IS the browser's ``coverage_stats_type()``.

    Pins the hand-written leaf list in ``lib/browser/schema.py`` to what
    ``get_coverage_agg_func`` + ``coverage_over_x_bins`` actually produce on a
    tiny dense matrix, so the browser's fallback type cannot silently drift from
    the util's output.
    """
    from gnomad_meb.lib.browser.schema import coverage_stats_type

    mt = hl.MatrixTable.from_parts(
        rows={"locus": [hl.locus("chr1", 100, reference_genome=REFERENCE_GENOME)]},
        cols={"s": ["s1", "s2"]},
        entries={"DP": [[10, 20]]},
    ).key_rows_by("locus")
    mt = mt.annotate_cols(hard_filters=hl.empty_array(hl.tstr))

    ht = compute_coverage_stats(
        mt,
        strata_expr=[
            {"filtered": hl.if_else(hl.len(mt.hard_filters) == 0, "pass", "filtered")}
        ],
    )

    assert ht.coverage_stats.dtype.element_type == coverage_stats_type()
