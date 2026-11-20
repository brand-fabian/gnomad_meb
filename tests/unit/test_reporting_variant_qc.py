"""Hail-free unit tests for the Variant QC reporting collectors and builders.

The collectors are thin hail reads over the reshape helpers, so the helpers are
driven here with hand-built ``pandas`` frames / plain dicts (the ``sample_qc.py``
precedent); the public surface of the hail collectors is pinned only. No
``hail.init()``.
"""

from __future__ import annotations

import pandas as pd
import plotnine as pn

from gnomad_meb import reporting
from gnomad_meb.reporting import variant_qc

# --- provenance / cutoffs --------------------------------------------------


def test_vqsr_provenance_reads_globals():
    provenance = variant_qc._vqsr_provenance(
        {
            "variant_qc_cutoffs": {"snp": {"target_truth_sensitivity": 99.7}},
            "vqsr_params": {"allele_specific": True},
            "gatk_version": {"version": "4.2.0.0"},
        },
        n_variants=42,
    )
    assert provenance["n_variants"] == 42
    assert provenance["vqsr_params"] == {"allele_specific": True}
    assert provenance["gatk_version"] == {"version": "4.2.0.0"}


def test_vqsr_provenance_defaults_missing_globals():
    provenance = variant_qc._vqsr_provenance({}, n_variants=0)
    assert provenance["variant_qc_cutoffs"] == {}
    assert provenance["vqsr_params"] == {}
    assert provenance["gatk_version"] == {}


def test_vqsr_cutoffs_frame():
    frame = variant_qc._vqsr_cutoffs_frame(
        {
            "snp": {"target_truth_sensitivity": 99.7, "vqslod_cutoff": 1.23},
            "indel": {"target_truth_sensitivity": 99.0, "vqslod_cutoff": -4.5},
        }
    )
    assert list(frame.columns) == [
        "mode",
        "target_truth_sensitivity",
        "vqslod_cutoff",
    ]
    assert frame["mode"].tolist() == ["snp", "indel"]
    assert frame["vqslod_cutoff"].tolist() == [1.23, -4.5]


def test_vqsr_cutoffs_frame_legacy_table():
    frame = variant_qc._vqsr_cutoffs_frame({})
    assert frame["mode"].tolist() == ["snp", "indel"]
    assert frame["vqslod_cutoff"].isna().all()


# --- overview --------------------------------------------------------------


def test_overview_computes_ti_tv():
    overview = variant_qc._overview({"n_ti": 6, "n_tv": 3})
    assert overview["ti_tv"] == 2.0


def test_overview_handles_zero_transversions():
    overview = variant_qc._overview({"n_ti": 6, "n_tv": 0})
    assert overview["ti_tv"] != overview["ti_tv"]  # nan


# --- VQSLOD ----------------------------------------------------------------


def test_vqslod_stats_frame():
    frame = variant_qc._vqslod_stats_frame(
        {
            "snp": {"n": 10, "min": -1.0, "p25": 0.0, "median": 1.0, "p75": 2.0,
                    "max": 3.0, "mean": 1.0},
            "indel": {"n": 4, "min": -5.0, "p25": -3.0, "median": -2.0, "p75": -1.0,
                      "max": 0.0, "mean": -2.2},
        }
    )
    assert list(frame.columns) == [
        "variant_class", "n", "min", "p25", "median", "p75", "max", "mean",
    ]
    assert frame["variant_class"].tolist() == ["snp", "indel"]
    assert frame["median"].tolist() == [1.0, -2.0]


def test_vqslod_histogram_frame_tags_class():
    frame = variant_qc._vqslod_histogram_frame(
        {
            "snp": {"bin_edges": [0.0, 1.0, 2.0], "bin_freq": [3, 4]},
            "indel": {"bin_edges": [0.0, 1.0, 2.0], "bin_freq": [1, 0]},
        }
    )
    assert list(frame.columns) == ["variant_class", "bin_center", "count"]
    assert frame["variant_class"].tolist() == ["snp", "snp", "indel", "indel"]
    assert frame["bin_center"].tolist() == [0.5, 1.5, 0.5, 1.5]
    assert frame["count"].tolist() == [3, 4, 1, 0]


def test_histogram_bounds_guards_nonfinite():
    # GATK writes ``Infinity``/``-Infinity``/``NaN`` AS_VQSLOD; the raw max is
    # infinite and the raw min is a finite but far outlier.
    assert variant_qc._histogram_bounds(-4.3e7, float("inf")) == (
        -4.3e7,
        -4.3e7 + 1.0,
    )
    assert variant_qc._histogram_bounds(float("-inf"), float("-inf")) == (0.0, 1.0)


def test_histogram_bounds_guards_degenerate_and_missing():
    assert variant_qc._histogram_bounds(2.0, 2.0) == (2.0, 3.0)
    assert variant_qc._histogram_bounds(None, None) == (0.0, 1.0)
    assert variant_qc._histogram_bounds(1.0, 5.0) == (1.0, 5.0)


# --- class / contig / culprit ----------------------------------------------


def test_variant_class_frame():
    frame = variant_qc._variant_class_frame(
        counts={("snv", "snv"): 8, ("ins", "indel"): 2},
        filtered={("snv", "snv"): 1, ("ins", "indel"): 2},
    )
    assert list(frame.columns) == [
        "allele_type", "variant_type", "n_variants", "n_pass", "n_filtered",
        "filter_rate",
    ]
    assert frame["allele_type"].tolist() == ["snv", "ins"]
    assert frame["n_pass"].tolist() == [7, 0]
    assert frame["filter_rate"].tolist() == [0.125, 1.0]


def test_filter_by_contig_frame():
    frame = variant_qc._filter_by_contig_frame(
        counts={"1": 10, "2": 5}, filtered={"1": 2, "2": 5}
    )
    assert list(frame.columns) == ["contig", "n_variants", "n_filtered", "filter_rate"]
    assert frame["filter_rate"].tolist() == [0.2, 1.0]


def test_culprit_frame():
    frame = variant_qc._culprit_frame({"QD": 3, "FS": 1}, n_filtered=4)
    assert list(frame.columns) == ["culprit", "n_variants", "fraction"]
    assert frame["culprit"].tolist() == ["QD", "FS"]
    assert frame["fraction"].tolist() == [0.75, 0.25]


def test_culprit_frame_empty():
    frame = variant_qc._culprit_frame({}, n_filtered=0)
    assert frame.empty
    assert list(frame.columns) == ["culprit", "n_variants", "fraction"]


def test_top_filtered_frame_formats_filters():
    rows = pd.DataFrame(
        {
            "locus": ["1:100", "2:200"],
            "filters": [{"VQSRTrancheSNP99.00to99.90"}, set()],
            "AS_VQSLOD": [1.0, -2.0],
        }
    )
    frame = variant_qc._top_filtered_frame(rows)
    assert frame["filters"].tolist() == ["VQSRTrancheSNP99.00to99.90", ""]
    assert "AS_VQSLOD" in frame.columns


# --- public surface --------------------------------------------------------


def test_collectors_are_exposed():
    for name in (
        "vqsr_provenance",
        "variant_overview",
        "vqsr_cutoffs",
        "filter_counts",
        "vqslod_summary",
        "vqslod_bins",
        "variant_class_counts",
        "filter_by_contig",
        "culprit_counts",
        "top_filtered_sites",
    ):
        assert callable(getattr(reporting, name))


# --- plot builders ---------------------------------------------------------


def test_filter_counts_bar():
    frame = pd.DataFrame({"filter": ["LowQual", "VQSR"], "n_variants": [2, 5]})
    figure = reporting.filter_counts_bar(frame)
    assert isinstance(figure, pn.ggplot)
    assert figure.data["n_variants"].tolist() == [2, 5]


def test_filter_rate_by_contig_orders_axis():
    frame = pd.DataFrame(
        {"contig": ["10", "2", "1"], "filter_rate": [0.1, 0.2, 0.3]}
    )
    figure = reporting.filter_rate_by_contig(frame)
    assert isinstance(figure, pn.ggplot)
    assert list(figure.data["contig"].cat.categories) == ["1", "2", "10"]


def test_vqslod_histogram_with_cutoffs():
    frame = pd.DataFrame(
        {
            "variant_class": ["snp", "snp", "indel", "indel"],
            "bin_center": [0.5, 1.5, 0.5, 1.5],
            "count": [3, 4, 1, 0],
        }
    )
    figure = reporting.vqslod_histogram(frame, {"snp": 1.2, "indel": None})
    assert isinstance(figure, pn.ggplot)
    assert figure.data["count"].tolist() == [3, 4, 1, 0]


def test_variant_class_bar():
    frame = pd.DataFrame(
        {
            "allele_type": ["snv", "ins"],
            "variant_type": ["snv", "indel"],
            "n_pass": [7, 0],
            "n_filtered": [1, 2],
        }
    )
    figure = reporting.variant_class_bar(frame)
    assert isinstance(figure, pn.ggplot)
    assert set(figure.data["outcome"]) == {"n_pass", "n_filtered"}


def test_culprit_bar():
    frame = pd.DataFrame({"culprit": ["QD", "FS"], "n_variants": [5, 2]})
    figure = reporting.culprit_bar(frame)
    assert isinstance(figure, pn.ggplot)
    assert figure.data["n_variants"].tolist() == [5, 2]
