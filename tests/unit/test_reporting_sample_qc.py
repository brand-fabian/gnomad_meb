"""Hail-free unit tests for the Sample QC reporting collectors and builders.

The collectors are thin hail reads over the reshape helpers, so the helpers are
driven here with hand-built ``pandas`` frames / plain dicts (the ``vds.py``
precedent); the public surface of the hail collectors is pinned only. No
``hail.init()``.
"""

from __future__ import annotations

import matplotlib.figure
import pandas as pd
import plotnine as pn
from upsetplot import from_memberships

from gnomad_meb import reporting
from gnomad_meb.reporting import sample_qc

# --- configuration ---------------------------------------------------------


def test_config_summary_cutoffs():
    config = sample_qc._config_summary(
        {"sample_qc_filter_cutoffs": {"max_kinship": 0.2}}, n_samples=7
    )
    assert config["n_samples"] == 7
    assert config["sample_qc_filter_cutoffs"] == {"max_kinship": 0.2}
    assert config["charr_params"] == {}
    assert config["ploidy_params"] == {}
    assert config["ancestry_params"] == {}


def test_as_plain_recurses_structs():
    from types import SimpleNamespace as NS

    plain = sample_qc._as_plain(NS(a=1, nested=NS(b=2)))
    assert plain == {"a": 1, "nested": {"b": 2}}


# --- hard filters ----------------------------------------------------------


def test_filter_counts_frame():
    frame = sample_qc._filter_counts_frame(
        {"high_n_singleton": 3, "low_coverage": 1}, n_samples=10
    )
    assert list(frame.columns) == ["filter", "n_samples", "fraction"]
    assert frame["filter"].tolist() == ["high_n_singleton", "low_coverage"]
    assert frame["fraction"].tolist() == [0.3, 0.1]


def test_filter_counts_frame_empty():
    frame = sample_qc._filter_counts_frame({}, n_samples=0)
    assert list(frame.columns) == ["filter", "n_samples", "fraction"]
    assert frame.empty


def test_membership_frame():
    rows = pd.DataFrame(
        {
            "hard_filters": [
                {"high_n_singleton"},
                {"high_n_singleton", "relatedness"},
                set(),
            ]
        },
        index=pd.Index(["s1", "s2", "s3"], name="s"),
    )
    frame = sample_qc._membership_frame(rows)
    assert list(frame.columns) == ["high_n_singleton", "relatedness"]
    assert bool(frame.loc["s2", "relatedness"]) is True
    assert bool(frame.loc["s1", "relatedness"]) is False
    assert not frame.loc["s3"].any()
    assert frame.index.tolist() == ["s1", "s2", "s3"]


def test_membership_frame_all_empty():
    rows = pd.DataFrame(
        {"hard_filters": [set(), set()]}, index=pd.Index(["s1", "s2"], name="s")
    )
    frame = sample_qc._membership_frame(rows)
    assert frame.shape[1] == 0
    assert frame.index.tolist() == ["s1", "s2"]


# --- coverage --------------------------------------------------------------


def test_coverage_frame_flattens_dp_threshold():
    rows = pd.DataFrame(
        {
            "autosomal_mean_dp": [30.0, 10.0],
            "bases_over_dp_threshold": [[0, 100, 200, 50, 0], [0, 5, 6, 7, 8]],
            "callrate": [0.9, 0.8],
            "callrate_adj": [0.85, 0.7],
            "n_singleton": [1, 2],
            "r_het_hom_var": [1.5, 2.0],
            "n_het": [10, 20],
            "n_hom_var": [5, 10],
        },
        index=pd.Index(["s1", "s2"], name="s"),
    )
    frame = sample_qc._coverage_frame(rows)
    assert "bases_over_dp_threshold" not in frame.columns
    assert frame["bases_over_dp_1"].tolist() == [100, 5]
    assert frame["bases_over_dp_20"].tolist() == [50, 7]


def test_histogram_frame():
    frame = sample_qc._histogram_frame(
        {"bin_edges": [0.0, 1.0, 2.0], "bin_freq": [3, 4]}, "m"
    )
    assert frame["metric"].tolist() == ["m", "m"]
    assert frame["bin_center"].tolist() == [0.5, 1.5]
    assert frame["count"].tolist() == [3, 4]


# --- sex / ancestry --------------------------------------------------------


def test_ancestry_frame_expands_scores():
    rows = pd.DataFrame(
        {
            "pop": ["nfe", "afr"],
            "prob_nfe": [0.9, 0.1],
            "prob_afr": [0.1, 0.9],
            "scores": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        },
        index=pd.Index(["s1", "s2"], name="s"),
    )
    frame = sample_qc._ancestry_frame(rows)
    assert list(frame.columns) == ["pop", "prob_nfe", "prob_afr", "pc1", "pc2", "pc3"]
    assert frame["pc1"].tolist() == [0.1, 0.4]
    assert frame["pc3"].tolist() == [0.3, 0.6]


# --- relatedness -----------------------------------------------------------


def test_relatedness_frame_joins_population_and_caps():
    pairs = pd.DataFrame(
        {
            "i": ["s1", "s2"],
            "j": ["s2", "s3"],
            "kin": [0.5, 0.2],
            "ibd0": [0.1, 0.2],
            "ibd1": [0.3, 0.4],
            "ibd2": [0.1, 0.2],
        }
    )
    pop = pd.Series({"s1": "nfe", "s2": "afr", "s3": "nfe"})
    frame = sample_qc._relatedness_frame(pairs, pop, cap=1)
    assert len(frame) == 1
    assert list(frame.columns) == [
        "i",
        "j",
        "kinship",
        "ibd0",
        "ibd1",
        "ibd2",
        "pop_i",
        "pop_j",
    ]
    assert frame.iloc[0]["kinship"] == 0.5
    assert frame.iloc[0]["pop_i"] == "nfe"
    assert frame.iloc[0]["pop_j"] == "afr"


def test_kinship_matrix_mirrors_pairs():
    pairs = pd.DataFrame({"i": ["s1"], "j": ["s2"], "kinship": [0.2]})
    matrix = sample_qc._kinship_matrix(pairs)
    assert set(zip(matrix["i"], matrix["j"], strict=True)) == {
        ("s1", "s2"),
        ("s2", "s1"),
    }
    assert matrix["kinship"].tolist() == [0.2, 0.2]


def test_kinship_matrix_empty():
    matrix = sample_qc._kinship_matrix(
        pd.DataFrame(columns=["i", "j", "kinship"])
    )
    assert matrix.empty
    assert list(matrix.columns) == ["i", "j", "kinship"]


# --- public surface --------------------------------------------------------


def test_collectors_are_exposed():
    for name in (
        "config_summary",
        "sample_overview",
        "hard_filter_counts",
        "hard_filter_membership",
        "coverage_metrics",
        "coverage_histograms",
        "sex_inference",
        "ancestry_scores",
        "contamination",
        "relatedness_pairs",
    ):
        assert callable(getattr(reporting, name))


# --- plot builders ---------------------------------------------------------


def test_hard_filter_counts_bar():
    frame = pd.DataFrame(
        {"filter": ["low_coverage", "high_n_singleton"], "n_samples": [2, 5]}
    )
    figure = reporting.hard_filter_counts_bar(frame)
    assert isinstance(figure, pn.ggplot)
    assert figure.data["n_samples"].tolist() == [2, 5]


def test_hard_filter_upset_returns_figure():
    # ``hard_filter_membership`` hands ``hard_filter_upset`` the boolean
    # MultiIndex frame ``upsetplot.from_memberships`` produces, not a plain
    # bool-column frame.
    membership = from_memberships(
        [["low_coverage"], ["relatedness"]], data=["s1", "s2"]
    )
    figure = reporting.hard_filter_upset(membership)
    assert isinstance(figure, matplotlib.figure.Figure)


def test_hard_filter_upset_handles_no_filters():
    figure = reporting.hard_filter_upset(pd.DataFrame(index=["s1"]))
    assert isinstance(figure, matplotlib.figure.Figure)


def test_ancestry_pca_scatter():
    frame = pd.DataFrame(
        {"pc1": [0.1, 0.2], "pc2": [0.3, 0.4], "pop": ["nfe", "afr"]}
    )
    figure = reporting.ancestry_pca_scatter(frame)
    assert isinstance(figure, pn.ggplot)
    assert figure.data["pop"].tolist() == ["nfe", "afr"]


def test_relatedness_heatmap_mirrors_pairs():
    frame = pd.DataFrame({"i": ["s1"], "j": ["s2"], "kinship": [0.3]})
    figure = reporting.relatedness_heatmap(frame, min_kinship=0.1)
    assert isinstance(figure, pn.ggplot)
    assert set(zip(figure.data["i"], figure.data["j"], strict=True)) == {
        ("s1", "s2"),
        ("s2", "s1"),
    }
    assert figure.data["kinship"].tolist() == [0.3, 0.3]


def test_relatedness_heatmap_handles_no_pairs():
    figure = reporting.relatedness_heatmap(
        pd.DataFrame(columns=["i", "j", "kinship"]), min_kinship=0.1
    )
    assert isinstance(figure, pn.ggplot)
    assert figure.data.empty
