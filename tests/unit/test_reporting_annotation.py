"""Hail-free unit tests for the Annotation reporting collectors and builders.

The collectors are thin hail reads over the reshape helpers, so the helpers are
driven here with hand-built ``pandas`` frames / plain dicts (the
``variant_qc.py`` precedent); the public surface of the hail collectors is pinned
only. No ``hail.init()``.
"""

from __future__ import annotations

import pandas as pd
import plotnine as pn

from gnomad_meb import reporting
from gnomad_meb.reporting import annotation

# --- presence / consequence -------------------------------------------------


def test_presence_frame():
    frame = annotation._presence_frame({"vep": 9, "clinvar": 3}, n_variants=10)
    assert list(frame.columns) == ["annotation", "n_present", "fraction"]
    assert frame["n_present"].tolist() == [9, 3]
    assert frame["fraction"].tolist() == [0.9, 0.3]


def test_presence_frame_zero_variants():
    frame = annotation._presence_frame({"vep": 0}, n_variants=0)
    assert frame["fraction"].tolist() == [0.0]


def test_consequence_frame_sorts_by_count():
    frame = annotation._consequence_frame(
        {"intron_variant": 3, "missense_variant": 7}
    )
    assert frame["most_severe_consequence"].tolist() == [
        "missense_variant",
        "intron_variant",
    ]
    assert frame["fraction"].tolist() == [0.7, 0.3]


def test_insilico_frame_uses_canonical_field_order():
    frame = annotation._insilico_frame({"cadd": 4, "phylop": 2}, n_variants=10)
    # sorted by presence, but every canonical field is represented
    assert set(frame["predictor"]) == set(annotation.IN_SILICO_FIELDS)
    assert frame.iloc[0]["predictor"] == "cadd"
    assert frame.loc[frame.predictor == "revel_max", "n_present"].item() == 0


def test_clinvar_frame():
    frame = annotation._clinvar_frame({"Benign": 8, "Pathogenic": 2}, n_variants=20)
    assert list(frame.columns) == [
        "clinical_significance",
        "n_variants",
        "fraction",
    ]
    assert frame["clinical_significance"].tolist() == ["Benign", "Pathogenic"]
    assert frame["fraction"].tolist() == [0.4, 0.1]


def test_clinvar_frame_empty():
    frame = annotation._clinvar_frame({}, n_variants=10)
    assert frame.empty
    assert list(frame.columns) == [
        "clinical_significance",
        "n_variants",
        "fraction",
    ]


# --- frequencies ------------------------------------------------------------


def test_freq_stratum_frame():
    # ``freq_meta`` is a ``Array[Dict[String,String]]`` global: ``hl.eval``
    # yields a list of plain dicts, one per stratum.
    frame = annotation._freq_stratum_frame(
        freq_meta=[
            {"group": "adj"},
            {"gen_anc": "nfe", "group": "adj"},
        ],
        sample_counts=[14, 6],
        stats={
            "0": {"n_defined": 90, "n_af_gt0": 60, "mean_af": 0.02},
            "1": {"n_defined": 80, "n_af_gt0": 40, "mean_af": 0.01},
        },
    )
    assert list(frame.columns) == list(annotation._FREQ_STRATUM_COLUMNS)
    assert frame["group"].tolist() == ["adj", "adj"]
    assert frame["gen_anc"].tolist() == [None, "nfe"]
    assert frame["n_samples"].tolist() == [14, 6]
    assert frame["n_af_gt0"].tolist() == [60, 40]


def test_gnomad_freq_frame_selects_gen_anc_adj_keys():
    frame = annotation._gnomad_freq_frame(
        index_dict={"adj": 0, "raw": 1, "nfe_adj": 2, "afr_adj": 8},
        stats={
            "adj": {"n_defined": 90, "n_af_gt0": 70, "mean_af": 0.03},
            "raw": {"n_defined": 90, "n_af_gt0": 90, "mean_af": 0.04},
            "nfe_adj": {"n_defined": 89, "n_af_gt0": 60, "mean_af": 0.02},
        },
    )
    assert list(frame.columns) == list(annotation._GNOMAD_FREQ_COLUMNS)
    assert "adj" in frame["population"].tolist()
    assert "nfe" in frame["population"].tolist()
    assert (
        frame.loc[frame.population == "afr", "index"].item() == 8
    )
    assert frame.loc[frame.population == "afr", "n_defined"].item() == 0


# --- coverage ---------------------------------------------------------------


def test_coverage_group_frame():
    frame = annotation._coverage_group_frame(
        meta=[{}, {"filtered": "pass"}, {"cohort": "c1", "filtered": "pass"}],
        stats={
            "0": {
                "n_sites": 100,
                "mean_coverage": 31.2,
                "median_coverage": 30,
                "fraction_over_10": 0.95,
            },
            "1": {
                "n_sites": 80,
                "mean_coverage": 29.0,
                "median_coverage": 28,
                "fraction_over_10": 0.9,
            },
        },
    )
    assert list(frame.columns) == list(annotation._COVERAGE_GROUP_COLUMNS)
    assert frame["stratum"].tolist() == [
        "all",
        "filtered=pass",
        "cohort=c1;filtered=pass",
    ]
    assert frame["filtered"].tolist() == [None, "pass", "pass"]
    assert frame["cohort"].tolist() == [None, None, "c1"]
    assert frame["n_sites"].tolist() == [100, 80, 0]
    assert frame["mean_coverage"].iloc[0] == 31.2


def test_coverage_contig_frame():
    frame = annotation._coverage_contig_frame(
        meta=[{"filtered": "pass"}],
        stats={
            "0": {
                "chr1": {
                    "n_sites": 10,
                    "mean_coverage": 30.0,
                    "median_coverage": 29,
                    "fraction_over_10": 0.9,
                },
                "chrX": {
                    "n_sites": 5,
                    "mean_coverage": 15.0,
                    "median_coverage": 14,
                    "fraction_over_10": 0.4,
                },
            }
        },
    )
    assert list(frame.columns) == list(annotation._COVERAGE_CONTIG_COLUMNS)
    assert set(frame["contig"]) == {"chr1", "chrX"}
    assert frame["stratum"].tolist() == ["filtered=pass", "filtered=pass"]
    assert frame.loc[frame.contig == "chr1", "mean_coverage"].item() == 30.0


# --- region / protein-changing ---------------------------------------------


def test_region_frame_formats_filters():
    rows = pd.DataFrame(
        {
            "locus": ["chr16:11255182"],
            "alleles": ["A,G"],
            "filters": [{"vqc_uninformative"}],
            "most_severe_consequence": ["missense_variant"],
        }
    )
    frame = annotation._region_frame(rows)
    assert frame["filters"].tolist() == ["vqc_uninformative"]
    assert "insilico_cadd" not in frame.columns


def test_unannotated_frame_formats_filters():
    rows = pd.DataFrame(
        {
            "locus": ["chr1:100", "chr1:200"],
            "alleles": ["A,C", "G,T"],
            "filters": [set(), {"VQSRTrancheSNP99.00to99.90+"}],
            "gnomad_af": [0.0, None],
        }
    )
    frame = annotation._unannotated_frame(rows)
    assert list(frame.columns) == list(annotation._UNANNOTATED_COLUMNS)
    assert frame["filters"].tolist() == ["", "VQSRTrancheSNP99.00to99.90+"]


def test_bed_records_skips_comments(tmp_path):
    path = tmp_path / "socs1.bed"
    path.write_text(
        "track name=socs1\n"
        "# a comment\n"
        "\n"
        "chr16\t11255181\t11255182\n"
    )
    assert annotation._bed_records(path) == [("chr16", 11255181, 11255182)]


# --- public surface --------------------------------------------------------


def test_collectors_are_exposed():
    for name in (
        "annotation_provenance",
        "annotation_overview",
        "annotation_presence",
        "vep_consequence_counts",
        "in_silico_presence",
        "clinvar_counts",
        "freq_by_stratum",
        "gnomad_freq_by_population",
        "coverage_by_group",
        "coverage_by_contig",
        "region_variants",
        "protein_changing_check",
        "unannotated_protein_changing",
        "bed_intervals",
    ):
        assert callable(getattr(reporting, name))
    assert "missense_variant" in reporting.PROTEIN_CHANGING_CONSEQUENCES


# --- plot builders ---------------------------------------------------------


def test_annotation_presence_bar():
    frame = pd.DataFrame(
        {"annotation": ["vep", "clinvar"], "n_present": [10, 3], "fraction": [1.0, 0.3]}
    )
    figure = reporting.annotation_presence_bar(frame)
    assert isinstance(figure, pn.ggplot)
    assert figure.data["fraction"].tolist() == [1.0, 0.3]


def test_vep_consequence_bar_caps_rows():
    frame = pd.DataFrame(
        {
            "most_severe_consequence": [f"csq{i}" for i in range(30)],
            "n_variants": list(range(30, 0, -1)),
        }
    )
    figure = reporting.vep_consequence_bar(frame, n=20)
    assert isinstance(figure, pn.ggplot)
    assert len(figure.data) == 20


def test_in_silico_presence_bar():
    frame = pd.DataFrame(
        {
            "predictor": ["cadd", "revel_max"],
            "n_present": [8, 1],
            "fraction": [0.8, 0.1],
        }
    )
    figure = reporting.in_silico_presence_bar(frame)
    assert isinstance(figure, pn.ggplot)
    assert figure.data["predictor"].tolist() == ["cadd", "revel_max"]


def test_clinvar_significance_bar():
    frame = pd.DataFrame(
        {"clinical_significance": ["Benign", "Pathogenic"], "n_variants": [8, 2]}
    )
    figure = reporting.clinvar_significance_bar(frame)
    assert isinstance(figure, pn.ggplot)
    assert figure.data["n_variants"].tolist() == [8, 2]


def test_freq_stratum_bar_computes_polymorphic_fraction():
    frame = pd.DataFrame(
        {
            "stratum": ["group=adj", "gen_anc=nfe;group=adj"],
            "n_defined": [100, 50],
            "n_af_gt0": [60, 40],
        }
    )
    figure = reporting.freq_stratum_bar(frame)
    assert isinstance(figure, pn.ggplot)
    assert figure.data["polymorphic_fraction"].tolist() == [0.6, 0.8]


def test_gnomad_freq_population_bar():
    frame = pd.DataFrame(
        {
            "population": ["adj", "nfe", "afr"],
            "n_defined": [100, 100, 0],
            "n_af_gt0": [70, 60, 0],
        }
    )
    figure = reporting.gnomad_freq_population_bar(frame)
    assert isinstance(figure, pn.ggplot)
    assert figure.data["polymorphic_fraction"].tolist() == [0.7, 0.6, 0.0]


def test_coverage_contig_bar_orders_contigs():
    frame = pd.DataFrame(
        {
            "contig": ["chr10", "chr2"],
            "stratum": ["all", "all"],
            "n_sites": [1, 1],
            "mean_coverage": [20.0, 30.0],
            "fraction_over_10": [0.8, 0.9],
        }
    )
    figure = reporting.coverage_contig_bar(frame)
    assert isinstance(figure, pn.ggplot)
    # semantic contig order: chr2 precedes chr10
    assert list(figure.data["contig"].cat.categories) == ["chr2", "chr10"]
