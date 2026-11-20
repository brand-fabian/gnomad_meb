"""In-process tests for the GWAS shared helpers (``lib/gwas``).

These are pure-Python: they pin the response-name normalization, the set-column
membership semantics, the region labels and the ``RegenieDataset`` manifest,
none of which need a hail session. Import-safety is asserted by the absence of
a live hail backend after importing the package.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from gnomad_meb.lib.gwas import (
    RegenieDataset,
    column_membership,
    membership_union,
    region_label,
    response_name_mapping,
    snake_case,
    to_hail_literal_value,
)


@dataclass
class _Column:
    """A stand-in for :class:`gnomad_meb.config.GwasResponseColumn`."""

    name: str
    exclude: list[str]

    @property
    def field(self) -> str:
        return self.name


def test_snake_case_normalizes_unsafe_characters() -> None:
    assert snake_case("other_autoimmune_comorbidities") == (
        "other_autoimmune_comorbidities"
    )
    assert snake_case("Alopecia Areata") == "alopecia_areata"
    assert snake_case("thyroid-disease") == "thyroid_disease"
    assert snake_case("  ") == "response"


def test_response_name_mapping_records_display_names() -> None:
    mapping = response_name_mapping(["Alopecia Areata", "thyroid-disease"])

    assert mapping == {
        "alopecia_areata": "Alopecia Areata",
        "thyroid_disease": "thyroid-disease",
    }


def test_column_membership_removes_exclusions() -> None:
    assert column_membership(
        {"hypertension", "thyroid", "psoriasis"}, ["hypertension", "thyroid"]
    ) == {"psoriasis"}


def test_membership_union_respects_per_column_exclusions() -> None:
    values = {
        "comorbidities": {"hypertension", "thyroid", "psoriasis"},
        "comorbidities_thyroid": {"other_ai_thyroid_disease", "graves"},
        "other_autoimmune_comorbidities": {"vitiligo"},
    }
    columns = [
        _Column("comorbidities", ["hypertension", "thyroid"]),
        _Column("comorbidities_thyroid", ["other_ai_thyroid_disease"]),
        _Column("other_autoimmune_comorbidities", []),
    ]

    assert membership_union(values, columns) == {"psoriasis", "graves", "vitiligo"}


def test_region_labels_are_stable_and_path_safe() -> None:
    assert region_label("gene") == "gene"
    assert region_label("window", 1_000) == "window_1000"
    assert "/" not in region_label("gene")


def test_regenie_dataset_derives_its_paths(tmp_path: Path) -> None:
    base = tmp_path / "per_gene"
    base.mkdir()
    (base / "aapad.phenotype.tsv.bgz").write_text("")
    dataset = RegenieDataset(
        base_path=base,
        basename="aapad",
        plink_files={"1": str(base / "aapad"), "2": str(base / "aapad")},
    )

    assert dataset["base_path"] == base
    assert dataset["basename"] == "aapad"
    assert dataset.phenotype == base / "aapad.phenotype.tsv.bgz"
    assert dataset.covariates == base / "aapad.covariates.tsv.bgz"
    assert dataset.aaf == base / "aapad.aaf.tsv.bgz"
    assert dataset.annotation == base / "aapad.annotation.tsv.bgz"
    assert dataset.variant_sets == base / "aapad.variant_sets.tsv.bgz"
    assert dataset.effect_masks == base / "aapad.effect_masks.tsv.bgz"
    assert set(dataset.contigs) == {"1", "2"}
    assert dataset["1"] == base / "aapad"


def test_to_hail_literal_value_drops_nulls_and_wraps_structs(hl) -> None:
    value = {
        "cutoffs": {"min_gnomad_af": 0.01, "max_gnomad_af": None},
        "filters": ["gnomad_freq", None, "lof"],
        "kept": {"n_variants": 0, "flag": False},
    }

    converted = to_hail_literal_value(value)
    assert hl.eval(converted) == {
        "cutoffs": {"min_gnomad_af": 0.01},
        "filters": ["gnomad_freq", "lof"],
        "kept": {"n_variants": 0, "flag": False},
    }


def test_to_hail_literal_value_handles_scalars_and_nested_lists(hl) -> None:
    assert hl.eval(to_hail_literal_value([[1, None], [], None])) == [[1]]
    assert hl.eval(to_hail_literal_value(0)) == 0
    assert hl.eval(to_hail_literal_value(False)) is False


def test_to_hail_literal_value_does_not_mutate_its_input(hl) -> None:
    value = {"cutoffs": {"min_pli": None, "min_cadd": 20.0}}
    to_hail_literal_value(value)
    assert value == {"cutoffs": {"min_pli": None, "min_cadd": 20.0}}


def test_hl_literal_accepts_filter_globals_with_nulls(hl) -> None:
    """Regression for the real failing metadata: nulls *and* partial unification.

    The export records every parameter, so a filter left at its ``None`` default
    serializes to JSON ``null`` and reads back as ``None``. ``hl.literal`` on
    that tree raises ``cannot impute`` (a null hole, or -- even with nulls gone
    -- a nested dict whose values only partially unify, e.g. ``all_lof``'s
    ``cutoffs``/``n_variants``). The helper must make the real shape literalizable.
    """
    import pytest

    value = {
        "filters": ["lof"],
        "lof": {
            "cutoffs": {
                "impact": ["MODERATE", "HIGH"],
                "min_cadd": None,
                "min_pli": None,
                "min_z_score": None,
            },
            "n_variants": {"cadd": 5, "default": 5, "final": 3, "lof": 5},
        },
    }

    with pytest.raises(Exception, match="cannot impute"):
        hl.literal(value)

    expr = hl.literal(to_hail_literal_value(value))
    assert hl.eval(expr)["lof"]["cutoffs"] == {"impact": ["MODERATE", "HIGH"]}
    assert hl.eval(expr)["lof"]["n_variants"]["final"] == 3


def test_importing_the_package_does_not_initialize_hail_or_snakemake() -> None:
    code = (
        "import gnomad_meb.lib.gwas\n"
        "import sys\n"
        "from hail.utils.java import Env\n"
        "assert 'snakemake' not in sys.modules, 'snakemake imported'\n"
        "assert not Env.is_fully_initialized(), 'hail initialized'\n"
        "print('ok')\n"
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
