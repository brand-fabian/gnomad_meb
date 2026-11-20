"""Tests for the GWAS metadata objects.

:class:`GwasMetadata` derives the gated, ordered response set from the
``[gwas].responses`` model and the study metadata CSV at construction, exactly
as the export writes it. :class:`GwasDatasetMetadata` reads and writes the
exported dataset's GWAS metadata file. These pin the derivation, the
``min_case_count`` gate and the file read/write round-trip without a hail
session.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas
import pytest

from gnomad_meb.config import GwasConfig, GwasResponseColumn, GwasResponses, Settings
from gnomad_meb.lib.gwas.metadata import (
    GwasDatasetMetadata,
    GwasMetadata,
    GwasMetadataException,
)


def _write_csv(path: Path) -> Path:
    pandas.DataFrame(
        {
            "sample_id": ["1", "2", "3", "4"],
            "cohort": ["case", "case", "control", "control"],
            "dosage": [1.5, None, 2.0, 3.0],
            "conditions": [{"a", "b"}, {"a"}, set(), set()],
            "bmi": [1.0, 0.0, 1.0, 0.0],
        }
    ).to_csv(path, index=False)
    return path


def _settings(csv: Path, *, min_case_count: int = 2, **response_overrides) -> Settings:
    columns = response_overrides.pop(
        "columns",
        [
            GwasResponseColumn(name="dosage", kind="quantitative"),
            GwasResponseColumn(name="conditions", kind="set"),
            GwasResponseColumn(name="bmi", kind="binary"),
        ],
    )
    return Settings(
        validate_paths=False,
        project="study",
        checkpoint="/ckpt",
        gwas=GwasConfig(
            metadata_csv=csv,
            case_cohort="case",
            control_cohorts=["control"],
            min_case_count=min_case_count,
            responses=GwasResponses(columns=columns, **response_overrides),
        ),
    )


def _metadata(csv: Path, **overrides) -> GwasMetadata:
    return GwasMetadata.from_settings(_settings(csv, **overrides))


def test_metadata_returns_the_gated_ordered_set(tmp_path: Path) -> None:
    csv = _write_csv(tmp_path / "md.csv")

    assert _metadata(csv).responses == ("case", "a", "dosage", "bmi")


def test_response_below_min_case_count_is_omitted(tmp_path: Path) -> None:
    csv = _write_csv(tmp_path / "md.csv")

    # ``b`` sums to 1 (< 2) so it is dropped; at min 3 ``case``/``a``/``bmi`` fall.
    assert "b" not in _metadata(csv).responses
    assert _metadata(csv, min_case_count=3).responses == ("dosage",)


def test_quantitative_gates_on_non_missing_count(tmp_path: Path) -> None:
    csv = _write_csv(tmp_path / "md.csv")

    # ``dosage`` sums to 6.5 but has only 3 non-missing values.
    assert "dosage" in _metadata(csv, min_case_count=3).responses
    assert "dosage" not in _metadata(csv, min_case_count=4).responses


def test_set_values_are_snake_cased_in_sorted_order(tmp_path: Path) -> None:
    csv = _write_csv(tmp_path / "md.csv")
    settings = _settings(
        csv,
        columns=[
            GwasResponseColumn(name="conditions", kind="set"),
        ],
    )

    assert GwasMetadata.from_settings(settings).responses == ("case", "a")


def test_kinds_and_names_are_restricted_to_responses(tmp_path: Path) -> None:
    csv = _write_csv(tmp_path / "md.csv")
    metadata = _metadata(csv)

    # ``b`` is below the gate, so it is absent from the maps too.
    assert metadata.kinds == {
        "case": "binary",
        "a": "binary",
        "dosage": "quantitative",
        "bmi": "binary",
    }
    assert metadata.names == {
        "case": "case",
        "a": "a",
        "dosage": "dosage",
        "bmi": "bmi",
    }
    assert set(metadata.responses) == set(metadata.kinds) == set(metadata.names)


def test_missing_responses_section_fails() -> None:
    settings = Settings(validate_paths=False, project="study", checkpoint="/ckpt")

    with pytest.raises(GwasMetadataException, match="gwas configuration"):
        GwasMetadata.from_settings(settings)


def test_dataset_readers_over_a_metadata_file(tmp_path: Path) -> None:
    dataset = tmp_path / "ds"
    dataset.mkdir()
    (dataset / "study.gwas.md.json").write_text(
        json.dumps(
            {
                "response_name_map": {"case": "Case"},
                "response_kind_map": {"case": "binary", "dosage": "quantitative"},
                "source": "filter",
                "name": "rare_lof",
                "n_variants": 42,
                "filter_high_quality_gnomad": {"variant_info": {"n_variants": 10}},
                "filter_variant_info": {"filters": ["lof"], "lof": {"n_variants": 3}},
                "effect_mask_info": {
                    "variant_class": {"lof": 3, "rare": 12},
                },
            }
        )
    )

    record = GwasDatasetMetadata.read(dataset)
    assert record.names == {"case": "Case"}
    assert record.kinds == {
        "case": "binary",
        "dosage": "quantitative",
    }
    assert record.filter_globals == {
        "filter_high_quality_gnomad": {"variant_info": {"n_variants": 10}},
        "filter_variant_info": {"filters": ["lof"], "lof": {"n_variants": 3}},
    }
    assert (record.source, record.name) == ("filter", "rare_lof")
    assert record.n_variants == 42
    assert record.effect_mask_info == {"variant_class": {"lof": 3, "rare": 12}}


def test_missing_metadata_file_defaults(tmp_path: Path) -> None:
    dataset = tmp_path / "ds"
    dataset.mkdir()

    record = GwasDatasetMetadata.read(dataset)
    assert record.kinds == {}
    assert record.names == {}
    assert record.filter_globals == {}
    assert (record.source, record.name) == (None, None)
    assert record.n_variants == 0

    # No recorded count: the variant count falls back to the PLINK ``.bim`` rows.
    (dataset / "study.bim").write_text("\n".join(f"{i}\t0\t0\tG\tA" for i in range(5)))
    assert GwasDatasetMetadata.read(dataset).n_variants == 5


def test_filter_globals_drop_absent_and_empty_values(tmp_path: Path) -> None:
    dataset = tmp_path / "ds"
    dataset.mkdir()
    (dataset / "study.gwas.md.json").write_text(
        json.dumps({"filter_variant_info": {}, "filter_region_info": None})
    )

    assert GwasDatasetMetadata.read(dataset).filter_globals == {}


def test_write_gwas_metadata_round_trips(tmp_path: Path) -> None:
    dataset = tmp_path / "ds"
    dataset.mkdir()
    # The export writes PLINK and the TSVs before the metadata file; the writer
    # recovers the dataset basename from one of those artifacts.
    (dataset / "study.phenotype.tsv.bgz").write_text("")

    record = GwasDatasetMetadata(
        names={"case": "Case"},
        kinds={"case": "binary", "dosage": "quantitative"},
        n_variants=42,
        source="filter",
        name="rare_lof",
        filter_globals={
            "filter_variant_info": {"filters": ["lof"]},
            "filter_region_info": None,
        },
    )
    path = record.write(dataset)

    assert path == dataset / "study.gwas.md.json"
    read_back = GwasDatasetMetadata.read(dataset)
    assert read_back.names == {"case": "Case"}
    assert read_back.kinds == {
        "case": "binary",
        "dosage": "quantitative",
    }
    assert read_back.filter_globals == {"filter_variant_info": {"filters": ["lof"]}}
    assert (read_back.source, read_back.name) == ("filter", "rare_lof")
    assert read_back.n_variants == 42


def test_write_gwas_metadata_round_trips_effect_mask_info(tmp_path: Path) -> None:
    dataset = tmp_path / "ds"
    dataset.mkdir()
    (dataset / "study.bim").write_text("")

    record = GwasDatasetMetadata(
        names={},
        kinds={},
        effect_mask_info={"variant_class": {"lof": 3, "rare": 12}},
    )
    record.write(dataset)

    read_back = GwasDatasetMetadata.read(dataset)
    assert read_back.effect_mask_info == {"variant_class": {"lof": 3, "rare": 12}}


def test_write_gwas_metadata_omits_absent_effect_mask_info(tmp_path: Path) -> None:
    dataset = tmp_path / "ds"
    dataset.mkdir()
    (dataset / "study.bim").write_text("")

    path = GwasDatasetMetadata(names={}, kinds={}).write(dataset)

    assert "effect_mask_info" not in json.loads(path.read_text())
    assert GwasDatasetMetadata.read(dataset).effect_mask_info == {}


def test_write_gwas_metadata_omits_absent_provenance(tmp_path: Path) -> None:
    dataset = tmp_path / "ds"
    dataset.mkdir()
    (dataset / "study.bim").write_text("")

    path = GwasDatasetMetadata(names={}, kinds={}).write(dataset)

    assert path == dataset / "study.gwas.md.json"
    assert json.loads(path.read_text()) == {
        "response_name_map": {},
        "response_kind_map": {},
    }


def test_importing_the_metadata_module_does_not_initialize_hail() -> None:
    code = (
        "import gnomad_meb.lib.gwas.metadata\n"
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
