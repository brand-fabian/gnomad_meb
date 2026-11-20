"""Tests for the GWAS pipeline stage declarations (export/run/load).

These exercise the declaration, not the hail computations: the per-region
fan-out, the checkpoint prefixes, the terminal export's artifacts and edges,
and the reconstruction-equality invariant (ADR-0017). No hail session is
started.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from gnomad_meb.cmd.task import GwasExport, GwasLoad, GwasPipeline, Task
from gnomad_meb.config import (
    GwasConfig,
    GwasRegion,
    GwasResponseColumn,
    GwasResponses,
    Settings,
)
from gnomad_meb.lib.checkpoint import CHECKPOINT_TYPE, Artifact
from gnomad_meb.lib.gwas.export import GwasExportPipeline
from gnomad_meb.lib.gwas.load import (
    GwasLoadPipeline,
    parse_response_frame,
    regenie_rename,
    response_from_filename,
)
from gnomad_meb.lib.gwas.metadata import GwasMetadata
from gnomad_meb.lib.gwas.run import GwasRunPipeline
from gnomad_meb.pipeline import PipelineFactory, PipelineRunner, StepRegistry


def make_settings(regions: list[GwasRegion] | None = None) -> Settings:
    """A settings object with a minimal, valid GWAS study model.

    Writes a fresh study CSV per call so no test depends on an import-time
    side effect.
    """
    study_csv = Path(tempfile.mkdtemp(prefix="gwas-export-study-")) / "study.md.csv"
    study_csv.write_text("sample_id,cohort\n1,case-cohort\n2,control-cohort\n")
    return Settings(
        validate_paths=False,
        project="study",
        checkpoint="/ckpt",
        gwas=GwasConfig(
            metadata_csv=study_csv,
            case_cohort="case-cohort",
            control_cohorts=["control-cohort"],
            drop_cohorts=["dropped"],
            min_n_snps=0,
            regions=regions or [GwasRegion(kind="gene")],
        ),
    )


def _export_pipeline(**overrides) -> GwasExportPipeline:
    return GwasExportPipeline(
        Task(settings=make_settings(), command=GwasExport(mt="in.mt", **overrides))
    )


def test_export_group_name_is_stable() -> None:
    assert GwasExportPipeline.name() == "gwas_export"


def test_export_requires_exactly_one_input() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        GwasExport()
    with pytest.raises(ValidationError):
        GwasExport(mt="in.mt", manifest="/m.json.gz")
    with pytest.raises(ValidationError):
        GwasExport(mt="in.mt", include=("high_quality",))


def test_export_declares_the_base_and_per_region_steps() -> None:
    _export_pipeline()
    registry = StepRegistry()

    assert registry.get("gwas_export.gwas_base") is not None
    assert registry.get("gwas_export.normalize_regenie") is not None
    assert registry.get("gwas_export.filter_regenie_chromosomes") is not None
    assert registry.get("gwas_export.phenotype") is not None
    assert registry.get("gwas_export.covariates") is not None
    assert registry.get("gwas_export.aaf") is not None
    assert registry.get("gwas_export.regenie_input.gene") is not None
    assert registry.get("gwas_export.export_regenie.gene") is not None


def test_export_fans_out_over_configured_windows() -> None:
    settings = make_settings(
        regions=[
            GwasRegion(kind="gene"),
            GwasRegion(kind="window", size=1_000),
            GwasRegion(kind="window", size=10_000),
        ]
    )
    GwasExportPipeline(Task(settings=settings, command=GwasExport(mt="in.mt")))
    registry = StepRegistry()

    for label in ("gene", "window_1000", "window_10000"):
        assert registry.get(f"gwas_export.regenie_input.{label}") is not None
        assert registry.get(f"gwas_export.export_regenie.{label}") is not None


def test_export_step_set_is_identical_across_constructions() -> None:
    regions = [
        GwasRegion(kind="gene"),
        GwasRegion(kind="window", size=10_000),
    ]
    first = GwasExportPipeline(
        Task(
            settings=make_settings(regions),
            command=GwasExport(mt="in.mt"),
        )
    )
    second = GwasExportPipeline(
        Task(
            settings=make_settings(regions),
            command=GwasExport(mt="in.mt"),
        )
    )

    assert first.current_state()["steps"] == second.current_state()["steps"]


def test_export_region_inputs_are_checkpointed_tables() -> None:
    _export_pipeline()
    registry = StepRegistry()

    artifact = registry.get("gwas_export.regenie_input.gene").artifacts
    assert artifact == (
        Artifact(
            prefix="{project}/intermediate/gwas/gene/annotation",
            kind=CHECKPOINT_TYPE.HAIL_TABLE,
        ),
        Artifact(
            prefix="{project}/intermediate/gwas/gene/variant_sets",
            kind=CHECKPOINT_TYPE.HAIL_TABLE,
        ),
        Artifact(
            prefix="{project}/intermediate/gwas/gene/effect_masks",
            kind=CHECKPOINT_TYPE.HAIL_TABLE,
        ),
    )


def test_export_terminal_declares_the_self_contained_dataset() -> None:
    settings = make_settings()
    _export_pipeline(base="/out")
    registry = StepRegistry()
    artifacts = registry.get("gwas_export.export_regenie.gene").artifacts

    paths = {a.path(settings) for a in artifacts}
    assert {f"/out/gene/study.{ext}" for ext in ("bed", "bim", "fam")} <= paths
    for table in (
        "phenotype",
        "covariates",
        "aaf",
        "annotation",
        "variant_sets",
        "effect_masks",
    ):
        assert f"/out/gene/study.{table}.tsv.bgz" in paths
    assert "/out/gene/study.gwas.md.json" in paths
    # The GWAS metadata file is an opaque single file, not a VariantDataset
    # handle: the artifact kind must never be inferred as one, or the cache
    # check would try to parse it as VDS params (and crash).
    metadata_artifact = next(
        a for a in artifacts if a.path(settings) == "/out/gene/study.gwas.md.json"
    )
    assert metadata_artifact.kind is CHECKPOINT_TYPE.FILE
    # The manifest is derived from the path, not persisted as an artifact.
    assert not any(a.prefix.endswith("manifest") for a in artifacts)


def test_export_terminal_wires_the_six_tables_and_the_region() -> None:
    _export_pipeline()
    registry = StepRegistry()
    step = registry.get("gwas_export.export_regenie.gene")

    assert step.depends_on["mt"] == "gwas_export.filter_regenie_chromosomes"
    assert step.depends_on["phenotype"] == "gwas_export.phenotype"
    assert step.depends_on["covariates"] == "gwas_export.covariates"
    assert step.depends_on["aaf"] == "gwas_export.aaf"
    assert step.depends_on["annotation"] == "gwas_export.regenie_input.gene"
    assert step.depends_on["variant_sets"] == "gwas_export.regenie_input.gene"
    assert step.depends_on["effect_masks"] == "gwas_export.regenie_input.gene"
    assert step.outputs["annotation"] == 0
    assert step.outputs["variant_sets"] == 1
    assert step.outputs["effect_masks"] == 2


def test_export_generated_steps_carry_their_region_identity() -> None:
    _export_pipeline()
    registry = StepRegistry()

    assert registry.get("gwas_export.regenie_input.gene").task.command.args == {
        "task": "gene"
    }
    assert registry.get("gwas_export.export_regenie.gene").task.command.args == {
        "task": "gene"
    }


def test_export_mt_input_resolves_to_the_configured_path() -> None:
    _export_pipeline()
    registry = StepRegistry()

    assert registry["gwas_export.mt"].value == "in.mt"


def test_export_is_factory_discoverable() -> None:
    assert "gwas_export" in PipelineFactory.list()


def _write_manifest(tmp_path: Path) -> Path:
    """A manifest with the basis plus one filter and one region."""

    import gzip

    from gnomad_meb.lib.filter.manifest import (
        FilterEntry,
        FilterManifest,
        FilterRegion,
        HighQuality,
    )

    manifest = FilterManifest(
        project="study",
        reference_genome="GRCh38",
        high_quality=HighQuality(
            mt=str(tmp_path / "high_quality.mt"),
            variant_info=str(tmp_path / "high_quality.variant_info.json.gz"),
            sample_info=str(tmp_path / "high_quality.sample_info.json.gz"),
            variant_info_summary={"n_variants": {"final": 100}},
        ),
        filters=[
            FilterEntry(
                name="rare_lof",
                mt=str(tmp_path / "rare_lof.mt"),
                variant_info=str(tmp_path / "rare_lof.variant_info.json.gz"),
                variant_info_summary={
                    "filters": ["lof"],
                    "lof": {"n_variants": {"final": 7}},
                },
                regions=[
                    FilterRegion(
                        name="integrated", mt=str(tmp_path / "rintegrated.mt")
                    )
                ],
            )
        ],
    )
    path = tmp_path / "manifest.json.gz"
    with gzip.open(path, "wt") as out_f:
        json.dump(manifest.model_dump(mode="json"), out_f)
    return path


def _manifest_export(tmp_path: Path, **overrides) -> GwasExportPipeline:
    settings = make_settings(regions=[GwasRegion(kind="gene")])
    return GwasExportPipeline(
        Task(
            settings=settings,
            command=GwasExport(
                manifest=str(_write_manifest(tmp_path)), base="/out", **overrides
            ),
        )
    )


def test_export_fans_out_over_manifest_tables_and_regions(tmp_path: Path) -> None:
    _manifest_export(tmp_path)
    registry = StepRegistry()

    for table in (
        "high_quality",
        "filter_rare_lof",
        "filter_rare_lof_region_integrated",
    ):
        assert registry.get(f"gwas_export.gwas_base.{table}") is not None
        assert registry.get(f"gwas_export.export_regenie.{table}.gene") is not None
        assert registry.get(f"gwas_export.mt.{table}") is not None


def test_export_per_table_base_chains_are_distinct(tmp_path: Path) -> None:
    _manifest_export(tmp_path)
    registry = StepRegistry()

    assert registry.get("gwas_export.phenotype.high_quality").depends_on["mt"] == (
        "gwas_export.filter_regenie_chromosomes.high_quality"
    )
    assert registry.get(
        "gwas_export.filter_regenie_chromosomes.high_quality"
    ).depends_on["mt"] == "gwas_export.normalize_regenie.high_quality"


def test_export_manifest_datasets_land_in_distinct_base_dirs(tmp_path: Path) -> None:
    settings = make_settings(regions=[GwasRegion(kind="gene")])
    _manifest_export(tmp_path)
    registry = StepRegistry()

    hq = registry.get("gwas_export.export_regenie.high_quality.gene")
    flt = registry.get("gwas_export.export_regenie.filter_rare_lof.gene")
    paths = {a.path(settings) for a in hq.artifacts}
    assert "/out/high_quality/gene/study.gwas.md.json" in paths
    assert "/out/filter_rare_lof/gene/study.bed" in {
        a.path(settings) for a in flt.artifacts
    }


def test_export_input_endpoints_carry_each_table(tmp_path: Path) -> None:
    _manifest_export(tmp_path)
    registry = StepRegistry()

    assert registry["gwas_export.mt.filter_rare_lof"].value.endswith("rare_lof.mt")


def test_export_include_and_exclude_narrow_the_tables(tmp_path: Path) -> None:
    StepRegistry._reset()
    _manifest_export(tmp_path, include=("high_quality",))
    registry = StepRegistry()

    assert registry.get("gwas_export.gwas_base.high_quality") is not None
    # The excluded table's base steps were not registered by this construction.
    assert registry.get("gwas_export.gwas_base.filter_rare_lof") is None



# --- run stage --------------------------------------------------------------


def _write_dataset(tmp_path: Path, basename: str = "study") -> Path:
    import pandas

    dataset = tmp_path / "per_gene"
    dataset.mkdir(parents=True, exist_ok=True)
    frame = pandas.DataFrame(
        {
            "FID": ["F1", "F2", "F3"],
            "IID": ["1", "2", "3"],
            "case": [1.0, 1.0, 0.0],
            "rare_response": [1.0, None, None],
        }
    )
    frame.to_csv(
        dataset / f"{basename}.phenotype.tsv.bgz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    return dataset


def _study_csv(tmp_path: Path) -> Path:
    """A metadata CSV whose case cohort gives two cases (min_case_count 2)."""
    import pandas

    path = tmp_path / "study.md.csv"
    pandas.DataFrame(
        {
            "sample_id": ["1", "2", "3"],
            "cohort": ["case", "case", "control"],
        }
    ).to_csv(path, index=False)
    return path


def _gwas_metadata(tmp_path: Path, **gwas_overrides) -> GwasMetadata:
    """A metadata object over ``_study_csv`` (min_case_count 2)."""
    return GwasMetadata.from_settings(
        Settings(
            validate_paths=False,
            project="study",
            checkpoint="/ckpt",
            gwas=GwasConfig(
                metadata_csv=_study_csv(tmp_path),
                case_cohort="case",
                control_cohorts=["control"],
                min_case_count=2,
                **gwas_overrides,
            ),
        )
    )


def _run_pipeline(tmp_path: Path, **overrides) -> GwasRunPipeline:
    dataset = overrides.pop("dataset", _write_dataset(tmp_path))
    return GwasRunPipeline(
        Task(
            settings=Settings(
                validate_paths=False,
                project="study",
                checkpoint="/ckpt",
                gwas=GwasConfig(
                    metadata_csv=_study_csv(tmp_path),
                    case_cohort="case",
                    control_cohorts=["control"],
                    min_case_count=2,
                ),
            ),
            command=GwasPipeline(
                dataset=str(dataset),
                output_dir="/out",
                **overrides,
            ),
        )
    )


def test_run_group_name_is_stable() -> None:
    assert GwasRunPipeline.name() == "gwas"


def test_run_derives_the_response_set_from_settings(tmp_path: Path) -> None:
    pipeline = _run_pipeline(tmp_path)

    # ``case`` (the cohort-derived response) sums to 2 (>= min 2).
    assert pipeline._responses == ["case"]


def test_run_gates_quantitative_on_its_sample_count(tmp_path: Path) -> None:
    import pandas

    csv = tmp_path / "quant.md.csv"
    pandas.DataFrame(
        {
            "sample_id": ["1", "2", "3"],
            "cohort": ["case", "case", "control"],
            "dosage": [1e-9, None, 0.0],
        }
    ).to_csv(csv, index=False)

    def _metadata(kind: str) -> GwasMetadata:
        return GwasMetadata.from_settings(
            Settings(
                validate_paths=False,
                project="study",
                checkpoint="/ckpt",
                gwas=GwasConfig(
                    metadata_csv=csv,
                    case_cohort="case",
                    control_cohorts=["control"],
                    min_case_count=2,
                    responses=GwasResponses(
                        columns=[GwasResponseColumn(name="dosage", kind=kind)]
                    ),
                ),
            )
        )

    # ``case`` sums to 2; a quantitative ``dosage`` sums to ~1e-9 but has 2
    # non-missing samples, so it passes. As a binary column it falls (sum < 2).
    assert _metadata("quantitative").responses == ("case", "dosage")
    assert _metadata("binary").responses == ("case",)


def test_run_declares_per_phenotype_outputs_and_provenance(tmp_path: Path) -> None:
    settings = Settings(
        validate_paths=False,
        project="study",
        checkpoint="/ckpt",
        gwas=GwasConfig(
            metadata_csv=_study_csv(tmp_path),
            case_cohort="case",
            control_cohorts=["control"],
            min_case_count=2,
        ),
    )
    GwasRunPipeline(
        Task(
            settings=settings,
            command=GwasPipeline(
                dataset=str(_write_dataset(tmp_path)), output_dir="/out"
            ),
        )
    )
    artifacts = {
        a.path(settings) for a in StepRegistry().get("gwas.run_gwas_pipeline").artifacts
    }

    assert "/out/gwas/case/study_Y1_case.regenie.gz" in artifacts
    assert "/out/burden/case/study_Y1_case.regenie.gz" in artifacts
    assert "/out/regenie.version.json" in artifacts


def test_run_is_factory_discoverable() -> None:
    assert "gwas" in PipelineFactory.list()


def test_resolve_n_chunks_targets_variants_per_split() -> None:
    from gnomad_meb.lib.gwas.run import resolve_n_chunks

    settings = Settings(
        validate_paths=False,
        project="study",
        checkpoint="/ckpt",
        gwas=GwasConfig(
            metadata_csv=Path("/data/study.md.csv"), variants_per_split=100
        ),
    )

    # Regenie needs at least two chunks, then one more per ``variants_per_split``.
    assert resolve_n_chunks(100, settings) == 2
    assert resolve_n_chunks(201, settings) == 3
    assert resolve_n_chunks(50, settings) == 2
    assert resolve_n_chunks(0, settings) == 2


def test_resolved_n_chunks_reaches_the_snakemake_config(tmp_path: Path) -> None:
    import yaml

    from gnomad_meb.config import set_config
    from gnomad_meb.regenie.regenie import _make_config

    settings = Settings(
        validate_paths=False,
        project="study",
        checkpoint="/ckpt",
        gwas=GwasConfig(metadata_csv=Path("/data/study.md.csv")),
    )
    set_config(settings)
    target = tmp_path / "config.yaml"
    _make_config(
        "bed",
        "cov",
        "pheno",
        "annot",
        "vsets",
        "aaf",
        "masks",
        n_chunks=7,
        target=target,
    )

    assert yaml.safe_load(target.read_text())["n_chunks"] == 7


def test_run_fans_out_one_step_per_dataset(tmp_path: Path) -> None:
    from gnomad_meb.cmd.task import GwasDatasetRef

    dataset = _write_dataset(tmp_path)
    settings = Settings(
        validate_paths=False,
        project="study",
        checkpoint="/ckpt",
        gwas=GwasConfig(
            metadata_csv=_study_csv(tmp_path),
            case_cohort="case",
            control_cohorts=["control"],
            min_case_count=2,
        ),
    )
    GwasRunPipeline(
        Task(
            settings=settings,
            command=GwasPipeline(
                output_dir="/run",
                datasets=[
                    GwasDatasetRef(
                        label="high_quality.gene",
                        dataset=str(dataset),
                        gwas="/run/high_quality/gene/gwas",
                        burden="/run/high_quality/gene/burden",
                    ),
                    GwasDatasetRef(
                        label="filter_rare_lof.gene",
                        dataset=str(dataset),
                        gwas="/run/filter_rare_lof/gene/gwas",
                        burden="/run/filter_rare_lof/gene/burden",
                    ),
                ],
            ),
        )
    )
    registry = StepRegistry()

    assert registry.get("gwas.run_gwas_pipeline.high_quality.gene") is not None
    assert registry.get("gwas.run_gwas_pipeline.filter_rare_lof.gene") is not None
    artifacts = {
        a.path(settings)
        for a in registry.get(
            "gwas.run_gwas_pipeline.high_quality.gene"
        ).artifacts
    }
    assert "/run/high_quality/gene/gwas/case/study_Y1_case.regenie.gz" in artifacts
    assert "/run/high_quality/gene/regenie.version.json" in artifacts


def test_vqsr_is_factory_discoverable() -> None:
    assert "vqsr" in PipelineFactory.list()


# --- load stage -------------------------------------------------------------


_GWAS_ROW = (
    "#comment\n"
    "CHROM GENPOS ID ALLELE0 ALLELE1 A1FREQ INFO N TEST BETA SE CHISQ LOG10P EXTRA\n"
    "1 1234 1:1234:A:G A G 0.1 0.5 100 ADD 0.5 0.1 25 3 -\n"
)
_BURDEN_ROW = (
    "#comment\n"
    "CHROM GENPOS ID ALLELE0 ALLELE1 A1FREQ N TEST BETA SE CHISQ LOG10P EXTRA\n"
    "1 1234 GENE.Mask-lof.0.01 A G 0.1 100 ADD 0.5 0.1 25 3 -\n"
)


def test_response_from_filename_recovers_underscored_responses() -> None:
    assert response_from_filename("study_Y1_case.regenie.gz") == "case"
    assert response_from_filename("study_Y2_other_ai.regenie.gz") == "other_ai"
    assert response_from_filename("aapad_Y10_alopecia_areata.regenie.gz") == (
        "alopecia_areata"
    )


def _write_regenie(path: Path, content: str) -> Path:
    import gzip

    with gzip.open(path, "wt") as handle:
        handle.write(content)
    return path


def test_parse_gwas_frame_splits_variant_id(tmp_path: Path) -> None:
    path = _write_regenie(tmp_path / "study_Y1_case.regenie.gz", _GWAS_ROW)

    frame = parse_response_frame(path, "case", id_cols=["variant_id"])

    assert frame["variant_id"].tolist() == ["1:1234:A:G"]
    assert frame["phenotype"].tolist() == ["case"]
    assert frame["INFO"].dtype == "float64"


def test_parse_gwas_frame_pins_info_dtype(tmp_path: Path) -> None:
    """An all-integral ``INFO`` must not read as ``int`` (breaks the union).

    pandas infers ``int64`` for a response whose ``INFO`` values are all
    integral (``1``) and ``float64`` for one with fractional values; the pinned
    read makes both responses' row types agree so ``load_gwas`` can union them.
    """
    integral = (
        "#comment\n"
        "CHROM GENPOS ID ALLELE0 ALLELE1 A1FREQ INFO N TEST BETA SE CHISQ "
        "LOG10P EXTRA\n"
        "1 1234 1:1234:A:G A G 0.1 1 100 ADD 0.5 0.1 25 3 -\n"
    )
    fractional = (
        "#comment\n"
        "CHROM GENPOS ID ALLELE0 ALLELE1 A1FREQ INFO N TEST BETA SE CHISQ "
        "LOG10P EXTRA\n"
        "1 1234 1:1234:A:G A G 0.1 0.98 100 ADD 0.5 0.1 25 3 -\n"
    )
    int_path = _write_regenie(tmp_path / "a_Y1_case.regenie.gz", integral)
    float_path = _write_regenie(tmp_path / "b_Y2_case.regenie.gz", fractional)

    int_frame = parse_response_frame(int_path, "case", id_cols=["variant_id"])
    float_frame = parse_response_frame(float_path, "case", id_cols=["variant_id"])

    assert int_frame["INFO"].dtype == "float64"
    assert float_frame["INFO"].dtype == "float64"


def test_parse_burden_frame_splits_region_mask_aaf(tmp_path: Path) -> None:
    path = _write_regenie(tmp_path / "study_Y1_case.regenie.gz", _BURDEN_ROW)

    frame = parse_response_frame(path, "case", id_cols=["region", "mask", "aaf"])

    assert frame["region"].tolist() == ["GENE"]
    assert frame["mask"].tolist() == ["Mask-lof"]
    assert frame["aaf"].tolist() == ["0.01"]


class _FakeTable:
    """A stand-in exposing just what :func:`regenie_rename` reads and calls."""

    def __init__(self, fields: list[str]) -> None:
        self._fields = dict.fromkeys(fields)
        self.renamed: dict[str, str] | None = None

    @property
    def row(self):
        return _FakeRow(self._fields)

    def rename(self, mapping: dict[str, str]):
        self.renamed = mapping
        return self


class _FakeRow:
    def __init__(self, fields: dict[str, str]) -> None:
        self.dtype = _FakeRowDtype(fields)


class _FakeRowDtype:
    def __init__(self, fields: dict[str, str]) -> None:
        self.fields = fields


def test_regenie_rename_drops_absent_columns() -> None:
    """Hail ``rename`` rejects keys not in the table, so they must be filtered.

    A burden table has no ``INFO`` and pandas can drop an all-NA column, so the
    map handed to hail may name a column the table lacks.
    """
    table = _FakeTable(["ID", "A1FREQ", "N", "CHISQ"])

    regenie_rename(table)

    assert table.renamed == {
        "A1FREQ": "a1_freq",
        "N": "n",
        "CHISQ": "chi_sq",
    }


def test_regenie_rename_includes_info_when_present() -> None:
    table = _FakeTable(["A1FREQ", "INFO", "N"])

    regenie_rename(table)

    assert table.renamed == {"A1FREQ": "a1_freq", "INFO": "info", "N": "n"}


def test_load_group_name_is_stable() -> None:
    assert GwasLoadPipeline.name() == "gwas_load"


def _load_pipeline(tmp_path: Path, **overrides) -> GwasLoadPipeline:
    dataset = overrides.pop("dataset", _write_dataset(tmp_path))
    return GwasLoadPipeline(
        Task(
            settings=Settings(
                validate_paths=False,
                project="study",
                checkpoint="/ckpt",
                gwas=GwasConfig(
                    metadata_csv=_study_csv(tmp_path),
                    case_cohort="case",
                    control_cohorts=["control"],
                    min_case_count=2,
                ),
            ),
            command=GwasLoad(
                dataset=str(dataset),
                gwas=str(tmp_path / "gwas"),
                burden=str(tmp_path / "burden"),
                **overrides,
            ),
        )
    )


def test_load_scatters_one_step_per_response(tmp_path: Path) -> None:
    _load_pipeline(tmp_path)
    registry = StepRegistry()

    assert registry.get("gwas_load.load_gwas.case") is not None
    assert registry.get("gwas_load.load_burden.case") is not None
    assert registry.get("gwas_load.merge_gwas") is not None
    assert registry.get("gwas_load.merge_burden") is not None


def test_load_merges_union_the_per_response_parts(tmp_path: Path) -> None:
    _load_pipeline(tmp_path)
    registry = StepRegistry()

    assert registry.get("gwas_load.merge_gwas").depends_on == {
        "case": "gwas_load.load_gwas.case"
    }
    assert registry.get("gwas_load.merge_burden").depends_on == {
        "case": "gwas_load.load_burden.case"
    }


def test_load_terminal_tables_use_the_regenie_prefixes(tmp_path: Path) -> None:
    _load_pipeline(tmp_path)
    registry = StepRegistry()

    assert registry.get("gwas_load.merge_gwas").artifacts == (
        Artifact(
            prefix="{project}/intermediate/regenie/gwas",
            kind=CHECKPOINT_TYPE.HAIL_TABLE,
        ),
    )
    assert registry.get("gwas_load.merge_burden").artifacts == (
        Artifact(
            prefix="{project}/intermediate/regenie/burden",
            kind=CHECKPOINT_TYPE.HAIL_TABLE,
        ),
    )


def test_load_is_factory_discoverable() -> None:
    assert "gwas_load" in PipelineFactory.list()


def test_gwas_manifest_path_matches_the_declared_artifact(tmp_path: Path) -> None:
    from gnomad_meb.lib.gwas.manifest import gwas_manifest_path

    settings = Settings(
        validate_paths=False,
        project="study",
        checkpoint="/ckpt",
        gwas=GwasConfig(
            metadata_csv=_study_csv(tmp_path),
            case_cohort="case",
            control_cohorts=["control"],
            min_case_count=2,
        ),
    )
    _load_pipeline(tmp_path)
    declared = StepRegistry().get("gwas_load.get_manifest").artifacts[0].path(settings)

    assert declared == gwas_manifest_path(settings)
    assert declared == "/ckpt/study/intermediate/regenie/manifest.json.gz"


def test_load_fans_out_per_dataset_with_terminal_prefixes(tmp_path: Path) -> None:
    from gnomad_meb.cmd.task import GwasDatasetRef

    StepRegistry._reset()
    dataset = _write_dataset(tmp_path)
    settings = Settings(
        validate_paths=False,
        project="study",
        checkpoint="/ckpt",
        gwas=GwasConfig(
            metadata_csv=_study_csv(tmp_path),
            case_cohort="case",
            control_cohorts=["control"],
            min_case_count=2,
        ),
    )
    GwasLoadPipeline(
        Task(
            settings=settings,
            command=GwasLoad(
                datasets=[
                    GwasDatasetRef(
                        label="high_quality.gene",
                        dataset=str(dataset),
                        gwas="/run/high_quality/gene/gwas",
                        burden="/run/high_quality/gene/burden",
                    ),
                    GwasDatasetRef(
                        label="filter_rare_lof.gene",
                        dataset=str(dataset),
                        gwas="/run/filter_rare_lof/gene/gwas",
                        burden="/run/filter_rare_lof/gene/burden",
                    ),
                ]
            ),
        )
    )
    registry = StepRegistry()

    assert registry.get("gwas_load.load_gwas.high_quality.gene.case") is not None
    assert registry.get("gwas_load.load_gwas.filter_rare_lof.gene.case") is not None
    assert registry.get("gwas_load.get_manifest").artifacts == (
        Artifact(
            prefix="{project}/intermediate/regenie/manifest",
            kind=CHECKPOINT_TYPE.DICTIONARY,
        ),
    )
    gwas_artifacts = registry.get(
        "gwas_load.merge_gwas.high_quality.gene"
    ).artifacts
    assert gwas_artifacts == (
        Artifact(
            prefix="{project}/intermediate/regenie/high_quality/gene/gwas",
            kind=CHECKPOINT_TYPE.HAIL_TABLE,
        ),
    )


def test_get_manifest_records_each_dataset(tmp_path: Path) -> None:
    import gzip

    from gnomad_meb.lib.gwas.manifest import read_gwas_manifest

    dataset = tmp_path / "ds"
    dataset.mkdir()
    (dataset / "study.gwas.md.json").write_text(
        json.dumps(
            {
                "source": "filter",
                "name": "rare_lof",
                "response_name_map": {"case": "Case"},
                "n_variants": 7,
                "filter_variant_info": {"filters": ["lof"]},
            }
        )
    )
    gwas_dir = tmp_path / "run" / "filter_rare_lof" / "gene" / "gwas"
    gwas_dir.mkdir(parents=True)
    (gwas_dir.parent / "regenie.version.json").write_text(
        json.dumps({"version": "3.3", "build": "py38", "url": "u"})
    )

    settings = Settings(
        validate_paths=False,
        project="study",
        checkpoint=str(tmp_path / "ckpt"),
        gwas=GwasConfig(
            metadata_csv=_study_csv(tmp_path),
            case_cohort="case",
            control_cohorts=["control"],
            min_case_count=2,
        ),
    )
    pipeline = GwasLoadPipeline(
        Task(
            settings=settings,
            command=GwasLoad(
                dataset=str(dataset),
                gwas=str(gwas_dir),
                burden=str(tmp_path / "run" / "filter_rare_lof" / "gene" / "burden"),
            ),
        )
    )

    payload = pipeline.get_manifest()
    path = tmp_path / "manifest.json.gz"
    with gzip.open(path, "wt") as out_f:
        json.dump(payload, out_f)
    restored = read_gwas_manifest(path)

    (entry,) = restored.datasets
    assert entry.dataset == str(dataset)
    assert entry.responses == ["case"]
    assert entry.regenie_version == {"version": "3.3", "build": "py38", "url": "u"}
    assert entry.gwas.endswith("/intermediate/regenie/gwas.ht")
    assert entry.filters == {"filter_variant_info": {"filters": ["lof"]}}



# --- acceptance: reconstruction parity and discovery ------------------------


def test_all_gwas_pipelines_are_factory_discoverable() -> None:
    names = PipelineFactory.list()

    assert {"gwas_export", "gwas", "gwas_load"} <= set(names)


def test_export_step_set_round_trips_through_json() -> None:
    regions = [GwasRegion(kind="gene"), GwasRegion(kind="window", size=1_000)]
    task = Task(
        settings=make_settings(regions),
        command=GwasExport(mt="in.mt"),
    )
    runner = PipelineRunner(task, [GwasExportPipeline(task)])
    dumped = json.loads(json.dumps(runner.current_state()))

    restored = PipelineRunner.load(dumped)

    # The same inputs rebuilt the identical step set -- no PipelineMismatchError.
    assert set(restored.current_state()["steps"]) == set(dumped["steps"])
