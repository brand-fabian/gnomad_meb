"""In-process tests for the settings layer.

These are pure, hail-free and reference-data-free: they assert the external
behaviour of the resolution seam (which value a field resolves to, and what is
raised for invalid input) using small, self-contained TOML files and the process
environment. See ``docs/adr/0005`` and ``docs/adr/0006``.
"""

from __future__ import annotations

import os
from pathlib import Path
from textwrap import dedent

import pytest
from pydantic import ValidationError

import gnomad_meb.config as config_module
from gnomad_meb.config import (
    INVALID_TOKEN,
    Resources,
    Settings,
    get_config,
    resolve,
)


@pytest.fixture(autouse=True)
def _clear_ambient_gnomeb(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop any ambient ``GMEB_`` variables so tests see a clean environment."""
    for key in [k for k in os.environ if k.startswith("GMEB_")]:
        monkeypatch.delenv(key, raising=False)
    config_module._config = None
    config_module._config_path = None


def _write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "gnomad_meb.toml"
    path.write_text(dedent(body).lstrip())
    return path


# --- resolution and precedence ----------------------------------------------


def test_defaults_are_used_without_a_config_file() -> None:
    settings = resolve()

    assert settings.project.startswith("gnomad_meb-")
    assert settings.s3.access_key == INVALID_TOKEN
    assert settings.s3.secret_key == INVALID_TOKEN
    assert settings.hail.partitions_per_core == 10


def test_precedence_is_override_then_env_then_file_then_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write_config(
        tmp_path,
        """
        [hail]
        partitions_per_core = 5
        """,
    )
    monkeypatch.setenv("GMEB_HAIL__PARTITIONS_PER_CORE", "7")

    assert resolve(config).hail.partitions_per_core == 7
    assert (
        resolve(
            config, overrides={"hail": {"partitions_per_core": 9}}
        ).hail.partitions_per_core
        == 9
    )

    monkeypatch.delenv("GMEB_HAIL__PARTITIONS_PER_CORE")
    assert resolve(config).hail.partitions_per_core == 5

    assert resolve().hail.partitions_per_core == 10


def test_gmeb_prefix_is_honoured_across_sections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GMEB_S3__ACCESS_KEY", "env-access")
    monkeypatch.setenv("GMEB_VQSR__OUTPUT_DIR", "/env/vqsr")
    monkeypatch.setenv("GMEB_HAIL__PARTITIONS_PER_CORE", "12")
    monkeypatch.setenv("GMEB_SAMPLE_QC__GQ_BINS", "[1, 2, 3]")

    settings = resolve()

    assert settings.s3.access_key == "env-access"
    assert settings.vqsr is not None and settings.vqsr.output_dir == Path("/env/vqsr")
    assert settings.hail.partitions_per_core == 12
    assert settings.sample_qc is not None
    assert settings.sample_qc.gq_bins == [1, 2, 3]


def test_unprefixed_variables_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROJECT", "not-prefixed")
    monkeypatch.setenv("HAIL__PARTITIONS_PER_CORE", "99")

    settings = resolve()

    assert settings.project.startswith("gnomad_meb-")
    assert settings.hail.partitions_per_core == 10


# --- coercion ---------------------------------------------------------------


def test_string_ints_and_bools_are_coerced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write_config(
        tmp_path,
        """
        [hail]
        partitions_per_core = "20"

        [s3]
        verify = "false"
        """,
    )

    settings = resolve(config)

    assert settings.hail.partitions_per_core == 20
    assert settings.s3.verify is False


def test_list_and_tuple_fields_are_coerced(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path,
        """
        [sample_qc]
        gq_bins = ["0", "30"]
        sex_contigs = ["chrX"]
        """,
    )

    settings = resolve(config)

    assert settings.sample_qc is not None
    assert settings.sample_qc.gq_bins == [0, 30]
    assert settings.sample_qc.sex_contigs == ("chrX",)


def test_snakemake_passthrough_resolves_from_file_then_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cluster passthrough is run config: ``[vqsr].snakemake`` / env override."""
    config = _write_config(
        tmp_path,
        """
        [vqsr]
        snakemake = ["--profile", "slurm", "--executor", "slurm"]

        [gwas]
        metadata_csv = "/res/gwas.md.csv"
        snakemake = ["--profile", "slurm"]
        """,
    )

    settings = resolve(config, overrides={"validate_paths": False})

    assert settings.vqsr is not None
    assert settings.vqsr.snakemake == ["--profile", "slurm", "--executor", "slurm"]
    assert settings.gwas is not None
    assert settings.gwas.snakemake == ["--profile", "slurm"]

    monkeypatch.setenv("GMEB_VQSR__SNAKEMAKE", '["--executor", "kubernetes"]')
    settings = resolve(config, overrides={"validate_paths": False})

    assert settings.vqsr is not None
    assert settings.vqsr.snakemake == ["--executor", "kubernetes"]


# --- optional sections ------------------------------------------------------


def test_absent_optional_sections_resolve_to_none(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path,
        """
        project = "partial"
        """,
    )

    settings = resolve(config)

    assert settings.project == "partial"
    assert settings.vqsr is None
    assert settings.gwas is None
    assert settings.annotation is None
    assert settings.variant_qc is None
    assert settings.resources is None
    assert settings.filter is None


def test_nested_sections_are_loaded_from_the_file(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path,
        """
        [resources]
        calling_intervals = "/r/calling.interval_list"
        telomeres_and_centromeres = "/r/tel.ht"
        lcr_intervals = "/r/lcr.ht"
        seg_dup_intervals = "/r/seg.ht"
        gnomad_pca_loadings = "/r/loadings.ht"
        gnomad_pca_model = "/r/model.onnx"
        gnomad_v4 = "/r/gnomad.ht"
        reference = "/r/reference.fasta"

        [filter.resources]
        constraints = "/r/constraints.ht"

        [[filter.filters]]
        name = "rare_lof"
        max_gnomad_af = 0.01
        impact = ["HIGH", "MODERATE"]

        [[filter.regions]]
        name = "integrated"
        intervals = "/r/integrated.bed"
        """,
    )

    settings = resolve(config, overrides={"validate_paths": False})

    assert settings.resources is not None
    assert settings.resources.reference == Path("/r/reference.fasta")
    assert settings.filter is not None
    assert [flt.name for flt in settings.filter.filters] == ["rare_lof"]
    assert settings.filter.regions[0].name == "integrated"
    assert settings.filter.resources is not None
    assert settings.filter.resources.constraints == "/r/constraints.ht"


# --- filter and region declarations -----------------------------------------


def test_filter_and_region_declarations_resolve_from_toml(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path,
        """
        [[filter.filters]]
        name = "rare_lof"
        max_gnomad_af = 0.01
        min_pLI = 0.9
        impact = ["HIGH", "MODERATE"]

        [[filter.filters]]
        name = "all_clinvar_pathogenic"
        filter_clinvar_pathogenic = true

        [[filter.regions]]
        name = "integrated"
        intervals = "/data/regions/integrated.bed"

        [[filter.regions]]
        name = "grch37"
        intervals = "/data/regions/grch37.bed"
        reference_genome = "GRCh37"
        """,
    )

    settings = resolve(config, overrides={"validate_paths": False})

    assert settings.filter is not None
    rare_lof = settings.filter.filters[0]
    assert rare_lof.name == "rare_lof"
    assert rare_lof.max_gnomad_af == 0.01
    assert rare_lof.min_pLI == 0.9
    assert rare_lof.impact == {"HIGH", "MODERATE"}
    assert rare_lof.ancestry_group is None

    clinvar = settings.filter.filters[1]
    assert clinvar.filter_clinvar_pathogenic is True

    integrated, grch37 = settings.filter.regions
    assert integrated.name == "integrated"
    assert integrated.intervals == "/data/regions/integrated.bed"
    assert integrated.reference_genome == "GRCh38"
    assert grch37.reference_genome == "GRCh37"

    assert settings.filter.load_filters() == settings.filter.filters
    assert settings.filter.load_regions() == settings.filter.regions


def test_missing_filter_name_fails_resolution(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path,
        """
        [[filter.filters]]
        max_gnomad_af = 0.01
        """,
    )

    with pytest.raises(ValidationError) as excinfo:
        resolve(config, overrides={"validate_paths": False})

    message = str(excinfo.value)
    assert "filter" in message
    assert "name" in message


def test_missing_region_name_fails_resolution(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path,
        """
        [[filter.regions]]
        intervals = "/data/regions/integrated.bed"
        """,
    )

    with pytest.raises(ValidationError) as excinfo:
        resolve(config, overrides={"validate_paths": False})

    message = str(excinfo.value)
    assert "region" in message
    assert "name" in message


def test_unknown_filter_key_fails_resolution(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path,
        """
        [[filter.filters]]
        name = "rare_lof"
        max_gnomad_af = 0.01
        max_af_gnomad = 0.01
        """,
    )

    with pytest.raises(ValidationError) as excinfo:
        resolve(config, overrides={"validate_paths": False})

    message = str(excinfo.value)
    assert "filter" in message
    assert "filters" in message
    assert "0" in message
    assert "max_af_gnomad" in message


def test_unknown_region_key_fails_resolution(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path,
        """
        [[filter.regions]]
        name = "integrated"
        intervals = "/data/regions/integrated.bed"
        reference_genom = "GRCh38"
        """,
    )

    with pytest.raises(ValidationError) as excinfo:
        resolve(config, overrides={"validate_paths": False})

    message = str(excinfo.value)
    assert "filter" in message
    assert "regions" in message
    assert "0" in message
    assert "reference_genom" in message


_CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"

#: The ten filter declarations migrated out of ``config/filters.yaml`` into the
#: global TOML. Expected values are transcribed from the migration ticket
#: (``.scratch/filter-regions/issues/05-migrate-filters-yaml.md``), not derived
#: from the config file, so the test disagrees if a value drifts.
_MIGRATED_FILTERS: list[tuple[str, dict[str, object]]] = [
    (
        "rare_clinvar_pathogenic",
        {"max_gnomad_af": 0.01, "filter_clinvar_pathogenic": True},
    ),
    ("all_clinvar_pathogenic", {"filter_clinvar_pathogenic": True}),
    (
        "rare_deleterious_missense",
        {
            "max_gnomad_af": 0.01,
            "min_cadd": 20.0,
            "impact": {"HIGH", "MODERATE"},
        },
    ),
    ("rare_high_impact", {"max_gnomad_af": 0.01, "impact": {"HIGH"}}),
    ("all_lof", {"impact": {"HIGH", "MODERATE"}}),
    (
        "rare_lof_constrained",
        {
            "max_gnomad_af": 0.01,
            "min_pLI": 0.9,
            "min_z_score": 5.0,
            "min_cadd": 20.0,
            "impact": {"HIGH", "MODERATE"},
        },
    ),
    ("common_variants", {"min_gnomad_af": 0.05}),
    ("rare_variants", {"max_gnomad_af": 0.01}),
    (
        "clinvar_pathogenic_lof",
        {
            "min_cadd": 20.0,
            "impact": {"HIGH", "MODERATE"},
            "filter_clinvar_pathogenic": True,
        },
    ),
    ("splice_site_variants", {"filter_splice_variants": True}),
]

#: The constraints resource each config points at. The two configs target
#: different machines, so the path is pinned per file rather than shared.
_CONSTRAINTS_RESOURCE = {
    "gnomad_meb.orig.toml": (
        "/home/brand/scratch/annotation/hail/gnomad.v4.1.constraint_metrics.ht"
    ),
    "gnomad_meb.brand-deskt.toml": (
        "/data/volumes/xcat-login/scratch/brand/annotation/hail/"
        "gnomad.v4.1.constraint_metrics.ht"
    ),
}


@pytest.mark.parametrize(
    "toml_name",
    ["gnomad_meb.orig.toml", "gnomad_meb.brand-deskt.toml"],
)
def test_filters_yaml_contents_are_migrated_into_the_toml(toml_name: str) -> None:
    settings = resolve(
        _CONFIG_ROOT / toml_name,
        overrides={"validate_paths": False},
    )

    assert settings.filter is not None
    filters = settings.filter.filters
    assert [flt.name for flt in filters] == [
        name for name, _ in _MIGRATED_FILTERS
    ]

    by_name = {flt.name: flt for flt in filters}
    for name, expected in _MIGRATED_FILTERS:
        for field, value in expected.items():
            assert getattr(by_name[name], field) == value, f"{name}.{field}"

    assert settings.filter.resources is not None
    assert (
        settings.filter.resources.constraints == _CONSTRAINTS_RESOURCE[toml_name]
    )


def test_load_filters_and_regions_accept_models_and_mappings() -> None:
    from gnomad_meb.config import FilterCheckpoint, RegionCheckpoint

    filter_model = FilterCheckpoint(name="rare_lof", max_gnomad_af=0.01)
    filter_mapping = {
        "name": "all_clinvar_pathogenic",
        "filter_clinvar_pathogenic": True,
    }
    filters = FilterCheckpoint.load_filters([filter_model, filter_mapping])
    assert filters == [
        filter_model,
        FilterCheckpoint(name="all_clinvar_pathogenic", filter_clinvar_pathogenic=True),
    ]

    region_model = RegionCheckpoint(name="integrated", intervals="/r/integrated.bed")
    region_mapping = {"name": "grch37", "reference_genome": "GRCh37"}
    regions = RegionCheckpoint.load_regions([region_model, region_mapping])
    assert regions == [
        region_model,
        RegionCheckpoint(name="grch37", reference_genome="GRCh37"),
    ]


def test_variant_qc_tunables_are_loaded_from_the_file(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path,
        """
        [variant_qc]
        import_header_fixed = "/r/vqsr.header.fixed"

        [variant_qc.target_truth_sensitivity]
        snp = 99.5
        indel = 98.0

        [variant_qc.vqsr_params]
        allele_specific = false
        max_gaussians = 6
        variants_to_sample = 5
        snp_tranche_values = [100.0, 99.0, 90.0]
        indel_tranche_values = [100.0, 95.0, 90.0]
        """,
    )

    settings = resolve(config, overrides={"validate_paths": False})

    assert settings.variant_qc is not None
    assert settings.variant_qc.target_truth_sensitivity.snp == 99.5
    assert settings.variant_qc.target_truth_sensitivity.indel == 98.0
    assert settings.variant_qc.vqsr_params.allele_specific is False
    assert settings.variant_qc.vqsr_params.max_gaussians == 6
    assert settings.variant_qc.vqsr_params.variants_to_sample == 5
    assert settings.variant_qc.vqsr_params.snp_tranche_values == [100.0, 99.0, 90.0]
    assert settings.variant_qc.vqsr_params.indel_tranche_values == [
        100.0,
        95.0,
        90.0,
    ]


# --- serialization ----------------------------------------------------------


def test_serialization_round_trip(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path,
        """
        project = "round-trip"

        [s3]
        access_key = "key"
        secret_key = "secret"

        [vqsr]
        output_dir = "./vqsr-output"

        [resources]
        calling_intervals = "/r/calling.interval_list"
        telomeres_and_centromeres = "/r/tel.ht"
        lcr_intervals = "/r/lcr.ht"
        seg_dup_intervals = "/r/seg.ht"
        gnomad_pca_loadings = "/r/loadings.ht"
        gnomad_pca_model = "/r/model.onnx"
        gnomad_v4 = "/r/gnomad.ht"
        reference = "/r/reference.fasta"
        """,
    )
    settings = resolve(config, overrides={"validate_paths": False})

    reloaded = Settings.model_validate_json(settings.model_dump_json())

    assert reloaded.model_dump() == settings.model_dump()


# --- the config file is explicit --------------------------------------------


def test_missing_config_file_is_a_hard_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        resolve(tmp_path / "does-not-exist.toml")


# --- get_config caching -----------------------------------------------------


def test_get_config_caches_the_first_resolution() -> None:
    first = get_config()
    second = get_config()

    assert first is second


def test_get_config_resolves_with_a_supplied_path(tmp_path: Path) -> None:
    config = _write_config(tmp_path, 'project = "from-file"\n')

    resolved = get_config(config=config)

    assert resolved.project == "from-file"
    assert get_config() is resolved


# --- resolution has no side effects -----------------------------------------


def test_resolution_does_not_create_the_checkpoint_dir(tmp_path: Path) -> None:
    checkpoint = tmp_path / "does-not-exist-yet"
    config = _write_config(tmp_path, f'checkpoint = "{checkpoint}"\n')

    resolve(config)

    assert not checkpoint.exists()


# --- validation hardening ----------------------------------------------------

_RESOURCE_FIELDS = tuple(Resources.model_fields)


def _real_resources(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name in _RESOURCE_FIELDS:
        path = root / name
        path.write_text("x")
        paths[name] = path
    return paths


def _resources_table(paths: dict[str, Path]) -> str:
    lines = [f'{name} = "{path}"' for name, path in paths.items()]
    return "[resources]\n" + "\n".join(lines) + "\n"


def test_existing_resource_paths_resolve(tmp_path: Path) -> None:
    paths = _real_resources(tmp_path / "ref")
    config = _write_config(tmp_path, _resources_table(paths))

    settings = resolve(config)

    assert settings.resources is not None
    assert settings.resources.reference == paths["reference"]


def test_missing_required_resource_names_the_field(tmp_path: Path) -> None:
    paths = _real_resources(tmp_path / "ref")
    del paths["reference"]
    config = _write_config(tmp_path, _resources_table(paths))

    with pytest.raises(ValidationError) as excinfo:
        resolve(config)

    assert "reference" in str(excinfo.value)


def test_nonexistent_resource_path_names_the_field(tmp_path: Path) -> None:
    paths = _real_resources(tmp_path / "ref")
    paths["reference"] = tmp_path / "does-not-exist.fasta"
    config = _write_config(tmp_path, _resources_table(paths))

    with pytest.raises(ValidationError) as excinfo:
        resolve(config)

    assert "resources" in str(excinfo.value)
    assert "reference" in str(excinfo.value)


def test_wrong_kind_resource_names_the_field(tmp_path: Path) -> None:
    paths = _real_resources(tmp_path / "ref")
    directory = tmp_path / "alpha_missense"
    directory.mkdir()
    config = _write_config(
        tmp_path,
        _resources_table(paths)
        + "\n[annotation.vep]\n"
        + f'vep = "{tmp_path / "vep.sh"}"\n'
        + f'cache_dir = "{tmp_path / "cache"}"\n'
        + f'alpha_missense = "{directory}"\n',
    )

    with pytest.raises(ValidationError) as excinfo:
        resolve(config)

    assert "alpha_missense" in str(excinfo.value)


def test_vep_required_fields_name_the_missing_one(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path,
        '[annotation.vep]\nvep = "/v/vep.sh"\n',
    )

    with pytest.raises(ValidationError) as excinfo:
        resolve(config, overrides={"validate_paths": False})

    assert "cache_dir" in str(excinfo.value)


def test_loftee_both_or_neither_is_enforced(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path,
        "[annotation.vep]\n"
        'vep = "/v/vep.sh"\n'
        'cache_dir = "/v/cache"\n'
        'loftee_base = "/v/loftee"\n',
    )

    with pytest.raises(ValidationError) as excinfo:
        resolve(config, overrides={"validate_paths": False})

    assert "loftee" in str(excinfo.value)


def test_cadd_both_or_neither_is_enforced(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path,
        "[annotation.vep]\n"
        'vep = "/v/vep.sh"\n'
        'cache_dir = "/v/cache"\n'
        'cadd = "/v/cadd.tsv"\n',
    )

    with pytest.raises(ValidationError) as excinfo:
        resolve(config, overrides={"validate_paths": False})

    assert "cadd" in str(excinfo.value)


def test_validate_paths_false_skips_only_the_filesystem_probes(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path,
        "[resources]\n"
        'calling_intervals = "/r/calling.interval_list"\n'
        'telomeres_and_centromeres = "/r/tel.ht"\n'
        'lcr_intervals = "/r/lcr.ht"\n'
        'seg_dup_intervals = "/r/seg.ht"\n'
        'gnomad_pca_loadings = "/r/loadings.ht"\n'
        'gnomad_pca_model = "/r/model.onnx"\n'
        'gnomad_v4 = "/r/gnomad.ht"\n'
        'reference = "/r/reference.fasta"\n',
    )

    settings = resolve(config, overrides={"validate_paths": False})
    assert settings.resources is not None
    assert settings.resources.reference == Path("/r/reference.fasta")

    bad = _write_config(
        tmp_path,
        "[annotation.vep]\n"
        'vep = "/v/vep.sh"\n'
        'cache_dir = "/v/cache"\n'
        'cadd = "/v/cadd.tsv"\n',
    )
    with pytest.raises(ValidationError, match="cadd"):
        resolve(bad, overrides={"validate_paths": False})


def test_blank_s3_credential_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GMEB_S3__ACCESS_KEY", "   ")

    with pytest.raises(ValidationError) as excinfo:
        resolve()

    assert "access_key" in str(excinfo.value)


# --- gwas study model -------------------------------------------------------


_STUDY_TOML = """
[gwas]
metadata_csv = "/data/study.md.csv"
case_cohort = "alopecia-areata"
control_cohorts = ["radar"]
drop_cohorts = ["covid"]
rare_variants_maf = 0.01
min_cadd = 20.0
intercept = true

[gwas.responses]
case = "case"
case_only = true
[[gwas.responses.columns]]
name = "comorbidities"
kind = "set"
exclude = ["hypertension", "thyroid"]
[[gwas.responses.columns]]
name = "comorbidities_thyroid"
kind = "set"
exclude = ["other_ai_thyroid_disease"]
[[gwas.responses.columns]]
name = "other_autoimmune_comorbidities"
kind = "set"
exclude = []
[[gwas.responses.columns]]
name = "exposition"
kind = "binary"
metadata_field = "exposition"
[[gwas.responses.columns]]
name = "exposition_sum_dosage"
kind = "quantitative"
metadata_field = "exposition_sum_dosage"

[[gwas.covariates]]
name = "age"
metadata_field = "birthdate"
[[gwas.covariates]]
name = "sex"
metadata_field = "sex"
[[gwas.covariates]]
name = "ancestry.pca_scores"
indices = [0, 1, 2]

[[gwas.regions]]
kind = "gene"
"""


def test_gwas_study_model_resolves_from_toml(tmp_path: Path) -> None:
    """The worked study block parses to the spec's response/covariate shape."""
    config = _write_config(tmp_path, _STUDY_TOML)

    settings = resolve(config, overrides={"validate_paths": False})

    gwas = settings.gwas
    assert gwas is not None
    assert gwas.metadata_csv == Path("/data/study.md.csv")
    assert gwas.case_cohort == "alopecia-areata"
    assert gwas.control_cohorts == ["radar"]
    assert gwas.drop_cohorts == ["covid"]
    assert gwas.responses.case == "case"
    assert gwas.responses.case_only is True
    assert [
        (c.name, c.kind, c.metadata_field) for c in gwas.responses.columns
    ] == [
        ("comorbidities", "set", None),
        ("comorbidities_thyroid", "set", None),
        ("other_autoimmune_comorbidities", "set", None),
        ("exposition", "binary", "exposition"),
        ("exposition_sum_dosage", "quantitative", "exposition_sum_dosage"),
    ]
    assert gwas.responses.columns[0].exclude == [
        "hypertension",
        "thyroid",
    ]
    assert gwas.intercept is True
    assert [(c.name, c.metadata_field, c.indices) for c in gwas.covariates] == [
        ("age", "birthdate", None),
        ("sex", "sex", None),
        ("ancestry.pca_scores", None, [0, 1, 2]),
    ]
    assert [r.kind for r in gwas.regions] == ["gene"]


def test_gwas_study_model_round_trips() -> None:
    from gnomad_meb.config import GwasConfig

    gwas = GwasConfig(metadata_csv=Path("/data/study.md.csv"))
    reloaded = GwasConfig.model_validate(gwas.model_dump())

    assert reloaded == gwas
    # Per-gene is the default region when none is configured.
    assert [r.kind for r in gwas.regions] == ["gene"]


def test_gwas_response_exclude_is_set_only() -> None:
    from gnomad_meb.config import GwasResponseColumn

    with pytest.raises(ValidationError, match="exclude"):
        GwasResponseColumn(name="exposition", kind="binary", exclude=["x"])


def test_gwas_window_region_requires_a_size() -> None:
    from gnomad_meb.config import GwasConfig

    with pytest.raises(ValidationError, match="size"):
        GwasConfig(
            metadata_csv=Path("/data/study.md.csv"),
            regions=[{"kind": "window"}],
        )


def test_gwas_metadata_csv_is_required() -> None:
    from gnomad_meb.config import GwasConfig

    with pytest.raises(ValidationError, match="metadata_csv"):
        GwasConfig()


# --- HailConfig.log_level (ticket 03) ----------------------------------------


def test_hail_log_level_defaults_to_warning() -> None:
    from gnomad_meb.config.settings import HailConfig

    assert HailConfig().log_level == "warning"


def test_hail_log_level_is_normalised() -> None:
    from gnomad_meb.config.settings import HailConfig

    assert HailConfig(log_level="WARNING").log_level == "warning"
    assert HailConfig(log_level="Info").log_level == "info"


def test_hail_log_level_rejects_unknown_levels() -> None:
    from gnomad_meb.config.settings import HailConfig

    with pytest.raises(ValidationError):
        HailConfig(log_level="loud")


def test_hail_log_level_resolves_from_the_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from gnomad_meb.config import Settings

    monkeypatch.setenv("GMEB_HAIL__LOG_LEVEL", "debug")
    assert Settings(validate_paths=False).hail.log_level == "debug"


# --- checkpoint staleness policy ---------------------------------------------


def test_staleness_policy_defaults_to_completeness() -> None:
    from gnomad_meb.lib.checkpoint.status import StalenessPolicy

    assert Settings(validate_paths=False).staleness_policy is (
        StalenessPolicy.COMPLETENESS
    )


def test_staleness_policy_resolves_from_the_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gnomad_meb.lib.checkpoint.status import StalenessPolicy

    monkeypatch.setenv("GMEB_STALENESS_POLICY", "source")
    assert Settings(validate_paths=False).staleness_policy is StalenessPolicy.SOURCE


def test_staleness_policy_rejects_an_invalid_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GMEB_STALENESS_POLICY", "nonsense")
    with pytest.raises(ValidationError) as excinfo:
        Settings(validate_paths=False)
    assert "staleness_policy" in str(excinfo.value)


def test_staleness_policy_round_trips_through_dump_and_validate() -> None:
    from gnomad_meb.lib.checkpoint.status import StalenessPolicy

    settings = Settings(
        validate_paths=False, staleness_policy=StalenessPolicy.SOURCE_OR_GIT
    )
    dumped = settings.model_dump(mode="json")
    assert dumped["staleness_policy"] == "source_or_git"

    reloaded = Settings.model_validate(dumped)
    assert reloaded.staleness_policy is StalenessPolicy.SOURCE_OR_GIT
