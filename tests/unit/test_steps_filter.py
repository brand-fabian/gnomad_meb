"""Tests for the Filter stage, a ``Pipeline`` subclass in ``lib.filter``.

``FilterPipeline`` declares two fixed ``@checkpoint`` steps plus one Step per
configured filter and one per ``(region, filter)`` pair. These tests pin down
the declared step names, the ``FromStep``/``ResourceRef`` edges, the class
resource set, and the declared artifacts, without running the hail computation.
"""

from __future__ import annotations

import pytest

from gnomad_meb.cmd.task import PipelineStep, Task
from gnomad_meb.config import (
    FilterCheckpoint,
    FilterConfig,
    FilterResources,
    RegionCheckpoint,
    Settings,
)
from gnomad_meb.hail import hail as hl
from gnomad_meb.lib.checkpoint import CHECKPOINT_TYPE
from gnomad_meb.lib.filter.pipeline import FilterPipeline
from gnomad_meb.pipeline import InputStep, StepRegistry

#: The fixed terminal step's name: the constraints-annotated callset.
_TERMINAL = "annotate_constraints_table"

_FILTERS = [
    FilterCheckpoint(name="rare_lof", min_pLI=0.9),
    FilterCheckpoint(name="all_clinvar_pathogenic", filter_clinvar_pathogenic=True),
]
_REGIONS = [RegionCheckpoint(name="integrated")]


def _settings(
    *,
    filters: list[FilterCheckpoint] | None = None,
    regions: list[RegionCheckpoint] | None = None,
    constraints: str | None = "/data/constraints.ht",
    reference_genome: str = "GRCh38",
) -> Settings:
    return Settings(
        validate_paths=False,
        project="proj",
        checkpoint="/tmp/opencode/filter-regions/ckpt",
        reference_genome=reference_genome,
        filter=FilterConfig(
            filters=list(_FILTERS) if filters is None else filters,
            regions=list(_REGIONS) if regions is None else regions,
            resources=FilterResources(constraints=constraints),
        ),
    )


def _pipeline(**kwargs) -> FilterPipeline:
    task = Task(settings=_settings(**kwargs), command=PipelineStep(name="filter"))
    return FilterPipeline(task)


def test_fixed_steps_are_declared() -> None:
    _pipeline()
    names = StepRegistry().names()

    assert "filter_high_quality_variants_gnomad" in names
    assert _TERMINAL in names


def test_mt_input_reads_the_matrix_table_checkpoint() -> None:
    _pipeline()
    step = StepRegistry().get("mt")

    assert isinstance(step, InputStep)
    assert step.value is None
    assert step.run is hl.read_matrix_table


def test_each_filter_is_a_step_off_the_terminal() -> None:
    _pipeline()
    registry = StepRegistry()

    terminal = registry.get(_TERMINAL)
    assert terminal.depends_on == {
        "mt": "filter.filter_high_quality_variants_gnomad"
    }
    assert terminal.outputs == {"mt": 0}

    for flt in _FILTERS:
        step = registry.get(flt.name)
        assert step.depends_on == {"mt": f"filter.{_TERMINAL}"}
        assert step.outputs == {"mt": 0}


def test_each_filter_declares_matrix_and_variant_info() -> None:
    pipeline = _pipeline()

    for flt in _FILTERS:
        step = StepRegistry().get(flt.name)
        assert [artifact.kind for artifact in step.artifacts] == [
            CHECKPOINT_TYPE.MATRIX_TABLE,
            CHECKPOINT_TYPE.DICTIONARY,
        ]
        assert (
            step.artifacts[0]
            .path(pipeline.settings)
            .endswith(f"/proj/intermediate/filter/{flt.name}.mt")
        )
        assert (
            step.artifacts[1]
            .path(pipeline.settings)
            .endswith(f"/proj/intermediate/filter/{flt.name}.variant_info.json.gz")
        )


def test_region_filter_step_consumes_the_filter_matrix() -> None:
    _pipeline()
    registry = StepRegistry()

    for flt in _FILTERS:
        for reg in _REGIONS:
            step = registry.get(f"filter_r{reg.name}_f{flt.name}")
            assert step.depends_on == {"mt": f"filter.{flt.name}"}
            assert step.outputs == {"mt": 0}
            assert [artifact.kind for artifact in step.artifacts] == [
                CHECKPOINT_TYPE.MATRIX_TABLE
            ]


def test_region_step_path_uses_the_checkpoint_prefix() -> None:
    pipeline = _pipeline()
    step = StepRegistry().get("filter_rintegrated_frare_lof")

    assert (
        step.artifacts[0]
        .path(pipeline.settings)
        .endswith("/proj/intermediate/filter/rintegrated/frare_lof.mt")
    )


def test_manifest_step_is_a_leaf_over_every_filter_and_region() -> None:
    pipeline = _pipeline()
    step = StepRegistry().get("get_manifest")

    assert [artifact.kind for artifact in step.artifacts] == [
        CHECKPOINT_TYPE.DICTIONARY
    ]
    assert (
        step.artifacts[0]
        .path(pipeline.settings)
        .endswith("/proj/intermediate/filter/manifest.json.gz")
    )
    assert step.depends_on == {
        "high_quality_variant_info": "filter.filter_high_quality_variants_gnomad",
        "high_quality_sample_info": "filter.filter_high_quality_variants_gnomad",
        **{f"variant_info_{flt.name}": f"filter.{flt.name}" for flt in _FILTERS},
        **{
            f"region_{reg.name}_{flt.name}": (
                f"filter.filter_r{reg.name}_f{flt.name}"
            )
            for flt in _FILTERS
            for reg in _REGIONS
        },
    }
    assert step.outputs == {
        "high_quality_variant_info": 1,
        "high_quality_sample_info": 2,
        **{f"variant_info_{flt.name}": 1 for flt in _FILTERS},
        **{
            f"region_{reg.name}_{flt.name}": 0
            for flt in _FILTERS
            for reg in _REGIONS
        },
    }


def test_manifest_step_declares_only_the_fixed_dependencies_without_filters() -> None:
    _pipeline(filters=[], regions=[], constraints=None)
    step = StepRegistry().get("get_manifest")

    assert step.depends_on == {
        "high_quality_variant_info": "filter.filter_high_quality_variants_gnomad",
        "high_quality_sample_info": "filter.filter_high_quality_variants_gnomad",
    }


def test_no_regions_declares_no_region_steps() -> None:
    _pipeline(regions=[])

    names = set(StepRegistry().names())
    assert {
        "mt",
        "filter_high_quality_variants_gnomad",
        _TERMINAL,
        "rare_lof",
        "all_clinvar_pathogenic",
    } <= names
    assert not any(name.startswith("filter_r") for name in names)


def test_no_configured_filters_leaves_the_fixed_steps() -> None:
    _pipeline(filters=[], regions=[], constraints=None)

    assert set(StepRegistry().names()) == {
        "mt",
        "filter_high_quality_variants_gnomad",
        _TERMINAL,
        "get_manifest",
    }


def test_constraints_is_a_declared_resource_when_configured() -> None:
    pipeline = _pipeline()

    assert len(pipeline.resources) == 1
    resource = pipeline.resources[0]
    assert resource.name == "constraints"
    assert resource.path == "/data/constraints.ht"
    assert resource.reference_genome == "GRCh38"
    assert resource.loader is hl.read_table

    step = StepRegistry().get(_TERMINAL)
    assert step.resources["constraints_ht"] == "resources.load-constraints"


def test_constraints_uses_the_run_reference_genome() -> None:
    pipeline = _pipeline(reference_genome="GRCh37")

    assert pipeline.resources[0].reference_genome == "GRCh37"


def test_a_filter_without_constraints_still_registers() -> None:
    pipeline = _pipeline(constraints=None)

    assert pipeline.resources == ()
    assert "rare_lof" in StepRegistry().names()


def test_filter_name_collision_is_rejected() -> None:
    with pytest.raises(ValueError, match="collides"):
        _pipeline(filters=[FilterCheckpoint(name=_TERMINAL)], regions=[])
