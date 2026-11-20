"""Tests for the Resource model and the :class:`ResourceRegistry`.

A resource is *referenced, not copied*: a pipeline declares each resource as a
``Resource(name, path, reference_genome, loader)`` and the registry indexes the
objects by name and path. An unmodified resource resolves to a load step that
reads its path in place; a resource whose path equals a sibling step's declared
artifact is *derived* and resolves to that producing step; a resource the
registry must adapt gains the registry's modification steps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gnomad_meb.cmd.task import PipelineStep, Task
from gnomad_meb.config import Settings, set_config
from gnomad_meb.lib.checkpoint import CHECKPOINT_TYPE, Artifact
from gnomad_meb.pipeline import (
    Resource,
    ResourceRegistry,
    Step,
    StepRegistry,
)


def _settings(**overrides: Any) -> Settings:
    return Settings(
        validate_paths=False,
        project=overrides.pop("project", "demo"),
        checkpoint=overrides.pop("checkpoint", "/ckpt"),
        **overrides,
    )


def _read_table(path: Path | str) -> bytes:
    """A plain-callable test loader (never serialized)."""
    return b"raw:" + str(path).encode()


def _resource(
    name: str = "res",
    *,
    path: str = "/data/raw.ht",
    reference_genome: str = "GRCh38",
    loader=_read_table,
) -> Resource:
    return Resource(
        name=name,
        path=path,
        reference_genome=reference_genome,
        loader=loader,
    )


def _task() -> Task:
    return Task(settings=_settings(), command=PipelineStep(name="test"))


class _Conform:
    """A test-only registry modification, conforming a resource to the run."""

    def applies(self, resource: Resource, run_reference: str) -> bool:
        return resource.reference_genome != run_reference

    def steps(self, resource: Resource, settings: Settings) -> tuple[Step, ...]:
        output = Artifact(
            prefix=f"{{project}}/resources/{resource.name}",
            kind=CHECKPOINT_TYPE.FILE,
        )
        return (
            Step(
                name=f"{resource.name}/conform",
                task=Task(
                    settings=settings,
                    command=PipelineStep(name=f"{resource.name}/conform"),
                ),
                artifacts=(output,),
                group="resources",
                run=lambda path: b"conformed:" + str(path).encode(),
            ),
        )


def test_registry_indexes_by_name_and_path() -> None:
    registry = ResourceRegistry()
    resource = _resource("lcr", path="/data/lcr.ht")
    registry.add(resource)

    assert "lcr" in registry
    assert registry.get_resource("lcr") is resource
    assert registry.get_resource(("/data/lcr.ht", "GRCh38")) is resource


def test_one_source_shared_under_two_names_dedupes_by_identity() -> None:
    registry = ResourceRegistry()
    registry.add(_resource("lcr_intervals", path="/data/lcr.ht"))
    registry.add(_resource("lcr_intervals_ht", path="/data/lcr.ht"))

    assert registry.names() == ("lcr_intervals", "lcr_intervals_ht")
    assert len(registry._resources) == 1


def test_an_unmodified_resource_resolves_to_a_load_step() -> None:
    registry = ResourceRegistry()
    registry.add(_resource("gnomad_v4", path="/data/gnomad_v4.ht"))

    steps = registry.get("gnomad_v4")

    assert steps == ["resources.load-gnomad_v4"]
    load = StepRegistry().get(steps[0])
    assert load.run() == _read_table("/data/gnomad_v4.ht")


def test_a_derived_resource_resolves_to_its_producing_step() -> None:
    set_config(_settings())
    produced = Artifact(prefix="{project}/sample_qc", kind=CHECKPOINT_TYPE.HAIL_TABLE)
    registry = ResourceRegistry()
    registry.add(_resource("sample_qc_ht", path=produced.path(_settings())))

    producer = Step(
        name="compute_hard_filters",
        task=_task(),
        artifacts=(produced,),
        group="sample_qc",
        run=lambda: None,
    )
    StepRegistry().add(producer)

    assert registry.get("sample_qc_ht") == ["sample_qc.compute_hard_filters"]


def test_a_resource_the_registry_must_adapt_gets_the_modification_step() -> None:
    registry = ResourceRegistry()
    registry.modifications = (_Conform(),)
    registry.add(_resource("grch37", path="/data/grch37.ht", reference_genome="GRCh37"))

    steps = registry.get("grch37")

    assert len(steps) == 2
    load, modify = (StepRegistry()[step] for step in steps)
    assert load.name == "load-grch37"
    assert modify.resources == {"grch37": load.id}


def test_a_conforming_resource_gets_no_modification_step() -> None:
    registry = ResourceRegistry()
    registry.modifications = (_Conform(),)
    registry.add(_resource("grch38", path="/data/grch38.ht", reference_genome="GRCh38"))

    assert registry.get("grch38") == ["resources.load-grch38"]


def test_current_state_records_path_and_reference_genome() -> None:
    registry = ResourceRegistry()
    registry.add(_resource("lcr", path="/data/lcr.ht"))

    assert registry.current_state()["lcr"] == {
        "name": "lcr",
        "path": "/data/lcr.ht",
        "reference_genome": "GRCh38",
    }
