"""Tests for the :class:`Pipeline` base and the central registries.

Constructing a pipeline registers its resources and steps; the
:class:`~gnomad_meb.pipeline.StepRegistry` then exposes the declared DAG: step
names and order, ``FromStep`` / ``RunInput`` / ``ResourceRef`` edges, and the
artifacts the decorator declares. The markers are parameter defaults; the class
``resources`` property is the run's resource set.
"""

from __future__ import annotations

import functools

import pytest

from gnomad_meb.cmd.task import PipelineStep, Task
from gnomad_meb.config import Settings
from gnomad_meb.lib.checkpoint import CHECKPOINT_TYPE, Artifact, checkpoint
from gnomad_meb.pipeline import (
    FromStep,
    InputStep,
    Pipeline,
    ReferenceGenomeMismatch,
    Resource,
    ResourceRef,
    RunInput,
    Step,
    StepRegistry,
)


def _settings(checkpoint: str | None = "/ckpt") -> Settings:
    return Settings(validate_paths=False, project="demo", checkpoint=checkpoint)


def _task() -> Task:
    return Task(settings=_settings(), command=PipelineStep(name="test"))


class SampleQc(Pipeline):
    """A small stage: a resource consumer, then a ``FromStep`` consumer."""

    @classmethod
    def name(cls) -> str:
        return "sample_qc"

    @property
    def resources(self) -> tuple[Resource, ...]:
        return (
            Resource(
                name="calling_intervals",
                path="/data/calling.interval_list",
                reference_genome="GRCh38",
                loader=lambda path: path,
            ),
            Resource(
                name="gnomad_v4",
                path="/data/gnomad_v4.ht",
                reference_genome="GRCh38",
                loader=lambda path: path,
            ),
        )

    def get_input(self) -> tuple[InputStep, ...]:
        return (
            InputStep(
                name="vds",
                task=self.task,
                group=type(self).name(),
                value=getattr(self.command, "vds", None),
            ),
        )

    @checkpoint(
        "{project}/intermediate/sample_qc/filter_intervals", CHECKPOINT_TYPE.HAIL_TABLE
    )
    def filter_intervals(
        self,
        vds=RunInput("vds"),  # noqa: B008
        calling_intervals=ResourceRef("calling_intervals"),  # noqa: B008
    ):
        return vds

    @checkpoint(
        "{project}/intermediate/sample_qc/base_sample_qc", CHECKPOINT_TYPE.HAIL_TABLE
    )
    def base_sample_qc(self, vds=FromStep("filter_intervals")):  # noqa: B008
        return vds


class Cyclic(Pipeline):
    @classmethod
    def name(cls) -> str:
        return "cyclic"

    @property
    def resources(self) -> tuple[Resource, ...]:
        return ()

    def get_input(self) -> tuple[InputStep, ...]:
        return ()

    @checkpoint("{project}/cyclic/a", CHECKPOINT_TYPE.HAIL_TABLE)
    def a(self, b=FromStep("b")):  # noqa: B008
        return b

    @checkpoint("{project}/cyclic/b", CHECKPOINT_TYPE.HAIL_TABLE)
    def b(self, a=FromStep("a")):  # noqa: B008
        return a


def test_decorated_methods_become_steps_in_definition_order() -> None:
    SampleQc(_task())
    registry = StepRegistry()

    assert registry.names()[:3] == (
        "vds",
        "filter_intervals",
        "base_sample_qc",
    )


def test_steps_get_group_prefixed_ids() -> None:
    SampleQc(_task())

    assert StepRegistry().get("filter_intervals").id == "sample_qc.filter_intervals"


def test_fromstep_default_becomes_a_dependency_edge() -> None:
    SampleQc(_task())
    registry = StepRegistry()

    assert registry.get("base_sample_qc").depends_on == {
        "vds": "sample_qc.filter_intervals"
    }


def test_runinput_default_resolves_to_the_input_endpoint() -> None:
    SampleQc(_task())

    assert StepRegistry().get("filter_intervals").depends_on == {"vds": "sample_qc.vds"}


def test_resourceref_default_resolves_to_the_load_step() -> None:
    SampleQc(_task())
    step = StepRegistry().get("filter_intervals")

    load_id = step.resources["calling_intervals"]
    assert load_id == "resources.load-calling_intervals"
    assert StepRegistry().get(load_id) is not None


def test_outputs_are_the_decorators_artifacts() -> None:
    SampleQc(_task())
    step = StepRegistry().get("filter_intervals")

    assert step.artifacts == (
        Artifact(
            prefix="{project}/intermediate/sample_qc/filter_intervals",
            kind=CHECKPOINT_TYPE.HAIL_TABLE,
        ),
    )
    assert (
        step.artifacts[0].path(_settings())
        == "/ckpt/demo/intermediate/sample_qc/filter_intervals.ht"
    )


class OutputSelector(Pipeline):
    """A consumer that selects an output index from a tuple producer."""

    @classmethod
    def name(cls) -> str:
        return "output_selector"

    @property
    def resources(self) -> tuple[Resource, ...]:
        return ()

    def get_input(self) -> tuple[InputStep, ...]:
        return ()

    @checkpoint("{project}/output_selector/make", CHECKPOINT_TYPE.FILE)
    def make(self):
        return ("a", "b")

    @checkpoint("{project}/output_selector/pick", CHECKPOINT_TYPE.FILE)
    def pick(self, value=FromStep("make", output=1)):  # noqa: B008
        return value


def test_fromstep_output_is_recorded_on_the_step() -> None:
    OutputSelector(_task())

    assert StepRegistry().get("pick").outputs == {"value": 1}


def test_step_state_carries_the_recorded_outputs() -> None:
    OutputSelector(_task())

    assert StepRegistry().get("pick").state().outputs == {"value": 1}


def test_a_dependency_cycle_is_rejected_when_ordered() -> None:
    Cyclic(_task())

    with pytest.raises(ValueError):
        StepRegistry().topological()


def test_topological_order_honours_depends_on() -> None:
    SampleQc(_task())
    order = [step.name for step in StepRegistry().topological()]

    assert order.index("filter_intervals") < order.index("base_sample_qc")


class MismatchStage(Pipeline):
    @classmethod
    def name(cls) -> str:
        return "mismatch"

    @property
    def resources(self) -> tuple[Resource, ...]:
        return (
            Resource(
                name="res",
                path="/data/grch37.ht",
                reference_genome="GRCh37",
                loader=lambda path: path,
            ),
        )

    def get_input(self) -> tuple[InputStep, ...]:
        return ()

    @checkpoint("{project}/mismatch", CHECKPOINT_TYPE.FILE)
    def mismatch(self, res=ResourceRef("res")):  # noqa: B008
        return res


def test_a_mismatched_reference_genome_is_rejected() -> None:
    with pytest.raises(ReferenceGenomeMismatch):
        MismatchStage(_task())


class Generated(Pipeline):
    """A stage that derives one step per split from its inputs in ``prepare``.

    It calls ``super().prepare()`` to keep the decorated methods and then
    appends loop-built ``@checkpoint`` methods -- the shape a per-split stage
    uses -- with no attribute on ``self`` and no hail import.
    """

    SPLITS = ("left", "right")

    @classmethod
    def name(cls) -> str:
        return "generated"

    @property
    def resources(self) -> tuple[Resource, ...]:
        return ()

    def get_input(self) -> tuple[InputStep, ...]:
        return ()

    @checkpoint("{project}/generated/base", CHECKPOINT_TYPE.FILE)
    def base(self):
        return "base"

    def prepare(self) -> None:
        super().prepare()
        for split in self.SPLITS:

            @checkpoint(f"{{project}}/generated/{split}", CHECKPOINT_TYPE.FILE)
            def gen(value=FromStep("base"), _split=split):  # noqa: B008
                return f"{value}:{_split}"

            self._methods.append((f"gen_{split}", gen, split))


def test_prepare_finalizes_the_methods_it_generates() -> None:
    Generated(_task())
    registry = StepRegistry()

    # The finalized list decides discovery: decorated, then generated, in order.
    assert registry.names()[:3] == ("base", "gen_left", "gen_right")

    for split in Generated.SPLITS:
        step = registry.get(f"gen_{split}")
        assert step is not None
        assert step.id == f"generated.gen_{split}"


def test_a_generated_step_carries_its_task_identity_in_args() -> None:
    Generated(_task())

    assert StepRegistry().get("gen_left").task.command.args == {"task": "left"}
    assert StepRegistry().get("base").task.command.args == {}


def test_a_generated_step_is_wired_from_its_marker_defaults() -> None:
    Generated(_task())

    assert StepRegistry().get("gen_left").depends_on == {"value": "generated.base"}


class PartialGenerated(Pipeline):
    """A stage whose generated methods carry their edges as ``partial`` keywords.

    Each per-split method is ``@checkpoint``-applied to a ``functools.partial``
    of a *plain* body: the marker is bound as a partial keyword, which is the
    only place a terminal ``**parts`` merge can carry its per-task edges.
    """

    SPLITS = ("left", "right")

    @classmethod
    def name(cls) -> str:
        return "partial_generated"

    @property
    def resources(self) -> tuple[Resource, ...]:
        return ()

    def get_input(self) -> tuple[InputStep, ...]:
        return ()

    @checkpoint("{project}/partial_generated/base", CHECKPOINT_TYPE.FILE)
    def base(self):
        return "base"

    def dense(self, task=None):
        return f"dense:{task}"

    def annotate(self, mt=FromStep("base"), task=None):  # noqa: B008
        return f"annotated:{mt}:{task}"

    def merge(self, **parts):
        return parts

    def prepare(self) -> None:
        super().prepare()
        for split in self.SPLITS:
            self._methods.append((
                f"dense_{split}",
                checkpoint(
                    f"{{project}}/partial_generated/dense_{split}", CHECKPOINT_TYPE.FILE
                )(functools.partial(self.dense, task=split)),
                split,
            ))
            self._methods.append((
                f"annotate_{split}",
                checkpoint(
                    f"{{project}}/partial_generated/annotate_{split}",
                    CHECKPOINT_TYPE.FILE,
                )(
                    functools.partial(
                        self.annotate,
                        mt=FromStep(f"dense_{split}"),
                        task=split,
                    )
                ),
                split,
            ))
        self._methods.append((
            "merge",
            checkpoint("{project}/partial_generated/merge", CHECKPOINT_TYPE.FILE)(
                functools.partial(
                    self.merge,
                    **{split: FromStep(f"annotate_{split}") for split in self.SPLITS},
                )
            ),
            None,
        ))


def test_a_partial_edge_is_wired_from_its_bound_keyword() -> None:
    PartialGenerated(_task())

    # ``mt=FromStep("dense_left")`` is the edge; ``task="left"`` is not a marker.
    assert StepRegistry().get("annotate_left").depends_on == {
        "mt": "partial_generated.dense_left"
    }


def test_a_partial_without_a_marker_keyword_wires_no_edge() -> None:
    PartialGenerated(_task())

    assert StepRegistry().get("dense_left").depends_on == {}


def test_a_terminal_partial_wires_every_task_kwarg_as_an_edge() -> None:
    PartialGenerated(_task())

    assert StepRegistry().get("merge").depends_on == {
        "left": "partial_generated.annotate_left",
        "right": "partial_generated.annotate_right",
    }


def test_current_state_is_the_provenance_metadata() -> None:
    pipeline = SampleQc(_task())
    state = pipeline.current_state()

    assert state["pipeline"] == "sample_qc"
    assert state["resources"] == ["calling_intervals", "gnomad_v4"]
    assert "sample_qc.filter_intervals" in state["steps"]


def _register(*steps: Step) -> StepRegistry:
    registry = StepRegistry()
    for step in steps:
        registry.add(step)
    return registry


def _step(name: str, **edges: object) -> Step:
    return Step(name=name, task=_task(), **edges)


def test_dependencies_of_a_linear_chain_is_the_whole_prefix() -> None:
    registry = _register(
        _step("a"),
        _step("b", depends_on={"a": "a"}),
        _step("c", depends_on={"b": "b"}),
    )

    assert registry.dependencies_of("c") == {"a", "b"}


def test_dependencies_of_a_diamond_is_the_deduplicated_union() -> None:
    registry = _register(
        _step("a"),
        _step("b", depends_on={"a": "a"}),
        _step("c", depends_on={"a": "a"}),
        _step("d", depends_on={"b": "b", "c": "c"}),
    )

    assert registry.dependencies_of("d") == {"a", "b", "c"}


def test_dependencies_of_a_root_step_is_empty() -> None:
    registry = _register(_step("a"), _step("b", depends_on={"a": "a"}))

    assert registry.dependencies_of("a") == set()


def test_a_present_optional_edge_is_followed() -> None:
    registry = _register(
        _step("a"),
        InputStep(name="b", task=_task(), source="a"),
    )

    assert registry.dependencies_of("b") == {"a"}


def test_an_absent_optional_edge_is_not_followed() -> None:
    registry = _register(
        _step("a"),
        InputStep(name="b", task=_task(), source="missing"),
    )

    assert registry.dependencies_of("b") == set()


def test_a_resources_edge_is_followed_like_a_depends_on_edge() -> None:
    registry = _register(
        _step("loader"),
        _step("consumer", resources={"res": "loader"}),
    )

    assert registry.dependencies_of("consumer") == {"loader"}


def test_a_pipeline_without_declared_resources_registers_none() -> None:
    class NoResources(Pipeline):
        @classmethod
        def name(cls) -> str:
            return "no_resources"

        @property
        def resources(self) -> tuple[Resource, ...]:
            return ()

        def get_input(self) -> tuple[InputStep, ...]:
            return ()

    pipeline = NoResources(_task())

    assert pipeline.resources == ()
    assert pipeline.current_state()["resources"] == []


# --- dependents_of: the forward closure (ticket 07) --------------------------


def test_dependents_of_a_linear_chain_is_the_whole_suffix() -> None:
    registry = _register(
        _step("a"),
        _step("b", depends_on={"a": "a"}),
        _step("c", depends_on={"b": "b"}),
    )

    assert registry.dependents_of("a") == {"b", "c"}


def test_dependents_of_a_diamond_is_the_deduplicated_union() -> None:
    registry = _register(
        _step("a"),
        _step("b", depends_on={"a": "a"}),
        _step("c", depends_on={"a": "a"}),
        _step("d", depends_on={"b": "b", "c": "c"}),
    )

    assert registry.dependents_of("a") == {"b", "c", "d"}


def test_dependents_of_a_leaf_is_empty() -> None:
    registry = _register(_step("a"), _step("b", depends_on={"a": "a"}))

    assert registry.dependents_of("b") == set()


def test_a_present_optional_edge_is_a_dependent() -> None:
    registry = _register(
        _step("a"),
        InputStep(name="b", task=_task(), source="a"),
    )

    assert registry.dependents_of("a") == {"b"}


def test_an_absent_optional_edge_is_not_a_dependent() -> None:
    registry = _register(
        _step("a"),
        InputStep(name="b", task=_task(), source="missing"),
    )

    assert registry.dependents_of("a") == set()


def test_a_resources_edge_is_a_dependent_like_a_depends_on_edge() -> None:
    registry = _register(
        _step("loader"),
        _step("consumer", resources={"res": "loader"}),
    )

    assert registry.dependents_of("loader") == {"consumer"}
