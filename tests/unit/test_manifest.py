"""The run envelope: :class:`RunManifest`, its builder and its loader.

Every inspection flag and the ``run`` command speak one declarative model. These
tests pin the envelope's legal shapes (a pipeline name iff runnable), the builder
that projects a task and optional steps onto it, the settings-excluding step
dumps and the re-injection that makes them loadable again, and ``load_payload``'s
JSON/YAML sniffing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gnomad_meb.cmd.output import (
    load_manifest,
    load_payload,
    manifest_payload,
    parse_manifest,
)
from gnomad_meb.cmd.task import PipelineStep, Task
from gnomad_meb.config import Settings
from gnomad_meb.pipeline import RunInput
from gnomad_meb.pipeline.execution import PipelineRunner
from gnomad_meb.pipeline.spec import RunManifest, Step


def _task() -> Task:
    return Task(
        settings=Settings(validate_paths=False, project="demo"),
        command=PipelineStep(name="test"),
    )


def _step() -> Step:
    return Step(name="first", task=_task(), depends_on={"v": "producer"})


# --- the model ---------------------------------------------------------------


def test_a_pipeline_backed_envelope_validates() -> None:
    payload = manifest_payload("chain", _task())

    manifest = RunManifest.model_validate(payload)

    assert manifest.pipeline == "chain"
    assert manifest.runnable is True
    assert manifest.steps is None
    assert manifest.overwrite is False


def test_a_pipelineless_envelope_validates_but_is_not_runnable() -> None:
    payload = manifest_payload(None, _task())

    manifest = RunManifest.model_validate(payload)

    assert manifest.pipeline is None
    assert manifest.runnable is False


def test_pipeline_none_with_runnable_true_is_rejected() -> None:
    payload = manifest_payload("chain", _task())
    payload["pipeline"] = None

    with pytest.raises(ValueError, match="runnable"):
        RunManifest.model_validate(payload)


def test_pipeline_named_with_runnable_false_is_rejected() -> None:
    payload = manifest_payload("chain", _task())
    payload["runnable"] = False

    with pytest.raises(ValueError, match="runnable"):
        RunManifest.model_validate(payload)


def test_overwrite_is_carried_through_the_envelope() -> None:
    payload = manifest_payload("chain", _task(), overwrite=True)

    manifest = RunManifest.model_validate(payload)

    assert manifest.overwrite is True


# --- the step dump and re-injection ------------------------------------------


def test_a_step_dump_omits_the_task_settings() -> None:
    dump = _step().model_dump(mode="json", exclude={"task": {"settings"}})

    assert "settings" not in dump["task"]


def test_a_step_dump_without_settings_is_not_a_valid_step() -> None:
    dump = _step().model_dump(mode="json", exclude={"task": {"settings"}})

    with pytest.raises(ValueError, match="settings"):
        Step.model_validate(dump)


def test_parsing_reinjects_the_root_settings_into_every_step() -> None:
    payload = manifest_payload(
        "chain",
        _task(),
        steps=[_step().model_dump(mode="json", exclude={"task": {"settings"}})],
    )

    manifest = parse_manifest(payload)

    assert manifest.steps is not None
    assert manifest.steps[0].task.settings.project == "demo"
    assert manifest.steps[0].depends_on == {"v": "producer"}


def test_parsing_a_manifest_without_steps_leaves_steps_none() -> None:
    manifest = parse_manifest(manifest_payload("chain", _task()))

    assert manifest.steps is None


# --- load_payload ------------------------------------------------------------


def test_load_payload_reads_a_json_file(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    payload = manifest_payload("chain", _task())
    path.write_text(json.dumps(payload))

    loaded = load_payload(path)

    assert loaded == payload


def test_load_payload_reads_a_yaml_file(tmp_path: Path) -> None:
    import yaml

    path = tmp_path / "manifest.yml"
    payload = manifest_payload("chain", _task())
    path.write_text(yaml.safe_dump(payload, sort_keys=False))

    loaded = load_payload(path)

    assert loaded == json.loads(json.dumps(payload))


def test_load_manifest_reads_a_file_into_a_validated_model(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest_payload("chain", _task())))

    manifest = load_manifest(path)

    assert manifest.pipeline == "chain"
    assert manifest.runnable is True


# --- PipelineRunner.manifest() -----------------------------------------------


def _runner() -> PipelineRunner:
    from gnomad_meb.lib.checkpoint import CHECKPOINT_TYPE, checkpoint
    from gnomad_meb.pipeline import InputStep, Pipeline

    class Demo(Pipeline):
        @classmethod
        def name(cls) -> str:
            return "demo"

        @property
        def resources(self) -> tuple:
            return ()

        def get_input(self) -> tuple[InputStep, ...]:
            return (InputStep(name="vds", task=self.task, group="demo"),)

        @checkpoint("{project}/demo/first", CHECKPOINT_TYPE.HAIL_TABLE)
        def first(self, vds=RunInput("vds")):  # noqa: B008
            return vds

    return PipelineRunner(_task(), [Demo(_task())])


def test_runner_manifest_is_the_envelope() -> None:
    manifest = _runner().manifest()

    assert manifest["pipeline"] == "demo"
    assert manifest["runnable"] is True
    assert manifest["overwrite"] is False
    assert [step["name"] for step in manifest["steps"]] == ["vds", "first"]
    # The step dumps omit settings, so the whole envelope is not yet loadable
    # until parse_manifest re-injects the root task's settings.
    assert "settings" not in manifest["steps"][0]["task"]
    assert parse_manifest(manifest).pipeline == "demo"


def test_runner_manifest_carries_the_overwrite_flag() -> None:
    from gnomad_meb.pipeline import PipelineRunner

    manifest = PipelineRunner(_task(), _runner().pipelines, overwrite=True).manifest()

    assert manifest["overwrite"] is True
