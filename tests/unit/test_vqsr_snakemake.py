"""Unit tests for the VQSR Snakemake config assembly.

The VQSR workflow itself is a Snakemake DAG covered by the container-only
integration suite (ADR-0002); these tests exercise the hail-free, pure-Python
seam: :func:`gnomad_meb.vqsr.vqsr._make_config` turning the per-task inputs the
stage computed into the ordered config the rules read. The order of the task
list (and of the per-task ``as_info`` / ``calling_intervals`` maps) is
load-bearing: it fixes the ``bcftools concat`` input order, so the concatenated
``site_only.vcf.bgz`` is deterministic.
"""

from __future__ import annotations

import importlib

import pytest
import yaml

from gnomad_meb.config import Settings
from gnomad_meb.vqsr.vqsr import _make_config

#: The module object itself: ``gnomad_meb.vqsr.__init__`` re-exports the
#: ``vqsr`` function, shadowing the submodule name, so import it explicitly.
vqsr_module = importlib.import_module("gnomad_meb.vqsr.vqsr")

#: Deliberately not in alphabetical order, so a ``sort_keys=True`` dump (or a
#: set-backed container) would reorder ``chr2`` before ``chr1`` and fail.
TASK_INPUTS = {
    "chr2": {
        "as_info": "/data/as_info/task-chr2/as_info.vcf.bgz",
        "calling_intervals": "/work/vqsr/task-chr2/calling_intervals.interval_list",
    },
    "chr1": {
        "as_info": "/data/as_info/task-chr1/as_info.vcf.bgz",
        "calling_intervals": "/work/vqsr/task-chr1/calling_intervals.interval_list",
    },
}


def _settings(**overrides: object) -> Settings:
    base = {
        "project": "proj",
        "checkpoint": "/ckpt",
        "validate_paths": False,
        "vqsr": {
            "output_dir": "/out/vqsr",
            "log_dir": "/logs",
            "resources": {
                "hapmap": "/r/hapmap",
                "omni": "/r/omni",
                "thousand_genome": "/r/1000g",
                "dbsnp": "/r/dbsnp",
                "mills": "/r/mills",
                "axiom_poly": "/r/axiom",
            },
        },
    }
    base.update(overrides)
    return Settings(**base)


@pytest.fixture(autouse=True)
def _config_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    settings = _settings()
    monkeypatch.setattr(vqsr_module, "get_config", lambda: settings)
    return settings


def test_make_config_preserves_task_order(tmp_path) -> None:
    target = tmp_path / "config.yaml"

    result = _make_config(TASK_INPUTS, "/out/vqsr", target=target)

    assert result == target
    config = yaml.safe_load(target.read_text())
    # insertion order, not sorted order
    assert config["tasks"] == ["chr2", "chr1"]
    assert list(config["as_info"]) == ["chr2", "chr1"]
    assert list(config["calling_intervals"]) == ["chr2", "chr1"]


def test_make_config_carries_the_stage_inputs(tmp_path) -> None:
    target = tmp_path / "config.yaml"

    _make_config(TASK_INPUTS, "/out/vqsr", target=target)

    config = yaml.safe_load(target.read_text())
    assert config["output_dir"] == "/out/vqsr"
    assert config["as_info"]["chr2"] == (
        "/data/as_info/task-chr2/as_info.vcf.bgz"
    )
    assert config["calling_intervals"]["chr1"] == (
        "/work/vqsr/task-chr1/calling_intervals.interval_list"
    )
    # the wrapper dir and resources come from settings
    assert config["wrapper_dir"].endswith("vqsr/wrapper")
    assert config["log_dir"] == "/logs"
    assert config["resources"]["hapmap"] == "/r/hapmap"


def test_make_config_drops_the_old_scatter_keys(tmp_path) -> None:
    target = tmp_path / "config.yaml"

    _make_config(TASK_INPUTS, "/out/vqsr", target=target)

    config = yaml.safe_load(target.read_text())
    # the split set is the scatter now: no ``scatter`` knob and no single
    # ``site_only_vcf`` input (the concat rule builds it from the tasks)
    assert "scatter" not in config
    assert "site_only_vcf" not in config
    # ``calling_intervals`` is now a per-task map, not the old global path
    assert config["calling_intervals"] == {
        "chr2": "/work/vqsr/task-chr2/calling_intervals.interval_list",
        "chr1": "/work/vqsr/task-chr1/calling_intervals.interval_list",
    }


def test_make_config_carries_the_variant_qc_tunables(tmp_path) -> None:
    """The VQSR cutoffs/params are projected from ``[variant_qc]`` settings.

    No ``variant_qc`` section is configured here, so the ``VariantQcConfig``
    defaults supply them.
    """
    target = tmp_path / "config.yaml"

    _make_config(TASK_INPUTS, "/out/vqsr", target=target)

    config = yaml.safe_load(target.read_text())
    assert config["target_truth_sensitivity"] == {"snp": 99.7, "indel": 99.0}
    assert config["vqsr_params"]["allele_specific"] is True
    assert config["vqsr_params"]["max_gaussians"] == 4
    assert config["vqsr_params"]["variants_to_sample"] == 10
    assert config["vqsr_params"]["snp_tranche_values"][:3] == [100.0, 99.95, 99.9]
    assert config["vqsr_params"]["indel_tranche_values"][:3] == [
        100.0,
        99.95,
        99.9,
    ]


def test_make_config_requires_vqsr_resources(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(vqsr={"output_dir": "/out/vqsr", "log_dir": "/logs"})
    monkeypatch.setattr(vqsr_module, "get_config", lambda: settings)

    with pytest.raises(AssertionError):
        _make_config(TASK_INPUTS, "/out/vqsr", target=tmp_path / "config.yaml")
