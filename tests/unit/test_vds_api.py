"""In-process tests for the single, authoritative VDS API surface.

``gnomad_meb.lib`` must resolve again (the old ``vds`` module used to shadow the
``vds`` package by name), expose the class-based API, and expose no retired
function-based name. See ``.scratch/vds-dataset-migration``.
"""

from __future__ import annotations

import importlib

import pytest


def test_library_package_imports() -> None:
    """The library package resolves despite the old module/package name clash."""
    lib = importlib.import_module("gnomad_meb.lib")

    assert lib is not None


@pytest.mark.parametrize(
    "name",
    [
        "checkpoint",
        "glob",
        "VariantDataset",
        "VariantDatasetParameters",
        "VariantDatasetTaskIntervals",
        "VdsLocation",
    ],
)
def test_new_vds_api_is_exported(name: str) -> None:
    lib = importlib.import_module("gnomad_meb.lib")

    assert hasattr(lib, name), f"missing export: {name}"


@pytest.mark.parametrize("name", ["create_vds", "inspect_vds"])
def test_retired_vds_names_are_gone(name: str) -> None:
    lib = importlib.import_module("gnomad_meb.lib")

    assert not hasattr(lib, name), f"retired export still present: {name}"
    assert name not in lib.__all__


def test_glob_helper_lives_in_utils() -> None:
    utils = importlib.import_module("gnomad_meb.lib.utils")

    assert callable(utils.glob)


def test_retired_reader_is_gone_from_hail() -> None:
    """Downstream stages read the dataset handle, not the old function reader."""
    hail_module = importlib.import_module("gnomad_meb.hail")

    assert not hasattr(hail_module, "read_vds")


@pytest.mark.parametrize(
    "name",
    [
        "filter_intervals",
        "filter_chromosomes",
        "filter_variants",
        "filter_samples",
        "filter_rows",
        "split_multi",
        "select_tasks",
        "filter_contigs",
        "to_dense_mt",
    ],
)
def test_per_split_transform_methods_are_exposed(name: str) -> None:
    """QC keeps transforms per split through wrappers on the handle."""
    dataset = importlib.import_module("gnomad_meb.lib.vds").VariantDataset

    assert hasattr(dataset, name), f"missing transform method: {name}"


@pytest.mark.parametrize("name", ["read_vds", "read"])
def test_retired_dataset_readers_are_gone(name: str) -> None:
    """The handle exposes ``open``/``to_hail``; the old reader names are retired."""
    dataset = importlib.import_module("gnomad_meb.lib.vds").VariantDataset

    assert not hasattr(dataset, name), f"retired dataset reader still present: {name}"
