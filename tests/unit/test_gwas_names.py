"""Tests for the dataset key value object (``lib/gwas/names``).

``DatasetKey`` owns a ``(table, region)`` identity and the names every GWAS
stage derives from it: the dataset label, its path segment, its step id and its
checkpoint prefix. These pin the empty-table rule (a raw ``--mt`` reproduces the
single-table names) and the table/region/empty shapes. Pure Python: no hail.
"""

from __future__ import annotations

import pytest

from gnomad_meb.lib.gwas.constants import INTERMEDIATE_PREFIX, REGENIE_PREFIX
from gnomad_meb.lib.gwas.names import DatasetKey


def test_of_normalizes_falsy_parts() -> None:
    assert DatasetKey.of("", "") == DatasetKey()
    assert DatasetKey.of("T", "R") == DatasetKey(table="T", region="R")


def test_label_and_segment_for_a_table_and_region() -> None:
    key = DatasetKey.of("filter_rare_lof", "gene")

    assert key.label == "filter_rare_lof.gene"
    assert key.segment == "filter_rare_lof/gene"


def test_label_and_segment_for_a_region_only_key() -> None:
    key = DatasetKey.of("", "gene")

    assert key.label == "gene"
    assert key.segment == "gene"


def test_label_and_segment_for_an_empty_key() -> None:
    key = DatasetKey.of("", "")

    assert key.label == ""
    assert key.segment == ""


def test_step_suffixes_the_kind_with_the_label() -> None:
    key = DatasetKey.of("filter_rare_lof", "gene")

    assert key.step("run_gwas_pipeline") == "run_gwas_pipeline.filter_rare_lof.gene"
    assert key.step("export_regenie") == "export_regenie.filter_rare_lof.gene"


def test_step_for_a_region_only_key() -> None:
    assert DatasetKey.of("", "gene").step("mt") == "mt.gene"


def test_step_is_bare_for_the_empty_key() -> None:
    assert DatasetKey.of("", "").step("run_gwas_pipeline") == "run_gwas_pipeline"


def test_prefix_for_a_table_and_region() -> None:
    key = DatasetKey.of("filter_rare_lof", "gene")

    assert key.prefix("gwas_base") == (
        f"{INTERMEDIATE_PREFIX}/filter_rare_lof/gwas_base"
    )
    assert key.prefix("regenie_input") == (
        f"{INTERMEDIATE_PREFIX}/filter_rare_lof/gene"
    )
    assert key.prefix("load_gwas") == (
        f"{REGENIE_PREFIX}/filter_rare_lof/gene/parts/gwas"
    )
    assert key.prefix("merge_gwas") == (f"{REGENIE_PREFIX}/filter_rare_lof/gene/gwas")


def test_prefix_for_a_region_only_key() -> None:
    key = DatasetKey.of("", "gene")

    assert key.prefix("gwas_base") == f"{INTERMEDIATE_PREFIX}/gwas_base"
    assert key.prefix("regenie_input") == f"{INTERMEDIATE_PREFIX}/gene"
    assert key.prefix("load_burden") == f"{REGENIE_PREFIX}/gene/parts/burden"
    assert key.prefix("merge_burden") == f"{REGENIE_PREFIX}/gene/burden"


def test_prefix_for_the_empty_key() -> None:
    key = DatasetKey.of("", "")

    assert key.prefix("gwas_base") == f"{INTERMEDIATE_PREFIX}/gwas_base"
    assert key.prefix("load_gwas") == f"{REGENIE_PREFIX}/parts/gwas"
    assert key.prefix("merge_gwas") == f"{REGENIE_PREFIX}/gwas"


def test_base_chain_prefixes_are_table_scoped_and_ignore_the_region() -> None:
    table_key = DatasetKey.of("T", "")
    region_key = DatasetKey.of("T", "gene")

    for kind in (
        "gwas_base",
        "normalize_regenie",
        "filter_regenie_chromosomes",
        "phenotype",
        "covariates",
        "aaf",
    ):
        assert table_key.prefix(kind) == region_key.prefix(kind)


def test_unknown_prefix_kind_fails_loudly() -> None:
    with pytest.raises(ValueError):
        DatasetKey.of("T", "gene").prefix("not_a_kind")


def test_parse_inverts_label() -> None:
    for key in (
        DatasetKey.of("filter_rare_lof", "gene"),
        DatasetKey.of("", "gene"),
        DatasetKey.of("", ""),
    ):
        assert DatasetKey.parse(key.label) == key


def test_parse_recovers_a_table_label_whole() -> None:
    assert DatasetKey.parse("filter_rare_lof.gene") == DatasetKey(
        table="filter_rare_lof", region="gene"
    )
    assert DatasetKey.parse("gene") == DatasetKey(region="gene")
    assert DatasetKey.parse("") == DatasetKey()
