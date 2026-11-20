"""Hail-free unit tests for the VDS reporting collectors.

The four metadata collectors read only ``VariantDataset.params``, so they are
exercised here with lightweight stand-ins instead of a real dataset: building
``hl.Interval`` objects would start Hail/Spark. The three Hail-bearing
collectors aggregate real splits and are exercised in a Hail-capable
environment, so here we only pin their public surface.
"""

from pathlib import Path
from types import SimpleNamespace as NS

import pandas as pd
import plotnine as pn
import pytest

from gnomad_meb import reporting


@pytest.fixture
def variants():
    return pd.DataFrame(
        {
            "split": ["chr1", "chr2"],
            "n_variant_rows": [100, 200],
            "n_snps": [80, 150],
            "n_indels": [20, 50],
            "n_other": [0, 0],
        }
    )


def _interval(contig, start, end):
    return NS(
        start=NS(contig=contig, position=start),
        end=NS(contig=contig, position=end),
    )


@pytest.fixture
def dataset(tmp_path):
    tasks = (
        NS(
            name="chr1",
            intervals=(
                _interval("chr1", 1, 1_200_000),
                _interval("chr1", 1_200_001, 2_400_000),
            ),
        ),
        NS(name="chr2", intervals=(_interval("chr2", 1, 2_000_000),)),
    )
    params = NS(
        name="vds",
        base=Path(tmp_path),
        gvcfs=(
            "/data/a/A.gvcf.gz",
            "/data/b/B.gvcf.gz",
            "s3://bucket/c/C.gvcf.gz",
        ),
        tasks=tasks,
        reference_genome="GRCh38",
        interval_size=1_200_000,
        n_splits=0,
        branch_factor=100,
        target_records=24_000,
        overwrite=False,
        task_status={"chr1": ("finished", ""), "chr2": ("error", "boom")},
    )
    return NS(params=params)


def test_build_parameters(dataset, tmp_path):
    params = reporting.build_parameters(dataset)
    assert params["name"] == "vds"
    assert params["base"] == str(tmp_path)
    assert params["reference_genome"] == "GRCh38"
    assert params["n_gvcfs"] == 3
    assert params["n_tasks"] == 2


def test_interval_coverage(dataset):
    frame = reporting.interval_coverage(dataset)
    assert frame["split"].tolist() == ["chr1", "chr2"]
    assert frame["n_intervals"].tolist() == [2, 1]
    assert frame["total_bp"].tolist() == [2_400_000, 2_000_000]
    assert frame.iloc[0]["first_contig"] == "chr1"
    assert frame.iloc[0]["last_pos"] == 2_400_000


def test_task_status(dataset):
    frame = reporting.task_status(dataset)
    assert frame["split"].tolist() == ["chr1", "chr2"]
    assert frame["status"].tolist() == ["finished", "error"]
    assert frame.iloc[1]["message"] == "boom"


def test_hail_collectors_are_exposed():
    for name in (
        "split_variant_stats",
        "split_reference_stats",
        "samples_summary",
    ):
        assert callable(getattr(reporting, name))


# --- plot builders ----------------------------------------------------------


def test_intervals_per_split_orders_contigs_semantically():
    frame = pd.DataFrame({"split": ["chr10", "chr2"], "n_intervals": [3, 1]})
    figure = reporting.intervals_per_split(frame)
    assert isinstance(figure, pn.ggplot)
    assert figure.data["n_intervals"].tolist() == [3, 1]
    assert list(figure.data["split"].cat.categories) == ["chr2", "chr10"]


def test_intervals_per_split_orders_task_names_by_id():
    frame = pd.DataFrame({"split": ["task-10", "task-2"], "n_intervals": [3, 1]})
    figure = reporting.intervals_per_split(frame)
    assert list(figure.data["split"].cat.categories) == ["task-2", "task-10"]


def test_variant_class_by_split(variants):
    figure = reporting.variant_class_by_split(variants)
    assert isinstance(figure, pn.ggplot)
    assert set(figure.data["variant_class"]) == {"n_snps", "n_indels", "n_other"}
    assert list(figure.data["split"].cat.categories) == ["chr1", "chr2"]


def test_split_sizes(variants):
    figure = reporting.split_sizes(variants)
    assert isinstance(figure, pn.ggplot)
    assert figure.data["n_variant_rows"].tolist() == [100, 200]
