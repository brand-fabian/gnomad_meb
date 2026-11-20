"""Tests for the browser-export pipeline steps.

The ``browser_sites`` step is a ``HAIL_TABLE`` checkpoint at
``{project}/browser/sites.ht`` that reshapes ``mt.rows()`` into the browser's
sites-only schema, keeps the source MatrixTable's globals
(``freq_meta``/``gnomad_freq_meta`` stay globals, never row columns), and
annotates one ``browser_export`` provenance global (the enabled optional field
set, the metadata globals kept and ``checkpoint_static()``).

The declaration tests run without Hail; the provenance tests use the local
single-core Hail session from the conftest ``hl`` fixture and a small in-memory
MatrixTable carrying every required source column, then read the written
checkpoint back.
"""

from __future__ import annotations

import pytest

from gnomad_meb.cmd.task import BrowserExportRun, Task
from gnomad_meb.config import BrowserExportConfig, Settings, set_config
from gnomad_meb.lib.browser import BrowserExportPipeline
from gnomad_meb.pipeline import StepRegistry


def _settings(
    *,
    project: str = "proj",
    checkpoint: str | None = "/ckpt",
    fields: list[str] | None = None,
) -> Settings:
    return Settings(
        validate_paths=False,
        project=project,
        checkpoint=checkpoint,
        browser_export=BrowserExportConfig(fields=list(fields or ())),
    )


# --- step declaration --------------------------------------------------------


def test_browser_sites_step_id() -> None:
    """The reshape step is ``browser_export.browser_sites``."""
    BrowserExportPipeline(
        Task(settings=_settings(), command=BrowserExportRun(mt="in.mt"))
    )

    step = StepRegistry().get("browser_export.browser_sites")

    assert step is not None
    assert step.id == "browser_export.browser_sites"


def test_browser_sites_declares_the_sites_hail_table() -> None:
    """The step declares exactly ``{checkpoint}/{project}/browser/sites.ht``."""
    settings = _settings()
    BrowserExportPipeline(Task(settings=settings, command=BrowserExportRun(mt="in.mt")))

    step = StepRegistry().get("browser_export.browser_sites")

    assert [artifact.prefix for artifact in step.artifacts] == [
        "{project}/browser/sites"
    ]
    assert [artifact.path(settings) for artifact in step.artifacts] == [
        "/ckpt/proj/browser/sites.ht"
    ]


def test_browser_sites_depends_on_the_mt_input() -> None:
    """The step's ``mt`` parameter is the ``RunInput("mt")`` endpoint edge."""
    BrowserExportPipeline(
        Task(settings=_settings(), command=BrowserExportRun(mt="in.mt"))
    )

    step = StepRegistry().get("browser_export.browser_sites")

    assert step.depends_on["mt"] == "browser_export.mt"


# --- provenance global -------------------------------------------------------


def _meta_expr(hl, meta: list[dict[str, str]]):
    """An ``Array[Struct]`` ``freq_meta`` literal; absent keys become null."""
    return hl.array(
        [
            hl.struct(
                group=m.get("group", hl.missing(hl.tstr)),
                gen_anc=m.get("gen_anc", hl.missing(hl.tstr)),
                sex=m.get("sex", hl.missing(hl.tstr)),
                cohort=m.get("cohort", hl.missing(hl.tstr)),
            )
            for m in meta
        ]
    )


def _hist(hl, marker: int):
    """One source ``hl.agg.hist`` struct."""
    return hl.struct(
        bin_edges=hl.array([0.0, 1.0]),
        bin_freq=hl.array([marker]),
        n_larger=marker,
        n_smaller=marker,
    )


def _histograms(hl):
    """The source ``histograms`` struct: ``{qual_hists, raw_qual_hists}``."""

    def group(offset: int):
        return hl.struct(
            gq_hist_all=_hist(hl, offset + 1),
            gq_hist_alt=_hist(hl, offset + 2),
            dp_hist_all=_hist(hl, offset + 3),
            dp_hist_alt=_hist(hl, offset + 4),
            ab_hist_alt=_hist(hl, offset + 5),
        )

    return hl.struct(qual_hists=group(0), raw_qual_hists=group(10))


def _coverage_element(hl):
    """One source ``coverage_stats`` element."""
    return hl.struct(
        mean=1.0,
        median_approx=1,
        total_DP=hl.int64(1),
        **{f"over_{x}": 0.5 for x in (100, 50, 30, 25, 20, 15, 10, 5, 1)},
    )


def _info(hl):
    """The site/AS INFO struct the site-quality reshape consumes."""
    return hl.struct(
        QUALapprox=hl.int64(100),
        AS_FS=0.5,
        AS_MQ=60.0,
        AS_MQRankSum=0.1,
        AS_pab_max=1.0,
        AS_QUALapprox=hl.int64(90),
        AS_QD=10.0,
        AS_ReadPosRankSum=0.0,
        AS_SOR=0.7,
        AS_VarDP=hl.int32(10),
        AS_VQSLOD=1.0,
    )


def _mt(hl):
    """A one-row MT carrying every required source column the reshape consumes."""
    mt = hl.MatrixTable.from_parts(
        rows={
            "locus": [hl.locus("chr1", 100, reference_genome="GRCh38")],
            "alleles": [["A", "T"]],
            "freq": [hl.array([hl.struct(AC=1, AN=2, AF=0.5, homozygote_count=0)])],
            "grpmax": [hl.struct(AC=1, AN=2, AF=0.5, gen_anc="afr")],
            "fafmax": [hl.struct(faf95_max=0.1, faf95_max_gen_anc="afr")],
            "histograms": [_histograms(hl)],
            "info": [_info(hl)],
            "inbreeding_coef": [hl.float64(0.01)],
            "filters": [hl.literal({"PASS"}, hl.tset(hl.tstr))],
            "coverage_stats": [hl.array([_coverage_element(hl)])],
        },
        cols={"s": ["S1"]},
        entries={"GT": [[hl.call(0, 1)]]},
    )
    mt = mt.key_rows_by("locus", "alleles")
    return mt.annotate_globals(
        freq_meta=_meta_expr(hl, [{"group": "adj"}]),
        coverage_stats_meta=hl.array([hl.literal({}, hl.tdict(hl.tstr, hl.tstr))]),
    )


def test_browser_sites_annotates_the_provenance_global(hl, tmp_path) -> None:
    """The written ``sites.ht`` carries a well-formed ``browser_export`` global."""
    fields = ["gnomad_freq"]
    settings = _settings(checkpoint=str(tmp_path), fields=fields)
    set_config(settings)
    pipeline = BrowserExportPipeline(
        Task(settings=settings, command=BrowserExportRun(mt="in.mt"))
    )

    table = pipeline.browser_sites(mt=_mt(hl))

    assert "browser_export" in table.globals.dtype.fields
    assert "browser_export" not in table.row.dtype.fields
    assert set(table.browser_export.dtype.fields) >= {
        "fields",
        "metadata_globals",
        "version",
        "git_revision",
        "host",
    }

    provenance = hl.eval(table.browser_export)
    assert list(provenance.fields) == fields
    assert list(provenance.metadata_globals) == ["freq_meta"]


def test_browser_sites_provenance_records_every_kept_metadata_global(
    hl, tmp_path
) -> None:
    """``freq_meta``/``gnomad_freq_meta`` both present are both recorded."""
    settings = _settings(checkpoint=str(tmp_path))
    set_config(settings)
    pipeline = BrowserExportPipeline(
        Task(settings=settings, command=BrowserExportRun(mt="in.mt"))
    )
    mt = _mt(hl).annotate_globals(gnomad_freq_meta=hl.array([hl.struct(group="adj")]))

    table = pipeline.browser_sites(mt=mt)

    assert list(hl.eval(table.browser_export).metadata_globals) == [
        "freq_meta",
        "gnomad_freq_meta",
    ]


def test_browser_sites_populates_the_synthesized_columns(hl, tmp_path) -> None:
    """The reshape writes populated ``joint``/``genome``/``coverage`` columns."""
    settings = _settings(checkpoint=str(tmp_path))
    set_config(settings)
    pipeline = BrowserExportPipeline(
        Task(settings=settings, command=BrowserExportRun(mt="in.mt"))
    )

    table = pipeline.browser_sites(mt=_mt(hl))
    row = table.collect()[0]

    assert row.joint.freq.all.ac == 1
    assert row.joint.grpmax.gen_anc == "afr"
    assert row.joint.fafmax.faf95_max == pytest.approx(0.1)
    assert row.joint.flags == ["PASS"]
    assert row.genome.quality_metrics.genotype_quality.all_adj.n_larger == 1
    assert row.coverage.genome.mean == pytest.approx(1.0)
    assert tuple(table.genome.dtype["freq"].fields) == ("all",)


def test_browser_sites_keeps_source_globals_and_drops_consumed_rows(
    hl, tmp_path
) -> None:
    """``freq_meta``/``freq`` stay globals, never row columns, on ``sites.ht``."""
    settings = _settings(checkpoint=str(tmp_path))
    set_config(settings)
    pipeline = BrowserExportPipeline(
        Task(settings=settings, command=BrowserExportRun(mt="in.mt"))
    )

    table = pipeline.browser_sites(mt=_mt(hl))

    assert "freq_meta" in table.globals.dtype.fields
    assert "freq_meta" not in table.row.dtype.fields
    assert "freq" not in table.row.dtype.fields
    assert "grpmax" not in table.row.dtype.fields
    assert "histograms" not in table.row.dtype.fields


def test_browser_sites_resumes_from_the_written_checkpoint(hl, tmp_path) -> None:
    """A second call reads the checkpoint back with the provenance intact."""
    settings = _settings(checkpoint=str(tmp_path), fields=["gnomad_freq"])
    set_config(settings)
    pipeline = BrowserExportPipeline(
        Task(settings=settings, command=BrowserExportRun(mt="in.mt"))
    )

    pipeline.browser_sites(mt=_mt(hl))
    resumed = pipeline.browser_sites(mt=_mt(hl))

    assert list(hl.eval(resumed.browser_export).fields) == ["gnomad_freq"]
