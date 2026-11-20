"""Feature-closeout acceptance for the browser export.

Three seams, all at public boundaries:

- **Config example** -- the worked ``[browser_export]`` block in
  ``config/gnomad_meb.orig.toml`` declares the default optional field set and
  resolves into :class:`Settings`.
- **Determinism** -- two constructions of ``browser_export`` yield the same step
  ids, one ``export_contig.<contig>`` per canonical contig (ADR-0017), checked
  against a hard-coded expectation rather than only against each other.
- **Acceptance** -- a real run of the ``browser_sites`` step and the per-contig
  export steps over a small in-memory MatrixTable carrying every required source
  column, with the genohype subprocess the only thing mocked (it writes real,
  small pyarrow parquet files), asserting the browser contract: the mandatory
  columns are present on ``sites.ht``, the synthesized ``joint`` /
  ``genome.quality_metrics`` / ``coverage`` columns are populated, ``xpos`` is
  materialized, ``genome.freq.all`` holds the cohort numbers, and every exported
  contig has a provenance JSON.

No test executes the genohype binary. Loadability through the browser's DuckDB
backend is the deferred, container-gated check documented in ``README.md``.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from gnomad_meb.cmd.task import BrowserExportRun, Task
from gnomad_meb.config import Settings, set_config
from gnomad_meb.config import resolve as resolve_config
from gnomad_meb.constants import DEFAULT_FIELDS, MANDATORY_FIELDS
from gnomad_meb.lib.browser import BrowserExportPipeline
from gnomad_meb.lib.browser import genohype as genohype_module
from gnomad_meb.lib.browser.constants import (
    CONTIG_FILENAME_TEMPLATE,
    CONTIG_ORDER,
    PARQUET_DATASET_NAME,
)
from gnomad_meb.lib.browser.export import BrowserExportError
from gnomad_meb.lib.browser.genohype import GenohypeExportResult
from gnomad_meb.pipeline import PipelineRunner, StepRegistry

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG = REPO_ROOT / "config" / "gnomad_meb.orig.toml"

_EXPORT_STEM = "browser_export.export_contig"

#: The full browser_export step set, in declaration order: one ``mt`` endpoint,
#: the ``browser_sites`` reshape and one export step per canonical contig.
_EXPECTED_STEP_IDS = (
    "browser_export.mt",
    "browser_export.browser_sites",
    *(f"{_EXPORT_STEM}.{contig}" for contig in CONTIG_ORDER),
)


def _settings(*, checkpoint: str | None = "/ckpt") -> Settings:
    """The default browser-export settings (``[browser_export].fields`` default)."""
    return Settings(validate_paths=False, project="proj", checkpoint=checkpoint)


def _task(settings: Settings, mt: str | None = "in.mt") -> Task:
    return Task(settings=settings, command=BrowserExportRun(mt=mt))


# --- the worked config example -----------------------------------------------


def test_example_config_declares_the_default_fields() -> None:
    """The worked example states the optional default field set explicitly."""
    data = tomllib.loads(EXAMPLE_CONFIG.read_text())

    assert data["browser_export"]["fields"] == list(DEFAULT_FIELDS)


def test_example_config_parses_into_settings() -> None:
    """The example config resolves, and its ``[browser_export]`` section wins."""
    settings = resolve_config(EXAMPLE_CONFIG, overrides={"validate_paths": False})

    assert settings.browser_export.fields == list(DEFAULT_FIELDS)


# --- determinism (ADR-0017) --------------------------------------------------


def test_full_step_set_is_deterministic() -> None:
    """The step set is a pure function of the reference genome (ADR-0017)."""
    BrowserExportPipeline(_task(_settings()))
    first = StepRegistry().steps()

    StepRegistry._reset()
    BrowserExportPipeline(_task(_settings()))
    second = StepRegistry().steps()

    assert first == second
    assert tuple(second) == _EXPECTED_STEP_IDS


def test_export_step_declares_the_per_contig_parquet_and_sidecar() -> None:
    """Each export step declares its parquet plus ``.md`` provenance artifacts."""
    settings = _settings()
    BrowserExportPipeline(_task(settings))

    step = StepRegistry().get(f"{_EXPORT_STEM}.chr1")

    assert [artifact.prefix for artifact in step.artifacts] == [
        "{project}/browser/variants.parquet/locus.contig=chr1",
        "{project}/browser/variants.parquet/locus.contig=chr1.md",
    ]
    assert [artifact.path(settings) for artifact in step.artifacts] == [
        f"/ckpt/proj/browser/{PARQUET_DATASET_NAME}/"
        f"{CONTIG_FILENAME_TEMPLATE.format(contig='chr1')}.parquet",
        f"/ckpt/proj/browser/{PARQUET_DATASET_NAME}/"
        f"{CONTIG_FILENAME_TEMPLATE.format(contig='chr1')}.md.json.gz",
    ]
    assert step.depends_on == {"browser_sites": "browser_export.browser_sites"}


def test_a_missing_canonical_contig_fails_loudly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reference without a canonical contig is an error, not a silent skip."""
    from typing import ClassVar

    class _Reference:
        lengths: ClassVar[dict[str, int]] = {"chr1": 1, "chrX": 2}

    monkeypatch.setattr("gnomad_meb.hail.hail.get_reference", lambda name: _Reference())

    with pytest.raises(BrowserExportError, match="chrM"):
        BrowserExportPipeline(_task(_settings()))


# --- acceptance over a small in-memory MatrixTable ---------------------------


def _meta_expr(hl):
    """An ``Array[Struct]`` ``freq_meta``: overall adj + one Ancestry stratum."""
    return hl.array(
        [
            hl.struct(
                group="adj",
                gen_anc=hl.missing(hl.tstr),
                sex=hl.missing(hl.tstr),
                cohort=hl.missing(hl.tstr),
            ),
            hl.struct(
                group="adj",
                gen_anc="afr",
                sex=hl.missing(hl.tstr),
                cohort=hl.missing(hl.tstr),
            ),
        ]
    )


def _stats(hl, ac: int, an: int, af: float, homozygote_count: int):
    return hl.struct(AC=ac, AN=an, AF=af, homozygote_count=homozygote_count)


def _hist(hl, marker: int):
    return hl.struct(
        bin_edges=hl.array([0.0, 1.0]),
        bin_freq=hl.array([marker]),
        n_larger=marker,
        n_smaller=marker,
    )


def _histograms(hl, base: int):
    def group(offset: int):
        return hl.struct(
            gq_hist_all=_hist(hl, offset + 1),
            gq_hist_alt=_hist(hl, offset + 2),
            dp_hist_all=_hist(hl, offset + 3),
            dp_hist_alt=_hist(hl, offset + 4),
            ab_hist_alt=_hist(hl, offset + 5),
        )

    return hl.struct(qual_hists=group(base), raw_qual_hists=group(base + 10))


def _info(hl, qualapprox: int):
    """The site/AS INFO struct the site-quality reshape consumes."""
    return hl.struct(
        QUALapprox=hl.int64(qualapprox),
        AS_FS=0.5,
        AS_MQ=60.0,
        AS_MQRankSum=0.1,
        AS_pab_max=1.0,
        AS_QUALapprox=hl.int64(qualapprox - 10),
        AS_QD=10.0,
        AS_ReadPosRankSum=0.0,
        AS_SOR=0.7,
        AS_VarDP=hl.int32(10),
        AS_VQSLOD=1.0,
    )


def _coverage_element(hl, mean: float):
    return hl.struct(
        mean=mean,
        median_approx=1,
        total_DP=hl.int64(1),
        **{f"over_{x}": 0.5 for x in (100, 50, 30, 25, 20, 15, 10, 5, 1)},
    )


def _grpmax(hl, ac: int, an: int):
    return hl.struct(AC=ac, AN=an, AF=0.1, gen_anc="afr")


def _fafmax(hl, faf95: float):
    return hl.struct(
        faf95_max=faf95,
        faf95_max_gen_anc="nfe",
        faf99_max=0.02,
        faf99_max_gen_anc="nfe",
    )


def _mt(hl):
    """A two-row MatrixTable (chr1, chrX) carrying every required source column."""
    mt = hl.MatrixTable.from_parts(
        rows={
            "locus": [
                hl.locus("chr1", 100, reference_genome="GRCh38"),
                hl.locus("chrX", 200, reference_genome="GRCh38"),
            ],
            "alleles": [["A", "T"], ["C", "G"]],
            "freq": [
                hl.array([_stats(hl, 10, 100, 0.1, 2), _stats(hl, 3, 30, 0.1, 1)]),
                hl.array([_stats(hl, 5, 50, 0.1, 0), _stats(hl, 1, 10, 0.1, 0)]),
            ],
            "grpmax": [_grpmax(hl, 10, 100), _grpmax(hl, 5, 50)],
            "fafmax": [_fafmax(hl, 0.05), _fafmax(hl, 0.06)],
            "histograms": [_histograms(hl, 0), _histograms(hl, 20)],
            "info": [_info(hl, 100), _info(hl, 50)],
            "inbreeding_coef": [hl.float64(0.01), hl.float64(-0.02)],
            "filters": [
                hl.literal({"PASS"}, hl.tset(hl.tstr)),
                hl.literal({"a", "b"}, hl.tset(hl.tstr)),
            ],
            "coverage_stats": [
                hl.array([_coverage_element(hl, 1.0)]),
                hl.array([_coverage_element(hl, 2.0)]),
            ],
            "in_silico_predictors": [
                hl.struct(cadd=hl.struct(phred=2.0)),
                hl.struct(cadd=hl.struct(phred=3.0)),
            ],
        },
        cols={"s": ["S1"]},
        entries={"GT": [[hl.call(0, 1)], [hl.call(1, 1)]]},
    )
    mt = mt.key_rows_by("locus", "alleles")
    return mt.annotate_globals(
        freq_meta=_meta_expr(hl),
        coverage_stats_meta=hl.array([hl.literal({}, hl.tdict(hl.tstr, hl.tstr))]),
    )


def _to_py(value):
    """Convert a collected Hail value to plain Python (``Struct`` -> ``dict``)."""
    if hasattr(value, "items"):
        try:
            return {key: _to_py(item) for key, item in value.items()}
        except (AttributeError, TypeError):
            pass
    if isinstance(value, list | tuple):
        return [_to_py(item) for item in value]
    return value


def _fake_export(hl):
    """A genohype stand-in that writes the slice's real columns to parquet.

    It reads the temp Hail table the export wrote, converts the required nested
    columns to plain Python and writes them with pyarrow, so the promoted
    per-contig parquet carries the same nested schema genohype would emit.
    """
    import pyarrow as pa

    columns = ("xpos", "joint", "genome", "coverage")

    def fake(input_ht, output_parquet):
        table = hl.read_table(str(input_ht))
        present = [name for name in columns if name in table.row.dtype.fields]
        rows = [
            {name: _to_py(hl.eval(row)[name]) for name in present}
            for row in table.select(*present).collect()
        ]
        pq.write_table(pa.Table.from_pylist(rows), output_parquet)
        return GenohypeExportResult(
            version="genohype 9.9.9 (deadbee)",
            args=(
                "export",
                "parquet",
                str(input_ht),
                str(output_parquet),
                "--width",
                "full",
            ),
            num_rows=len(rows),
        )

    return fake


@pytest.fixture
def exported_browser_run(hl, tmp_path, monkeypatch):
    """Run the pipeline for chr1/chrX/chrY over a small MT; return the dirs."""
    monkeypatch.setattr(genohype_module, "export_parquet", _fake_export(hl))

    mt_path = str(tmp_path / "in.mt")
    _mt(hl).write(mt_path)

    settings = _settings(checkpoint=str(tmp_path))
    set_config(settings)
    task = _task(settings, mt=mt_path)
    pipeline = BrowserExportPipeline(task)
    runner = PipelineRunner(
        task,
        [pipeline],
        select=(f"{_EXPORT_STEM}.chr1", f"{_EXPORT_STEM}.chrX", f"{_EXPORT_STEM}.chrY"),
    )
    runner.run()

    browser_dir = tmp_path / "proj" / "browser"
    return {
        "hl": hl,
        "sites": browser_dir / "sites.ht",
        "dataset": browser_dir / PARQUET_DATASET_NAME,
    }


def test_acceptance_sites_table_has_the_mandatory_columns(exported_browser_run) -> None:
    """``sites.ht`` is the browser's fixed physical schema, in column order."""
    hl = exported_browser_run["hl"]
    sites = hl.read_table(str(exported_browser_run["sites"]))

    assert tuple(sites.row.dtype.fields)[: len(MANDATORY_FIELDS)] == MANDATORY_FIELDS
    assert "xpos" in sites.row.dtype.fields
    assert sites.xpos.dtype == hl.tint64
    assert "freq_meta" in sites.globals.dtype.fields
    assert "freq" not in sites.row.dtype.fields
    assert list(hl.eval(sites.browser_export).metadata_globals) == ["freq_meta"]


def test_acceptance_sites_table_materializes_xpos_and_genome_freq(
    exported_browser_run,
) -> None:
    """``xpos`` is computed and ``genome.freq.all`` holds the cohort numbers."""
    hl = exported_browser_run["hl"]
    sites = hl.read_table(str(exported_browser_run["sites"]))

    chr1 = sites.filter(sites.locus.contig == "chr1").collect()[0]
    chr_x = sites.filter(sites.locus.contig == "chrX").collect()[0]

    assert chr1.xpos == 1_000_000_100
    assert chr_x.xpos == 23_000_000_200
    assert chr1.genome.freq.all.ac == 10
    assert chr1.genome.freq.all.an == 100
    assert chr1.genome.freq.all.af == pytest.approx(0.1)
    assert tuple(sites.genome.dtype["freq"].fields) == ("all",)


def test_acceptance_sites_table_carries_joint_and_quality_metrics(
    exported_browser_run,
) -> None:
    """The synthesized ``joint``/``quality_metrics``/``coverage`` are populated."""
    hl = exported_browser_run["hl"]
    sites = hl.read_table(str(exported_browser_run["sites"]))

    chr1 = sites.filter(sites.locus.contig == "chr1").collect()[0]
    chr_x = sites.filter(sites.locus.contig == "chrX").collect()[0]

    assert [group.id for group in chr1.joint.freq.all.ancestry_groups] == ["afr"]
    assert chr1.joint.freq.all.ancestry_groups[0].ac == 3
    assert chr1.joint.grpmax.gen_anc == "afr"
    assert chr1.joint.fafmax.faf95_max == pytest.approx(0.05)
    assert chr1.joint.flags == ["PASS"]
    assert chr_x.joint.flags == ["a", "b"]
    assert chr1.genome.quality_metrics.genotype_quality.all_adj.n_larger == 1
    assert chr1.genome.quality_metrics.genotype_depth.all_raw.n_larger == 13
    site_quality = {
        metric.metric: metric.value
        for metric in chr1.genome.quality_metrics.site_quality_metrics
    }
    assert site_quality["SiteQuality"] == pytest.approx(100.0)
    assert site_quality["inbreeding_coeff"] == pytest.approx(0.01)
    assert chr1.genome.ac == 10
    assert chr1.genome.hemizygote_count == 0
    assert chr1.coverage.genome.mean == pytest.approx(1.0)
    assert "total_DP" not in sites.coverage.dtype["genome"].fields


def test_acceptance_joint_schema_is_fixed(exported_browser_run) -> None:
    """The required nested members survive with a fixed, input-independent dtype."""
    hl = exported_browser_run["hl"]
    sites = hl.read_table(str(exported_browser_run["sites"]))

    joint = sites.joint.dtype
    assert tuple(joint["freq"]["all"].fields) == (
        "ac",
        "an",
        "af",
        "homozygote_count",
        "hemizygote_count",
        "ancestry_groups",
    )
    assert tuple(joint["grpmax"].fields) == ("ac", "an", "af", "gen_anc")
    assert tuple(joint["fafmax"].fields) == ("faf95_max", "faf95_max_gen_anc")
    assert joint["flags"] == hl.tarray(hl.tstr)
    assert tuple(sites.genome.dtype["quality_metrics"].fields) == (
        "genotype_quality",
        "genotype_depth",
        "allele_balance",
        "site_quality_metrics",
    )


def test_acceptance_optional_fields_pass_through_and_drop(exported_browser_run) -> None:
    """The default passthrough is present; consumed/absent fields are not."""
    hl = exported_browser_run["hl"]
    sites = hl.read_table(str(exported_browser_run["sites"]))

    fields = sites.row.dtype.fields
    assert "in_silico_predictors" in fields
    assert "gnomad_freq" not in fields
    assert "grpmax" not in fields
    assert "fafmax" not in fields
    assert "histograms" not in fields
    assert "clinvar" not in fields


def test_acceptance_writes_one_parquet_per_selected_contig(
    exported_browser_run,
) -> None:
    """Every selected contig is promoted to its final ``.parquet`` (empty OK)."""
    dataset = exported_browser_run["dataset"]

    chr1 = dataset / f"{CONTIG_FILENAME_TEMPLATE.format(contig='chr1')}.parquet"
    chr_y = dataset / f"{CONTIG_FILENAME_TEMPLATE.format(contig='chrY')}.parquet"

    assert pq.read_metadata(str(chr1)).num_rows == 1
    assert pq.read_table(str(chr1)).column("xpos").to_pylist() == [1_000_000_100]
    assert pq.read_metadata(str(chr_y)).num_rows == 0


def test_acceptance_exported_parquet_has_the_nested_fields(
    exported_browser_run,
) -> None:
    """The required nested members survive parquet encoding with a fixed shape."""
    dataset = exported_browser_run["dataset"]
    chr1 = dataset / f"{CONTIG_FILENAME_TEMPLATE.format(contig='chr1')}.parquet"

    schema = pq.read_table(str(chr1)).schema
    joint = schema.field("joint").type
    all_stats = joint.field("freq").type.field("all").type
    assert set(all_stats.names) == {
        "ac",
        "an",
        "af",
        "homozygote_count",
        "hemizygote_count",
        "ancestry_groups",
    }
    assert set(all_stats.field("ancestry_groups").type.value_type.names) == {
        "id",
        "ac",
        "an",
        "af",
        "homozygote_count",
        "hemizygote_count",
    }
    assert set(joint.field("grpmax").type.names) == {"ac", "an", "af", "gen_anc"}
    assert set(joint.field("fafmax").type.names) == {
        "faf95_max",
        "faf95_max_gen_anc",
    }
    quality = (
        schema.field("genome")
        .type.field("quality_metrics")
        .type.field("genotype_quality")
        .type
    )
    assert set(quality.names) == {"all_adj", "alt_adj", "all_raw", "alt_raw"}
    coverage = schema.field("coverage").type.field("genome").type
    assert "mean" in coverage.names
    assert "total_DP" not in coverage.names


def test_acceptance_per_contig_provenance_records_the_contract(
    exported_browser_run,
) -> None:
    """Each contig's provenance records its source fields, args and counts."""
    import gzip
    import json

    dataset = exported_browser_run["dataset"]
    provenance_file = (
        dataset / f"{CONTIG_FILENAME_TEMPLATE.format(contig='chr1')}.md.json.gz"
    )

    provenance = json.loads(gzip.decompress(provenance_file.read_bytes()))

    assert provenance["contig"] == "chr1"
    assert provenance["genohype_version"] == "genohype 9.9.9 (deadbee)"
    assert provenance["genohype_args"][0:2] == ["export", "parquet"]
    assert provenance["fields"] == list(DEFAULT_FIELDS)
    assert provenance["num_rows"] == 1


# --- temp cleanup ------------------------------------------------------------


def _site_table(hl):
    """A one-row site table on chr1, keyed by locus."""
    return hl.Table.parallelize(
        [{"locus": hl.locus("chr1", 100, reference_genome="GRCh38"), "xpos": 100}],
        key="locus",
    )


def test_export_contig_removes_the_temp_paths_on_success(
    hl, tmp_path, monkeypatch
) -> None:
    """A successful export removes the temp ``.ht`` and ``.parquet``."""
    recorded: list[tuple[Path, Path]] = []

    def fake(input_ht, output_parquet):
        import pyarrow as pa

        recorded.append((Path(input_ht), Path(output_parquet)))
        pq.write_table(
            pa.table({"xpos": pa.array([100], type=pa.int64())}), output_parquet
        )
        return GenohypeExportResult(
            version="genohype 9.9.9 (deadbee)",
            args=("export", "parquet", str(input_ht), str(output_parquet)),
            num_rows=1,
        )

    monkeypatch.setattr(genohype_module, "export_parquet", fake)
    settings = _settings(checkpoint=str(tmp_path))
    set_config(settings)
    pipeline = BrowserExportPipeline(_task(settings))

    pipeline.export_contig(browser_sites=_site_table(hl), contig="chr1")

    temp_ht, temp_pq = recorded[0]
    assert temp_ht.suffix == ".ht"
    assert temp_pq.suffix == ".parquet"
    assert not temp_ht.exists()
    assert not temp_pq.exists()


def test_export_contig_removes_the_temp_paths_on_failure(
    hl, tmp_path, monkeypatch
) -> None:
    """A genohype failure still removes the temp ``.ht`` and ``.parquet``."""
    recorded: list[tuple[Path, Path]] = []

    def boom(input_ht, output_parquet):
        recorded.append((Path(input_ht), Path(output_parquet)))
        raise genohype_module.GenohypeExportError("boom")

    monkeypatch.setattr(genohype_module, "export_parquet", boom)
    settings = _settings(checkpoint=str(tmp_path))
    set_config(settings)
    pipeline = BrowserExportPipeline(_task(settings))

    with pytest.raises(genohype_module.GenohypeExportError):
        pipeline.export_contig(browser_sites=_site_table(hl), contig="chr1")

    temp_ht, temp_pq = recorded[0]
    assert not temp_ht.exists()
    assert not temp_pq.exists()
