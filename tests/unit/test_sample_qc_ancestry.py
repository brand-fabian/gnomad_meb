"""Hail-expression test for the Sample QC ancestry PC-count guard (bug 07).

``SampleQcPipeline.apply_ancestry_classification`` projects the cohort onto the
gnomAD reference loadings and hands the first ``gnomad_num_pcs`` scores to the
ONNX ancestry classifier; only a projection carrying the *requested* number of
PCs may take that path. Anything else must return the all-missing ancestry table
so a degenerate projection degrades gracefully instead of feeding the model the
wrong number of features.

The declared-DAG tests in ``test_steps_sample_qc.py`` cannot see this branch, so
it is driven here through the checkpoint's ``__wrapped__`` body with the
projection and the classifier stubbed: no cohort, loadings file or ONNX model is
needed. The conftest ``hl`` fixture supplies the local Hail session (skipped when
the host has no working JVM).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gnomad_meb.cmd.task import SampleQcRun, Task
from gnomad_meb.config import Settings
from gnomad_meb.config.settings import Resources, SampleQcConfig
from gnomad_meb.lib.sample_qc import SampleQcPipeline

#: The empty-ancestry fields the fallback table always carries.
FALLBACK_FIELDS = {
    "pop",
    "prob_other",
    "prob_afr",
    "prob_ami",
    "prob_amr",
    "prob_eas",
    "prob_mid",
    "prob_nfe",
    "prob_sas",
}


def _settings(gnomad_num_pcs: int) -> Settings:
    return Settings(
        validate_paths=False,
        project="demo",
        checkpoint="/ckpt",
        resources=Resources(
            calling_intervals="/res/calling_intervals.bed",
            telomeres_and_centromeres="/res/telomeres_and_centromeres.ht",
            lcr_intervals="/res/lcr_intervals.ht",
            seg_dup_intervals="/res/seg_dup_intervals.ht",
            gnomad_pca_loadings="/res/pca_loadings.ht",
            gnomad_pca_model="/res/pca_model.onnx",
            gnomad_v4="/res/gnomad_v4.ht",
            reference="/res/reference.fa",
        ),
        sample_qc=SampleQcConfig(gnomad_num_pcs=gnomad_num_pcs),
    )


def _pipeline(gnomad_num_pcs: int) -> SampleQcPipeline:
    return SampleQcPipeline(
        Task(settings=_settings(gnomad_num_pcs), command=SampleQcRun(vds="in.vds"))
    )


class _StubMt:
    """The minimal MatrixTable surface ``apply_ancestry_classification`` touches.

    ``GT`` is handed straight to the stubbed ``pc_project``; ``cols()`` is a real
    Hail table because the fallback branch selects from it.
    """

    def __init__(self, cols) -> None:
        self.GT = None
        self._cols = cols

    def count_rows(self) -> int:
        return 1

    def cols(self):
        return self._cols


def _projected_scores(hl, n_pcs: int):
    """A one-sample projection table whose ``scores`` array has ``n_pcs`` PCs."""
    return hl.Table.parallelize(
        [{"s": "S1", "scores": [float(index) for index in range(n_pcs)]}],
        hl.tstruct(s=hl.tstr, scores=hl.tarray(hl.tfloat64)),
    ).key_by("s")


def _run(hl, monkeypatch, *, projected: int, configured: int):
    """Call the ancestry step body with stubbed projection and classifier.

    Returns the result table and the number of times the ancestry classifier
    was invoked (``0`` means the all-missing fallback was taken).
    """
    pipeline = _pipeline(configured)
    pca_ht = _projected_scores(hl, projected)
    cols = hl.Table.parallelize([{"s": "S1"}], hl.tstruct(s=hl.tstr)).key_by("s")

    monkeypatch.setattr(hl.experimental, "pc_project", lambda *args, **kwargs: pca_ht)

    calls: list[dict] = []

    def _fake_assign(scores_ht, **kwargs):
        calls.append(kwargs)
        return scores_ht.annotate(pop=hl.literal("nfe")), None

    import gnomad.sample_qc.ancestry as ancestry

    monkeypatch.setattr(ancestry, "assign_genetic_ancestry_pcs", _fake_assign)

    result = SampleQcPipeline.apply_ancestry_classification.__wrapped__(
        pipeline,
        mt=_StubMt(cols),
        fit=None,
        pca_loadings_ht=SimpleNamespace(loadings=None, pca_af=None),
    )
    return result, len(calls)


def test_requested_pc_count_takes_the_classifier_path(hl, monkeypatch) -> None:
    """A projection with the configured number of PCs is classified normally."""
    result, n_calls = _run(hl, monkeypatch, projected=20, configured=20)

    assert n_calls == 1
    assert "scores" in result.row.dtype.fields
    assert result.aggregate(hl.agg.take(result.pop, 1)) == ["nfe"]


def test_short_projection_falls_back_to_all_missing(hl, monkeypatch) -> None:
    """Too few PCs (incompatible loadings) degrade to the all-missing table."""
    result, n_calls = _run(hl, monkeypatch, projected=10, configured=20)

    assert n_calls == 0
    assert "scores" not in result.row.dtype.fields
    assert FALLBACK_FIELDS <= set(result.row.dtype.fields)
    assert result.aggregate(hl.agg.take(result.pop, 1)) == [None]


def test_the_target_is_the_configured_count_not_a_literal(hl, monkeypatch) -> None:
    """The guard compares against ``gnomad_num_pcs``, not a hard-coded value.

    A non-default target proves the branch follows the setting: 5 projected PCs
    are accepted when 5 are configured, and 2 projected PCs are rejected when 5
    are configured. A guard pinned to any constant (such as the historical ``1``)
    fails the first case.
    """
    accepted, accepted_calls = _run(hl, monkeypatch, projected=5, configured=5)
    assert accepted_calls == 1
    assert accepted.aggregate(hl.agg.take(accepted.pop, 1)) == ["nfe"]

    rejected, rejected_calls = _run(hl, monkeypatch, projected=2, configured=5)
    assert rejected_calls == 0
    assert rejected.aggregate(hl.agg.take(rejected.pop, 1)) == [None]


if __name__ == "__main__":  # pragma: no cover - manual debugging aid
    raise SystemExit(pytest.main([__file__, "-q"]))
