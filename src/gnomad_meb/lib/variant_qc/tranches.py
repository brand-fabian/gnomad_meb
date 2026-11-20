"""Parse GATK VQSR tranches files into recalibration provenance.

A tranches file (the ``GatherTranches`` output declared by the VQSR stage's
``rule all``) is a comment-prefixed CSV: ``#`` header lines, one column header
row, then one row per tranche. :func:`select_vqslod_cutoff` reproduces the
cutoff ``ApplyVQSR`` applies at a requested truth-sensitivity level, so the
``variant_qc`` table can record the ``minVQSLod`` the ``filters`` were set at.

``ApplyVQSR`` (``onTraversalStart``) keeps every tranche whose
``targetTruthSensitivity`` is at or above the requested filter level, then
(``generateFilterString``) passes a variant when its ``AS_VQSLOD`` is at least
the ``minVQSLod`` of the *lowest* kept target. That lowest kept target's
``minVQSLod`` is therefore the effective cutoff.
"""

from __future__ import annotations

from pathlib import Path

import pandas


def read_tranches(path: str | Path) -> pandas.DataFrame:
    """Read a GATK tranches CSV into a :class:`pandas.DataFrame`.

    The ``#`` header lines are skipped (``comment="#"``); the first remaining
    line is the comma-separated column header.
    """
    return pandas.read_csv(path, comment="#")


def select_vqslod_cutoff(
    path: str | Path,
    target_truth_sensitivity: float,
    target_col: str = "targetTruthSensitivity",
    min_vqslod_col: str = "minVQSLod",
) -> float:
    """The ``minVQSLod`` ``ApplyVQSR`` selects at ``target_truth_sensitivity``.

    Keeps the tranches with ``targetTruthSensitivity >= target`` and returns the
    ``minVQSLod`` of the lowest such target. Raises :class:`ValueError` when no
    tranche reaches the requested level, mirroring GATK's refusal to run.
    """
    tranches = read_tranches(path)
    eligible = tranches[tranches[target_col] >= target_truth_sensitivity]
    if eligible.empty:
        raise ValueError(
            f"no tranche at or above target truth sensitivity "
            f"{target_truth_sensitivity} in {path}"
        )
    return float(eligible.loc[eligible[target_col].idxmin(), min_vqslod_col])
