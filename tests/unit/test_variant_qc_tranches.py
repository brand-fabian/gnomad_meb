"""Unit tests for parsing GATK VQSR tranches files.

The parser reproduces the ``minVQSLod`` ``ApplyVQSR`` selects at a requested
truth-sensitivity level: it keeps the tranches at or above the target and takes
the lowest such target's ``minVQSLod``. The fixtures are the real
``GatherTranches`` output shape (``#`` header lines, a column header, then one
row per tranche).
"""

from __future__ import annotations

import pytest

from gnomad_meb.lib.variant_qc.tranches import (
    read_tranches,
    select_vqslod_cutoff,
)

#: A SNP tranches file: targets 90/99/99.9, as ``GatherTranches`` writes them.
SNP_TRANCHES = """\
# Variant quality score tranches file
# Version number 5
targetTruthSensitivity,numKnown,numNovel,knownTiTv,novelTiTv,minVQSLod,filterName,model,accessibleTruthSites,callsAtTruthSites,truthSensitivity
90.00,41320,289,2.7861,3.2008,4.9900,VQSRTrancheSNP0.00to90.00,SNP,33368,30043,0.9004
99.00,50238,1677,2.6828,2.0916,1.1700,VQSRTrancheSNP90.00to99.00,SNP,33368,33034,0.9900
99.90,55111,5204,2.6034,1.4291,-10.0000,VQSRTrancheSNP99.00to99.90,SNP,33368,33320,0.9986
"""

#: An INDEL tranches file with the same columns.
INDEL_TRANCHES = """\
# Variant quality score tranches file
# Version number 5
targetTruthSensitivity,numKnown,numNovel,knownTiTv,novelTiTv,minVQSLod,filterName,model,accessibleTruthSites,callsAtTruthSites,truthSensitivity
90.00,6261,10507,0.0000,0.0000,1.8800,VQSRTrancheINDEL0.00to90.00,INDEL,2857,2570,0.8995
99.00,7679,17758,0.0000,0.0000,-4.7000,VQSRTrancheINDEL90.00to99.00,INDEL,2857,2828,0.9898
99.90,7776,18454,0.0000,0.0000,-10.0000,VQSRTrancheINDEL99.00to99.90,INDEL,2857,2845,0.9958
"""


def _write(tmp_path, text: str, name: str = "tranches.txt"):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_read_tranches_skips_comments_and_parses_rows(tmp_path) -> None:
    tranches = read_tranches(_write(tmp_path, SNP_TRANCHES))

    assert list(tranches["targetTruthSensitivity"]) == [90.0, 99.0, 99.9]
    assert tranches.loc[1, "minVQSLod"] == 1.17
    assert tranches.loc[1, "model"] == "SNP"


def test_selects_the_lowest_target_at_or_above_the_level(tmp_path) -> None:
    """Requests below a target fall to the next-higher tranche, as GATK does.

    Target ``99.7`` is between the file's ``99.0`` and ``99.9`` tranches, so
    GATK keeps both and applies the ``99.9`` tranche's ``minVQSLod``.
    """
    snp = _write(tmp_path, SNP_TRANCHES, "snp.txt")

    assert select_vqslod_cutoff(snp, 99.7) == -10.0
    assert select_vqslod_cutoff(snp, 99.0) == 1.17
    assert select_vqslod_cutoff(snp, 90.0) == 4.99


def test_selects_an_exact_target(tmp_path) -> None:
    indel = _write(tmp_path, INDEL_TRANCHES, "indel.txt")

    assert select_vqslod_cutoff(indel, 99.0) == -4.7


def test_raises_when_no_tranche_reaches_the_level(tmp_path) -> None:
    snp = _write(tmp_path, SNP_TRANCHES, "snp.txt")

    with pytest.raises(ValueError):
        select_vqslod_cutoff(snp, 99.95)
