"""The fixture oracle: loading and comparing the expected values.

The integration tests assert selected pipeline outputs against hand-maintained
expected values, one file per stage (see ``tests/integration/data``). This module
owns the whole **Expected values** concept: it loads each file, keys it the way
the stage is keyed (by sample or by site), and compares actual values against it.

The oracle files are filled stage by stage, so a blank cell means "not filled in
yet" and is dropped from the comparison rather than compared against a missing
value; a header-only column therefore compares nothing. The only hard error is a
column name the oracle file does not have at all. The ``Oracle`` is pure — it
takes a plain mapping and never imports hail — and carries its
:class:`harness.Reference` for failure provenance only.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import harness
import pandas as pd
import pytest

SAMPLE_QC_FILE = "sample_qc.tsv"
VQSR_FILE = "vqsr.tsv"
ANNOTATION_FILE = "annotation.tsv"

#: The four columns that together identify a site in the site-keyed files.
SITE_KEY_COLUMNS = ("chrom", "pos", "ref", "alt")

#: Kinship at or above which two samples are treated as related when the family
#: structure is inferred from the relatedness matrix (``king.mt``). Shared by the
#: oracle-refresh helper and the integration test so the two cannot drift.
RELATEDNESS_KINSHIP_THRESHOLD = 0.1


def variant_key(chrom: str, pos: int | str, ref: str, alt: str) -> str:
    """The ``chrom:pos:ref:alt`` key used to identify an asserted variant."""
    return f"{chrom}:{pos}:{ref}:{alt}"


def format_hard_filters(filters: set[str] | None) -> str:
    """Render a hail ``hard_filters`` set as the oracle string.

    An empty set (or a set of only missing/blank entries) is ``"pass"``;
    otherwise the active filter names are sorted and joined with ``";"``.
    """
    active = sorted(str(f) for f in (filters or set()) if f)
    return "pass" if not active else ";".join(active)


def format_flag(value: bool | None) -> str | None:
    """Render a hail boolean region flag as the oracle string.

    A missing value stays ``None`` so it can be told apart from a real
    ``False``; otherwise the boolean renders as ``"True"``/``"False"``.
    """
    if value is None:
        return None
    return str(bool(value))


def sex_from_karyotype(karyotype: str | None) -> str | None:
    """Map an inferred sex karyotype onto the pedigree ``M``/``F`` vocabulary.

    The fixture's recorded sexes are ``M``/``F``; the pipeline reports an
    inferred karyotype (for example ``"XX"``/``"XY"``). The presence of a Y
    chromosome is treated as male, an X-only karyotype as female.
    """
    if karyotype is None:
        return None
    upper = str(karyotype).upper()
    if "Y" in upper:
        return "M"
    if "X" in upper:
        return "F"
    return None


def _value_or_none(value: Any) -> str | None:
    """A non-blank oracle cell as a string, or ``None`` when the cell is blank.

    The oracle files are filled stage by stage, so a blank cell means "not
    filled in yet" and must be distinguishable from an actual expectation.
    """
    blank = pd.isna(value) or str(value).strip() == ""
    return None if blank else str(value)


def _render(value: Any) -> str | None:
    return None if value is None else str(value)


@dataclass(frozen=True)
class Oracle:
    """Expected values for one stage, keyed by sample or by site.

    ``expected`` maps a column name to that column's expected value per key
    (``sample_id`` for Sample QC, ``chrom:pos:ref:alt`` for the site-keyed
    stages). Construct it directly from an in-memory mapping in tests, or via
    :func:`sample_qc`, :func:`vqsr` and :func:`annotation` to load a file.
    """

    reference: harness.Reference
    expected: Mapping[str, Mapping[str, str | None]]
    path: Path | None = None

    def expect(self, column: str, actual: Mapping[str, Any]) -> None:
        """Assert every key's actual value equals the expected value.

        Blank (``None``) expectations are dropped, so a column may be filled
        incrementally; an entirely blank column compares nothing. A mismatch
        lists the offending keys, and a column name absent from the oracle is a
        hard failure naming it.
        """
        expected = self._values(column)
        mismatches = [
            f"{key}: expected={expected[key]!r} actual={actual.get(key)!r}"
            for key in expected
            if _render(actual.get(key)) != _render(expected[key])
        ]
        self._fail(column, mismatches)

    def expect_numeric(
        self, column: str, actual: Mapping[str, Any], tolerance: float
    ) -> None:
        """Assert every key's actual number is within ``tolerance`` of expected."""
        expected = self._values(column)
        mismatches = []
        for key, value in expected.items():
            actual_value = actual.get(key)
            if (
                actual_value is None
                or abs(float(actual_value) - float(value)) > tolerance
            ):
                mismatches.append(f"{key}: expected={value!r} actual={actual_value!r}")
        self._fail(column, mismatches)

    def _fail(self, column: str, mismatches: list[str]) -> None:
        assert not mismatches, (
            f"expected values column '{column}' differs for reference "
            f"{self.reference.name}:\n  " + "\n  ".join(mismatches)
        )

    def _values(self, column: str) -> Mapping[str, str]:
        """The expected values of ``column``.

        Note
        ----
        Expected columns may be empty, returning an empty dictionary if
        the column is present. If the column is not present in the expected
        dictionary the test fails immediately.
        """
        if column not in self.expected:
            pytest.fail(
                f"expected values have no column '{column}' for reference "
                f"{self.reference.name}"
            )
        expected = self.expected[column]
        if not expected:
            return {}

        if any(value is None for value in expected.values()):
            return {k: v for k, v in expected.items() if v is not None}
        else:
            return expected


def _load(reference: harness.Reference, filename: str) -> tuple[Path, pd.DataFrame]:
    path = harness.oracle_path(reference, filename)
    return path, pd.read_csv(path, sep="\t", dtype=str)


def _canonical_families(
    values: Mapping[str, str | None],
) -> dict[str, str | None]:
    """Re-label a family partition by each component's smallest member.

    ``inferred_family_id`` asserts a *partition*, not the labels: the README
    says the label may be any consistent family id. Canonicalising both sides to
    the same convention (the one ``inferred_families`` already uses) turns
    partition equality into ordinary value equality, so the oracle's ``expect``
    can compare it like any other column.
    """
    members: dict[str, list[str]] = {}
    for sample, label in values.items():
        if label is not None:
            members.setdefault(label, []).append(sample)
    canonical: dict[str, str | None] = {}
    for sample, label in values.items():
        canonical[sample] = None if label is None else min(members[label])
    return canonical


def sample_qc(reference: harness.Reference) -> Oracle:
    """The Sample QC expected values (``sample_qc.tsv``), keyed by sample."""
    path, table = _load(reference, SAMPLE_QC_FILE)
    expected: dict[str, Mapping[str, str | None]] = {
        column: {
            sample: _value_or_none(value)
            for sample, value in zip(table["sample_id"], table[column], strict=True)
        }
        for column in table.columns
        if column != "sample_id"
    }
    if "inferred_family_id" in expected:
        expected["inferred_family_id"] = _canonical_families(
            expected["inferred_family_id"]
        )
    return Oracle(reference, expected, path)


def _site_keyed(reference: harness.Reference, filename: str) -> Oracle:
    path, table = _load(reference, filename)
    keys = [
        variant_key(row["chrom"], row["pos"], row["ref"], row["alt"])
        for _, row in table.iterrows()
    ]
    expected = {
        column: dict(
            zip(keys, (_value_or_none(value) for value in table[column]), strict=True)
        )
        for column in table.columns
        if column not in SITE_KEY_COLUMNS
    }
    return Oracle(reference, expected, path)


def vqsr(reference: harness.Reference) -> Oracle:
    """The VQSR expected values (``vqsr.tsv``), keyed by site."""
    return _site_keyed(reference, VQSR_FILE)


def annotation(reference: harness.Reference) -> Oracle:
    """The annotation expected values (``annotation.tsv``), keyed by site."""
    return _site_keyed(reference, ANNOTATION_FILE)
