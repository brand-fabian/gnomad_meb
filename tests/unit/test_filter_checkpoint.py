"""Hail-free unit tests for the gnomAD sub-filter decision in ``filter_checkpoint``.

The predicate construction is factored into pure helpers so the two mode bugs
from issue 04 can be pinned without a hail/JVM session: a ``None`` AF bound must
contribute no condition, and the ancestry group must reach the
``gnomad_freq_meta`` index lookup. Full hail evaluation stays in the Java-11
container per ``docs/adr/0002``.
"""

from __future__ import annotations

from gnomad_meb.config import FilterCheckpoint
from gnomad_meb.lib.filter.filter_checkpoint import (
    gnomad_af_conditions,
    gnomad_freq_meta_key,
)


def test_unset_af_bounds_contribute_no_condition() -> None:
    assert gnomad_af_conditions(None, None, filter_snps=False) == frozenset()
    assert gnomad_af_conditions(None, 0.01, filter_snps=False) == frozenset({"max_af"})
    assert gnomad_af_conditions(0.0, None, filter_snps=False) == frozenset({"min_af"})
    assert gnomad_af_conditions(None, None, filter_snps=True) == frozenset({"snps"})
    assert gnomad_af_conditions(0.0, 0.01, filter_snps=True) == frozenset(
        {"min_af", "max_af", "snps"}
    )


def test_a_zero_bound_is_still_a_bound() -> None:
    """``0.0`` must be distinguishable from ``None`` (the old coercion bug)."""
    assert "min_af" in gnomad_af_conditions(0.0, None, filter_snps=False)
    assert "min_af" not in gnomad_af_conditions(None, None, filter_snps=False)


def test_ancestry_group_alone_does_not_enable_the_subfilter() -> None:
    flt = FilterCheckpoint(name="afr_only", ancestry_group="afr")

    assert not gnomad_af_conditions(
        flt.min_gnomad_af, flt.max_gnomad_af, flt.filter_snps
    )


def test_ancestry_group_reaches_the_meta_lookup() -> None:
    assert gnomad_freq_meta_key("afr") == {"gen_anc": "afr", "group": "adj"}
    assert gnomad_freq_meta_key("nfe") == {"gen_anc": "nfe", "group": "adj"}


def test_missing_ancestry_group_falls_back_to_the_reference_group() -> None:
    assert gnomad_freq_meta_key(None) == {"gen_anc": "nfe", "group": "adj"}
