"""Hail-free tests for the sample-QC sex-karyotype settings seam.

The pinned X/Y/hom-alt cutoffs and the aneuploidy-screening cutoffs configure
``infer_sex_karyotype`` (see :mod:`gnomad_meb.lib.sample_qc.utils`). Hail is
unavailable on this host, so the settings' public shape and the positional-tuple
conversion they promise are verified here; the Hail arithmetic itself is
exercised at the integration seam (ADR-0002).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from gnomad_meb.config.settings import (
    ChrXFracHomAltCutoffs,
    SampleQcConfig,
    SampleQcHardFilterCutoffs,
    XPloidyCutoffs,
    YPloidyCutoffs,
)


def test_sample_qc_sex_cutoff_defaults() -> None:
    cfg = SampleQcConfig()

    assert cfg.x_ploidy_cutoffs is None
    assert cfg.y_ploidy_cutoffs is None
    assert cfg.chr_x_frac_hom_alt_cutoffs is None
    assert cfg.normal_ploidy_cutoff == 5
    assert cfg.aneuploidy_cutoff == 6
    assert cfg.hard_filter.include_sex_filter is True


def test_pinned_x_ploidy_cutoffs_convert_to_positional_tuple() -> None:
    cutoffs = XPloidyCutoffs(
        upper_cutoff_X=1.4,
        lower_cutoff_XX=1.7,
        upper_cutoff_XX=2.3,
        lower_cutoff_XXX=2.9,
    )

    assert cutoffs.as_tuple() == (1.4, (1.7, 2.3), 2.9)


def test_pinned_y_ploidy_cutoffs_convert_to_positional_tuple() -> None:
    cutoffs = YPloidyCutoffs(
        lower_cutoff_Y=0.2,
        upper_cutoff_Y=1.8,
        lower_cutoff_YY=2.5,
    )

    assert cutoffs.as_tuple() == ((0.2, 1.8), 2.5)


def test_pinned_hom_alt_cutoffs_convert_to_positional_tuple() -> None:
    cutoffs = ChrXFracHomAltCutoffs(
        lower_cutoff_more_than_one_X=0.1,
        upper_cutoff_more_than_one_X=0.6,
        lower_cutoff_single_X=0.4,
    )

    assert cutoffs.as_tuple() == ((0.1, 0.6), 0.4)


def test_pinned_cutoffs_must_be_configured_together() -> None:
    with pytest.raises(ValidationError):
        SampleQcConfig(
            x_ploidy_cutoffs=XPloidyCutoffs(
                upper_cutoff_X=1.4,
                lower_cutoff_XX=1.7,
                upper_cutoff_XX=2.3,
                lower_cutoff_XXX=2.9,
            )
        )


def test_pinned_cutoffs_configured_together_are_accepted() -> None:
    cfg = SampleQcConfig(
        x_ploidy_cutoffs=XPloidyCutoffs(
            upper_cutoff_X=1.4,
            lower_cutoff_XX=1.7,
            upper_cutoff_XX=2.3,
            lower_cutoff_XXX=2.9,
        ),
        y_ploidy_cutoffs=YPloidyCutoffs(
            lower_cutoff_Y=0.2,
            upper_cutoff_Y=1.8,
            lower_cutoff_YY=2.5,
        ),
        chr_x_frac_hom_alt_cutoffs=ChrXFracHomAltCutoffs(
            lower_cutoff_more_than_one_X=0.1,
            upper_cutoff_more_than_one_X=0.6,
            lower_cutoff_single_X=0.4,
        ),
    )

    assert cfg.x_ploidy_cutoffs is not None
    assert cfg.x_ploidy_cutoffs.as_tuple() == (1.4, (1.7, 2.3), 2.9)
    assert cfg.y_ploidy_cutoffs is not None
    assert cfg.y_ploidy_cutoffs.as_tuple() == ((0.2, 1.8), 2.5)
    assert cfg.chr_x_frac_hom_alt_cutoffs is not None
    assert cfg.chr_x_frac_hom_alt_cutoffs.as_tuple() == ((0.1, 0.6), 0.4)


def test_hard_filter_include_sex_filter_can_be_disabled() -> None:
    assert SampleQcHardFilterCutoffs().include_sex_filter is True
    assert (
        SampleQcHardFilterCutoffs(include_sex_filter=False).include_sex_filter is False
    )
