"""In-process tests that a task VDS carries ``DP`` on both entry sets.

Coverage aggregation (``annotation.utils.compute_coverage_stats``, ticket 01)
reads ``dp_field="DP"`` from the dense matrix, which realizes both VDS entry
types. A reference-block entry without ``DP`` densifies to a missing depth that
coverage silently treats as zero, so the VDS build must retain ``DP`` on the
reference data as well as on the (pass-through) variant data. See
``.scratch/annotation-coverage/spec.md`` (D4) and ticket 03.

The combiner's default keeps the entry fields *defined* in the first 100,000
reference rows of the first gVCF (see
``hail/vds/combiner/variant_dataset_combiner.py``), so a declared-but-sparse
``DP`` can be dropped. These tests build a tiny real VDS through the production
:meth:`VariantDataset.process` path: the first gVCF declares ``DP`` in its FORMAT
header but leaves it missing on reference blocks, exactly the shape that makes
the default drop the field.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("hail")

from gnomad_meb.lib.vds import (
    VariantDataset,
    VariantDatasetParameters,
    VariantDatasetTaskIntervals,
)

pysam = pytest.importorskip("pysam")

REFERENCE_GENOME = "GRCh38"

GVCF_HEADER = "\n".join(
    [
        "##fileformat=VCFv4.2",
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Depth">',
        '##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype quality">',
        '##FORMAT=<ID=MIN_DP,Number=1,Type=Integer,Description="Minimum depth">',
        '##INFO=<ID=END,Number=1,Type=Integer,Description="End">',
        "##contig=<ID=chr1,length=248956422>",
    ]
)


def _gvcf_text(sample: str, *, reference_dp: str) -> bytes:
    """One-sample gVCF with a reference block and a variant.

    ``reference_dp`` is the FORMAT/DP value on the reference block; pass ``"."``
    to declare the field but leave it missing, the shape GATK gVCFs commonly
    produce while carrying depth in ``MIN_DP``.
    """
    lines = [
        GVCF_HEADER,
        f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample}",
        f"chr1\t1\t.\tA\t<NON_REF>\t.\t.\tEND=100\tGT:DP:GQ:MIN_DP\t0/0:{reference_dp}:99:30",
        "chr1\t150\t.\tA\tC\t.\t.\t.\tGT:DP:GQ\t0/1:12:99",
    ]
    return ("\n".join(lines) + "\n").encode()


def _write_indexed_gvcf(path: Path, sample: str, *, reference_dp: str) -> None:
    """bgzip + tabix a gVCF so the hail combiner can read it."""
    plain = path.with_suffix(path.suffix + ".plain")
    plain.write_bytes(_gvcf_text(sample, reference_dp=reference_dp))
    pysam.tabix_compress(str(plain), str(path), force=True)
    pysam.tabix_index(str(path), preset="vcf", force=True)


@pytest.fixture(scope="module")
def vds(tmp_path_factory, hl):
    """A tiny VDS built once through the production build path."""
    tmp_path = tmp_path_factory.mktemp("vds-dp")
    first = tmp_path / "a.g.vcf.gz"
    second = tmp_path / "b.g.vcf.gz"
    _write_indexed_gvcf(first, "s1", reference_dp=".")
    _write_indexed_gvcf(second, "s2", reference_dp="30")

    interval = hl.Interval(
        hl.Locus("chr1", 1, REFERENCE_GENOME),
        hl.Locus("chr1", 1000, REFERENCE_GENOME),
        includes_start=True,
        includes_end=True,
    )
    task = VariantDatasetTaskIntervals("chr1", (interval,), REFERENCE_GENOME)
    params = VariantDatasetParameters(
        base=tmp_path,
        name="vds",
        gvcfs=(str(first), str(second)),
        tasks=(task,),
    )
    VariantDataset(tmp_path, "vds", params).process()

    return hl.vds.read_vds(str(tmp_path / "vds.chr1.vds"))


def test_reference_entries_keep_dp(vds) -> None:
    """A declared-but-sparse reference ``DP`` survives the build."""
    assert "DP" in vds.reference_data.entry
    reference_dp = vds.reference_data.DP.collect()
    assert any(value is not None for value in reference_dp), reference_dp


def test_variant_entries_keep_dp(vds) -> None:
    """Variant ``DP`` passes through the combiner untouched."""
    assert "DP" in vds.variant_data.entry
    variant_dp = vds.variant_data.DP.collect()
    assert variant_dp and all(value is not None for value in variant_dp), variant_dp
