"""Unit tests for detecting the deployed GATK version.

Snakemake writes each conda rule's environment spec under
``{workdir}/.snakemake/conda`` and deploys it to a sibling prefix.
``_detect_gatk_version`` finds the ``gatk4`` environment and reads the installed
package's ``conda-meta`` record, which the VQSR step writes next to the VCFs.
"""

from __future__ import annotations

import json

import yaml

from gnomad_meb.vqsr.vqsr import _detect_gatk_version

#: A conda-meta record as conda writes it (the fields we read, plus extras).
GATK_RECORD = {
    "name": "gatk4",
    "version": "4.5.0.0",
    "build": "hdfd78af_0",
    "url": "https://conda.anaconda.org/bioconda/noarch/gatk4-4.5.0.0-hdfd78af_0.tar.bz2",
    "subdir": "noarch",
}

EXPECTED = {
    "version": "4.5.0.0",
    "build": "hdfd78af_0",
    "url": (
        "https://conda.anaconda.org/bioconda/noarch/gatk4-4.5.0.0-hdfd78af_0.tar.bz2"
    ),
}


def _deploy_environment(
    workdir, hash_: str, dependencies: list[object], records: dict[str, dict]
) -> None:
    conda = workdir / ".snakemake" / "conda"
    conda.mkdir(parents=True, exist_ok=True)
    (conda / f"{hash_}.yaml").write_text(yaml.safe_dump({"dependencies": dependencies}))
    meta = conda / hash_ / "conda-meta"
    meta.mkdir(parents=True, exist_ok=True)
    for name, record in records.items():
        (meta / name).write_text(json.dumps(record))


def test_detect_gatk_version_reads_the_deployed_environment(tmp_path) -> None:
    _deploy_environment(
        tmp_path,
        "abc123",
        ["gatk4>=4.5.0.0", "bcftools", "pysam"],
        {"gatk4-4.5.0.0-hdfd78af_0.json": GATK_RECORD},
    )

    assert _detect_gatk_version(tmp_path) == EXPECTED


def test_detect_gatk_version_ignores_environments_without_gatk(tmp_path) -> None:
    _deploy_environment(
        tmp_path,
        "abc123",
        ["bcftools", "samtools"],
        {"bcftools-1.20-h81e9296_1.json": {"version": "1.20"}},
    )

    assert _detect_gatk_version(tmp_path) is None


def test_detect_gatk_version_returns_none_without_conda_environments(tmp_path) -> None:
    assert _detect_gatk_version(tmp_path) is None
