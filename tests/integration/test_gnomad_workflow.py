"""The happy-path ``gnomad-workflow`` example.

One test run drives the whole composite pipeline exactly as an operator does,
through the CLI subprocess seam: constructing ``gnomad-workflow`` composes every
stage and the runner executes the DAG (see ``conftest.gnomad_workflow``). This
module asserts a short list of stable **terminal artifacts** and a few Expected
values from the tracked oracle files; it deliberately does not re-derive
production's split-merge, concat or interval arithmetic.

The composite includes the VQSR stage, whose external tools (GATK, bcftools,
picard, …) are provisioned at runtime and so need the network on a first run.
The module therefore carries the ``vqsr`` marker alongside ``integration``:
``pytest -m "integration and not vqsr"`` runs the fast harness subset only.

Tests name only terminal artifacts, the fixtures and Expected values — never an
intermediate, a per-split path, a step id or a dependency edge.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import harness
import oracle
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.vqsr]

#: The VQSR recalibration score is compared as a float, not a string.
VQSLOD_TOLERANCE = 1e-3

#: The annotation value columns compared as floats rather than strings, mapped
#: to the absolute tolerance they are compared with. The frequencies are small
#: (order 1e-5) and must not drift, so they get a tight tolerance; the CADD
#: score is order 1 and is compared more loosely.
NUMERIC_COLUMNS = {
    "gnomad_af": 1e-6,
    "grpmax_af": 1e-6,
    "fafmax_faf95_max": 1e-6,
    "cadd": 1e-3,
}

#: The problematic-region boolean flag columns.
REGION_FLAG_COLUMNS = ("region_lcr", "region_segdup", "region_non_par")


def _actual(sites: Mapping[str, Mapping[str, Any]], column: str) -> dict[str, Any]:
    """One selected column of the per-site values, keyed by site/sample."""
    return {key: values[column] for key, values in sites.items()}


# --- terminal artifacts -----------------------------------------------------


def test_gnomad_workflow_produces_terminal_artifacts(
    gnomad_workflow: Path, session_dir: Path, reference: harness.Reference
) -> None:
    """The composite's terminal Annotation matrix and Filter manifest exist.

    ``gnomad_workflow`` is requested to run the composite once; its return value
    (the Filter stage's high-quality matrix) is the deferred GWAS example's
    ``--mt`` input and is not asserted here.
    """
    annotation = (
        harness.project_checkpoint_dir(session_dir, reference) / "annotation.mt"
    )
    assert annotation.is_dir(), f"no annotation checkpoint at '{annotation}'"
    assert annotation.suffix == ".mt", f"unexpected checkpoint type: {annotation}"
    for marker in ("_SUCCESS", "metadata.json.gz"):
        assert (annotation / marker).is_file(), f"missing '{marker}' in checkpoint"
    for part in ("rows", "cols", "entries"):
        assert (annotation / part).is_dir(), (
            f"checkpoint is not a matrix table (missing '{part}')"
        )

    manifest = (
        harness.project_checkpoint_dir(session_dir, reference)
        / "intermediate/filter/manifest.json.gz"
    )
    assert manifest.is_file(), f"no Filter manifest at '{manifest}'"

    with gzip.open(manifest, "rb") as handle:
        payload = json.load(handle)

    assert isinstance(payload["schema_version"], int), payload.get("schema_version")

    high_quality = payload["high_quality"]
    assert Path(high_quality["mt"]).exists(), (
        f"manifest high-quality matrix is absent: {high_quality['mt']}"
    )

    filters = payload["filters"]
    assert filters, "manifest records no configured filters"
    for entry in filters:
        assert entry["name"], f"filter entry has no name: {entry}"
        assert Path(entry["mt"]).exists(), (
            f"manifest filter '{entry['name']}' matrix is absent: {entry['mt']}"
        )


# --- expected values --------------------------------------------------------


def test_sample_qc_expected_values(
    gnomad_workflow: Path,
    session_dir: Path,
    reference: harness.Reference,
    hail: Any,
) -> None:
    """Filled Sample QC expected values of the known samples match the oracle.

    ``inferred_sex`` is the pipeline's karyotype mapped onto the pedigree's
    ``M``/``F`` vocabulary; ``sex_karyotype`` pins the raw karyotype; ``ancestry``
    is the assigned genetic ancestry group; ``hard_filter`` is the hard-filter
    outcome. Blank oracle cells are dropped by the Oracle, so the aneuploid's
    blank ``inferred_sex`` is not compared.
    """
    table = hail.read_table(
        str(harness.project_checkpoint_dir(session_dir, reference) / "sample_qc.ht")
    )
    expected = oracle.sample_qc(reference)

    rows = table.select("ploidy", "ancestry", "hard_filters").collect()
    inferred_sex = {
        row.s: oracle.sex_from_karyotype(row.ploidy.sex_karyotype) for row in rows
    }
    expected.expect("inferred_sex", inferred_sex)

    sex_karyotype = {row.s: row.ploidy.sex_karyotype for row in rows}
    expected.expect("sex_karyotype", sex_karyotype)

    ancestry = {row.s: row.ancestry.gen_anc for row in rows}
    expected.expect("ancestry", ancestry)

    hard_filter = {row.s: oracle.format_hard_filters(row.hard_filters) for row in rows}
    expected.expect("hard_filter", hard_filter)


def _related_component_labels(pairs: Iterable[tuple[str, str]]) -> dict[str, str]:
    """Label each sample by its connected component over related pairs.

    Union-find over the ``(i, j)`` pairs; each component is labelled by its
    lexicographically smallest member, so the labels are stable but arbitrary.
    The oracle canonicalises ``inferred_family_id`` the same way, so the labels
    compare directly.
    """
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    def union(a: str, b: str) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_a] = root_b

    for a, b in pairs:
        union(a, b)

    groups: dict[str, list[str]] = {}
    for node in parent:
        groups.setdefault(find(node), []).append(node)

    labels: dict[str, str] = {}
    for members in groups.values():
        label = min(members)
        for member in members:
            labels[member] = label
    return labels


def _inferred_families(
    pairs: Iterable[tuple[str, str]], samples: Iterable[str]
) -> dict[str, str]:
    """Infer a family label per sample from related pairs.

    A sample that appears in no related pair is its own singleton family.
    """
    labels = _related_component_labels(pairs)
    for sample in samples:
        labels.setdefault(sample, sample)
    return labels


def _partition(labels: Mapping[str, str]) -> set[frozenset[str]]:
    """The set of groups a sample-to-label mapping induces."""
    groups: dict[str, set[str]] = {}
    for sample, label in labels.items():
        groups.setdefault(label, set()).add(sample)
    return {frozenset(group) for group in groups.values()}


def test_sample_qc_relatedness_expected_values(
    gnomad_workflow: Path,
    session_dir: Path,
    reference: harness.Reference,
    hail: Any,
) -> None:
    """The relatedness matrix recovers the pedigree families.

    The family structure is inferred from the relatedness matrix's ``phi`` at
    :data:`oracle.RELATEDNESS_KINSHIP_THRESHOLD` — each pair at or above the
    threshold joins its samples. The resulting partition must equal the
    pedigree's, and the ``inferred_family_id`` filled in ``sample_qc.tsv``.
    """
    king = hail.read_matrix_table(
        str(
            harness.project_checkpoint_dir(session_dir, reference)
            / "intermediate/sample_qc/king.mt"
        )
    )
    row_key = next(iter(king.row_key))
    col_key = next(iter(king.col_key))
    entries = king.entries()
    pairs_ht = entries.select(i=entries[col_key], j=entries[row_key], kin=entries.phi)
    pairs_ht = pairs_ht.filter(
        (pairs_ht.kin >= oracle.RELATEDNESS_KINSHIP_THRESHOLD)
        & (pairs_ht.i < pairs_ht.j)
    )
    pairs = [(row.i, row.j) for row in pairs_ht.collect()]

    pedigree = harness.load_pedigree(reference)
    inferred = _inferred_families(pairs, pedigree["sample_id"])
    inferred_partition = _partition(inferred)

    pedigree_partition = {
        frozenset(group) for _, group in pedigree.groupby("family_id")["sample_id"]
    }
    assert inferred_partition == pedigree_partition, (
        "inferred family structure differs from the pedigree:\n"
        f"  inferred={sorted(sorted(g) for g in inferred_partition)}\n"
        f"  pedigree={sorted(sorted(g) for g in pedigree_partition)}"
    )

    oracle.sample_qc(reference).expect("inferred_family_id", inferred)


def test_variant_qc_expected_values(
    gnomad_workflow: Path,
    session_dir: Path,
    reference: harness.Reference,
    hail: Any,
) -> None:
    """VQSR expected values of known sites match the oracle."""
    table = hail.read_table(
        str(harness.project_checkpoint_dir(session_dir, reference) / "variant_qc.ht")
    )
    expected = oracle.vqsr(reference)

    sites: dict[str, dict[str, Any]] = {}
    for row in table.select("filters", "info", "a_index", "was_split").collect():
        key = oracle.variant_key(
            row.locus.contig,
            row.locus.position,
            row.alleles[0],
            row.alleles[1],
        )
        sites[key] = {
            "a_index": None if row.a_index is None else str(row.a_index),
            "was_split": None if row.was_split is None else str(row.was_split),
            "variant_qc_filter": oracle.format_hard_filters(row.filters),
            "vqslod": row.info.AS_VQSLOD,
        }

    expected.expect("variant_qc_filter", _actual(sites, "variant_qc_filter"))
    expected.expect("a_index", _actual(sites, "a_index"))
    expected.expect("was_split", _actual(sites, "was_split"))
    expected.expect_numeric("vqslod", _actual(sites, "vqslod"), VQSLOD_TOLERANCE)


def test_annotation_expected_values(
    gnomad_workflow: Path,
    session_dir: Path,
    reference: harness.Reference,
    hail: Any,
) -> None:
    """Annotation expected values of known variants match the oracle.

    ``annotation.tsv`` is header-only until a known-good run fills it, so blank
    cells are dropped by the Oracle and a still-blank column compares nothing.
    """
    matrix = hail.read_matrix_table(
        str(harness.project_checkpoint_dir(session_dir, reference) / "annotation.mt")
    )
    expected = oracle.annotation(reference)

    freq_meta = hail.eval(matrix.gnomad_freq_meta)
    gnomad_adj_index = next(
        (
            index
            for index, meta in enumerate(freq_meta)
            if dict(meta) == {"group": "adj"}
        ),
        None,
    )
    if gnomad_adj_index is None:
        pytest.fail(
            "annotation matrix table has no overall 'adj' frequency entry in "
            f"gnomad_freq_meta; present={[dict(meta) for meta in freq_meta]}"
        )

    sites: dict[str, dict[str, Any]] = {}
    rows = (
        matrix.select_rows(
            "filters",
            "region_flags",
            "vep",
            "in_silico_predictors",
            "gnomad_freq",
            "grpmax",
            "fafmax",
        )
        .rows()
        .collect()
    )
    for row in rows:
        key = oracle.variant_key(
            row.locus.contig,
            row.locus.position,
            row.alleles[0],
            row.alleles[1],
        )
        gnomad_adj_freq = None
        if row.gnomad_freq is not None and gnomad_adj_index < len(row.gnomad_freq):
            gnomad_adj_freq = row.gnomad_freq[gnomad_adj_index]
        sites[key] = {
            "annotation_filter": oracle.format_hard_filters(row.filters),
            "region_lcr": oracle.format_flag(row.region_flags.lcr),
            "region_segdup": oracle.format_flag(row.region_flags.segdup),
            "region_non_par": oracle.format_flag(row.region_flags.non_par),
            "most_severe_consequence": (
                None
                if row.vep is None or row.vep.most_severe_consequence is None
                else str(row.vep.most_severe_consequence)
            ),
            "gnomad_af": None if gnomad_adj_freq is None else gnomad_adj_freq.AF,
            "grpmax_af": None if row.grpmax is None else row.grpmax.AF,
            "fafmax_faf95_max": (None if row.fafmax is None else row.fafmax.faf95_max),
            "cadd": (
                None
                if row.in_silico_predictors is None
                or row.in_silico_predictors.cadd is None
                else row.in_silico_predictors.cadd.phred
            ),
        }

    expected.expect("annotation_filter", _actual(sites, "annotation_filter"))
    for column in REGION_FLAG_COLUMNS:
        expected.expect(column, _actual(sites, column))
    expected.expect(
        "most_severe_consequence", _actual(sites, "most_severe_consequence")
    )
    for column, tolerance in NUMERIC_COLUMNS.items():
        expected.expect_numeric(column, _actual(sites, column), tolerance)


# --- general annotation coverage --------------------------------------------


def test_annotation_annotates_every_variant(
    gnomad_workflow: Path,
    session_dir: Path,
    reference: harness.Reference,
    hail: Any,
) -> None:
    """Every variant and sample carries the annotations of each pipeline stage.

    Rows: the VEP consequence (``annotate_vep``), the gnomAD/ClinVar/in-silico
    joins (``annotate_resources``), the region flags and produced frequencies
    (``annotate_regions_and_freq``). Columns: the Sample QC fields
    (``annotate_resources``). Globals: the per-stage provenance.
    """
    matrix = hail.read_matrix_table(
        str(harness.project_checkpoint_dir(session_dir, reference) / "annotation.mt")
    )
    n_rows = matrix.count_rows()
    assert n_rows > 0, "annotation matrix table is empty"

    missing_rows = matrix.aggregate_rows(
        hail.struct(
            vep=hail.agg.count_where(
                hail.is_missing(matrix.vep.most_severe_consequence)
            ),
            clinvar=hail.agg.count_where(hail.is_missing(matrix.clinvar)),
            in_silico=hail.agg.count_where(
                hail.is_missing(matrix.in_silico_predictors)
            ),
            region_flags=hail.agg.count_where(
                hail.is_missing(matrix.region_flags.lcr)
                | hail.is_missing(matrix.region_flags.segdup)
                | hail.is_missing(matrix.region_flags.non_par)
            ),
            freq=hail.agg.count_where(hail.is_missing(matrix.freq)),
        )
    )
    assert all(count == 0 for count in missing_rows.values()), (
        f"unannotated rows out of {n_rows}: {missing_rows}"
    )

    missing_cols = matrix.aggregate_cols(
        hail.struct(
            hard_filters=hail.agg.count_where(hail.is_missing(matrix.hard_filters)),
            ancestry=hail.agg.count_where(hail.is_missing(matrix.ancestry)),
        )
    )
    assert all(count == 0 for count in missing_cols.values()), (
        f"unannotated samples: {missing_cols}"
    )

    # The data-dependent sources must have joined at least some rows.
    covered = matrix.aggregate_rows(
        hail.struct(
            gnomad_freq=hail.agg.count_where(hail.is_defined(matrix.gnomad_freq)),
            clinvar=hail.agg.count_where(
                hail.len(matrix.clinvar.clinical_significance) > 0
            ),
            cadd=hail.agg.count_where(
                hail.is_defined(matrix.in_silico_predictors.cadd)
            ),
        )
    )
    assert all(count > 0 for count in covered.values()), (
        f"annotation sources did not cover any row: {covered}"
    )

    # Every in-silico predictor the pipeline wires is present in the schema.
    assert set(matrix.in_silico_predictors.dtype.fields) >= {
        "cadd",
        "revel_max",
        "spliceai_ds_max",
        "pangolin_largest_ds",
        "phylop",
        "sift_max",
        "polyphen_max",
        "alphamissense",
    }

    globals_seen = hail.eval(matrix.index_globals())
    for name in (
        "sample_qc",
        "variant_qc",
        "vep_globals",
        "gnomad_vep_globals",
        "in_silico_predictors_globals",
    ):
        assert globals_seen.get(name) is not None, f"missing annotation global: {name}"


def test_annotation_retains_every_variant_qc_site(
    gnomad_workflow: Path,
    session_dir: Path,
    reference: harness.Reference,
    hail: Any,
) -> None:
    """No Variant QC site is dropped between the VQSR import and the callset."""
    checkpoint = harness.project_checkpoint_dir(session_dir, reference)
    variant_qc = hail.read_table(str(checkpoint / "variant_qc.ht"))
    annotation = hail.read_matrix_table(str(checkpoint / "annotation.mt"))

    missing = variant_qc.select().anti_join(annotation.rows().select())
    n_missing = missing.count()
    assert n_missing == 0, (
        f"{n_missing} variant_qc site(s) are absent from the annotation callset: "
        f"{missing.head(10).collect()}"
    )
