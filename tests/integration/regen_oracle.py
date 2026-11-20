"""Regenerate the fixture oracles from a run.

The oracle files are header-only until a known-good run exists (see
``data/README.md``). This helper fills them from the produced tables. It is
**not** a test — it writes tracked files — so it is run deliberately, inside the
container (ADR-0002) after a full ``vqsr``-marked run:

    # Variant QC (``vqsr.tsv``), from the produced ``variant_qc`` table.
    python tests/integration/regen_oracle.py \\
        --reference GRCh38 \\
        --variant-qc <run>/…/checkpoint/<project>/variant_qc.ht \\
        --variant-subset-frac 0.05

    # Sample QC (``sample_qc.tsv``), from the produced ``sample_qc`` table and
    # the relatedness matrix (``--relatedness`` fills ``inferred_family_id``).
    python tests/integration/regen_oracle.py \\
        --reference GRCh38 \\
        --sample-qc <run>/…/checkpoint/<project>/sample_qc.ht \\
        --relatedness <run>/…/intermediate/sample_qc/king.mt

    # Annotation (``annotation.tsv``), from the produced ``annotation`` matrix.
    python tests/integration/regen_oracle.py \\
        --reference GRCh38 \\
        --annotation <run>/…/checkpoint/<project>/annotation.mt \\
        --variant-subset-frac 0.05

It reads the same values the load tests assert and writes one row per asserted
key (sites sorted by contig, position, ref, alt; samples sorted by id). Review
the diff before committing it.

The variant QC subset is configurable — ``--variant-subset-n`` (a count) or
``--variant-subset-frac`` (a fraction), never both — and reproducible given
``--variant-subset-seed``. Selection takes one variant per chromosome first, so
every contig present in the table is represented, then fills the remainder with
a seeded shuffle.

TODO(scatter-gather VQSR, ticket 03): the workflow now trains the SNP and INDEL
models once globally then scatters the apply per VDS task
(``--output-tranches-for-scatter``) with ``GatherTranches``, replacing the old
single global ``COMBINED`` INDEL recalibration and fixing the INDEL cross-wiring
(the INDEL path no longer reads the SNP tranches). This changes INDEL — and,
through the scattered apply, SNP — ``AS_VQSLOD`` / filters relative to the old
workflow, so the first oracle filled in the container **must be regenerated
after that fix is present**.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import harness
import oracle


def _site_sort_key(
    row: dict[str, str],
    contig_index: Sequence[str] = (
        *(f"chr{c}" for c in range(23)),
        "chrX",
        "chrY",
        "chrM",
    ),
) -> tuple[int, int, str, str]:
    """The contig/position/ref/alt sort key for a site row.

    ``contig_index`` orders the contigs (default: chr1-chr22, chrX, chrY, chrM).
    """
    return (contig_index.index(row["chrom"]), int(row["pos"]), row["ref"], row["alt"])


def _rows(variant_qc_ht: Any) -> list[dict[str, str]]:
    """The oracle rows for every site in the ``variant_qc`` table."""
    rows: list[dict[str, str]] = []
    for row in variant_qc_ht.select(
        "filters", "info", "a_index", "was_split"
    ).collect():
        rows.append(
            {
                "chrom": row.locus.contig,
                "pos": str(row.locus.position),
                "ref": row.alleles[0],
                "alt": row.alleles[1],
                "a_index": "" if row.a_index is None else str(row.a_index),
                "was_split": ("" if row.was_split is None else str(row.was_split)),
                "vqslod": (
                    "" if row.info.AS_VQSLOD is None else str(row.info.AS_VQSLOD)
                ),
                "variant_qc_filter": oracle.format_hard_filters(row.filters),
            }
        )
    rows.sort(key=_site_sort_key)
    return rows


def _subset(
    rows: list[dict[str, str]],
    *,
    n: int | None = None,
    frac: float | None = None,
    seed: int = 0,
) -> list[dict[str, str]]:
    """A reproducible subset that covers every chromosome present.

    ``count`` is ``n`` when given, else ``round(len(rows) * frac)``. The rows
    (already sorted by site key) are shuffled with a ``seed``-seeded RNG; one
    variant per chromosome is taken first so no contig is dropped, then the
    remainder is filled in shuffle order. The result is re-sorted by site key so
    the written file is stable. With neither ``n`` nor ``frac`` every row is
    returned unchanged.
    """
    if n is None and frac is None:
        return rows

    total = len(rows)
    count = n if n is not None else round(total * frac)
    count = max(1, min(count, total))

    chromosomes = {row["chrom"] for row in rows}
    if count < len(chromosomes):
        print(
            f"warning: subset count {count} < {len(chromosomes)} chromosomes; "
            "not every chromosome can be covered",
            file=sys.stderr,
        )

    order = list(range(total))
    # A seeded, reproducible shuffle — not a security-sensitive draw.
    random.Random(seed).shuffle(order)  # noqa: S311

    chosen: list[int] = []
    covered: set[str] = set()
    for index in order:
        if len(chosen) >= count:
            break
        chromosome = rows[index]["chrom"]
        if chromosome not in covered:
            chosen.append(index)
            covered.add(chromosome)
    picked = set(chosen)
    for index in order:
        if len(chosen) >= count:
            break
        if index not in picked:
            chosen.append(index)
            picked.add(index)

    selected = sorted(
        (rows[index] for index in chosen),
        key=_site_sort_key,
    )
    return selected


def write_vqsr_oracle(
    variant_qc_path: str | Path,
    out: Path,
    *,
    subset_n: int | None = None,
    subset_frac: float | None = None,
    subset_seed: int = 0,
) -> Path:
    """Write the VQSR oracle to ``out`` from a ``variant_qc`` table."""
    import hail as hl

    variant_qc_ht = hl.read_table(str(variant_qc_path))
    rows = _subset(
        _rows(variant_qc_ht),
        n=subset_n,
        frac=subset_frac,
        seed=subset_seed,
    )

    columns = (
        "chrom",
        "pos",
        "ref",
        "alt",
        "a_index",
        "was_split",
        "vqslod",
        "variant_qc_filter",
    )
    return _write(columns, rows, out)


def _related_pairs(king_path: str | Path, threshold: float) -> list[tuple[str, str]]:
    """The related pairs (``i < j``, ``phi >= threshold``) of a ``king.mt``.

    Mirrors production's ``get_king_pairs``: the entries' ``phi`` is the
    kinship, read as ``i`` from the column key and ``j`` from the row key.
    """
    import hail as hl

    king_mt = hl.read_matrix_table(str(king_path))
    row_key = next(iter(king_mt.row_key))
    col_key = next(iter(king_mt.col_key))
    entries = king_mt.entries()
    pairs = entries.select(i=entries[col_key], j=entries[row_key], kin=entries.phi)
    pairs = pairs.filter((pairs.kin >= threshold) & (pairs.i < pairs.j))
    return [(row.i, row.j) for row in pairs.collect()]


def _family_labels(pairs: list[tuple[str, str]], samples: list[str]) -> dict[str, str]:
    """Infer a family label per sample from related pairs.

    Union-find over the pairs; each component is labelled by its smallest
    member, the convention :func:`oracle._canonical_families` canonicalises to,
    so the labels compare directly. A sample in no related pair is its own
    singleton family.
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
    for sample in samples:
        labels.setdefault(sample, sample)
    return labels


def write_sample_qc_oracle(
    sample_qc_path: str | Path,
    out: Path,
    *,
    king_path: str | Path | None = None,
    threshold: float = oracle.RELATEDNESS_KINSHIP_THRESHOLD,
) -> Path:
    """Write the Sample QC oracle to ``out`` from a ``sample_qc`` table.

    ``king_path`` (the relatedness ``king.mt``) fills ``inferred_family_id``;
    without it that column is left blank.
    """
    import hail as hl

    sample_qc_ht = hl.read_table(str(sample_qc_path))
    collected = sample_qc_ht.select("ploidy", "ancestry", "hard_filters").collect()
    samples = [row.s for row in collected]

    labels: dict[str, str] = {}
    if king_path is not None:
        labels = _family_labels(_related_pairs(king_path, threshold), samples)

    rows: list[dict[str, str]] = []
    for row in collected:
        rows.append(
            {
                "sample_id": row.s,
                "inferred_sex": (
                    oracle.sex_from_karyotype(row.ploidy.sex_karyotype) or ""
                ),
                "sex_karyotype": row.ploidy.sex_karyotype or "",
                "inferred_family_id": labels.get(row.s, ""),
                "ancestry": row.ancestry.gen_anc or "",
                "hard_filter": oracle.format_hard_filters(row.hard_filters),
            }
        )
    rows.sort(key=lambda r: r["sample_id"])

    columns = (
        "sample_id",
        "inferred_sex",
        "sex_karyotype",
        "inferred_family_id",
        "ancestry",
        "hard_filter",
    )
    return _write(columns, rows, out)


def _annotation_rows(annotation_mt: Any) -> list[dict[str, str]]:
    """The oracle rows for every site in the ``annotation`` matrix table."""
    import hail as hl

    freq_meta = hl.eval(annotation_mt.gnomad_freq_meta)
    adj_index = next(
        (i for i, meta in enumerate(freq_meta) if dict(meta) == {"group": "adj"}),
        None,
    )
    assert adj_index is not None, (
        f"no overall 'adj' entry in gnomad_freq_meta: {[dict(m) for m in freq_meta]}"
    )

    rows: list[dict[str, str]] = []
    selected = annotation_mt.select_rows(
        "filters",
        "region_flags",
        "vep",
        "in_silico_predictors",
        "gnomad_freq",
        "grpmax",
        "fafmax",
    ).rows()
    for row in selected.collect():
        adj_freq = (
            row.gnomad_freq[adj_index]
            if row.gnomad_freq is not None and adj_index < len(row.gnomad_freq)
            else None
        )
        cadd = (
            row.in_silico_predictors.cadd
            if row.in_silico_predictors is not None
            else None
        )
        rows.append(
            {
                "chrom": row.locus.contig,
                "pos": str(row.locus.position),
                "ref": row.alleles[0],
                "alt": row.alleles[1],
                "annotation_filter": oracle.format_hard_filters(row.filters),
                "region_lcr": oracle.format_flag(row.region_flags.lcr) or "",
                "region_segdup": oracle.format_flag(row.region_flags.segdup) or "",
                "region_non_par": oracle.format_flag(row.region_flags.non_par) or "",
                "most_severe_consequence": (
                    ""
                    if row.vep is None or row.vep.most_severe_consequence is None
                    else str(row.vep.most_severe_consequence)
                ),
                "gnomad_af": "" if adj_freq is None else str(adj_freq.AF),
                "grpmax_af": "" if row.grpmax is None else str(row.grpmax.AF),
                "fafmax_faf95_max": (
                    "" if row.fafmax is None else str(row.fafmax.faf95_max)
                ),
                "cadd": "" if cadd is None else str(cadd.phred),
            }
        )
    rows.sort(key=_site_sort_key)
    return rows


def write_annotation_oracle(
    annotation_path: str | Path,
    out: Path,
    *,
    subset_n: int | None = None,
    subset_frac: float | None = None,
    subset_seed: int = 0,
) -> Path:
    """Write the Annotation oracle to ``out`` from an ``annotation`` matrix table."""
    import hail as hl

    annotation_mt = hl.read_matrix_table(str(annotation_path))

    # Subset here, materializing all .mt rows into dicts blows up RAM:
    if subset_n is not None:
        n_rows = annotation_mt.count_rows()
        frac = subset_n / n_rows
        assert 0.0 < frac < 1.0, f"invalid variant subset specification: n = {subset_n}, n_rows = {n_rows}"
    else:
        frac = subset_frac
    annotation_mt = annotation_mt.sample_rows(frac, seed=subset_seed)

    rows = _annotation_rows(annotation_mt)

    columns = (
        "chrom",
        "pos",
        "ref",
        "alt",
        "annotation_filter",
        "region_lcr",
        "region_segdup",
        "region_non_par",
        "most_severe_consequence",
        "gnomad_af",
        "grpmax_af",
        "fafmax_faf95_max",
        "cadd",
    )
    return _write(columns, rows, out)


def _write(columns: tuple[str, ...], rows: list[dict[str, str]], out: Path) -> Path:
    """Write the tab-separated oracle rows to ``out``."""
    lines = ["\t".join(columns)]
    lines += ["\t".join(row[column] for column in columns) for row in rows]
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--reference",
        required=True,
        help="Reference genome name, e.g. GRCh38 (selects data/<reference>/).",
    )
    parser.add_argument(
        "--variant-qc",
        default=None,
        help="Path to the produced {project}/variant_qc.ht table.",
    )
    parser.add_argument(
        "--sample-qc",
        default=None,
        help="Path to the produced {project}/sample_qc.ht table.",
    )
    parser.add_argument(
        "--annotation",
        default=None,
        help="Path to the produced {project}/annotation.mt matrix table.",
    )
    parser.add_argument(
        "--relatedness",
        default=None,
        help=(
            "Path to the produced {project}/intermediate/sample_qc/king.mt "
            "matrix (fills inferred_family_id)."
        ),
    )
    parser.add_argument(
        "--relatedness-threshold",
        type=float,
        default=oracle.RELATEDNESS_KINSHIP_THRESHOLD,
        help="Kinship at or above which a pair is related (default: %(default)s).",
    )
    parser.add_argument(
        "--variant-subset-n",
        type=int,
        default=None,
        help="Write at most this many variants of the site-keyed oracles.",
    )
    parser.add_argument(
        "--variant-subset-frac",
        type=float,
        default=None,
        help="Write this fraction of variants of the site-keyed oracles (0 < F <= 1).",
    )
    parser.add_argument(
        "--variant-subset-seed",
        type=int,
        default=0,
        help="Seed for the variant subset selection (default: %(default)s).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="VQSR output TSV path (default: data/<reference>/vqsr.tsv).",
    )
    parser.add_argument(
        "--sample-qc-out",
        default=None,
        help="Sample QC output TSV path (default: data/<reference>/sample_qc.tsv).",
    )
    parser.add_argument(
        "--annotation-out",
        default=None,
        help="Annotation output TSV path (default: data/<reference>/annotation.tsv).",
    )
    args = parser.parse_args()

    if args.variant_qc is None and args.sample_qc is None and args.annotation is None:
        parser.error(
            "at least one of --variant-qc, --sample-qc and --annotation is required"
        )
    if args.variant_subset_n is not None and args.variant_subset_frac is not None:
        parser.error(
            "--variant-subset-n and --variant-subset-frac are mutually exclusive"
        )
    if args.variant_subset_frac is not None and not 0 < args.variant_subset_frac <= 1:
        parser.error("--variant-subset-frac must be in (0, 1]")
    if args.variant_subset_n is not None and args.variant_subset_n < 1:
        parser.error("--variant-subset-n must be positive")

    reference = harness.Reference(args.reference)

    if args.variant_qc is not None:
        out = (
            Path(args.out)
            if args.out is not None
            else harness.oracle_dir(reference) / oracle.VQSR_FILE
        )
        written = write_vqsr_oracle(
            args.variant_qc,
            out,
            subset_n=args.variant_subset_n,
            subset_frac=args.variant_subset_frac,
            subset_seed=args.variant_subset_seed,
        )
        print(f"wrote {written}")

    if args.sample_qc is not None:
        out = (
            Path(args.sample_qc_out)
            if args.sample_qc_out is not None
            else harness.oracle_dir(reference) / oracle.SAMPLE_QC_FILE
        )
        written = write_sample_qc_oracle(
            args.sample_qc,
            out,
            king_path=args.relatedness,
            threshold=args.relatedness_threshold,
        )
        print(f"wrote {written}")

    if args.annotation is not None:
        out = (
            Path(args.annotation_out)
            if args.annotation_out is not None
            else harness.oracle_dir(reference) / oracle.ANNOTATION_FILE
        )
        written = write_annotation_oracle(
            args.annotation,
            out,
            subset_n=args.variant_subset_n,
            subset_frac=args.variant_subset_frac,
            subset_seed=args.variant_subset_seed,
        )
        print(f"wrote {written}")


if __name__ == "__main__":
    main()
