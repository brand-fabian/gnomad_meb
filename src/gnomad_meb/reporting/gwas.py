"""The GWAS workflow stage: collectors over the load manifest and the loaded
``gwas`` / ``burden`` tables, and the ``plotnine`` builders over their frames.

The workflow's declared terminal output is the **GWAS load manifest**
(``{checkpoint}/intermediate/regenie/manifest.json.gz``, written by
``GwasLoadPipeline.get_manifest``), the one index of every loaded ``(table,
region)`` dataset: its ``gwas``/``burden`` hail-table paths and its provenance
(``regenie_version``, ``responses``, the filter globals and the effect-mask
info). It is the GWAS analog of the Filter manifest, and being pure Python it
is read without a hail session -- see
:func:`gnomad_meb.lib.gwas.manifest.read_gwas_manifest`.

The report therefore has three kinds of collector:

* **manifest collectors** (hail-free): reshape the manifest into the dataset
  table, the ``(table x region)`` fan-out and the run provenance;
* **filter/effect-mask collectors** (hail-free): reshape each dataset's
  ``filters`` / ``effect_mask_info`` provenance into tidy frames -- the "which
  tests and filters ran, and what did they keep" content;
* **table collectors** (hail reads): summarise each loaded ``gwas``/``burden``
  table into the summary frames and the (downsampled) manhattan frames.

Two deliberate choices, both from the ticket this report answers:

* **Manhattan rows are downsampled.** A GWAS table can hold millions of rows,
  so :func:`gwas_manhattan` keeps the top-``top_n`` rows by ``log10_p`` --
  globally, so every genome-wide-significant variant survives -- plus a random
  background sample, the whole roughly capped at ``max_rows``, rather than
  pulling the whole table onto the driver.
* **Callset metrics that grow with filters x region sets are plotted, not
  tabulated.** The effect-mask class counts, the per-filter variant/sample
  counts and the region-set fan-out are rendered as scatter plots / heatmaps
  (:func:`effect_mask_heatmap`, :func:`filter_variant_scatter`,
  :func:`dataset_region_heatmap`) so an arbitrary number of datasets stays
  legible.

Each collector is stateless, returns a tidy ``pandas.DataFrame`` (or a plain
dict for scalars) with stable columns, and does not write to disk. Following
:mod:`gnomad_meb.reporting.variant_qc`, each Hail collector is a thin
``hl.Table`` read plus a **pure reshape helper**, so the shaping is
unit-testable without a Spark session. No collector unions a callset: each
``gwas``/``burden`` table is already the terminal merge (ADR-0015).

TODO: add BH FDR correction (``q_value``) to the summary frames at a later
stage; the frames carry ``log10_p`` today.

Burden region coordinates: a burden row is keyed by a region id (a gene symbol
or a ``contig:pos // window + 1`` window), not by coordinates, so a burden
manhattan needs a genomic position per region. :func:`region_coordinates` reads
it from the exported dataset's ``variant_sets`` TSV (the manifest's ``dataset``
path carries it), which maps every region to its ``contig`` and a
representative ``position``. A gene id with no exported mapping is placed at
its region's first variant position if present, else dropped from the
coordinate axis.

TODO: gene symbols (and the ``<none>`` fallback) currently carry no interval
end, so a gene region's "window" is just its first variant position. Convert a
gene id back to a genomic interval (``pyensembl``, already a dependency) and
key on the gene's own coordinates when the variant-sets mapping is absent.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import plotnine as pn

from gnomad_meb.hail import hail as hl
from gnomad_meb.lib.gwas.manifest import GwasManifest
from gnomad_meb.reporting.sample_qc import _as_plain
from gnomad_meb.reporting.theme import CONTIG_ORDER, THEME, ordered_splits

__all__ = [
    "burden_manhattan",
    "burden_mask_scatter",
    "burden_overview",
    "burden_summary",
    "dataset_matrix",
    "dataset_region_heatmap",
    "effect_mask_bar",
    "effect_mask_counts",
    "effect_mask_heatmap",
    "filter_cutoffs",
    "filter_sample_counts",
    "filter_sample_scatter",
    "filter_variant_counts",
    "filter_variant_scatter",
    "gwas_lambda",
    "gwas_manhattan",
    "gwas_overview",
    "gwas_qq",
    "lambda_bar",
    "manhattan_plot",
    "manifest_datasets",
    "qq_plot",
    "region_coordinates",
    "run_provenance",
]

#: The genome-wide significance threshold (p) drawn on every GWAS manhattan.
GWAS_SIGNIFICANCE = 5e-8

#: The χ² 1-df median a genomic-inflation factor λ is normalised by.
_CHI2_1DF_MEDIAN = 0.45493642311957174  # ``scipy.stats.chi2.ppf(0.5, 1)``

#: Alternating contig colours for the manhattan's two-tone chromosome bands.
_ALTERNATE_COLORS = ("#1f3b73", "#8c8c8c")

# --- pure reshape helpers ---------------------------------------------------
# Each manifest/table collector is a thin read; the shaping below is ordinary
# Python so it is unit-testable without a Spark session (the ``variant_qc.py``
# precedent).


def _regenie_version_label(version: dict[str, Any] | None) -> str:
    """A one-line regenie version label (``version`` or ``-``)."""
    if not version:
        return "-"
    return str(version.get("version") or "-")


def _manifest_frame(manifest: GwasManifest) -> pd.DataFrame:
    """One row per loaded dataset from the manifest.

    Columns: ``label``, ``source``, ``name``, ``region``, ``responses`` (a
    comma-joined string), ``regenie_version``, ``dataset``, ``gwas``,
    ``burden``.
    """
    rows = []
    for entry in manifest.datasets:
        rows.append(
            {
                "label": entry.label,
                "source": entry.source,
                "name": entry.name,
                "region": entry.region,
                "responses": ",".join(entry.responses),
                "regenie_version": _regenie_version_label(entry.regenie_version),
                "dataset": entry.dataset,
                "gwas": entry.gwas,
                "burden": entry.burden,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "label",
            "source",
            "name",
            "region",
            "responses",
            "regenie_version",
            "dataset",
            "gwas",
            "burden",
        ],
    )


def _dataset_matrix_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """A boolean ``(source x region)`` fan-out from the dataset frame.

    ``source`` is the filter/table that produced the exported dataset (its
    human name when present), ``region`` the region set it was run over; a cell
    is True when that ``(source, region)`` pair was loaded.
    """
    if frame.empty:
        return pd.DataFrame(columns=["source", "region", "present"])
    work = frame.copy()
    work["source"] = work["name"].where(work["name"].notna(), work["source"])
    return work[["source", "region"]].assign(present=True)


def _variant_counts_frame(filters: dict[str, Any], label: str) -> pd.DataFrame:
    """Long ``label``/``filter``/``step``/``n_variants`` rows from filter globals.

    Every filter global carries a ``variant_info.n_variants`` mapping (its
    ``default``/``final`` and any intermediate gate counts), one row per step.
    """
    rows = []
    for filter_name, info in (filters or {}).items():
        counts = ((info or {}).get("variant_info") or {}).get("n_variants") or {}
        for step, n in counts.items():
            rows.append(
                {
                    "label": label,
                    "filter": filter_name,
                    "step": step,
                    "n_variants": n,
                }
            )
    return pd.DataFrame(rows, columns=["label", "filter", "step", "n_variants"])


def _sample_counts_frame(filters: dict[str, Any], label: str) -> pd.DataFrame:
    """Long ``label``/``filter``/``step``/``n_samples`` rows from filter globals."""
    rows = []
    for filter_name, info in (filters or {}).items():
        sample_info = (info or {}).get("sample_info") or {}
        counts = sample_info.get("n_samples") or {}
        for step, n in counts.items():
            rows.append(
                {
                    "label": label,
                    "filter": filter_name,
                    "step": step,
                    "n_samples": n,
                }
            )
    return pd.DataFrame(rows, columns=["label", "filter", "step", "n_samples"])


def _cutoffs_frame(filters: dict[str, Any], label: str) -> pd.DataFrame:
    """Long ``label``/``filter``/``cutoff``/``value`` rows from filter globals."""
    rows = []
    for filter_name, info in (filters or {}).items():
        cutoffs = ((info or {}).get("variant_info") or {}).get("cutoffs") or {}
        for cutoff, value in cutoffs.items():
            rows.append(
                {
                    "label": label,
                    "filter": filter_name,
                    "cutoff": cutoff,
                    "value": value,
                }
            )
    return pd.DataFrame(rows, columns=["label", "filter", "cutoff", "value"])


def _effect_mask_frame(effect_mask_info: dict[str, Any], label: str) -> pd.DataFrame:
    """Long ``label``/``mask``/``class``/``n`` rows from ``effect_mask_info``.

    ``effect_mask_info`` is a mapping of mask name (``cadd``, ``clinvar``,
    ``lof``, ``missense``, ``rare``, ``variant_class``) to a mapping of class
    (``True``/``False``/``None`` or the variant-class labels) to a count.
    """
    rows = []
    for mask, classes in (effect_mask_info or {}).items():
        for klass, n in (classes or {}).items():
            rows.append({"label": label, "mask": mask, "class": str(klass), "n": n})
    return pd.DataFrame(rows, columns=["label", "mask", "class", "n"])


def _manhattan_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """A tidy manhattan frame from a pulled ``phenotype``/``log10_p``/``locus`` frame.

    The collector's transformed table already carries ``contig``, ``position``,
    ``log10_p``, ``phenotype`` and ``test``; here the ``global_position``
    (cumulative offset so contigs lay end to end) and the two-colour contig
    parity are derived in one place, unit-testable on a hand-built frame.
    """
    if rows.empty:
        return pd.DataFrame(
            columns=[
                "contig",
                "position",
                "phenotype",
                "test",
                "log10_p",
                "global_position",
                "is_significant",
                "is_alternate",
            ]
        )
    frame = rows.copy()
    frame["contig"] = frame["contig"].astype(str)
    # The export strips the ``chr`` prefix (``1``..``22``, ``X``, ``Y``, ``MT``);
    # normalise to the canonical ``chr``-prefixed spelling the theme knows.
    frame["contig"] = frame["contig"].map(_canonical_contig)
    # Cumulative span: contigs in canonical order, each shifted past the last.
    order = {contig: index for index, contig in enumerate(CONTIG_ORDER)}
    frame["_rank"] = frame["contig"].map(order).fillna(len(order)).astype(int)
    spans = frame.groupby("contig", sort=False)["position"].max().rename("span")
    spans = spans.reset_index()
    spans["_rank"] = spans["contig"].map(order).fillna(len(order)).astype(int)
    spans = spans.sort_values("_rank")
    offsets = {}
    running = 0
    for row in spans.itertuples():
        offsets[row.contig] = running
        running += int(row.span) + 1
    frame["global_position"] = frame["position"] + frame["contig"].map(offsets)
    frame["is_significant"] = frame["log10_p"] >= -_log10(GWAS_SIGNIFICANCE)
    # Alternating contig colours: parity of the contig's rank.
    frame["is_alternate"] = frame["_rank"] % 2 == 1
    frame = frame.drop(columns=["_rank"])
    return frame


def _canonical_contig(contig: str) -> str:
    """The ``chr``-prefixed spelling of a bare or prefixed contig name.

    The export writes contigs without the ``chr`` prefix and folds ``MT`` onto
    ``MT``; the report's contig order (:data:`~gnomad_meb.reporting.theme.CONTIG_ORDER`)
    is ``chr``-prefixed, so the two spellings are reconciled here.
    """
    text = str(contig)
    if text.startswith("chr"):
        return text
    if text == "MT":
        return "chrM"
    return f"chr{text}"


def _log10(value: float) -> float:
    """``log10`` that tolerates a zero/negative (returns ``-inf``)."""
    try:
        return math.log10(value)
    except ValueError:
        return float("-inf")


def _lambda_frame(chi_sq: dict[str, Any]) -> pd.DataFrame:
    """One row per ``(phenotype, test)``: the genomic-inflation factor λ.

    λ is the median χ² divided by the 1-df χ² median (0.4549). It is computed
    from the two ``median_chi_sq``/``n`` counters :func:`gwas_lambda` collects.
    """
    rows = []
    for (phenotype, test), stats in (chi_sq or {}).items():
        median = stats.get("median_chi_sq") if isinstance(stats, dict) else stats
        n = stats.get("n") if isinstance(stats, dict) else None
        rows.append(
            {
                "phenotype": phenotype,
                "test": test,
                "n": n,
                "median_chi_sq": median,
                "lambda_": (median / _CHI2_1DF_MEDIAN)
                if median is not None
                else float("nan"),
            }
        )
    return pd.DataFrame(
        rows, columns=["phenotype", "test", "n", "median_chi_sq", "lambda_"]
    )


def _qq_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """A QQ frame from ranked ``log10_p`` rows (``expected`` vs ``observed``).

    ``rows`` carries ``phenotype``/``test``/``log10_p``; the p-values are
    re-ranked within each group and the expected order statistic
    ``-log10((rank - 0.5) / n)`` derived.
    """
    if rows.empty:
        return pd.DataFrame(columns=["phenotype", "test", "expected", "observed"])
    frame = rows.copy()
    if "observed" not in frame.columns and "log10_p" in frame.columns:
        frame = frame.rename(columns={"log10_p": "observed"})
    frames = []
    for (phenotype, test), group in frame.groupby(["phenotype", "test"]):
        ordered = group.sort_values("observed", ascending=False).reset_index(drop=True)
        n = len(ordered)
        ranks = range(1, n + 1)
        ordered = ordered.assign(
            expected=[-_log10((rank - 0.5) / n) for rank in ranks],
            phenotype=phenotype,
            test=test,
        )
        frames.append(ordered[["phenotype", "test", "expected", "observed"]])
    return pd.concat(frames, ignore_index=True)


def _overview(stats: dict[str, Any], kind: str) -> dict[str, Any]:
    """A plain overview dict from a scalar aggregate struct."""
    plain = _as_plain(stats) or {}
    plain["kind"] = kind
    return plain


def _burden_summary_frame(stats: Any) -> pd.DataFrame:
    """One row per ``(region, mask, aaf, phenotype)`` from a burden group-by.

    ``stats`` is the plain view of a ``group_by(...).aggregate`` keyed by the
    four identity fields, each row a partition with its counts and extreme p.
    """
    plain = _as_plain(stats) or {}
    rows = []
    for key, value in plain.items():
        if isinstance(key, tuple):
            region, mask, aaf, phenotype = (*key, None, None, None)[:4]
        else:
            region, mask, aaf, phenotype = key, None, None, None
        value = value or {}
        rows.append(
            {
                "region": region,
                "mask": mask,
                "aaf": aaf,
                "phenotype": phenotype,
                **value,
            }
        )
    return pd.DataFrame(rows)


# --- manifest collectors (hail-free) ----------------------------------------


def manifest_datasets(manifest: GwasManifest) -> pd.DataFrame:
    """One row per loaded dataset: identity, paths, responses and regenie version."""
    return _manifest_frame(manifest)


def dataset_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """The ``(source x region)`` fan-out presence frame (for the heatmap)."""
    return _dataset_matrix_frame(frame)


def run_provenance(manifest: GwasManifest) -> dict[str, Any]:
    """The run-level scalars: project, reference, versions and dataset count."""
    versions = sorted(
        {_regenie_version_label(entry.regenie_version) for entry in manifest.datasets}
    )
    responses = sorted(
        {response for entry in manifest.datasets for response in entry.responses}
    )
    regions = sorted({entry.region for entry in manifest.datasets})
    sources = sorted({entry.source for entry in manifest.datasets})
    return {
        "project": manifest.project,
        "reference_genome": manifest.reference_genome,
        "checkpoint_root": manifest.checkpoint_root,
        "n_datasets": len(manifest.datasets),
        "regenie_versions": versions,
        "responses": responses,
        "regions": regions,
        "sources": sources,
    }


# --- filter / effect-mask collectors (hail-free) ----------------------------


def _concat_or_empty(manifest: GwasManifest, reshape, columns) -> pd.DataFrame:
    """Concatenate a filter-global reshape over every manifest dataset."""
    frames = [reshape(entry, entry.label) for entry in manifest.datasets]
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return pd.DataFrame(columns=columns)
    return pd.concat(non_empty, ignore_index=True)


def filter_variant_counts(manifest: GwasManifest) -> pd.DataFrame:
    """Long variant counts per dataset's filter steps, from the manifest.

    One row per (dataset, filter global, step): the ``n_variants`` mapping each
    filter global records (its ``default``/``final`` plus any intermediate gates).
    """
    return _concat_or_empty(
        manifest,
        lambda entry, label: _variant_counts_frame(entry.filters, label),
        ["label", "filter", "step", "n_variants"],
    )


def filter_sample_counts(manifest: GwasManifest) -> pd.DataFrame:
    """Long sample counts per dataset's filter steps, from the manifest."""
    return _concat_or_empty(
        manifest,
        lambda entry, label: _sample_counts_frame(entry.filters, label),
        ["label", "filter", "step", "n_samples"],
    )


def filter_cutoffs(manifest: GwasManifest) -> pd.DataFrame:
    """Long filter-cutoff rows over all datasets, from the manifest."""
    return _concat_or_empty(
        manifest,
        lambda entry, label: _cutoffs_frame(entry.filters, label),
        ["label", "filter", "cutoff", "value"],
    )


def effect_mask_counts(manifest: GwasManifest) -> pd.DataFrame:
    """Long effect-mask class counts over all datasets, from the manifest."""
    return _concat_or_empty(
        manifest,
        lambda entry, label: _effect_mask_frame(entry.effect_mask_info, label),
        ["label", "mask", "class", "n"],
    )


# --- GWAS table collectors (hail reads) -------------------------------------


def gwas_overview(ht: hl.Table) -> dict[str, Any]:
    """Scalars over one ``gwas`` table: rows, phenotypes, tests, af/n ranges."""
    stats = ht.aggregate(
        hl.struct(
            n_rows=hl.agg.count(),
            phenotypes=hl.agg.collect_as_set(ht.phenotype),
            tests=hl.agg.collect_as_set(ht.test),
            min_log10_p=hl.agg.min(ht.log10_p),
            max_log10_p=hl.agg.max(ht.log10_p),
            n_sig=hl.agg.count_where(ht.log10_p >= -_log10(GWAS_SIGNIFICANCE)),
            min_a1_freq=hl.agg.min(ht.a1_freq),
            max_a1_freq=hl.agg.max(ht.a1_freq),
            max_n=hl.agg.max(ht.n),
        )
    )
    plain = _overview(stats, "gwas")
    # ``phenotypes``/``tests`` come back as sets; order them for the header.
    plain["phenotypes"] = sorted(plain.get("phenotypes") or [])
    plain["tests"] = sorted(plain.get("tests") or [])
    return plain


def _manhattan_select(table: hl.Table) -> hl.Table:
    """The four columns :func:`_manhattan_frame` needs, plus the locus split."""
    return table.select(
        contig=table.locus.contig,
        position=table.locus.position,
        phenotype=table.phenotype,
        test=table.test,
        log10_p=table.log10_p,
    )


def gwas_manhattan(
    ht: hl.Table,
    top_n: int = 10_000,
    max_rows: int = 100_000,
    seed: int = 0,
) -> pd.DataFrame:
    """A downsampled manhattan frame for one ``gwas`` table.

    Keeps the top-``top_n`` rows by ``log10_p`` -- so every genome-wide-
    significant variant survives, whatever panel it is in -- plus a random
    background of the table, the whole roughly capped at ``max_rows`` rows (the
    two parts may overlap, which a plot tolerates). The frame is what
    :func:`manhattan_plot` draws. Downsampling matters because a production
    ``gwas`` table holds millions of rows; the driver pull here is bounded
    whatever the table's size.
    """
    total = ht.count()
    if total == 0:
        return _manhattan_frame(pd.DataFrame())
    top = ht.order_by(hl.desc(ht.log10_p)).head(min(top_n, max_rows))
    n_background = max(max_rows - top.count(), 0)
    if n_background:
        background = ht.sample(min(1.0, n_background / total), seed=seed)
        background = background.key_by()  # order_by unkeys the table
        combined = top.union(background)
    else:
        combined = top
    rows = _manhattan_select(combined).to_pandas()
    return _manhattan_frame(rows)


def gwas_lambda(ht: hl.Table) -> pd.DataFrame:
    """The genomic-inflation factor λ per ``(phenotype, test)``."""
    stats = ht.aggregate(
        hl.agg.group_by(
            hl.tuple([ht.phenotype, ht.test]),
            hl.struct(
                median_chi_sq=hl.agg.approx_quantiles(ht.chi_sq, 0.5),
                n=hl.agg.count(),
            ),
        )
    )
    return _lambda_frame(_as_plain(stats))


def gwas_qq(ht: hl.Table, max_rows: int = 100_000) -> pd.DataFrame:
    """A QQ frame from a bounded pull of each panel's ``log10_p`` rows.

    QQ needs the ranked distribution, so the most-significant ``max_rows`` rows
    are kept (the tail the plot must show) and the rest discarded; the expected
    order statistic is then computed over the retained sample. Pass a large
    ``max_rows`` (or the table's own row count) for an exact QQ on a small
    callset.
    """
    top = ht.order_by(hl.desc(ht.log10_p)).head(max_rows)
    rows = top.select(
        phenotype=top.phenotype, test=top.test, observed=top.log10_p
    ).to_pandas()
    return _qq_frame(rows)


# --- burden table collectors (hail reads) -----------------------------------


def burden_overview(ht: hl.Table) -> dict[str, Any]:
    """Scalars over one ``burden`` table: rows, regions, masks, aafs, phenotypes."""
    stats = ht.aggregate(
        hl.struct(
            n_rows=hl.agg.count(),
            regions=hl.agg.collect_as_set(ht.region),
            masks=hl.agg.collect_as_set(ht.mask),
            aafs=hl.agg.collect_as_set(ht.aaf),
            phenotypes=hl.agg.collect_as_set(ht.phenotype),
            tests=hl.agg.collect_as_set(ht.test),
            max_log10_p=hl.agg.max(ht.log10_p),
            min_a1_freq=hl.agg.min(ht.a1_freq),
            max_a1_freq=hl.agg.max(ht.a1_freq),
            max_n=hl.agg.max(ht.n),
        )
    )
    plain = _overview(stats, "burden")
    plain["n_regions"] = len(plain.get("regions") or [])
    for field in ("regions", "masks", "aafs", "phenotypes", "tests"):
        plain[field] = sorted(value for value in (plain.get(field) or []) if value)
    return plain


def burden_summary(ht: hl.Table) -> pd.DataFrame:
    """One row per ``(region, mask, aaf, phenotype)``: count and extreme p.

    This is the burden analog of the Variant QC per-class summary -- the table
    that would grow with the region set, so the report plots its columns rather
    than rendering it whole when it is large.
    """
    stats = ht.aggregate(
        hl.agg.group_by(
            hl.tuple([ht.region, ht.mask, ht.aaf, ht.phenotype]),
            hl.struct(
                n_tests=hl.agg.count(),
                max_log10_p=hl.agg.max(ht.log10_p),
                min_a1_freq=hl.agg.min(ht.a1_freq),
                max_a1_freq=hl.agg.max(ht.a1_freq),
                max_n=hl.agg.max(ht.n),
            ),
        )
    )
    return _burden_summary_frame(stats)


def burden_manhattan(ht: hl.Table, coordinates: pd.DataFrame) -> pd.DataFrame:
    """A burden manhattan frame, one point per region test.

    ``coordinates`` is :func:`region_coordinates`'s ``contig``/``position``
    frame; a region absent from it is dropped (see the module TODO on gene
    ids).
    """
    rows = ht.key_by()
    rows = rows.select(
        region=rows.region,
        mask=rows.mask,
        aaf=rows.aaf,
        phenotype=rows.phenotype,
        log10_p=rows.log10_p,
    ).to_pandas()
    if rows.empty:
        return _manhattan_frame(pd.DataFrame())
    merged = rows.merge(coordinates, on="region", how="inner")
    merged["position"] = merged["position"].astype(int)
    return _manhattan_frame(merged)


def region_coordinates(dataset: str | Any) -> pd.DataFrame:
    """The ``region``/``contig``/``position`` mapping for one exported dataset.

    Reads the dataset's ``variant_sets`` TSV (``contig``/``region``/
    ``position``/``variants``), which the export writes for every region set.
    The representative position is the region's minimum variant position; for a
    ``window_N`` region the window's own start is recovered from the region id.
    A missing file returns an empty frame (the caller then plots a categorical
    axis).
    """
    import gzip
    from pathlib import Path

    directory = Path(dataset)
    matches = sorted(directory.glob("*.variant_sets.tsv.bgz"))
    if not matches:
        return pd.DataFrame(columns=["region", "contig", "position"])
    opener = gzip.open if matches[0].suffix == ".bgz" else open
    frame = pd.read_csv(opener(matches[0], "rt"), sep="\t")
    if frame.empty:
        return pd.DataFrame(columns=["region", "contig", "position"])
    grouped = frame.groupby(["region", "contig"], as_index=False)["position"].min()
    # One position per region: the first contig it touches.
    grouped = grouped.sort_values(["region", "position"]).drop_duplicates("region")
    return grouped[["region", "contig", "position"]]


# --- plot builders ----------------------------------------------------------


def _label_contigs(frame: pd.DataFrame) -> pd.DataFrame:
    """Order and label the contig axis semantically; blank mid-contig labels."""
    out = frame.copy()
    out["contig"] = ordered_splits(out["contig"])
    return out


def manhattan_plot(
    df: pd.DataFrame,
    *,
    title: str = "Manhattan",
    significance: float = GWAS_SIGNIFICANCE,
    facet: str | None = None,
) -> pn.ggplot:
    """A GWAS manhattan plot from :func:`gwas_manhattan`'s frame.

    Alternating contig colours, a dashed genome-wide-significance line, and an
    optional facet (``"phenotype"``/``"test"``) for small multiples across all
    panels.
    """
    if df.empty:
        return pn.ggplot(df) + pn.labs(title=title) + THEME
    frame = _label_contigs(df)
    frame["_color"] = frame["is_alternate"].map(
        {True: _ALTERNATE_COLORS[0], False: _ALTERNATE_COLORS[1]}
    )
    plot = (
        pn.ggplot(frame, pn.aes(x="global_position", y="log10_p", color="_color"))
        + pn.geom_point(size=0.6, alpha=0.7, stroke=0)
        + pn.scale_color_identity()
        + pn.geom_hline(
            yintercept=_log10(significance) * -1,
            linetype="dashed",
            color="red",
        )
        + pn.labs(
            x="Chromosome",
            y=r"$-\log_{10}(p)$",
            title=title,
        )
        + THEME
        + pn.theme(legend_position="none")
    )
    if facet:
        plot = plot + pn.facet_wrap(f"~ {facet}", scales="free_x")
    return plot


def qq_plot(df: pd.DataFrame, *, facet: str | None = None) -> pn.ggplot:
    """A QQ plot from :func:`gwas_qq`'s frame (expected vs observed)."""
    if df.empty:
        return pn.ggplot(df) + pn.labs(title="QQ") + THEME
    plot = (
        pn.ggplot(df, pn.aes(x="expected", y="observed"))
        + pn.geom_point(size=0.6, alpha=0.6)
        + pn.geom_abline(intercept=0, slope=1, linetype="dashed", color="red")
        + pn.labs(
            x=r"Expected $-\log_{10}(p)$",
            y=r"Observed $-\log_{10}(p)$",
            title="QQ",
        )
        + THEME
    )
    if facet:
        plot = plot + pn.facet_wrap(f"~ {facet}")
    return plot


def lambda_bar(df: pd.DataFrame) -> pn.ggplot:
    """The genomic-inflation factor λ per ``(phenotype, test)``."""
    if df.empty:
        return pn.ggplot(df) + pn.labs(title="Genomic inflation") + THEME
    frame = df.copy()
    frame["panel"] = frame["phenotype"].astype(str) + " / " + frame["test"].astype(str)
    return (
        pn.ggplot(frame, pn.aes(x="reorder(panel, lambda_)", y="lambda_"))
        + pn.geom_col(fill="#2b5c8f")
        + pn.geom_hline(yintercept=1.0, linetype="dashed", color="red")
        + pn.coord_flip()
        + pn.labs(x="", y="λ", title="Genomic inflation (λ)")
        + THEME
    )


def effect_mask_bar(df: pd.DataFrame) -> pn.ggplot:
    """Effect-mask class counts, stacked per mask, faceted by dataset."""
    if df.empty:
        return pn.ggplot(df) + pn.labs(title="Effect masks") + THEME
    return (
        pn.ggplot(df, pn.aes(x="mask", y="n", fill="class"))
        + pn.geom_col()
        + pn.facet_wrap("~ label")
        + pn.coord_flip()
        + pn.labs(x="", y="Variants", fill="Class", title="Effect-mask classes")
        + THEME
    )


def effect_mask_heatmap(df: pd.DataFrame) -> pn.ggplot:
    """A ``(label x mask)`` heatmap of a mask's positive class count.

    Replaces the wide class-count table when it grows with the dataset count.
    """
    if df.empty:
        return pn.ggplot(df) + pn.labs(title="Effect-mask heatmap") + THEME
    frame = df[~df["class"].isin(["False", "None"])].copy()
    if frame.empty:
        frame = df.copy()
    return (
        pn.ggplot(frame, pn.aes(x="mask", y="label", fill="n"))
        + pn.geom_tile()
        + pn.scale_fill_gradient(low="#eef2f7", high="#1f3b73")
        + pn.labs(x="", y="", fill="Variants", title="Effect-mask counts")
        + THEME
    )


def filter_variant_scatter(df: pd.DataFrame) -> pn.ggplot:
    """Variants retained at each filter step, as points across datasets.

    A scatter (not a wide table), so an arbitrary set of filters x datasets
    stays legible; a line joins each dataset's steps.
    """
    if df.empty:
        return pn.ggplot(df) + pn.labs(title="Filter variant counts") + THEME
    frame = df.copy()
    frame["filter"] = ordered_splits(frame["filter"])
    return (
        pn.ggplot(frame, pn.aes(x="filter", y="n_variants", color="label"))
        + pn.geom_point(size=2)
        + pn.geom_line(pn.aes(group="label"), alpha=0.4)
        + pn.coord_flip()
        + pn.labs(x="", y="Variants", color="Dataset", title="Variants per filter step")
        + THEME
    )


def filter_sample_scatter(df: pd.DataFrame) -> pn.ggplot:
    """Samples retained at each filter step, as points across datasets."""
    if df.empty:
        return pn.ggplot(df) + pn.labs(title="Filter sample counts") + THEME
    frame = df.copy()
    frame["filter"] = ordered_splits(frame["filter"])
    return (
        pn.ggplot(frame, pn.aes(x="filter", y="n_samples", color="label"))
        + pn.geom_point(size=2)
        + pn.coord_flip()
        + pn.labs(x="", y="Samples", color="Dataset", title="Samples per filter step")
        + THEME
    )


def dataset_region_heatmap(df: pd.DataFrame) -> pn.ggplot:
    """The ``(source x region)`` fan-out: which (filter, region set) was loaded."""
    if df.empty:
        return pn.ggplot(df) + pn.labs(title="Dataset fan-out") + THEME
    return (
        pn.ggplot(df, pn.aes(x="region", y="source", fill="present"))
        + pn.geom_tile()
        + pn.scale_fill_manual(values={True: "#1f3b73"})
        + pn.labs(x="Region set", y="", title="Dataset fan-out")
        + THEME
        + pn.theme(legend_position="none")
    )


def burden_mask_scatter(df: pd.DataFrame) -> pn.ggplot:
    """The burden landscape: ``-log10(p)`` per region, coloured by mask/aaf.

    The scatter that stands in for the (region x mask x aaf) table, faceted by
    phenotype.
    """
    if df.empty:
        return pn.ggplot(df) + pn.labs(title="Burden landscape") + THEME
    frame = df.copy()
    frame["mask_aaf"] = frame["mask"].astype(str) + " / " + frame["aaf"].astype(str)
    return (
        pn.ggplot(frame, pn.aes(x="mask_aaf", y="max_log10_p", color="mask_aaf"))
        + pn.geom_jitter(width=0.3, size=1.2, alpha=0.6)
        + pn.facet_wrap("~ phenotype", scales="free_y")
        + pn.coord_flip()
        + pn.labs(
            x="",
            y=r"Best $-\log_{10}(p)$",
            color="Mask / AAF",
            title="Burden landscape",
        )
        + THEME
    )
