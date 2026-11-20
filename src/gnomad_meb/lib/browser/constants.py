"""Public constants for the browser-export stage.

Hail-free: importing this module must not import the pipeline or Hail.
"""

from __future__ import annotations

from gnomad_meb.constants import DEFAULT_FIELDS, MANDATORY_FIELDS

#: The browser export root: the ``sites.ht`` table and the ``variants.parquet/``
#: dataset live under it.
BROWSER_PREFIX = "{project}/browser"

#: The dataset directory the browser's DuckDB backend opens. DuckDB treats a
#: directory path as "read every ``.parquet`` under it recursively", so the
#: per-contig files inside it are one first-class table.
PARQUET_DATASET_NAME = "variants.parquet"

#: The per-contig parquet filename template; the kind supplies the extension.
CONTIG_FILENAME_TEMPLATE = "locus.contig={contig}"

#: The per-contig provenance filename template, a sibling of the parquet.
CONTIG_PROVENANCE_TEMPLATE = "locus.contig={contig}.provenance.json"

#: The canonical contig order: one export step is generated per entry, and a
#: reference contig absent from this order fails loudly. Only the canonical
#: contigs are enumerated, never the reference's alt/decoy/random sequences.
CONTIG_ORDER = (
    *(f"chr{i}" for i in range(1, 23)),
    "chrX",
    "chrY",
    "chrM",
)

__all__ = [
    "BROWSER_PREFIX",
    "CONTIG_FILENAME_TEMPLATE",
    "CONTIG_ORDER",
    "CONTIG_PROVENANCE_TEMPLATE",
    "DEFAULT_FIELDS",
    "MANDATORY_FIELDS",
    "PARQUET_DATASET_NAME",
]
