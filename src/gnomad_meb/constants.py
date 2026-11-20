from typing import Literal

REFERENCE_GENOMES = Literal["GRCh38", "GRCh37", "CHM13"]
"""Known reference genome names."""

DEFAULT_REFERENCE_GENOME: REFERENCE_GENOMES = "GRCh38"
"""The reference genome a run uses unless it declares another."""

#: The browser site table's mandatory physical columns, in the order the
#: browser's DuckDB SQL reads them. The first two come from the input; the rest
#: are synthesized, typed null or empty when the input lacks them.
MANDATORY_FIELDS = (
    "locus",
    "alleles",
    "variant_id",
    "xpos",
    "rsids",
    "caid",
    "exome",
    "genome",
    "joint",
    "transcript_consequences",
    "in_silico_predictors",
    "coverage",
)
"""The browser's fixed column set, always materialized (see ``lib.browser``)."""

#: The default ``[browser_export].fields``: optional passthrough annotations
#: populated when present. A listed field absent on the input is dropped.
#: ``grpmax``/``fafmax``/``histograms`` are consumed into ``joint``/``genome``
#: and must not be listed; ``gnomad_freq``/``clinvar`` are backend-ignored
#: passthroughs and are opt-in.
DEFAULT_FIELDS = ("in_silico_predictors",)
"""The default browser-export passthrough annotations."""


CONTIGS = (
    "chr1",
    "chr2",
    "chr3",
    "chr4",
    "chr5",
    "chr6",
    "chr7",
    "chr8",
    "chr9",
    "chr10",
    "chr11",
    "chr12",
    "chr13",
    "chr14",
    "chr15",
    "chr16",
    "chr17",
    "chr18",
    "chr19",
    "chr20",
    "chr21",
    "chr22",
    "chrX",
    "chrY",
)
