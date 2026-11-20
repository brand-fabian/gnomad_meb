"""The ``browser_export`` pipeline.

Reads a MatrixTable's rows, reshapes them into the browser's sites-only variant
schema, checkpoints that as a Browser site table, and exports one parquet per
reference contig into a directory the browser's DuckDB backend opens directly.
The input endpoint's ``--mt`` value is the source when set; otherwise a composed
run supplies its ``annotation.merge`` result, and a run with neither fails
loudly.

The reshape is :meth:`BrowserExportPipeline.browser_sites`, a ``HAIL_TABLE``
checkpoint at ``{project}/browser/sites.ht``: it keeps the source MatrixTable's
globals (``freq_meta``/``gnomad_freq_meta`` stay globals, never row columns) and
annotates one ``browser_export`` provenance global (the enabled optional field
set, the metadata globals kept and ``checkpoint_static()``).

Per-contig export
-----------------
:meth:`BrowserExportPipeline.prepare` enumerates the **canonical contigs**
(:data:`~gnomad_meb.lib.browser.constants.CONTIG_ORDER`: chr1..chr22, chrX,
chrY, chrM) from the reference genome's metadata and appends one ``PARQUET``
checkpointed step per contig. The GRCh38 reference carries thousands of
alt/decoy/random sequences; those are deliberately **not** exported -- only the
canonical set -- and a canonical contig absent from the reference fails loudly.
Because the contigs come from the reference genome, not the data, the step set
is a pure function of the reference genome name (ADR-0017), so a direct run and
a serialized reconstruction rebuild the same steps.

Each step filters the site table to the whole contig (an inclusive
``1..length`` interval), writes a temporary per-contig Hail table, runs
``genohype export parquet`` with no filter/interval/limit (the parallel path
preserves our materialized ``xpos``) into a temporary parquet, then removes both
temp files in a ``finally``. Both temporarily live **outside**
``variants.parquet/`` -- a Hail table is a directory of ``part-*.parquet`` files
and the browser's DuckDB backend reads every ``.parquet`` under
``variants.parquet/`` recursively -- and the temp parquet is read back for
:class:`~gnomad_meb.lib.checkpoint.serializers.parquet_file.ParquetFileSerializer`
to write the final path atomically (a sibling temp whose name does not end in
``.parquet``, then ``os.replace``).

Invalidation
------------
The pipeline does **not** fingerprint the upstream MatrixTable, the field
configuration, the reference genome or the genohype checkout, so it cannot
detect that any of them changed. Regenerate proactively with ``--overwrite``
when ``--mt``, ``[browser_export].fields``, the reference genome or the genohype
checkout changes; when the contig set itself changes, remove the old
``variants.parquet/`` directory first so files from the previous contig set are
not mixed in. The pipeline never deletes files outside the current contig set.

Importing this module must not initialize Hail: ``browser_sites`` imports Hail
lazily inside its body, ``get_input`` imports its loader lazily, and
``prepare``/``export_contig`` import Hail only when they run. Reading the
reference lengths in ``prepare`` does not call ``hail_init`` and reads metadata
only, never data.
"""

from __future__ import annotations

import secrets
import shutil
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gnomad_meb.lib.browser.constants import (
    BROWSER_PREFIX,
    CONTIG_FILENAME_TEMPLATE,
    CONTIG_ORDER,
    PARQUET_DATASET_NAME,
)
from gnomad_meb.lib.checkpoint import (
    CHECKPOINT_TYPE,
    Artifact,
    checkpoint,
    checkpoint_static,
)
from gnomad_meb.pipeline import FromStep, InputStep, Pipeline, Resource, RunInput

if TYPE_CHECKING:
    import pyarrow as pa

    from gnomad_meb.hail import hail as hl

#: The source metadata globals the Browser site table keeps and records in its
#: export provenance. Spec decision 13: ``freq_meta``/``gnomad_freq_meta`` stay
#: globals and are never exported as parquet columns.
METADATA_GLOBALS = ("freq_meta", "gnomad_freq_meta")


class BrowserExportError(ValueError):
    """The browser export cannot proceed (a canonical contig is missing)."""


class BrowserExportPipeline(Pipeline):
    """Declare and run the browser-export stage.

    The ``mt`` input endpoint is ``--mt`` when set, else the run's configured
    annotation checkpoint (a missing path fails loudly when the first body reads
    it). :meth:`browser_sites` reshapes that MatrixTable into the Browser site
    table; :meth:`prepare` appends one :meth:`export_contig` step per canonical
    reference contig.
    """

    @classmethod
    def name(cls) -> str:
        return "browser_export"

    @property
    def resources(self) -> tuple[Resource, ...]:
        return ()

    def get_input(self) -> tuple[InputStep, ...]:
        """The run's MatrixTable: ``--mt`` when set, else the annotation table."""
        from gnomad_meb.hail import hail as hl

        return (
            InputStep(
                name="mt",
                task=self.task,
                group=type(self).name(),
                source="annotation.merge",
                value=getattr(self.command, "mt", None),
                run=hl.read_matrix_table,
            ),
        )

    # --- generated per-contig steps ------------------------------------------

    def prepare(self) -> None:
        """Append one ``PARQUET`` export step per canonical reference contig.

        Reads only the reference genome's contig lengths (metadata, no data read)
        and enumerates the canonical order, so the step set is a pure function of
        the reference genome name and a reconstruction rebuilds it (ADR-0017). A
        canonical contig absent from the reference fails loudly; the reference's
        alt/decoy/random sequences are never exported.
        """
        super().prepare()

        from gnomad_meb.hail import hail as hl

        lengths = hl.get_reference(self.settings.reference_genome).lengths
        missing = [contig for contig in CONTIG_ORDER if contig not in lengths]
        if missing:
            raise BrowserExportError(
                f"reference genome {self.settings.reference_genome!r} is missing "
                f"canonical contigs: missing = {missing}"
            )

        for contig in CONTIG_ORDER:
            self._methods.append(
                (
                    f"export_contig.{contig}",
                    checkpoint(
                        [
                            self._contig_artifact(contig).prefix,
                            f"{self._contig_artifact(contig).prefix}.md",
                        ],
                        [CHECKPOINT_TYPE.PARQUET, CHECKPOINT_TYPE.DICTIONARY],
                    )(
                        partial(
                            self.export_contig,
                            contig=contig,
                            browser_sites=FromStep("browser_sites"),
                        )
                    ),
                    None,
                )
            )

    def export_contig(
        self,
        browser_sites: hl.Table,
        contig: str,
    ) -> tuple[pa.Table, dict[str, Any]]:
        """Slice the site table to one whole contig and export it with genohype.

        The slice is the inclusive ``1..reference_length`` interval for
        ``contig``. The filtered table is written to a unique temp Hail table
        outside ``variants.parquet/`` and genohype writes a temp parquet there
        too; the temp parquet is read back into a ``pyarrow`` table that the
        ``PARQUET`` artifact serializes to its final path, and the per-contig
        provenance is the sibling ``DICTIONARY`` artifact. Both temp files are
        removed in a ``finally``, on success or failure. Zero-row contigs still
        produce a valid (empty) parquet.
        """
        import pyarrow.parquet as pq

        from gnomad_meb.hail import hail as hl
        from gnomad_meb.lib.browser import genohype
        from gnomad_meb.lib.checkpoint import get_checkpoint_path

        reference = hl.get_reference(self.settings.reference_genome)
        interval = hl.locus_interval(
            contig,
            1,
            reference.lengths[contig],
            reference_genome=reference,
            includes_end=True,
        )
        contig_table = hl.filter_intervals(browser_sites, [interval])

        temp_ht_path = Path(
            str(
                get_checkpoint_path(
                    f"{self.settings.project}/browser/browser_export.{contig}.{secrets.token_hex(4)}",
                    CHECKPOINT_TYPE.HAIL_TABLE,
                )
            )
        )
        temp_pq_path = Path(
            str(
                get_checkpoint_path(
                    f"{self.settings.project}/browser/browser_export.{contig}.{secrets.token_hex(4)}",
                    CHECKPOINT_TYPE.PARQUET,
                )
            )
        )
        try:
            contig_table.write(str(temp_ht_path))
            result = genohype.export_parquet(temp_ht_path, temp_pq_path)
            tbl = pq.read_table(temp_pq_path)
        finally:
            shutil.rmtree(temp_ht_path, ignore_errors=True)
            temp_pq_path.unlink(missing_ok=True)

        provenance = {
            "contig": contig,
            "genohype_version": result.version,
            "genohype_args": list(result.args),
            "fields": list(self.settings.browser_export.fields),
            "num_rows": result.num_rows,
        }
        return tbl, provenance

    def _contig_artifact(self, contig: str) -> Artifact:
        """The per-contig parquet artifact declaration for ``contig``."""
        prefix = (
            f"{BROWSER_PREFIX}/{PARQUET_DATASET_NAME}/"
            f"{CONTIG_FILENAME_TEMPLATE.format(contig=contig)}"
        )
        return Artifact(prefix=prefix, kind=CHECKPOINT_TYPE.PARQUET)

    @checkpoint(f"{BROWSER_PREFIX}/sites", CHECKPOINT_TYPE.HAIL_TABLE)
    def browser_sites(
        self,
        mt: hl.MatrixTable = RunInput("mt"),  # noqa: B008
    ) -> hl.Table:
        """Reshape the input MatrixTable into the Browser site table.

        ``schema.to_browser_sites`` drops the entries and columns, synthesizes
        ``joint``/``genome.quality_metrics``/``coverage`` from the required
        source columns, and keeps the source globals (``freq_meta``/
        ``gnomad_freq_meta`` stay globals). The resulting table carries one
        ``browser_export`` provenance global: the enabled optional field set,
        which metadata globals were kept, and ``checkpoint_static()``.
        """
        from gnomad_meb.hail import hail as hl
        from gnomad_meb.lib.browser import schema

        enabled = list(self.settings.browser_export.fields)
        table = schema.to_browser_sites(
            mt,
            fields=enabled,
            reference_genome=self.settings.reference_genome,
        )
        kept_metadata_globals = [
            name for name in METADATA_GLOBALS if name in table.globals.dtype.fields
        ]
        provenance = {
            "fields": hl.literal(enabled, hl.tarray(hl.tstr)),
            "metadata_globals": hl.literal(kept_metadata_globals, hl.tarray(hl.tstr)),
            **checkpoint_static(),
        }
        return table.annotate_globals(browser_export=hl.struct(**provenance))
