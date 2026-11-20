"""Serializer for one per-contig browser parquet file.

The producer (the per-contig export step) hands this serializer an in-memory
``pyarrow.Table``. :meth:`ParquetFileSerializer.write` serializes it to the
final path — atomically for a local path, by writing a sibling temp file whose
name does **not** end in ``.parquet`` (so the browser's recursive directory scan
cannot pick up a partial file) and then ``os.replace``-ing it onto the final
path. The parent directory is created when absent. The standard checkpoint
sidecar is written by the decorator afterwards, so a contig is advertised as
present only once its parquet and its sidecar exist.

:meth:`ParquetFileSerializer.metadata` is the completeness probe: it reads only
the parquet footer (never a row). ``data_md5`` is a cheap content fingerprint
derived from the footer (row count, column count, column names/types) for a
local file, or the object's ETag for an S3 object — hashing the parquet bytes
themselves is not done, as that is expensive at genome scale (ADR-0020).
"""

from __future__ import annotations

import os
import secrets
from datetime import datetime
from hashlib import md5
from pathlib import Path
from traceback import format_exc

import pyarrow as pa
import pyarrow.parquet as pq

from gnomad_meb.config import get_config
from gnomad_meb.lib.checkpoint.metadata import CheckpointMetadata, checkpoint_static
from gnomad_meb.lib.checkpoint.paths import is_s3_path
from gnomad_meb.lib.checkpoint.serializers.base import Serializer


class ParquetFileSerializer(Serializer[pa.Table]):
    """Serialize one per-contig browser parquet artifact."""

    def __init__(self, *args, s3_uri_schemes: list[str] = ["s3", "s3a"], **kwargs):
        super().__init__(*args, **kwargs)
        self.s3_uri_schemes = s3_uri_schemes

    def metadata(self, source: str | None = None) -> CheckpointMetadata:
        """The sidecar for this per-contig browser parquet artifact.

        ``exists`` is the completeness probe: the file is present and its
        parquet footer parses, read without materializing a single row. A
        truncated or unreadable file is reported absent so the step recomputes
        rather than being advertised complete.

        ``data_md5`` is a cheap content fingerprint: for a local file it is
        derived from the footer (row count, column count, column names/types),
        and for an S3 object it is the object's ETag. ``source_md5`` is the
        wrapped step's source, like the single-file base.
        """
        source_md5 = (
            md5(source.encode("utf-8")).hexdigest() if source is not None else None
        )
        exists, mtime, data_md5 = self._footer_state()
        return CheckpointMetadata(
            path=self.path,
            exists=exists,
            mtime=mtime,
            data_md5=data_md5,
            source_md5=source_md5,
            **checkpoint_static(),
        )

    def _footer_state(self) -> tuple[bool, str | None, str | None]:
        """``(footer parses, mtime, data_md5)`` for this artifact.

        Returns ``(False, None, None)`` when the file is absent or its footer
        cannot be read. Reads only the parquet footer; never a row.
        """
        if is_s3_path(self.path, s3_uri_schemes=self.s3_uri_schemes):
            cfg = get_config()
            fs = cfg.s3fs.fs
            if not fs.exists(self.path):
                return False, None, None
            try:
                with fs.open(self.path, "rb") as in_f:
                    pq.ParquetFile(in_f)
            except Exception:
                return False, None, None
            info = fs.info(self.path)
            etag = info.get("ETag")
            data_md5 = etag.strip('"') if etag is not None else None
            last_modified = info.get("LastModified")
            mtime = (
                last_modified.strftime("%Y-%m-%d %H:%M:%S")
                if last_modified is not None
                else None
            )
            return True, mtime, data_md5

        target = Path(self.path)
        if not (target.exists() and target.is_file()):
            return False, None, None
        try:
            with open(target, "rb") as in_f:
                parquet = pq.ParquetFile(in_f)
                signature = "|".join(
                    [
                        str(parquet.metadata.num_rows),
                        str(parquet.metadata.num_columns),
                        *(
                            f"{field.name}:{field.type}"
                            for field in parquet.schema_arrow
                        ),
                    ]
                )
                data_md5 = md5(signature.encode("utf-8")).hexdigest()
        except Exception:
            return False, None, None
        mtime = datetime.utcfromtimestamp(os.path.getmtime(target)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        return True, mtime, data_md5

    @classmethod
    def read(cls, path: str | Path, **kwargs) -> pa.Table:
        """Return the parquet table at ``path``."""
        from gnomad_meb.logging import get_logger

        self = cls(path, **kwargs)
        get_logger().debug(f"reading parquet checkpoint {self.path}")
        if is_s3_path(self.path, s3_uri_schemes=self.s3_uri_schemes):
            cfg = get_config()
            with cfg.s3fs.fs.open(self.path, "rb") as in_f:
                return pq.read_table(in_f)
        with open(self.path, "rb") as in_f:
            return pq.read_table(in_f)

    def write(self, **kwargs) -> None:
        """Serialize ``self.data`` to ``self.path``.

        Local writes are atomic: a sibling temp file whose name does not end in
        ``.parquet`` is written and then ``os.replace``-d onto ``self.path``, so
        the browser's recursive scan never sees a partial file. The parent
        directory is created when absent. A failed write removes its temp file
        and re-raises.
        """
        from gnomad_meb.logging import get_logger

        logger = get_logger()
        if self.data is None:
            logger.warning("checkpoint data is missing in call to write")
            return

        target = Path(self.path)
        if is_s3_path(target, s3_uri_schemes=self.s3_uri_schemes):
            cfg = get_config()
            if not self.overwrite and cfg.s3fs.fs.exists(target):
                logger.error(
                    f"browser parquet checkpoint exists at {self.path}, "
                    f"overwrite = {self.overwrite}"
                )
                return
            with cfg.s3fs.fs.open(target, "wb") as out_f:
                pq.write_table(self.data, out_f, **kwargs)
            return

        if not self.overwrite and os.path.exists(target):
            logger.error(
                f"browser parquet checkpoint exists at {self.path}, "
                f"overwrite = {self.overwrite}"
            )
            return

        target.parent.mkdir(parents=True, exist_ok=True)
        temp_name = target.with_name(f"{target.name}.{secrets.token_hex(4)}")
        try:
            with open(temp_name, "wb") as out_f:
                pq.write_table(self.data, out_f, **kwargs)
            os.replace(temp_name, target)
        except Exception as error:
            logger.error(
                f"failed to write parquet file: path = {target}, error = {error}"
            )
            logger.error(format_exc())
            if temp_name.is_file():
                os.unlink(temp_name)
            raise
