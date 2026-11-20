"""Serializer for a hail ``Table`` (directory format)."""

from hashlib import md5
from pathlib import Path

import hail as hl

from ..completeness import has_success_file, is_incomplete_hail_table
from ..metadata import CheckpointMetadata, checkpoint_static
from .base import Serializer


class HailTableSerializer(Serializer[hl.Table]):
    @classmethod
    def read(cls, path: str | Path, **kwargs):
        if isinstance(path, Path):
            path = str(path.resolve())
        from gnomad_meb.logging import get_logger

        get_logger().debug(f"reading hail table checkpoint {path}")
        return hl.read_table(path, **kwargs)

    def write(self, **kwargs):
        from gnomad_meb.logging import get_logger

        logger = get_logger()
        if self.data is None:
            logger.warning("checkpoint data is missing in call to write")
            return
        if is_incomplete_hail_table(self.path):
            if not self.overwrite:
                logger.error(
                    f"encountered incomplete checkpoint at {self.path}, overwrite = {self.overwrite}"
                )
            else:
                logger.warning(f"overwriting incomplete checkpoint at {self.path}")
        logger.debug(f"writing hail table checkpoint {self.path}")
        self.data.write(self.path, overwrite=self.overwrite)

    def metadata(self, source: str | None = None):
        sfi = has_success_file(self.path)

        if source is not None:
            source_md5 = md5()
            source_md5.update(source.encode("utf-8"))
            source_md5 = source_md5.hexdigest()
        else:
            source_md5 = None

        return CheckpointMetadata(
            path=self.path,
            exists=sfi.exists,
            mtime=sfi.mtime,
            data_md5=sfi.metadata,
            source_md5=source_md5,
            **checkpoint_static(),
        )
