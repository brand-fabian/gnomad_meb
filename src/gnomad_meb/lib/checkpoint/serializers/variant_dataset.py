"""Serializer for a ``hl.vds.VariantDataset`` (hail's directory format)."""

import os
from hashlib import md5
from pathlib import Path

import hail as hl

from ..completeness import has_success_file, is_incomplete_hail_table
from ..metadata import CheckpointMetadata, checkpoint_static
from .base import Serializer


class VariantDatasetSerializer(Serializer[hl.vds.VariantDataset]):
    @classmethod
    def read(cls, path: str | Path, **kwargs):
        if isinstance(path, Path):
            path = str(path.resolve())
        from gnomad_meb.logging import get_logger

        get_logger().debug(f"reading hail vds checkpoint {path}")
        return hl.vds.read_vds(path, **kwargs)

    def write(self, **kwargs):
        from gnomad_meb.logging import get_logger

        logger = get_logger()
        if self.data is None:
            logger.warning("checkpoint data is missing in call to write")
            return
        if any(
            is_incomplete_hail_table(p)
            for p in [
                f"{self.path}/variant_data",
                f"{self.path}/reference_data",
            ]
        ):
            if not self.overwrite:
                logger.error(
                    f"encountered incomplete checkpoint at {self.path}, overwrite = {self.overwrite}"
                )
            else:
                logger.warning(f"overwriting incomplete checkpoint at {self.path}")
        logger.debug(f"writing hail vds checkpoint {self.path}")
        self.data.write(self.path, overwrite=self.overwrite, **kwargs)

    def metadata(self, source: str | None = None):
        vd = os.path.join(self.path, "variant_data")
        rd = os.path.join(self.path, "reference_data")

        vdi = has_success_file(vd)
        rdi = has_success_file(rd)

        if vdi.exists and rdi.exists:
            csum = md5()
            csum.update(vdi.metadata.encode("utf-8"))
            csum.update(rdi.metadata.encode("utf-8"))
            csum = csum.hexdigest()
        else:
            csum = None

        if source is not None:
            source_md5 = md5()
            source_md5.update(source.encode("utf-8"))
            source_md5 = source_md5.hexdigest()
        else:
            source_md5 = None

        return CheckpointMetadata(
            path=self.path,
            exists=vdi.exists and rdi.exists,
            mtime=vdi.mtime if vdi.exists and rdi.exists else None,
            data_md5=csum,
            source_md5=source_md5,
            **checkpoint_static(),
        )
