"""Serializer for the project's own :class:`VariantDataset` metadata handle.

The artifact path is the handle's metadata file. ``read`` reopens it with
:meth:`VariantDataset.open`; ``write`` delegates to :meth:`VariantDataset.write`,
which persists only the changed splits (an unchanged handle right after
``process()`` writes metadata only) and materializes an ``apply()`` result at
the checkpoint location. The VDS API is imported lazily to keep the checkpoint
package importable without it.
"""

import os
from datetime import UTC, datetime
from hashlib import md5
from pathlib import Path

from ..completeness import md5_file
from ..metadata import CheckpointMetadata, checkpoint_static
from .base import Serializer


class NativeVariantDatasetSerializer(Serializer):
    """Serialize the project's own :class:`VariantDataset` metadata handle."""

    @classmethod
    def read(cls, path: str | Path, **kwargs):
        from gnomad_meb.lib.vds import VariantDataset

        return VariantDataset.open(path)

    def write(self, **kwargs):
        from gnomad_meb.logging import get_logger

        logger = get_logger()
        if self.data is None:
            logger.warning("checkpoint data is missing in call to write")
            return
        # The artifact path is ``<checkpoint>/<prefix>.metadata.json``; its
        # directory and stem are the handle's base and name, so writing there
        # lands both the split VDSs and the metadata file where read() looks.
        md_path = Path(str(self.path))
        logger.debug(f"writing native variant dataset checkpoint {self.path}")
        self.data.write(
            base=md_path.parent,
            name=md_path.name.removesuffix(".metadata.json"),
        )

    def metadata(self, source: str | None = None) -> CheckpointMetadata:
        from gnomad_meb.lib.vds import VariantDataset

        md_path = Path(str(self.path))
        if md_path.is_file():
            vds = VariantDataset.open(md_path)

            if source is not None:
                source_md5 = md5()
                source_md5.update(source.encode("utf-8"))
                source_md5 = source_md5.hexdigest()
            else:
                source_md5 = None

            return CheckpointMetadata(
                path=self.path,
                exists=vds.is_complete,
                mtime=datetime.fromtimestamp(os.path.getmtime(md_path)).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                # The metadata file records the task paths and status, i.e. the
                # identity of the split set; hashing it avoids materializing the
                # data (as inspect() would) just to answer a cache lookup.
                data_md5=md5_file(md_path),
                source_md5=source_md5,
                **checkpoint_static(),
            )
        else:
            # The VariantDataset has not yet been created
            return CheckpointMetadata(
                path=self.path,
                exists=False,
                mtime=datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
                data_md5=None,
                source_md5=None,
                **checkpoint_static(),
            )
