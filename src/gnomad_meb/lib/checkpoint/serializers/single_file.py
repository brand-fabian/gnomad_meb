"""The single-file serializer base (pickle, dataframe, dictionary, file, plot)."""

import os
from datetime import datetime
from hashlib import md5
from pathlib import Path

from gnomad_meb.config import get_config

from ..completeness import md5_file
from ..metadata import CheckpointMetadata, checkpoint_static
from ..paths import is_s3_path, s3_base_path
from .base import Serializer


class SingleFileSerializer[T](Serializer[T]):
    """Specialization of the Serializer base class to handle types that serialize to a single file.

    This base class specializes the Serializer class to provide default functionality
    for classes that serialize objects to a single file (e.g. pickle, dataframes),
    as oppossed to directories like hail objects.
    """

    def metadata(
        self,
        source: str | None = None,
    ):
        """Retrieve metadata for the given object.

        Compute the checkpoint metadata object for the current source path, optionally
        adding an md5 hash of the source code of the function that was passed in.
        """
        if source is not None:
            source_md5 = md5()
            source_md5.update(source.encode("utf-8"))
            source_md5 = source_md5.hexdigest()
        else:
            source_md5 = None

        if is_s3_path(self.path, s3_uri_schemes=self.s3_uri_schemes):
            cfg = get_config()

            bucket, base = s3_base_path(
                self.path, s3_uri_schemes=self.s3_uri_schemes
            )
            files = cfg.s3fs.ls(bucket, force=True)

            return CheckpointMetadata(
                path=self.path,
                exists=base in files,
                mtime=cfg.s3fs.info(bucket, base)["LastModified"].strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                if base in files
                else None,
                data_md5=cfg.s3fs.info(bucket, base)["ETag"].strip('"')
                if base in files
                else None,
                source_md5=source_md5,
                **checkpoint_static(),
            )
        else:
            target = Path(self.path)
            exists = target.exists() and target.is_file()
            return CheckpointMetadata(
                path=self.path,
                exists=exists,
                mtime=datetime.utcfromtimestamp(os.path.getmtime(target)).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                if exists
                else None,
                data_md5=md5_file(target) if exists else None,
                source_md5=source_md5,
                **checkpoint_static(),
            )
