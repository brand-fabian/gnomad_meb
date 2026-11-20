"""Manage files on S3 storage."""

import os
from collections.abc import Generator
from pathlib import PurePath
from typing import NamedTuple

import s3fs

from gnomad_meb.logging import get_logger

logger = get_logger()


class S3ConnectionOptions(NamedTuple):
    access_key: str
    secret_key: str
    endpoint: str = "https://s3.intra.imbie"
    verify: bool = True


class S3Provider:
    """S3 provider for listing and retrieving files."""

    def __init__(self, options: S3ConnectionOptions):
        self.options = options

        self._fs = s3fs.S3FileSystem(
            endpoint_url=self.options.endpoint,
            key=self.options.access_key,
            secret=self.options.secret_key,
            client_kwargs={"verify": self.options.verify},
        )
        self._files: dict[str, dict[str, s3fs.S3File]] = {}

    @property
    def fs(self):
        return self._fs

    def ls(self, bucket: str, force: bool = False):
        """Retrieve all contents of the given bucket."""
        if force and bucket in self._files:
            del self._files[bucket]
        if bucket not in self._files:
            logger.debug(f"reading bucket {bucket}")
            self._files[bucket] = {
                p.removeprefix(f"{bucket}/"): None
                for p in self._fs.find(bucket, detail=False)
            }
        return list(self._files[bucket].keys())

    def lsf(
        self, bucket: str, key: str, force: bool = False
    ) -> tuple[bool, dict | None]:
        """Retrieve information about a specific key in a bucket.

        Returns
        -------
        bool
            True, if the file or path exists in the given bucket.
        dict | None
            A file info dict, if the detailed information for this file has been
            retrieved and cached by a previous call, else None. Use `info` to retrieve
            the metadata for this object.
        """
        if bucket not in self._files:
            self._files[bucket] = {}

        if key not in self._files or force:
            try:
                for path in self._fs.ls(f"{bucket}/{key}", detail=False):
                    self._files[bucket][path.removeprefix(f"{bucket}/")] = None
            except FileNotFoundError:
                return False, None

        if key in self._files[bucket]:
            return True, self._files[bucket][key]
        return False, None

    def info(
        self,
        bucket: str,
        key: str,
        force: bool = False,
        missing_ok: bool = False,
    ):
        if bucket not in self._files or key not in self._files[bucket]:
            self.lsf(bucket, key, force=force)
        assert missing_ok or key in self._files[bucket], (
            f"missing file {key} in bucket {bucket}"
        )
        if key in self._files[bucket] and self._files[bucket][key] is None:
            logger.debug(f"getting file info for {bucket}/{key}")
            self._files[bucket][key] = self._fs.info(os.path.join(bucket, key))
        elif key not in self._files[bucket]:
            logger.warning(f"key {key} not found in bucket {bucket}")
            return None
        return self._files[bucket][key]

    def rm(self, bucket: str, key: str, recursive: bool = False):
        if bucket not in self._files:
            self.ls(bucket)
        assert key in self._files[bucket], f"missing file {key} in bucket {bucket}"
        logger.debug(f"deleting {key} in {bucket}")
        self._fs.rm(f"{bucket}/{key}", recursive=recursive)
        del self._files[bucket][key]

    def glob(
        self,
        buckets: list[str],
        patterns: list[str] = [],
        endings: list[str] = [],
        return_url: bool = True,
        s3_url_scheme: str = "s3",
    ) -> Generator[tuple[str, str] | str, None, None]:
        patterns = [
            *patterns,
            *[f"*{ending}" for ending in endings],
        ]
        logger.info(
            f"searching for S3 files in {len(buckets)} buckets,"
            f" matching any of {len(patterns)} patterns."
        )

        for bucket in buckets:
            self.ls(bucket)

        for bucket in buckets:
            for key in self._files[bucket].keys():
                path = PurePath(key)
                if any(path.match(p) for p in patterns):
                    if return_url:
                        yield f"{s3_url_scheme}://{bucket}/{key}"
                    else:
                        yield bucket, key
