"""Completeness probes for hail (directory-format) checkpoints.

Hail writes MatrixTables and Tables as directories, finished by a ``_SUCCESS``
marker. These helpers answer "is the directory present and complete" for both
local and S3 paths, so the completeness rule lives in one place.
"""

import os
from datetime import datetime
from hashlib import md5
from pathlib import Path

from gnomad_meb.config import get_config
from gnomad_meb.logging import get_logger

from .paths import is_s3_path, s3_base_path
from .types import SuccessFileInfo

logger = get_logger()


def md5_file(path: str | Path):
    if isinstance(path, Path):
        path = str(path)
    with open(os.path.join(path), "rb") as in_f:
        md5sum = md5()
        while True:
            data = in_f.read(1024**2)
            if not data:
                break
            md5sum.update(data)
    return md5sum.hexdigest()


def is_incomplete_hail_table(
    obj: str | Path, s3_uri_schemes: list[str] = ["s3", "s3a"]
):
    """True, if the `obj` pointed to is an incomplete hail table.

    Hail writes MatrixTables and Tables in a directory format, creating many output
    parts on the fly on different workers. When finished it writes a `_SUCCESS` marker
    file. A directory is deemed incomplete if it exists, but no success file has
    been written to it.

    Returns
    -------
    bool
        True, if the directory exists, but does not contain the success marker.
    """
    if isinstance(obj, Path):
        obj = str(obj)

    if is_s3_path(obj, s3_uri_schemes=s3_uri_schemes):
        cfg = get_config()

        bucket, base = s3_base_path(obj, s3_uri_schemes=s3_uri_schemes)
        base_exists, _ = cfg.s3fs.lsf(bucket, base)
        success_exists, _ = cfg.s3fs.lsf(bucket, os.path.join(base, "_SUCCESS"))
        return base_exists and not success_exists
    else:
        obj = Path(obj)
        return obj.exists() and not (
            (obj / "_SUCCESS").exists() and (obj / "_SUCCESS").is_file()
        )


def has_success_file(root: str | Path, s3_uri_schemes: list[str] = ["s3", "s3a"]):
    if isinstance(root, Path):
        root = str(root)

    if is_s3_path(root, s3_uri_schemes=s3_uri_schemes):
        cfg = get_config()

        bucket, base = s3_base_path(root, s3_uri_schemes=s3_uri_schemes)

        success_file = os.path.join(base, "_SUCCESS")
        metadata_file = os.path.join(base, "metadata.json.gz")

        exists, _ = cfg.s3fs.lsf(bucket, success_file, force=True)
        if exists and not cfg.s3fs.lsf(bucket, metadata_file, force=True)[0]:
            logger.warning(
                "success file exists, but metadata is missing. The hail directory is likely corrupted"
            )
            exists = False

        return SuccessFileInfo(
            path=success_file,
            exists=exists,
            mtime=cfg.s3fs.info(bucket, success_file)["LastModified"].strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            if exists
            else None,
            metadata=cfg.s3fs.info(bucket, metadata_file)["ETag"].strip('"')
            if exists
            else None,
        )
    else:
        root = Path(root)
        success_file = root / "_SUCCESS"
        metadata_file = root / "metadata.json.gz"

        exists = success_file.exists() and success_file.is_file()
        if exists and not (metadata_file.exists() and metadata_file.is_file()):
            logger.warning(
                "success file exists, but metadata is missing. The hail directory is likely corrupted"
            )
            exists = False

        return SuccessFileInfo(
            path=str(success_file),
            exists=exists,
            mtime=datetime.utcfromtimestamp(os.path.getmtime(success_file)).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            if exists
            else None,
            metadata=md5_file(metadata_file) if exists else None,
        )
