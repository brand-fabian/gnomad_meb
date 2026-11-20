"""Path helpers for checkpoint artifacts (local and S3)."""

from pathlib import Path
from urllib.parse import urlparse

from gnomad_meb.config import get_config
from gnomad_meb.logging import get_logger

from .types import CHECKPOINT_EXTENSION, CHECKPOINT_TYPE

logger = get_logger()


def is_s3_path(root: str | Path, s3_uri_schemes: list[str] = ["s3", "s3a"]):
    if isinstance(root, Path):
        root = str(root)

    return any(root.startswith(x) for x in s3_uri_schemes)


def s3_base_path(obj: str | Path, s3_uri_schemes: list[str] = ["s3", "s3a"]):
    if isinstance(obj, Path):
        obj = str(obj)

    url_o = urlparse(obj, allow_fragments=False)
    bucket = url_o.netloc

    object_base = obj
    for scheme in s3_uri_schemes:
        object_base = object_base.lstrip(f"{scheme}://")
    object_base = object_base.lstrip(bucket + "/")

    return bucket, object_base


def path_exists(obj: str | Path, s3_uri_schemes: list[str] = ["s3", "s3a"]):
    if isinstance(obj, Path):
        obj = str(obj)

    if is_s3_path(obj, s3_uri_schemes=s3_uri_schemes):
        cfg = get_config()

        bucket, base = s3_base_path(obj, s3_uri_schemes=s3_uri_schemes)
        exists, _ = cfg.s3fs.lsf(bucket, base)
        return exists
    else:
        obj = Path(obj)
        return obj.exists() and obj.is_file()


def get_checkpoint_path(prefix: str, checkpoint_type: CHECKPOINT_TYPE):
    cfg = get_config()
    log = get_logger()
    if (
        checkpoint_type in CHECKPOINT_EXTENSION
        and CHECKPOINT_EXTENSION[checkpoint_type] is not None
    ):
        return f"{cfg.checkpoint}/{prefix}.{CHECKPOINT_EXTENSION[checkpoint_type]}"
    elif (
        checkpoint_type in CHECKPOINT_EXTENSION
        and CHECKPOINT_EXTENSION[checkpoint_type] is None
    ):
        log.debug(
            f"writing checkpoint with no defined extension, please add them to the path. (checkpoint_type = {checkpoint_type})"
        )
        return f"{cfg.checkpoint}/{prefix}"
    else:
        log.warning(f"invalid checkpoint type: checkpoint_type = {checkpoint_type}")
        return f"{cfg.checkpoint}/{prefix}"
