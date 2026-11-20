"""Serializer for a ``dict`` (gzip-compressed JSON, single file)."""

import gzip
import json
from pathlib import Path

from gnomad_meb.hail import HailStructEncoder

from ..paths import is_s3_path, path_exists
from .single_file import SingleFileSerializer


class DictionarySerializer(SingleFileSerializer[dict]):
    def __init__(self, *args, s3_uri_schemes: list[str] = ["s3", "s3a"], **kwargs):
        super().__init__(*args, **kwargs)
        self.s3_uri_schemes = s3_uri_schemes

    @classmethod
    def read(cls, path: str | Path, **kwargs):
        from gnomad_meb.config import get_config
        from gnomad_meb.logging import get_logger

        logger = get_logger()
        logger.debug(f"reading dictionary checkpoint {path}")
        self = cls(path, **kwargs)
        if is_s3_path(self.path, s3_uri_schemes=self.s3_uri_schemes):
            if isinstance(self.path, Path):
                path = str(self.path.resolve())
            else:
                path = str(self.path)

            cfg = get_config()

            try:
                with cfg.s3fs.fs.open(path, "rb") as raw_f:
                    with gzip.open(raw_f, "rb") as in_f:
                        return json.load(in_f)
            except json.JSONDecodeError:
                logger.error(f"failed to parse dictionary checkpoint file {path}")
                return {}
        else:
            try:
                with gzip.open(self.path, "rb") as in_f:
                    return json.load(in_f)
            except json.JSONDecodeError:
                logger.error(f"failed to parse dictionary checkpoint file {path}")
                return {}

    def write(self, **kwargs):
        from gnomad_meb.config import get_config
        from gnomad_meb.logging import get_logger

        logger = get_logger()
        if self.data is None:
            logger.warning("checkpoint data is missing in call to write")
            return
        logger.debug(f"writing dictionary checkpoint {self.path}")
        if is_s3_path(self.path, s3_uri_schemes=self.s3_uri_schemes):
            cfg = get_config()

            exists = path_exists(self.path, s3_uri_schemes=self.s3_uri_schemes)
            if exists and self.overwrite:
                logger.warning(f"overwriting dictionary checkpoint {self.path}")
            elif exists:
                logger.error(
                    f"dictionary checkpoint exists at {self.path}, overwrite = {self.overwrite}"
                )
                return

            with cfg.s3fs.fs.open(self.path, "wb") as raw_f:
                with gzip.open(raw_f, "wb") as out_f:
                    json_d = json.dumps(self.data, indent=4, cls=HailStructEncoder)
                    out_f.write(json_d.encode())
        else:
            target = Path(self.path)
            if target.exists() and target.is_file() and self.overwrite:
                logger.warning(f"overwriting dictionary checkpoint {self.path}")
            elif target.exists() and target.is_file():
                logger.error(
                    f"dictionary checkpoint exists at {self.path}, overwrite = {self.overwrite}"
                )
                return

            if not target.parent.exists():
                target.parent.mkdir(exist_ok=True, parents=True)

            with gzip.open(target, "wb") as out_f:
                json_d = json.dumps(self.data, indent=4, cls=HailStructEncoder)
                out_f.write(json_d.encode())

    def is_checkpoint(self) -> bool:
        return False
