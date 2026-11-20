"""Serializer for an arbitrary picklable Python object (gzip pickle)."""

import gzip
import json
import pickle
from pathlib import Path

from ..paths import is_s3_path, path_exists
from .single_file import SingleFileSerializer


class PickleSerializer(SingleFileSerializer):
    def __init__(self, *args, s3_uri_schemes: list[str] = ["s3", "s3a"], **kwargs):
        super().__init__(*args, **kwargs)
        self.s3_uri_schemes = s3_uri_schemes

    @classmethod
    def read(cls, path: str | Path, **kwargs):
        from gnomad_meb.config import get_config
        from gnomad_meb.logging import get_logger

        logger = get_logger()
        logger.debug(f"reading pickle checkpoint {path}")
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
                        return pickle.load(in_f)
            except json.JSONDecodeError:
                logger.error(f"failed to parse pickle checkpoint file {path}")
                return None
        else:
            try:
                with gzip.open(self.path, "rb") as in_f:
                    return pickle.load(in_f)
            except json.JSONDecodeError:
                logger.error(f"failed to parse pickle checkpoint file {path}")
                return None

    def write(self, **kwargs):
        from gnomad_meb.config import get_config
        from gnomad_meb.logging import get_logger

        logger = get_logger()
        if self.data is None:
            logger.warning("checkpoint data is missing in call to write")
            return
        logger.debug(f"writing pickle checkpoint {self.path}")
        if is_s3_path(self.path, s3_uri_schemes=self.s3_uri_schemes):
            cfg = get_config()

            exists = path_exists(self.path, s3_uri_schemes=self.s3_uri_schemes)
            if exists and self.overwrite:
                logger.warning(f"overwriting pickle checkpoint {self.path}")
            elif exists:
                logger.error(
                    f"pickle checkpoint exists at {self.path}, overwrite = {self.overwrite}"
                )
                return

            with cfg.s3fs.fs.open(self.path, "wb") as raw_f:
                with gzip.open(raw_f, "wb") as out_f:
                    pickle.dump(self.data, out_f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            target = Path(self.path)
            if target.exists() and target.is_file() and self.overwrite:
                logger.warning(f"overwriting pickle checkpoint {self.path}")
            elif target.exists() and target.is_file():
                logger.error(
                    f"pickle checkpoint exists at {self.path}, overwrite = {self.overwrite}"
                )
                return

            if not target.parent.exists():
                target.parent.mkdir(exist_ok=True, parents=True)

            with gzip.open(target, "wb") as out_f:
                pickle.dump(self.data, out_f, protocol=pickle.HIGHEST_PROTOCOL)
