"""Serializer for a pandas ``DataFrame`` (parquet, single file)."""

from pathlib import Path

import pandas

from ..paths import is_s3_path, path_exists
from .single_file import SingleFileSerializer


class DataFrameSerializer(SingleFileSerializer[pandas.DataFrame]):
    def __init__(self, *args, s3_uri_schemes: list[str] = ["s3", "s3a"], **kwargs):
        super().__init__(*args, **kwargs)
        self.s3_uri_schemes = s3_uri_schemes

    @classmethod
    def read(cls, path: str | Path, **kwargs):
        from gnomad_meb.config import get_config
        from gnomad_meb.logging import get_logger

        get_logger().debug(f"reading pandas dataframe checkpoint {path}")
        self = cls(path, **kwargs)
        if is_s3_path(self.path, s3_uri_schemes=self.s3_uri_schemes):
            if isinstance(self.path, Path):
                path = str(self.path.resolve())
            else:
                path = str(self.path)

            cfg = get_config()

            with cfg.s3fs.fs.open(path, "rb") as in_f:
                return pandas.read_parquet(in_f, **kwargs)
        else:
            return pandas.read_parquet(self.path, **kwargs)

    def write(self, **kwargs):
        from gnomad_meb.config import get_config
        from gnomad_meb.logging import get_logger

        logger = get_logger()
        if self.data is None:
            logger.warning("checkpoint data is missing in call to write")
            return
        logger.debug(f"writing pandas dataframe checkpoint {self.path}")
        if is_s3_path(self.path, s3_uri_schemes=self.s3_uri_schemes):
            cfg = get_config()

            exists = path_exists(self.path, s3_uri_schemes=self.s3_uri_schemes)
            if exists and self.overwrite:
                logger.warning(f"overwriting pandas dataframe checkpoint {self.path}")
            elif exists:
                logger.error(
                    f"pandas dataframe checkpoint exists at {self.path}, overwrite = {self.overwrite}"
                )
                return

            with cfg.s3fs.fs.open(self.path, "wb") as out_f:
                self.data.to_parquet(out_f, **kwargs)
        else:
            target = Path(self.path)
            if target.exists() and target.is_file() and self.overwrite:
                logger.warning(f"overwriting pandas dataframe checkpoint {self.path}")
            elif target.exists() and target.is_file():
                logger.error(
                    f"pandas dataframe checkpoint exists at {self.path}, overwrite = {self.overwrite}"
                )
                return

            if not target.parent.exists():
                target.parent.mkdir(exist_ok=True, parents=True)

            self.data.to_parquet(self.path, **kwargs)

    def is_checkpoint(self) -> bool:
        return False
