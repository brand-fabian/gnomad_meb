"""Serializer for a ``plotnine`` plot (image file, not restorable)."""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile, gettempdir

import plotnine as pn

from ..paths import is_s3_path
from .single_file import SingleFileSerializer


class PlotnineSerializer(SingleFileSerializer[pn.ggplot]):
    def __init__(self, *args, s3_uri_schemes: list[str] = ["s3", "s3a"], **kwargs):
        super().__init__(*args, **kwargs)
        self.s3_uri_schemes = s3_uri_schemes

    @classmethod
    def read(cls, path: str | Path, **kwargs):
        raise ValueError("cannot restore from PlotnineSerializer checkpoint")

    def write(self, **kwargs):
        from gnomad_meb.config import get_config
        from gnomad_meb.logging import get_logger

        logger = get_logger()
        if self.data is None:
            logger.warning("checkpoint data is missing in call to write")
            return

        logger.debug(f"writing plotnine checkpoint {self.path}")

        # Store the plot to a temporary file and copy it to the final location:
        suffix = self.path.split(".")[-1]
        with NamedTemporaryFile(
            dir=os.getenv("SCRATCH_DIR", gettempdir()),
            suffix=f".{suffix}",
            prefix="plotnine-checkpoint-",
        ) as tempf:
            self.data.save(tempf.name, **kwargs)

            if is_s3_path(self.path, s3_uri_schemes=self.s3_uri_schemes):
                cfg = get_config()
                with (
                    cfg.s3fs.fs.open(self.path, "wb") as out_f,
                    open(tempf.name, "rb") as in_f,
                ):
                    out_f.write(in_f.read())
            else:
                target = Path(self.path)
                if target.exists() and target.is_file() and self.overwrite:
                    logger.warning(f"overwriting plotnine checkpoint {self.path}")
                elif target.exists() and target.is_file():
                    logger.error(
                        f"plotnine checkpoint exists at {self.path}, overwrite = {self.overwrite}"
                    )
                    return

                with open(target, "wb") as out_f, open(tempf.name, "rb") as in_f:
                    out_f.write(in_f.read())

    def is_checkpoint(self) -> bool:
        return False

    def is_restorable(self) -> bool:
        return False
