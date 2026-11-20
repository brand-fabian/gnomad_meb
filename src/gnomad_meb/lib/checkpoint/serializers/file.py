"""Serializer for an arbitrary single-file artifact (raw bytes).

The canonical kind for arbitrary files (``*.vcf.bgz``, ``*.onnx``, a VDS
``*.metadata.json`` handle). It reuses :class:`SingleFileSerializer` for
completeness and metadata (so ``_SUCCESS``-free single files and S3 are
handled in one place) and reads/writes the raw bytes.
"""

from pathlib import Path

from ..paths import is_s3_path
from .single_file import SingleFileSerializer


class FileSerializer(SingleFileSerializer):
    """A single-file artifact whose contents are not otherwise interpreted."""

    def __init__(self, *args, s3_uri_schemes: list[str] = ["s3", "s3a"], **kwargs):
        super().__init__(*args, **kwargs)
        self.s3_uri_schemes = s3_uri_schemes

    @classmethod
    def read(cls, path: str | Path, **kwargs):
        from gnomad_meb.config import get_config
        from gnomad_meb.logging import get_logger

        get_logger().debug(f"reading file checkpoint {path}")
        self = cls(path, **kwargs)
        if is_s3_path(self.path, s3_uri_schemes=self.s3_uri_schemes):
            cfg = get_config()
            with cfg.s3fs.fs.open(self.path, "rb") as in_f:
                return in_f.read()
        with open(self.path, "rb") as in_f:
            return in_f.read()

    def write(self, **kwargs):
        from gnomad_meb.config import get_config
        from gnomad_meb.logging import get_logger

        logger = get_logger()
        if self.data is None:
            logger.warning("checkpoint data is missing in call to write")
            return
        logger.debug(f"writing file checkpoint {self.path}")
        data = self.data
        payload = (
            bytes(data) if isinstance(data, bytes | bytearray) else str(data).encode()
        )
        if is_s3_path(self.path, s3_uri_schemes=self.s3_uri_schemes):
            cfg = get_config()
            with cfg.s3fs.fs.open(self.path, "wb") as out_f:
                out_f.write(payload)
        else:
            target = Path(self.path)
            if not target.parent.exists():
                target.parent.mkdir(exist_ok=True, parents=True)
            target.write_bytes(payload)
