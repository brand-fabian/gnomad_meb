from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from gnomad_meb.logging import get_logger


logger = get_logger()


@dataclass(frozen=True)
class VdsLocation:
    base: Path | str
    name: str
    is_s3: bool = False
    s3_uri_scheme: str = "s3a"

    def __str__(self):
        if self.is_s3:
            return f"{self.s3_uri_scheme}://{self.base}/{self.name}"
        else:
            return str(self.as_path())

    def as_path(self):
        if self.is_s3:
            logger.warning(f"can not convert s3 location to path object: {self!s}")
            return str(self)
        else:
            return (Path(self.base) / self.name).resolve()

    def with_name(self, name: str):
        return VdsLocation(
            base=self.base,
            name=name,
            is_s3=self.is_s3,
        )

    def with_scheme(self, scheme: str):
        return VdsLocation(
            base=self.base,
            name=self.name,
            is_s3=self.is_s3,
            s3_uri_scheme=scheme,
        )

    @property
    def bucket(self):
        if self.is_s3:
            url_o = urlparse(str(self.with_scheme("s3")), allow_fragments=False)
            return url_o.netloc
        else:
            raise ValueError("non s3 paths can not have a bucket associated.")

    @classmethod
    def from_path(cls, base: str | Path, vds_name: str):
        if isinstance(base, str) and base.startswith("s3"):
            url_o = urlparse(base, allow_fragments=False)

            return cls(
                base=f"{url_o.netloc}{url_o.path}",
                name=vds_name,
                is_s3=True,
                s3_uri_scheme=url_o.scheme,
            )
        elif isinstance(base, str):
            target = Path(base)
        else:
            target = base

        vds_dir = (target / vds_name).resolve()
        vds_dir.mkdir(parents=True, exist_ok=True)
        return cls(base=target, name=vds_name, is_s3=False)
