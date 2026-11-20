"""The checkpoint provenance sidecar model.

``CheckpointMetadata`` is the content of one ``<path>.gnomad_meb.md.json``
sidecar. It is a **versioned pydantic model**: the on-disk keys are unchanged
(absent fields default to current values, unknown keys are ignored), and a new
``schema_version`` lets a future reader detect and upgrade an older file without
ever breaking the format.

The ``to_dict``/``from_dict``/``to_json_file``/``from_json_file`` surface is the
on-disk contract.
"""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from gnomad_meb.config import get_config
from gnomad_meb.lib.utils import get_git_revision, get_hostname, get_version

from .paths import is_s3_path

#: The current sidecar schema version. Written into every new sidecar; a reader
#: refuses a version it does not know rather than silently mis-reading fields.
CHECKPOINT_SCHEMA_VERSION = 1


class UnsupportedSchemaVersion(ValueError):
    """Raised when a sidecar declares a ``schema_version`` we cannot read."""


def checkpoint_static():
    """Return kwargs for checkpoints that are equal accross all types."""
    return {
        "version": get_version(),
        "git_revision": get_git_revision(),
        "host": get_hostname(),
    }


class CheckpointMetadata(BaseModel):
    """The provenance sidecar of one checkpoint artifact.

    Parsed from ``<path>.gnomad_meb.md.json``. Fields absent in an older sidecar
    default to current values, and ``schema_version`` lets a future reader detect
    and upgrade an older file, so the on-disk format never has to break. Unknown
    keys are ignored.
    """

    model_config = ConfigDict(extra="ignore")

    schema_version: int = CHECKPOINT_SCHEMA_VERSION
    path: str
    exists: bool = False
    mtime: str | None = None
    data_md5: str | None = None
    source_md5: str | None = None
    host: str = Field(default_factory=get_hostname)
    version: str = Field(default_factory=get_version)
    git_revision: str = Field(default_factory=get_git_revision)

    def matches_path(self, path: str | Path) -> bool:
        """Whether this sidecar was written for ``path``.

        A sidecar whose recorded ``path`` disagrees with the artifact path it
        was found beside is a foreign or stale file and must not be trusted.
        """
        if isinstance(path, Path):
            path = str(path.resolve())
        return self.path == str(path)

    def to_dict(self) -> dict[str, Any]:
        """The sidecar's on-disk dict: the eight historical keys plus the version."""
        return {
            "schema_version": self.schema_version,
            "path": self.path,
            "exists": self.exists,
            "mtime": self.mtime,
            "data_md5": self.data_md5,
            "source_md5": self.source_md5,
            "host": self.host,
            "version": self.version,
            "git_revision": self.git_revision,
        }

    def to_json_file(self, path: str | Path, s3_uri_schemes: list[str] = ["s3", "s3a"]):
        if isinstance(path, Path):
            path = str(path.resolve())

        if is_s3_path(path, s3_uri_schemes=s3_uri_schemes):
            cfg = get_config()

            json_b = json.dumps(self.to_dict()).encode("utf-8")
            with cfg.s3fs.fs.open(path, "wb") as out_f:
                out_f.write(json_b)
        else:
            with open(path, "w") as out_f:
                json.dump(self.to_dict(), out_f)

    @classmethod
    def from_dict(cls, obj: dict[str, Any]):
        """Build the model from a parsed sidecar dict.

        Absent keys default to current values; an explicitly declared
        ``schema_version`` newer than :data:`CHECKPOINT_SCHEMA_VERSION` is
        refused (a reader must not guess at unknown fields' meaning).
        """
        version = obj.get("schema_version", CHECKPOINT_SCHEMA_VERSION)
        if version > CHECKPOINT_SCHEMA_VERSION:
            raise UnsupportedSchemaVersion(
                f"unsupported checkpoint schema_version: found = {version}, "
                f"supported = {CHECKPOINT_SCHEMA_VERSION}"
            )
        return cls(**obj)

    @classmethod
    def from_json_file(
        cls, path: str | Path, s3_uri_schemes: list[str] = ["s3", "s3a"]
    ):
        if isinstance(path, Path):
            path = str(path.resolve())

        if is_s3_path(path, s3_uri_schemes=s3_uri_schemes):
            cfg = get_config()

            with cfg.s3fs.fs.open(path, "rb") as in_f:
                data = in_f.read().decode("utf-8")
            return cls._from_json_text(data, path)
        else:
            with open(path) as in_f:
                data = in_f.read()
            return cls._from_json_text(data, path)

    @classmethod
    def _from_json_text(cls, data: str, path: str):
        """Parse ``data``, naming ``path`` on a corrupt-JSON failure."""
        try:
            obj = json.loads(data)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"corrupt checkpoint sidecar JSON: path = {path}, error = {exc}"
            ) from exc
        return cls.from_dict(obj)
