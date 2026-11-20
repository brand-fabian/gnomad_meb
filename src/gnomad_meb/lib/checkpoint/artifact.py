"""The declared outputs of a checkpointed step.

``Artifact`` is the single artifact vocabulary (ADR-0011) and ``CheckpointSpec``
is the declaration a decorated callable carries: its artifacts and the
read/write options the wrapper uses.
"""

import os
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from gnomad_meb.config import Settings, get_config

from .serializers import get_serializer
from .types import CHECKPOINT_EXTENSION, CHECKPOINT_TYPE


@dataclass(frozen=True)
class Artifact:
    """A declared output: the single artifact vocabulary of the pipeline.

    ``prefix`` is relative to the run's ``checkpoint`` root and may carry a
    ``{project}``/``{checkpoint}`` template resolved against the run's settings.
    ``kind`` names how the artifact is serialized; ``suffix`` overrides the
    kind's default extension for artifacts that keep a different one
    (``.vcf.bgz``, ``.onnx``, ``.metadata.json``).
    """

    prefix: str
    kind: CHECKPOINT_TYPE
    suffix: str | None = None

    def extension(self) -> str | None:
        """The effective extension: the ``suffix`` override, else the kind's.

        ``None`` for an extension-less kind (``FILE``, ``PLOTNINE``), whose
        prefix carries its own extension.
        """
        return (
            self.suffix
            if self.suffix is not None
            else CHECKPOINT_EXTENSION.get(self.kind)
        )

    @classmethod
    def from_path(cls, path: str, kind: CHECKPOINT_TYPE | None = None) -> "Artifact":
        """Declare an artifact for an already-resolved ``path``.

        If ``kind`` is None, then it is inferred by the path suffix of the
        given output file. If the suffix is unknown a default FILE kind is used.

        The path's directory and stem become the ``prefix`` and its extension
        becomes the ``suffix`` (``out/as_info.vcf.bgz`` → prefix ``out/as_info``
        with suffix ``vcf.bgz``), so :meth:`path` reconstructs it exactly.
        """
        if kind is None:
            for kind, extension in CHECKPOINT_EXTENSION.items():
                if extension is not None and path.endswith(extension):
                    break
            else:
                kind = CHECKPOINT_TYPE.FILE

        directory, _, name = str(path).rpartition("/")
        stem, dot, suffix = name.partition(".")
        prefix = f"{directory}/{stem}" if directory else stem
        if dot:
            return cls(prefix=prefix, kind=kind, suffix=suffix)
        return cls(prefix=path, kind=kind)

    def path(self, cfg: Settings | None = None) -> str:
        """The on-disk path, ``{checkpoint}/{prefix}.<ext>``.

        The prefix template is formatted against ``cfg`` (the ambient config by
        default) so a declaration made at class-definition time follows the
        settings the run installs. A prefix that carries no ``{...}`` template is
        used verbatim, so an already-resolved path (an absolute output location)
        is preserved.
        """
        cfg = cfg if cfg is not None else get_config()
        prefix = self.prefix.format(project=cfg.project, checkpoint=cfg.checkpoint)
        if os.path.isabs(prefix):
            base = prefix
        else:
            base = f"{cfg.checkpoint}/{prefix}"
        ext = self.extension()
        return f"{base}.{ext}" if ext else base

    @property
    def serializer(self) -> type:
        """The serializer class for this artifact's kind."""
        return get_serializer(self.kind)

    def exists(self, cfg: Settings | None = None) -> bool:
        """Whether the artifact is present and complete.

        Delegates to the kind's serializer, so ``_SUCCESS``, the metadata
        sidecar and S3 are handled by ``lib.checkpoint`` rather than
        re-implemented by callers.
        """
        return self.serializer(self.path(cfg)).exists()


class CheckpointSpec(BaseModel):
    """The declaration a ``checkpoint``-decorated callable carries.

    Attached to the wrapper as ``__checkpoint__`` so a decorated step method is
    introspectable: its artifacts, and the read/write options the wrapper uses.
    """

    artifacts: tuple[Artifact, ...]
    write_metadata: bool = True
    read_kwargs: dict[str, Any] = {}
    write_kwargs: dict[str, Any] = {}
