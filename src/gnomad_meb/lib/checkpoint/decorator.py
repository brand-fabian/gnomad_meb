"""The ``checkpoint`` decorator: cache a function's outputs to artifacts.

The decorator returns a :class:`CheckpointedCallable` — a callable instance that
is introspectable **without invoking the step**. It carries the declaration, the
wrapped body and its keywords (so edge parsing lives here, not in the pipeline),
and the status query surface the runner uses to skip a step before scheduling it
(ADR-0020).
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from hashlib import md5
from typing import Any

from gnomad_meb.config import Settings, get_config
from gnomad_meb.lib.constants import DEFAULT_METADATA_SUFFIX
from gnomad_meb.logging import get_logger

from .artifact import Artifact, CheckpointSpec
from .metadata import CheckpointMetadata
from .serializers import get_serializer
from .status import (
    ArtifactResolution,
    CheckpointEvaluation,
    CheckpointStatus,
    StalenessPolicy,
    StalenessReason,
    verdict,
)
from .types import CHECKPOINT_TYPE

logger = get_logger()


def _unwrap_for_source(fn: Any) -> Any:
    """Follow ``functools.partial``/``functools.wraps`` to the body to hash.

    A generated step is a ``functools.partial`` of a plain body; its ``.func``
    chain (and any ``__wrapped__``) leads to the real source. Following it makes
    ``source()`` stable across equivalent partial instances and independent of
    bound values/addresses.
    """
    seen: set[int] = set()
    while id(fn) not in seen:
        seen.add(id(fn))
        if isinstance(fn, functools.partial):
            fn = fn.func
            continue
        wrapped = getattr(fn, "__wrapped__", None)
        if wrapped is not None:
            fn = wrapped
            continue
        break
    return fn


def callable_source(fn: Any) -> str:
    """The source of ``fn`` for its checkpoint metadata (the ``source_md5`` input).

    ``inspect.getsource`` needs a module/class/function/code object; a
    ``functools.partial`` has none. Follow the partial / ``__wrapped__`` chain to
    the underlying body and hash *its* source, so a generated step hashes
    identically across runs (a ``repr`` fallback would leak object addresses and
    make every generated checkpoint stale). Only when nothing sourceable is found
    does this fall back to ``repr``.
    """
    target = _unwrap_for_source(fn)
    try:
        return inspect.getsource(target)
    except (OSError, TypeError):
        return repr(target)


def declares_step(fn):
    """Mark ``fn`` as a pipeline step without wrapping it in a checkpoint.

    Some steps are side effects or self-resuming (a VCF export, the VQSR
    Snakemake delegation, the VDS build). None should run through the
    ``@checkpoint`` cache wrapper, so this returns a
    :class:`CheckpointedCallable` with **no artifacts** — the same declaration
    the DAG reads, but a direct pass-through whose ``evaluate().empty`` is True
    and is never a cache hit. Returning the class (rather than the bare ``fn``)
    keeps discovery uniform.
    """
    return CheckpointedCallable(
        fn=fn,
        spec=CheckpointSpec(artifacts=()),
        keywords=dict(getattr(fn, "keywords", {}) or {}),
    )


def _checkpoint_declarations(
    prefix: str | Iterable[str],
    checkpoint_type: CHECKPOINT_TYPE | Iterable[CHECKPOINT_TYPE],
) -> list[tuple[str, CHECKPOINT_TYPE]]:
    """Normalise the ``(prefix, kind)`` arguments into a flat declaration list."""
    if isinstance(prefix, Iterable) and not isinstance(prefix, str):
        if isinstance(checkpoint_type, Iterable):
            prefixes = tuple(prefix)
            types = tuple(checkpoint_type)
            assert len(prefixes) == len(types), (
                f"mismatching checkpoint prefixes and checkpoint types at "
                f"{prefixes[0]}: {len(prefixes)} != {len(types)}"
            )
            return list(zip(prefixes, types, strict=False))
        return [(p, checkpoint_type) for p in prefix]
    if isinstance(checkpoint_type, Iterable):
        checkpoint_type = next(iter(checkpoint_type))
    return [(prefix, checkpoint_type)]


#: Whether the ``checkpoint`` wrapper should ignore an existing checkpoint and
#: recompute. Scoped as a :class:`~contextvars.ContextVar` so a caller (the run
#: engine) can request an overwrite for the duration of one step without
#: turning cache-busting into a process-global.
_CHECKPOINT_OVERWRITE: ContextVar[bool] = ContextVar(
    "checkpoint_overwrite", default=False
)


@contextmanager
def checkpoint_overwrite(flag: bool = True):
    """Recompute inside this block, ignoring an existing checkpoint.

    While active, a ``@checkpoint``-decorated call skips the cache-hit read,
    calls the wrapped function again, and writes its result with
    ``overwrite=True``. The flag is a :class:`~contextvars.ContextVar`, so it
    is scoped to the current context (and reset on exit).
    """
    token = _CHECKPOINT_OVERWRITE.set(flag)
    try:
        yield
    finally:
        _CHECKPOINT_OVERWRITE.reset(token)


class CheckpointedCallable:
    """A ``@checkpoint``-decorated callable, introspectable without invoking it.

    Carries the declaration (``spec``), the wrapped body (``fn``) and its
    keywords (exposed as attributes so edge parsing lives here, not in the
    pipeline), the resolved artifacts, and the status query surface the runner
    uses to skip a step before scheduling it. A per-instance stale flag
    (``force_recompute``) makes a runner verdict of ``STALE`` re-execute on
    ``__call__`` instead of returning the stale artifact.
    """

    def __init__(
        self,
        fn: Any,
        spec: CheckpointSpec,
        keywords: Mapping[str, Any] | None = None,
        metadata_suffix: str = DEFAULT_METADATA_SUFFIX,
    ) -> None:
        self.fn = fn
        self.spec = spec
        self.keywords: Mapping[str, Any] = dict(keywords or {})
        self.metadata_suffix = metadata_suffix
        self._force_recompute = False

        functools.update_wrapper(self, fn, updated=())
        # ``__signature__`` lets ``inspect.signature`` surface the wrapped body's
        # parameters even when the body is hidden behind a partial.
        try:
            self.__signature__ = inspect.signature(fn)
        except (TypeError, ValueError):
            pass

    # --- descriptor protocol: bind an instance as ``self`` -------------------

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        return CheckpointedCallable(
            fn=functools.partial(self.fn, instance),
            spec=self.spec,
            keywords=self.keywords,
            metadata_suffix=self.metadata_suffix,
        )

    # --- introspection without invoking --------------------------------------

    def source(self) -> str:
        """A stable source of the wrapped body (the ``source_md5`` input)."""
        return callable_source(self.fn)

    def _resolutions(
        self,
        cfg: Settings,
        policy: StalenessPolicy,
        invalidated_by_lineage: bool,
    ) -> tuple[ArtifactResolution, ...]:
        """Resolve every declared artifact's status once (the shared read)."""
        src = self.source()
        current_source_md5 = md5(src.encode("utf-8")).hexdigest()
        resolutions: list[ArtifactResolution] = []

        for artifact in self.spec.artifacts:
            path = artifact.path(cfg)
            serializer_cls = get_serializer(artifact.kind)
            ser = serializer_cls(path, overwrite=True)
            # ``metadata`` reads the artifact's own completeness (``_SUCCESS``,
            # file presence) and the current run's provenance.
            current = ser.metadata(source=src)
            stored = current.exists

            stored_metadata: CheckpointMetadata | None = None
            sidecar = f"{path}.{self.metadata_suffix}"
            if self._sidecar_exists(sidecar):
                try:
                    stored_metadata = CheckpointMetadata.from_json_file(sidecar)
                    if not stored_metadata.matches_path(path):
                        stored_metadata = None
                except (ValueError, OSError):
                    # A corrupt or unreadable sidecar is treated as absent
                    # provenance: the artifact is present but unverified.
                    stored_metadata = None

            status, reasons = verdict(
                stored_metadata,
                stored=stored,
                policy=policy,
                invalidated_by_lineage=invalidated_by_lineage,
                current_source_md5=current_source_md5,
                current_git_revision=current.git_revision,
                current_version=current.version,
            )
            resolutions.append(
                ArtifactResolution(
                    artifact=artifact,
                    path=path,
                    metadata=stored_metadata,
                    status=status,
                    reasons=reasons,
                )
            )
        return tuple(resolutions)

    @staticmethod
    def _sidecar_exists(sidecar: str) -> bool:
        from .paths import is_s3_path

        if is_s3_path(sidecar):
            from .paths import path_exists

            return path_exists(sidecar)
        from pathlib import Path

        return Path(sidecar).is_file()

    def evaluate(
        self,
        cfg: Settings | None = None,
        *,
        policy: StalenessPolicy | None = None,
        invalidated_by_lineage: bool = False,
    ) -> CheckpointEvaluation:
        """Resolve paths + sidecars once; return status, reasons and resolutions."""
        cfg = cfg if cfg is not None else get_config()
        policy = policy if policy is not None else getattr(
            cfg, "staleness_policy", StalenessPolicy.COMPLETENESS
        )
        if not self.spec.artifacts:
            return CheckpointEvaluation(
                status=CheckpointStatus.PRESENT,
                reasons=(),
                resolutions=(),
                reusable=False,
                empty=True,
            )

        resolutions = self._resolutions(cfg, policy, invalidated_by_lineage)
        aggregate = _aggregate(resolutions)
        reasons = tuple(r for res in resolutions for r in res.reasons)
        reusable = aggregate is CheckpointStatus.PRESENT or (
            aggregate is CheckpointStatus.UNKNOWN
            and policy is StalenessPolicy.COMPLETENESS
        )
        return CheckpointEvaluation(
            status=aggregate,
            reasons=reasons,
            resolutions=resolutions,
            reusable=reusable,
            empty=False,
        )

    def status(
        self,
        cfg: Settings | None = None,
        *,
        policy: StalenessPolicy | None = None,
        invalidated_by_lineage: bool = False,
    ) -> CheckpointStatus:
        """The aggregate status (the enum half of ``evaluate``)."""
        return self.evaluate(
            cfg, policy=policy, invalidated_by_lineage=invalidated_by_lineage
        ).status

    def metadata(
        self, cfg: Settings | None = None
    ) -> tuple[CheckpointMetadata | None, ...]:
        """The sidecar metadata per declared artifact (``None`` when absent)."""
        cfg = cfg if cfg is not None else get_config()
        return tuple(
            resolution.metadata
            for resolution in self.evaluate(cfg).resolutions
        )

    def read(self, cfg: Settings | None = None) -> Any:
        """Load the declared artifacts (one value, or a tuple) — no invocation."""
        cfg = cfg if cfg is not None else get_config()
        values = []
        for artifact in self.spec.artifacts:
            serializer_cls = get_serializer(artifact.kind)
            ser = serializer_cls(artifact.path(cfg), overwrite=True)
            if not ser.is_restorable():
                raise ValueError(
                    f"artifact kind is not restorable: kind = {artifact.kind}"
                )
            values.append(
                serializer_cls.read(artifact.path(cfg), **self.spec.read_kwargs)
            )
        if not values:
            return None
        return values[0] if len(values) == 1 else tuple(values)

    # --- runner verdict hooks ------------------------------------------------

    def mark_stale(self, reasons: tuple[StalenessReason, ...]) -> None:
        """Record that the next ``__call__`` must recompute (runner verdict)."""
        self._force_recompute = True
        self._stale_reasons: tuple[StalenessReason, ...] = tuple(reasons)

    def mark_reusable(self) -> None:
        """Clear the stale flag after a clean ``PRESENT``/reuse verdict."""
        self._force_recompute = False

    @property
    def force_recompute(self) -> bool:
        """Whether the next ``__call__`` must recompute regardless of cache."""
        return self._force_recompute

    # --- the call ------------------------------------------------------------

    def __call__(self, *args, **kwargs) -> Any:
        """Run through the checkpoint; honours overwrite *and* the stale flag."""
        cfg = get_config()
        log = get_logger()
        if cfg.checkpoint is None:
            log.warning(
                "checkpointing is disabled in config, set 'checkpoint' to "
                "write outputs"
            )
            return self.fn(*args, **kwargs)

        if not self.spec.artifacts:
            return self.fn(*args, **kwargs)

        policy = getattr(cfg, "staleness_policy", StalenessPolicy.COMPLETENESS)
        overwrite = _CHECKPOINT_OVERWRITE.get() or self._force_recompute

        ops = [
            (
                artifact.path(cfg),
                f"{artifact.path(cfg)}.{self.metadata_suffix}",
                get_serializer(artifact.kind),
            )
            for artifact in self.spec.artifacts
        ]
        src = self.source()

        if len(ops) == 1:
            target_path, md_path, serializer_cls = ops[0]
            ser = serializer_cls(target_path, overwrite=True)
            if not overwrite:
                evaluation = self.evaluate(cfg, policy=policy)
                if evaluation.reusable:
                    if evaluation.status is CheckpointStatus.UNKNOWN:
                        log.warning(
                            f"reusing checkpoint with no provenance sidecar: "
                            f"path = {target_path}"
                        )
                    return serializer_cls.read(
                        target_path, **self.spec.read_kwargs
                    )
            ser.data = self.fn(*args, **kwargs)
            ser.write(**self.spec.write_kwargs)
            if self.spec.write_metadata:
                ser.metadata(source=src).to_json_file(md_path)
            self.mark_reusable()
            if ser.is_checkpoint():
                return serializer_cls.read(target_path, **self.spec.read_kwargs)
            return ser.data

        existing_data: list[Any] = []
        evaluation = self.evaluate(cfg, policy=policy)
        if not overwrite and evaluation.reusable:
            for target_path, _, serializer_cls in ops:
                existing_data.append(
                    serializer_cls.read(target_path, **self.spec.read_kwargs)
                )
            if evaluation.status is CheckpointStatus.UNKNOWN:
                log.warning(
                    f"reusing checkpoints with no provenance sidecar: "
                    f"paths = {[op[0] for op in ops]}"
                )
            return tuple(existing_data)

        result = self.fn(*args, **kwargs)
        assert len(result) == len(ops), (
            f"mismatch between the number of expected return values "
            f"of the function ({len(ops)}), and the "
            f"actual number of return values ({len(result)})"
        )
        final_result = []
        for partial_result, op in zip(result, ops, strict=False):
            target_path, md_path, serializer_cls = op
            ser = serializer_cls(target_path, overwrite=True)
            ser.data = partial_result
            ser.write(**self.spec.write_kwargs)
            if self.spec.write_metadata:
                ser.metadata(source=src).to_json_file(md_path)
            if ser.is_checkpoint():
                final_result.append(
                    serializer_cls.read(target_path, **self.spec.read_kwargs)
                )
            else:
                final_result.append(partial_result)
        self.mark_reusable()
        return tuple(final_result)


def _aggregate(resolutions: tuple[ArtifactResolution, ...]) -> CheckpointStatus:
    """Local copy of the aggregate precedence (kept import-light)."""
    from .status import aggregate_status

    return aggregate_status(resolutions)


def checkpoint(
    prefix: str | Iterable[str],
    checkpoint_type: CHECKPOINT_TYPE | Iterable[CHECKPOINT_TYPE],
    write_metadata: bool = True,
    metadata_suffix: str = DEFAULT_METADATA_SUFFIX,
    read_kwargs: dict[str, Any] = {},
    write_kwargs: dict[str, Any] = {},
):
    """Cache a function's outputs to ``Artifact`` checkpoints.

    Returns a :class:`CheckpointedCallable`. Paths resolve at *call* time against
    the ambient :func:`get_config`, so a method decorated at class-definition
    time follows the settings the run installs. When no checkpoint root is
    configured the callable is an identity pass-through.
    """
    spec = CheckpointSpec(
        artifacts=tuple(
            Artifact(prefix=p, kind=ct)
            for p, ct in _checkpoint_declarations(prefix, checkpoint_type)
        ),
        write_metadata=write_metadata,
        read_kwargs=read_kwargs,
        write_kwargs=write_kwargs,
    )

    def decorator(fn):
        return CheckpointedCallable(
            fn=fn,
            spec=spec,
            keywords=dict(getattr(fn, "keywords", {}) or {}),
            metadata_suffix=metadata_suffix,
        )

    return decorator
