"""The single artifact **and status** vocabulary of the pipeline (ADR-0011/0020).

This package owns how a ``@checkpoint``-decorated step declares its outputs,
resolves their paths, serializes them, records their provenance and reports their
status. It replaces the former 1291-line ``lib/checkpoint.py`` module: kinds and
extensions (:mod:`.types`), path helpers (:mod:`.paths`), completeness probes
(:mod:`.completeness`), the provenance sidecar (:mod:`.metadata`), the status
vocabulary (:mod:`.status`), the declared
:class:`~gnomad_meb.lib.checkpoint.artifact.Artifact`, the decorator
(:mod:`.decorator`, whose :class:`~gnomad_meb.lib.checkpoint.decorator.CheckpointedCallable`
is the one status interface) and one serializer per kind under
:mod:`~gnomad_meb.lib.checkpoint.serializers`.

Every re-export is lazy (PEP 562), including the light ones. Importing this
package must not import ``gnomad_meb.config`` or hail: :mod:`gnomad_meb.config`
imports :mod:`.status`, and an eager re-export would run this module's imports
mid-config-import and cycle. A caller importing a name still works exactly as it
did from the flat module.
"""

from importlib import import_module

#: name → the submodule that defines it. Resolved only on attribute access.
_EXPORTS = {
    "CHECKPOINT_EXTENSION": "types",
    "CHECKPOINT_TYPE": "types",
    "SuccessFileInfo": "types",
    "Artifact": "artifact",
    "CheckpointSpec": "artifact",
    "CheckpointMetadata": "metadata",
    "checkpoint_static": "metadata",
    "CHECKPOINT_SCHEMA_VERSION": "metadata",
    "is_s3_path": "paths",
    "s3_base_path": "paths",
    "path_exists": "paths",
    "get_checkpoint_path": "paths",
    "has_success_file": "completeness",
    "is_incomplete_hail_table": "completeness",
    "md5_file": "completeness",
    "CheckpointStatus": "status",
    "StalenessPolicy": "status",
    "StalenessReason": "status",
    "ArtifactResolution": "status",
    "CheckpointEvaluation": "status",
    "checkpoint": "decorator",
    "declares_step": "decorator",
    "checkpoint_overwrite": "decorator",
    "callable_source": "decorator",
    "CheckpointedCallable": "decorator",
    "Serializer": "serializers",
    "SingleFileSerializer": "serializers.single_file",
    "get_serializer": "serializers",
}


def __getattr__(name: str):
    """Resolve the lazily re-exported names (PEP 562)."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f"{__name__}.{module_name}"), name)


__all__ = list(_EXPORTS)
