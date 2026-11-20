"""Configuration: pydantic-settings models and the resolution entry points.

``resolve(config=None, overrides=None)`` is the pure entry point; it returns a
resolved :class:`Settings` purely from its inputs and the process environment,
with no hail session. ``get_config(config=None)`` is the process-global accessor
the rest of the library reads: a thin cache over :func:`resolve`.

Precedence, for every field, is a single declarative order:

    command-line override > ``GMEB_`` environment variable > TOML file > default
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gnomad_meb.config.settings import (
    ENV_NESTED_DELIMITER,
    ENV_PREFIX,
    INVALID_TOKEN,
    AnnotationConfig,
    AnnotationResources,
    BrowserExportConfig,
    FilterCheckpoint,
    FilterConfig,
    FilterResources,
    GwasConfig,
    GwasCovariate,
    GwasRegion,
    GwasResponseColumn,
    GwasResponses,
    HailConfig,
    RegionCheckpoint,
    Resources,
    S3Config,
    SampleQcConfig,
    SampleQcHardFilterCutoffs,
    Settings,
    VariantQcConfig,
    VariantQcSensitivityCutoffs,
    VariantQcVqsrParams,
    VepAnnotationResources,
    VqsrConfig,
    VqsrResources,
    current_config_file,
)

__all__ = [
    "ENV_NESTED_DELIMITER",
    "ENV_PREFIX",
    "INVALID_TOKEN",
    "AnnotationConfig",
    "AnnotationResources",
    "BrowserExportConfig",
    "FilterCheckpoint",
    "FilterConfig",
    "FilterResources",
    "GwasConfig",
    "GwasCovariate",
    "GwasRegion",
    "GwasResponseColumn",
    "GwasResponses",
    "HailConfig",
    "RegionCheckpoint",
    "Resources",
    "S3Config",
    "SampleQcConfig",
    "SampleQcHardFilterCutoffs",
    "Settings",
    "VariantQcConfig",
    "VariantQcSensitivityCutoffs",
    "VariantQcVqsrParams",
    "VepAnnotationResources",
    "VqsrConfig",
    "VqsrResources",
    "get_config",
    "resolve",
    "set_config",
]

_config: Settings | None = None
_config_path: Path | None = None


def resolve(
    config: Path | str | None = None,
    overrides: dict[str, Any] | None = None,
) -> Settings:
    """Resolve the settings from the config file, environment and overrides.

    The single TOML file is optional; when given it must be an existing file.
    The process environment is read under the ``GMEB_`` prefix, and ``overrides``
    (command-line values) win over everything.
    """
    path = Path(config) if config is not None else None
    if path is not None and not path.is_file():
        raise FileNotFoundError(f"configuration file not found: {path}")

    token = current_config_file.set(path)
    try:
        return Settings(**(overrides or {}))
    finally:
        current_config_file.reset(token)


def get_config(config: Path | str | None = None) -> Settings:
    """Return the process-global settings, resolving on first use.

    Supplying a ``config`` path resolves with it; a later call without a path
    returns the cached result rather than re-resolving without the file.
    """
    global _config, _config_path

    if config is not None:
        path = Path(config)
        if _config is None or path != _config_path:
            _config = resolve(path)
            _config_path = path
    elif _config is None:
        _config = resolve()

    return _config


def set_config(
    settings: Settings, config: Path | str | None = None
) -> Settings:
    """Install ``settings`` as the process-global configuration.

    The CLI resolves the settings (including command-line overrides) and installs
    the result so the ambient :func:`get_config` used by library code returns it.
    """
    global _config, _config_path
    _config = settings
    _config_path = Path(config) if config is not None else None
    return settings
