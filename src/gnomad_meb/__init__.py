"""gnomad-meb, a hail-backed genomics pipeline tool.

The package root deliberately does **not** import hail: importing any submodule
— notably the hail-free :mod:`gnomad_meb.pipeline` core — must not pull in the
hail stack. Hail-dependent symbols are resolved lazily on attribute access.
"""

from __future__ import annotations

import importlib

from .s3 import S3ConnectionOptions, S3Provider

__all__ = ["S3ConnectionOptions", "S3Provider"]


def __getattr__(name: str):
    """Resolve the hail re-exports lazily, without importing hail at package import.

    ``importlib`` is used rather than ``from . import hail``: the latter goes
    through attribute access and would re-enter this hook before the submodule
    is bound, recursing.
    """
    if name == "hail":
        return importlib.import_module("gnomad_meb.hail")
    if name == "reporting":
        return importlib.import_module("gnomad_meb.reporting")
    if name == "_hl":
        return importlib.import_module("gnomad_meb.hail").hail
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
