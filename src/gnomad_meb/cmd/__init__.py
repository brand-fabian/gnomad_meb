"""The ``gmeb`` command-line interface.

The root click group (defined in :mod:`.cli`) and its subcommands are assembled
lazily (PEP 562). Eager assembly would import :mod:`.run`, which imports
:mod:`gnomad_meb.pipeline`, whose modules import :mod:`gnomad_meb.cmd.task` — a
package cycle that only resolves under one particular import order. Deferring
the assembly until ``main`` is first accessed removes that cycle from the import
path of everything that merely imports a submodule (``gnomad_meb.cmd.task``),
while ``from gnomad_meb.cmd import main`` still yields the assembled group.

The click group lives in :mod:`.cli`, not a ``main`` submodule, so the package
attribute ``main`` is never shadowed by a submodule of the same name.
"""

from __future__ import annotations

from importlib import import_module

#: The subcommands assembled onto the root group, as ``(module, attribute)``.
_SUBCOMMANDS = (
    ("vds", "vds"),
    ("vqsr", "vqsr"),
    ("variant_qc", "variant_qc"),
    ("sample_qc", "sample_qc"),
    ("annotation", "annotation"),
    ("filter", "filter"),
    ("browser", "browser"),
    ("gwas", "gwas"),
    ("gnomad_workflow", "gnomad_workflow"),
    ("run", "run"),
)


def _build_main():
    """Assemble the root group with its subcommands and return it."""
    from .cli import main

    for module_name, attribute in _SUBCOMMANDS:
        module = import_module(f"{__name__}.{module_name}")
        main.add_command(getattr(module, attribute))
    return main


def __getattr__(name: str):
    """Resolve ``main`` and the subcommand modules lazily (PEP 562)."""
    if name == "main":
        return _build_main()
    if name in ("cli", *[module for module, _ in _SUBCOMMANDS]) or name in (
        "output",
        "task",
    ):
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["main"]
