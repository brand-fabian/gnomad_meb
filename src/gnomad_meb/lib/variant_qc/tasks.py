"""Shared derivation of the per-split task list for the Variant QC stages.

Every Variant QC stage scatters over the run's :class:`VariantDataset` task
list: ``prepare`` reads the task list from the VDS handle and generates one step
per split. The handle is read through :meth:`VariantDataset.open`, which reads
metadata only (no data), so ``--describe`` and reconstruction rebuild the same
step set from the same input. A missing handle fails loudly, so the two
invocation scenarios can never disagree about the steps.
"""

from __future__ import annotations


def vds_tasks(command: object) -> tuple:
    """The run's VDS split tasks, or a loud failure when the handle is absent."""
    from gnomad_meb.lib.vds import VariantDataset

    vds_path = getattr(command, "vds", None)
    if vds_path is None:
        raise ValueError(
            "variant qc pipeline needs the run's VDS handle to finalize its "
            f"per-task steps: command = {type(command).__name__}"
        )
    return VariantDataset.open(vds_path).params.tasks
