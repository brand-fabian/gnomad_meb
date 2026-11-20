"""The VDS build declared as a :class:`Pipeline`.

The build chunks the reference genome into splits, imports each split from the
input GVCFs and finalises the dataset metadata handle the downstream stages
read. It is a single step: the per-split fan-out and its resume live inside
:meth:`VariantDataset.process`, which tracks a per-task status in the dataset
metadata and is therefore its own cache; the declared artifact is the metadata
handle the dataset writes.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from gnomad_meb.lib.checkpoint import Artifact, declares_step
from gnomad_meb.logging import get_logger
from gnomad_meb.pipeline import InputStep, Pipeline, Resource

if TYPE_CHECKING:
    from gnomad_meb.cmd.task import VdsCreate

logger = get_logger()


class VdsCreatePipeline(Pipeline):
    """Create a :class:`VariantDataset` from a set of GVCFs."""

    @classmethod
    def name(cls) -> str:
        return "vds"

    @property
    def resources(self) -> tuple[Resource, ...]:
        """The VDS build declares no external resources."""
        return ()

    def get_input(self) -> tuple[InputStep, ...]:
        """The VDS build consumes its GVCFs from the command, not an endpoint."""
        return ()

    def _metadata_path(self) -> str:
        cmd: VdsCreate = self.command
        return os.path.abspath(f"{cmd.output_base}/{cmd.vds_name}.metadata.json")

    def _artifacts(self, name, method):
        if name == "create_variant_dataset":
            return (Artifact.from_path(self._metadata_path()),)
        return super()._artifacts(name, method)

    @declares_step
    def create_variant_dataset(self):
        """Search for the input GVCFs, import them and return the dataset handle."""
        from gnomad_meb.lib.utils import glob
        from gnomad_meb.lib.vds.dataset import VariantDataset
        from gnomad_meb.s3 import S3ConnectionOptions, S3Provider

        cfg = self.settings
        cmd: VdsCreate = self.command

        files = list(cmd.gvcf)
        if len(cmd.bucket) > 0:
            logger.info(f"searching for files in {len(cmd.bucket)} buckets")
            s3 = S3Provider(
                S3ConnectionOptions(
                    cfg.s3.access_key,
                    cfg.s3.secret_key,
                    cfg.s3.endpoint_url,
                    cfg.s3.verify,
                )
            )
            files.extend(
                list(
                    s3.glob(
                        cmd.bucket,
                        patterns=cmd.s3_search_pattern,
                        return_url=True,
                        s3_url_scheme="s3a",
                    )
                )
            )
        if len(cmd.search) > 0:
            logger.info(f"searching for files in {len(cmd.search)} local directories")
            files.extend(glob(cmd.search, patterns=cmd.local_search_pattern))

        if len(files) < 1:
            raise ValueError("no gvcf files specified for VDS creation")

        path = self._metadata_path()
        logger.info(f"creating VDS from {len(files)} files")
        return VariantDataset.create_from_gvcf(
            os.path.dirname(path),
            os.path.basename(path).removesuffix(".metadata.json"),
            files,
            reference_genome=cmd.reference_genome,
            n_splits=cmd.n_splits,
            interval_size=cmd.interval_size,
            overwrite=cmd.overwrite,
        )
