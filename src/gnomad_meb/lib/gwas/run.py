"""Run the regenie GWAS workflow once per exported dataset.

``GwasRunPipeline`` mirrors :class:`~gnomad_meb.lib.variant_qc.vqsr.VqsrPipeline`:
one Snakemake-backed step per exported dataset, each declaring every
per-response ``gwas``/``burden`` output plus the dataset's regenie provenance
file. The response set comes from the shared, hail-free
:class:`~gnomad_meb.lib.gwas.metadata.GwasMetadata`, so the step set is a
pure function of the settings and reconstruction rebuilds the same set.

The chunk count is data-driven: each step reads its dataset's recorded variant
count (the export GWAS metadata file) and derives
``ceil(n_variants / variants_per_split)`` chunks, so the split tracks the
dataset's size rather than a fixed config value.
"""

from __future__ import annotations

import math
from functools import partial
from pathlib import Path

from gnomad_meb.cmd.task import GwasDatasetRef, Task
from gnomad_meb.config import Settings
from gnomad_meb.lib.checkpoint import Artifact, declares_step
from gnomad_meb.lib.gwas.constants import (
    REGENIE_VERSION_FILENAME,
    TABLE_NAMES,
)
from gnomad_meb.lib.gwas.metadata import GwasDatasetMetadata, GwasMetadata
from gnomad_meb.lib.gwas.names import DatasetKey
from gnomad_meb.pipeline import InputStep, Pipeline, Resource, RunInput


class GwasRunException(Exception):
    """The exported dataset or its configuration is missing."""


def _phenotype_file(dataset: str | Path) -> Path:
    """The single ``*.phenotype.tsv.bgz`` in an exported region directory."""
    matches = sorted(Path(dataset).glob("*.phenotype.tsv.bgz"))
    if len(matches) != 1:
        raise GwasRunException(
            "expected exactly one phenotype table in the dataset directory: "
            f"dataset = {dataset}, matches = {[m.name for m in matches]}"
        )
    return matches[0]


def resolve_n_chunks(n_variants: int | None, settings: Settings, default: int = 10) -> int:
    """The chunk count for ``n_variants``, targeting ``variants_per_split``."""
    if n_variants is None:
        return default
    gwas = settings.gwas
    per_split = gwas.variants_per_split if gwas is not None else 100_000
    # Regenie chunk needs at least two chunks
    return max(2, math.ceil(n_variants / per_split))


def dataset_refs(command) -> list[GwasDatasetRef]:
    """The run's dataset refs: the explicit list, else the legacy single dataset."""
    datasets = list(getattr(command, "datasets", ()) or ())
    if datasets:
        return datasets
    dataset = getattr(command, "dataset", None)
    output_dir = getattr(command, "output_dir", None)
    if not dataset or not output_dir:
        raise GwasRunException("missing dataset directory")
    return [
        GwasDatasetRef(
            label="",
            dataset=str(dataset),
            gwas=f"{output_dir}/gwas",
            burden=f"{output_dir}/burden",
        )
    ]


class GwasRunPipeline(Pipeline):
    """Run the regenie workflow once per exported region dataset."""

    @classmethod
    def name(cls) -> str:
        return "gwas"

    @property
    def resources(self) -> tuple[Resource, ...]:
        return ()

    @property
    def snakemake_args(self) -> tuple[str, ...]:
        """The Snakemake pass-through: the command override, else the config."""
        override = getattr(self.command, "snakemake", ()) or ()
        if override:
            return tuple(override)
        gwas = self.settings.gwas
        return tuple(gwas.snakemake) if gwas is not None else ()

    def get_input(self) -> tuple[InputStep, ...]:
        """One ``dataset`` endpoint per exported region directory."""
        return tuple(
            InputStep(
                name=DatasetKey.parse(ref.label).step("dataset"),
                task=Task(settings=self.settings, command=self.command),
                group=type(self).name(),
                value=ref.dataset,
            )
            for ref in self._refs
        )

    def prepare(self) -> None:
        """Scatter one regenie step per dataset, with a deterministic response set."""
        super().prepare()

        # ``run_gwas_pipeline`` is declared once and rebuilt per dataset, so drop
        # the default (label-less) entry ``super().prepare`` collected.
        self._methods = [m for m in self._methods if m[0] != "run_gwas_pipeline"]

        self._refs = dataset_refs(self.command)
        self._ref_by_label = {ref.label: ref for ref in self._refs}
        self._metadata = GwasMetadata.from_settings(self.settings)
        self._responses = list(self._metadata.responses)

        for ref in self._refs:
            label = ref.label
            self._methods.append(
                (
                    DatasetKey.parse(label).step("run_gwas_pipeline"),
                    declares_step(
                        partial(
                            self.run_gwas_pipeline,
                            dataset=RunInput(
                                DatasetKey.parse(ref.label).step("dataset")
                            ),
                            ref=ref,
                        )
                    ),
                    None,
                )
            )

    def _artifacts(self, name: str, method):
        """Every per-response regenie output plus the dataset's provenance file."""
        if not name.startswith("run_gwas_pipeline"):
            return super()._artifacts(name, method)

        label = name.partition(".")[2] if "." in name else ""
        ref = self._ref_by_label[label]
        project = self.settings.project
        artifacts = []
        for index, response in enumerate(self._responses, start=1):
            for base in (ref.gwas, ref.burden):
                artifacts.append(
                    Artifact.from_path(
                        f"{base}/{response}"
                        f"/{project}_Y{index}_{response}.regenie.gz"
                    )
                )
        artifacts.append(
            Artifact.from_path(
                f"{Path(ref.gwas).parent}/{REGENIE_VERSION_FILENAME}"
            )
        )
        return tuple(artifacts)

    @declares_step
    def run_gwas_pipeline(self, dataset=RunInput("dataset"), ref=None):  # noqa: B008
        """Run the regenie Snakemake DAG for one exported dataset.

        The runner is imported lazily, so constructing (and reconstructing)
        this pipeline never pulls in Snakemake.
        """
        from gnomad_meb.regenie import regenie as run_regenie

        # Remove any extensions that are carried after the PipelineRunner
        # resolved the artifact names
        dataset = Path(dataset).resolve()
        if dataset.is_file():
            dataset = dataset.parent

        record = GwasDatasetMetadata.read(dataset)

        phenotype_file = _phenotype_file(dataset)
        basename = phenotype_file.name[: -len(".phenotype.tsv.bgz")]
        tables = {
            table: str(Path(dataset) / f"{basename}.{table}.tsv.bgz")
            for table in TABLE_NAMES
        }
        bed = str(Path(dataset) / f"{basename}.bed")

        n_chunks = resolve_n_chunks(record.n_variants, self.settings)

        run_regenie(
            bed,
            tables["covariates"],
            tables["phenotype"],
            tables["annotation"],
            tables["variant_sets"],
            tables["aaf"],
            tables["effect_masks"],
            self.snakemake_args,
            n_chunks,
            response_kinds=record.kinds,
            output_dir=str(Path(ref.gwas).parent),
            log_dir=str(
                self.settings.gwas.log_dir
                / (record.filter_name or "default")
                / (record.region_name or "default")
            ),
        )
        # No result: composed consumers read this step's artifacts, so the load
        # endpoints keep their own deterministic values.
        return None
