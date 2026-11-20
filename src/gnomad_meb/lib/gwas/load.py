"""Load the regenie GWAS results back into keyed hail tables.

``GwasLoadPipeline`` mirrors
:class:`~gnomad_meb.lib.variant_qc.load.VariantQcImportPipeline`: for each
exported dataset it scatters one ``load_gwas.<dataset>.<response>`` /
``load_burden.<dataset>.<response>`` checkpointed step per response (regenie
writes one file per response), plus two terminal merges that union the
row-disjoint parts and attach the dataset's provenance globals and any filter
metadata the export GWAS metadata file carried. A final ``get_manifest`` step writes the
GWAS load manifest, the index of every loaded table and its provenance.

The response set comes from the shared, hail-free
:class:`~gnomad_meb.lib.gwas.metadata.GwasMetadata`, so the step set is a
pure function of the settings and reconstruction rebuilds the same set. The
parse is reimplemented here (lifted from ``playbooks/regenie.py``); ``src/``
never imports ``playbooks/``.
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

from gnomad_meb.cmd.task import GwasDatasetRef, Task
from gnomad_meb.lib.checkpoint import CHECKPOINT_TYPE, Artifact, checkpoint
from gnomad_meb.lib.gwas.constants import (
    REGENIE_MANIFEST_PREFIX,
    REGENIE_VERSION_FILENAME,
    BURDEN_ID_PATTERN
)
from gnomad_meb.lib.gwas.manifest import GwasManifest, GwasTableEntry
from gnomad_meb.lib.gwas.metadata import GwasDatasetMetadata, GwasMetadata
from gnomad_meb.lib.gwas.names import DatasetKey
from gnomad_meb.lib.gwas.utils import to_hail_literal_value
from gnomad_meb.pipeline import FromStep, InputStep, Pipeline, Resource, RunInput
from gnomad_meb.logging import get_logger


class GwasLoadException(Exception):
    """A required regenie result file, response table or config is missing."""


#: regenie's output columns, renamed to snake_case (lifted from the playbook).
REGENIE_RENAME: dict[str, str] = {
    "CHROM": "contig",
    "GENPOS": "position",
    "ALLELE0": "allele_0",
    "ALLELE1": "allele_1",
    "A1FREQ": "a1_freq",
    "INFO": "info",
    "N": "n",
    "TEST": "test",
    "BETA": "beta",
    "SE": "se",
    "CHISQ": "chi_sq",
    "LOG10P": "log10_p",
    "EXTRA": "extra",
}


def regenie_rename(ht):
    """Rename the regenie columns the table actually has to snake_case.

    Regenie emits ``INFO`` in its genome-wide results but not its burden ones,
    and pandas can drop an all-``NA`` column, so a table may be missing any
    column :data:`REGENIE_RENAME` names. Hail's ``rename`` requires every source
    field to exist, so the map is intersected with the table's fields first.
    """
    fields = ht.row.dtype.fields
    return ht.rename({k: v for k, v in REGENIE_RENAME.items() if k in fields})


def response_from_filename(name: str) -> str:
    """The response a regenie output file belongs to.

    Files are ``{project}_Y{id}_{response}.regenie.gz``; the response is
    everything after the ``_Y{id}_`` marker, so responses containing ``_`` are
    recovered intact.
    """
    stem = name.removesuffix(".regenie.gz")
    _, _, after = stem.partition("_Y")
    _, _, response = after.partition("_")
    if not response:
        raise GwasLoadException(f"cannot recover a response from filename: {name}")
    return response


def parse_response_frame(path: str | Path, response: str, id_cols: list[str]):
    """Read one regenie output file into a renamed pandas frame.

    Assigns ``response`` to the ``phenotype`` column and splits ``ID`` into
    ``id_cols`` (a single id col copies ``ID`` verbatim; otherwise the id is
    split on ``.``). The numeric regenie fields are read with pinned dtypes
    (``A1FREQ``/``INFO`` as float, so pandas does not infer ``int`` for a
    response whose values happen to be all integral), so every response's frame
    -- and the typed empty frame :meth:`GwasLoadPipeline._empty_burden_ht` --
    agrees column-for-column and unions cleanly.
    """
    import pandas

    df = pandas.read_csv(path, sep=" ", comment="#", na_values=["NA"], dtype={
        "ID": "string",
        "CHROM": "string",
        "GENPOS": "int",
        "ALLELE0": "string",
        "ALLELE1": "string",
        "A1FREQ": "float64",
        # ``INFO`` only appears in the genome-wide (non-burden) files. It is
        # pinned like ``A1FREQ`` because pandas otherwise infers ``int64`` for a
        # response whose INFO values happen to be all integral and ``float64``
        # for one with fractional values -- making the two responses' row types
        # disagree and their union fail.
        "INFO": "float64",
        "N": "int64",
        "TEST": "string",
        "BETA": "float64",
        "SE": "float64",
        "CHISQ": "float64",
        "LOG10P": "float64",
    })
    df["phenotype"] = response
    df["phenotype"] = df["phenotype"].astype("string")
    if len(id_cols) == 1:
        df[id_cols[0]] = df["ID"]
    elif id_cols == ["region", "mask", "aaf"]:
        parsed = df["ID"].str.extract(BURDEN_ID_PATTERN)
        no_mask = parsed["mask"].isna()
        parsed.loc[no_mask, "region"] = df.loc[no_mask, "ID"]
        parsed.loc[no_mask, "mask"] = None
        parsed.loc[no_mask, "aaf"] = None
        df[id_cols] = parsed[id_cols]
    elif len(id_cols) < 1:
        raise GwasLoadException(f"invalid id columns: id_cols = {id_cols}")
    else:
        raise GwasLoadException(f"unsupported id columns: id_cols = {id_cols}, "
                                f"expected a single column or ['region', 'mask',"
                                "'aaf']")

    if len(df) == 0:
        get_logger().warning(f"got empty results dataframe: path = {path}, response = {response}")
        return df

    cols_all = set(df.columns)
    df = df.dropna(axis=1, how="all")
    cols_all_na = cols_all - set(df.columns)
    if cols_all_na:
        get_logger().warning("dropped all-na columns from response frame: "
                             f"columns = {list(sorted(cols_all_na))}")
    
    return df


def _single_file(base: str | Path, response: str) -> Path:
    """The one ``*.regenie.gz`` under ``{base}/{response}/``."""
    matches = sorted(Path(base, response).glob("*.regenie.gz"))
    if len(matches) != 1:
        raise GwasLoadException(
            "expected exactly one regenie file for a response: "
            f"base = {base}, response = {response}, "
            f"matches = {[m.name for m in matches]}"
        )
    return matches[0]


def constant_loader(value):
    """An input loader that ignores its argument and returns ``value``.

    A composed workflow wires a load endpoint's ``source`` to a producing step
    purely for ordering; the endpoint must still resolve to its own deterministic
    value (the exported dataset directory, the regenie ``gwas``/``burden``
    directory), so its loader returns that value rather than the producer's
    artifact path.
    """
    return lambda _ignored, _value=value: _value


def load_dataset_refs(command) -> list[GwasDatasetRef]:
    """The load's dataset refs: the explicit list, else the legacy single form."""
    datasets = list(getattr(command, "datasets", ()) or ())
    if datasets:
        return datasets
    dataset = getattr(command, "dataset", None)
    gwas = getattr(command, "gwas", None)
    burden = getattr(command, "burden", None)
    if not dataset or not gwas or not burden:
        raise GwasLoadException("missing dataset/gwas/burden directories")
    return [
        GwasDatasetRef(label="", dataset=dataset, gwas=gwas, burden=burden)
    ]


class GwasLoadPipeline(Pipeline):
    """Import every dataset's regenie ``gwas`` and ``burden`` results."""

    @classmethod
    def name(cls) -> str:
        return "gwas_load"

    @property
    def resources(self) -> tuple[Resource, ...]:
        return ()

    def get_input(self) -> tuple[InputStep, ...]:
        """One ``dataset``/``gwas``/``burden`` endpoint per ref."""
        cmd = self.command
        group = type(self).name()

        def _input(name: str, value) -> InputStep:
            return InputStep(
                name=name,
                task=Task(settings=self.settings, command=cmd),
                group=group,
                value=value,
                run=constant_loader(value),
            )

        steps: list[InputStep] = []
        for ref in self._refs:
            suffix = f".{ref.label}" if ref.label else ""
            steps.append(_input(f"dataset{suffix}", ref.dataset))
            steps.append(_input(f"gwas{suffix}", ref.gwas))
            steps.append(_input(f"burden{suffix}", ref.burden))
        return tuple(steps)

    def prepare(self) -> None:
        """Scatter one load step per dataset/response plus each dataset's merges."""
        super().prepare()

        # The decorated load/merge methods are rebuilt per dataset.
        rebuilt = {
            "load_gwas",
            "load_burden",
            "merge_gwas",
            "merge_burden",
            "get_manifest",
        }
        self._methods = [m for m in self._methods if m[0] not in rebuilt]

        self._refs = load_dataset_refs(self.command)
        self._ref_by_label = {ref.label: ref for ref in self._refs}
        self._metadata = GwasMetadata.from_settings(self.settings)
        self._responses = list(self._metadata.responses)

        for ref in self._refs:
            label = ref.label
            key = DatasetKey.parse(label)
            suffix = f".{label}" if label else ""
            for response in self._responses:
                self._methods.append(
                    (
                        f"{key.step('load_gwas')}.{response}",
                        checkpoint(
                            f"{key.prefix('load_gwas')}/{response}",
                            CHECKPOINT_TYPE.HAIL_TABLE,
                        )(
                            partial(
                                self.load_gwas,
                                response=response,
                                gwas=RunInput(f"gwas{suffix}"),
                            )
                        ),
                        response,
                    )
                )
                self._methods.append(
                    (
                        f"{key.step('load_burden')}.{response}",
                        checkpoint(
                            f"{key.prefix('load_burden')}/{response}",
                            CHECKPOINT_TYPE.HAIL_TABLE,
                        )(
                            partial(
                                self.load_burden,
                                response=response,
                                burden=RunInput(f"burden{suffix}"),
                            )
                        ),
                        response,
                    )
                )
            self._methods.append(
                (
                    key.step("merge_gwas"),
                    checkpoint(key.prefix("merge_gwas"), CHECKPOINT_TYPE.HAIL_TABLE)(
                        partial(
                            self.merge_gwas,
                            ref=ref,
                            **{
                                response: FromStep(
                                    f"{key.step('load_gwas')}.{response}"
                                )
                                for response in self._responses
                            },
                        )
                    ),
                    None,
                )
            )
            self._methods.append(
                (
                    key.step("merge_burden"),
                    checkpoint(key.prefix("merge_burden"), CHECKPOINT_TYPE.HAIL_TABLE)(
                        partial(
                            self.merge_burden,
                            ref=ref,
                            **{
                                response: FromStep(
                                    f"{key.step('load_burden')}.{response}"
                                )
                                for response in self._responses
                            },
                        )
                    ),
                    None,
                )
            )

        deps = {
            f"merged_gwas_{ref.label or 'single'}": FromStep(
                DatasetKey.parse(ref.label).step("merge_gwas")
            )
            for ref in self._refs
        }
        deps.update(
            {
                f"merged_burden_{ref.label or 'single'}": FromStep(
                    DatasetKey.parse(ref.label).step("merge_burden")
                )
                for ref in self._refs
            }
        )
        self._methods.append(
            (
                "get_manifest",
                checkpoint(
                    REGENIE_MANIFEST_PREFIX, CHECKPOINT_TYPE.DICTIONARY
                )(partial(self.get_manifest, **deps)),
                None,
            )
        )

    # --- work steps ----------------------------------------------------------

    def load_gwas(self, response: str, gwas=RunInput("gwas")):  # noqa: B008
        """Import one response's genome-wide results, keyed by locus/alleles."""
        import hail as hl

        path = _single_file(gwas, response)
        frame = parse_response_frame(path, response, id_cols=["variant_id"])
        ht = hl.Table.from_pandas(frame)

        # Variant id is set in utils:variant_id_expr, so we can use this instead
        # of CHROM / GENPOS to determine the locus. CHROM might contain the
        # collapsed regenie contig chr23, which is invalid.
        variant_id_expr = ht.variant_id.split(":")
        ht = ht.annotate(
            locus=hl.locus(
                hl.str("chr") + variant_id_expr[0],
                hl.int(variant_id_expr[1]),
                reference_genome=self.settings.reference_genome,
            ),
            alleles=hl.array([ht.ALLELE0, ht.ALLELE1]),
        )
        # Use locus or alleles instead of the regenie fields:
        ht = regenie_rename(ht).drop("contig", "position", "allele_0", "allele_1")
        return ht.key_by("locus", "alleles", "phenotype")

    def load_burden(self, response: str, burden=RunInput("burden")):  # noqa: B008
        """Import one response's burden results, keyed by region/mask/aaf."""
        from gnomad_meb.hail import hail as hl

        path = _single_file(burden, response)
        id_cols = ["region", "mask", "aaf"]
        frame = parse_response_frame(path, response, id_cols=id_cols)
        for id_col in id_cols:
            frame[id_col] = frame[id_col].astype("string")
        if len(frame) == 0:
            # A header-only burden file: build the empty table with the schema a
            # populated one produces, so the per-response parts union cleanly.
            # parse_response_frame returns the empty frame before dropping
            # all-NA columns (an empty frame would lose every column), so the
            # row type cannot be inferred from it.
            return self._empty_burden_ht()
        ht = hl.Table.from_pandas(frame)
        ht = regenie_rename(ht).drop("contig", "position")
        # ``extra`` is regenie's free-form column; it is unused downstream and
        # all-NA-and-dropped for real burden results, but can survive with other
        # values. Drop it explicitly so every part (and the typed empty table)
        # shares one row type.
        if "extra" in ht.row.dtype.fields:
            ht = ht.drop("extra")
        return ht.key_by("region", "mask", "aaf", "phenotype")

    @staticmethod
    def _empty_burden_ht():
        """An empty burden table matching the schema :meth:`load_burden` builds.

        The explicit row type mirrors a populated burden result after
        :func:`parse_response_frame` drops its all-``NA`` columns and
        :meth:`load_burden` drops ``extra`` (``ID`` survives the rename). An
        empty frame cannot be type-inferred by ``from_pandas``, so the types are
        stated here to keep the empty part's row type identical to its sibling
        parts' -- hail's union requires identical row types.
        """
        from gnomad_meb.hail import hail as hl
        row_type = hl.tstruct(
            ID=hl.tstr,
            allele_0=hl.tstr,
            allele_1=hl.tstr,
            a1_freq=hl.tfloat64,
            n=hl.tint64,
            test=hl.tstr,
            beta=hl.tfloat64,
            se=hl.tfloat64,
            chi_sq=hl.tfloat64,
            log10_p=hl.tfloat64,
            phenotype=hl.tstr,
            region=hl.tstr,
            mask=hl.tstr,
            aaf=hl.tstr,
        )
        return hl.Table.parallelize(
            [],
            schema=row_type,
            key=["region", "mask", "aaf", "phenotype"],
        )

    def _regenie_version(self, ref: GwasDatasetRef):
        """The dataset's regenie provenance value, or ``None`` when absent."""
        import hail as hl

        from gnomad_meb.logging import get_logger

        provenance = Path(ref.gwas).parent / REGENIE_VERSION_FILENAME
        if provenance.exists():
            return hl.struct(**json.loads(provenance.read_text()))
        get_logger().warning(
            f"missing regenie provenance file, no regenie version: {provenance}"
        )
        return hl.missing(hl.tstruct(version=hl.tstr, build=hl.tstr, url=hl.tstr))

    def _provenance_globals(self, ref: GwasDatasetRef) -> dict:
        """The terminal globals: the run provenance plus any filter metadata."""
        import hail as hl

        from gnomad_meb.logging import get_logger

        logger = get_logger()
        responses = self._responses
        record = GwasDatasetMetadata.read(ref.dataset)
        name_map: dict[str, str] = {response: response for response in responses}
        if record.names:
            name_map.update(record.names)
        else:
            logger.warning(
                f"missing dataset metadata file, using identity: {ref.dataset}"
            )

        globals_: dict = {
            "regenie_version": self._regenie_version(ref),
            "dataset": str(ref.dataset),
            "responses": hl.tuple(tuple(responses)),
            "response_name_map": hl.dict(name_map),
            "effect_mask_info": hl.literal(
                to_hail_literal_value(record.effect_mask_info)
            ),
        }
        for name, value in record.filter_globals.items():
            globals_[name] = hl.literal(to_hail_literal_value(value))
        return globals_

    def _merge_parts(self, ref: GwasDatasetRef | None, parts: dict):
        """Union the per-response tables, then attach the dataset provenance."""
        from gnomad_meb.hail import get_default_partitions

        split_parts = tuple(parts.values())
        ht = split_parts[0].union(*split_parts[1:])
        ht = ht.repartition(get_default_partitions(), shuffle=True)
        return ht.annotate_globals(**self._provenance_globals(ref))

    def merge_gwas(self, ref: GwasDatasetRef | None = None, **parts):
        """Union the per-response GWAS tables, then attach the provenance."""
        return self._merge_parts(ref, parts)

    def merge_burden(self, ref: GwasDatasetRef | None = None, **parts):
        """Union the per-response burden tables, then attach the provenance."""
        return self._merge_parts(ref, parts)

    # --- the manifest --------------------------------------------------------

    def get_manifest(self, **outputs) -> dict:
        """Build the GWAS load manifest: every loaded table and its provenance.

        ``**outputs`` are the wired merge dependencies; they are consumed only to
        order this step after every merge, so a present manifest implies the
        terminal tables exist. Paths come from the same prefixes the merge
        checkpoints declare, resolved through :class:`Artifact`.
        """
        settings = self.settings
        entries = []
        for ref in self._refs:
            record = GwasDatasetMetadata.read(ref.dataset)
            source = record.source
            if source is None:
                source = "mt" if not ref.label else "filter"
            provenance = Path(ref.gwas).parent / REGENIE_VERSION_FILENAME
            entries.append(
                GwasTableEntry(
                    label=ref.label,
                    source=source,
                    name=record.filter_name,
                    region=DatasetKey.parse(ref.label).region,
                    dataset=str(ref.dataset),
                    gwas=Artifact(
                        DatasetKey.parse(ref.label).prefix("merge_gwas"),
                        CHECKPOINT_TYPE.HAIL_TABLE,
                    ).path(settings),
                    burden=Artifact(
                        DatasetKey.parse(ref.label).prefix("merge_burden"),
                        CHECKPOINT_TYPE.HAIL_TABLE,
                    ).path(settings),
                    responses=list(self._responses),
                    regenie_version=(
                        json.loads(provenance.read_text())
                        if provenance.exists()
                        else None
                    ),
                    filters=record.filter_globals,
                    effect_mask_info=record.effect_mask_info,
                )
            )
        return GwasManifest(
            project=settings.project,
            reference_genome=settings.reference_genome,
            checkpoint_root=settings.checkpoint,
            datasets=entries,
        ).model_dump(mode="json")
