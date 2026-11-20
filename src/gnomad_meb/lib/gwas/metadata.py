"""The GWAS metadata: the study CSV, the response gate and the dataset file.

Two objects own everything study-metadata shaped. :class:`GwasMetadata` reads
the external study-metadata CSV **once** at construction, applies the
``min_case_count`` gate to the derived responses and reports only the gated
set, with the display-name and kind maps that go with it. The export's hail
``study_md`` projection is a deferred-hail method (:meth:`GwasMetadata.study_md`,
which imports hail inside its body), so importing this module never initializes
hail.

:class:`GwasDatasetMetadata` is one exported dataset's GWAS metadata file -- the
``{basename}.gwas.md.json`` the export writes next to each dataset -- as a
record: :meth:`GwasDatasetMetadata.read` parses the file and
:meth:`GwasDatasetMetadata.write` serializes it. Reading is pure Python, so the
run and load stages' ``prepare`` can consult it before any hail session.

``responses`` is exactly the gated :attr:`response_df` columns minus
``cohort``: the case response and the direct columns in config order, then the
per-value set indicators. Set values fan out into ``{column}_{value}``
indicator columns that are missing for non-cases when
``[gwas.responses].case_only`` (the default), so only cases contribute. The
gate itself is the Snakemake rule (``regenie/meta.smk``): a ``quantitative``
response is gated on its non-missing sample count, every other on its case
count (the column sum, which for the case-only indicators is the case count).
The raw ``cohort`` column rides along in :attr:`response_df` so the rebuilt
``study_md`` keeps it for ``drop_cohorts`` and the PLINK ``pheno``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

import pandas

from gnomad_meb.config import GwasConfig, Settings
from gnomad_meb.lib.gwas.constants import (
    EFFECT_MASK_INFO_KEY,
    FILTER_GLOBAL_NAMES,
    METADATA_SUFFIX,
    N_VARIANTS_KEY,
    REGION_NAME_KEY,
    RESPONSE_KIND_MAP_KEY,
    RESPONSE_NAME_MAP_KEY,
    TABLE_NAME_KEY,
    TABLE_SOURCE_KEY,
)
from gnomad_meb.lib.gwas.utils import read_study_md, snake_case
from gnomad_meb.logging import get_logger

if TYPE_CHECKING:
    import hail as hl


class GwasMetadataException(ValueError):
    """The GWAS metadata model or its study CSV is missing or invalid."""


class GwasMetadata:
    """The gated, ordered responses and the export's ``study_md`` projection.

    Construct from config -- :meth:`from_settings` is the convenience for a
    resolved :class:`~gnomad_meb.config.Settings`. Construction reads the study
    CSV once and builds the frames the stages consume: :attr:`response_df` (the
    gated responses plus the raw ``cohort`` column), :attr:`covariate_df` and
    the set-value one-hot frame behind it. Everything is pure pandas and
    hail-free; the only hail-touching method is :meth:`study_md`. A dataset's
    GWAS metadata *file* is a separate record: :class:`GwasDatasetMetadata`.
    """

    def __init__(
        self, cfg: GwasConfig, *, metadata_csv: str | Path | None = None
    ) -> None:
        self._cfg = cfg
        self._metadata_csv = (
            Path(metadata_csv) if metadata_csv is not None else cfg.metadata_csv
        )

        self._parse()

    @classmethod
    def from_settings(
        cls, settings: Settings, *, metadata_csv: str | Path | None = None
    ) -> GwasMetadata:
        """Build from a resolved settings object (the config-backed path)."""
        gwas = settings.gwas
        if gwas is None:
            raise GwasMetadataException("missing gwas configuration section")
        return cls(gwas, metadata_csv=metadata_csv)

    # --- construction --------------------------------------------------------

    def _parse(self) -> None:
        """Read the CSV once and build the covariate and one-hot frames."""
        cfg = self._cfg

        if any(col.kind == "quantitative" for col in cfg.responses.columns) and any(
            col.kind in ("set", "binary") for col in cfg.responses.columns
        ):
            get_logger().warning(
                "regenie does not support regressing binary and quantitative "
                "traits at once"
            )

        covariate_columns = [
            covariate.metadata_field
            for covariate in cfg.covariates
            if covariate.metadata_field is not None
        ]

        self._md_df = read_study_md(
            self._metadata_csv,
            set_columns=[
                col.field
                for col in cfg.responses.columns
                if col.kind == "set"
            ],
            required_columns=covariate_columns,
            response_columns=[
                col.field
                for col in cfg.responses.columns
                if col.kind in ("binary", "quantitative")
            ],
            cohort_col=cfg.responses.case_metadata_field,
        )

        covariate_sources: dict[str, str] = {}
        for covariate in cfg.covariates:
            if covariate.metadata_field is None:
                continue
            source = (
                "age"
                if covariate.metadata_field == "birthdate"
                else covariate.metadata_field
            )
            covariate_sources[source] = covariate.name
        self._covariate_df = self._md_df[list(covariate_sources)].rename(
            columns=covariate_sources
        )

        self._set_value_names: dict[str, str] = {}
        oh_dfs: list[pandas.DataFrame] = []
        for col in cfg.responses.columns:
            if col.kind != "set":
                continue
            oh_df, column_names = self.one_hot_encode(
                col.field,
                set(col.exclude),
                case_only=cfg.responses.case_only,
            )
            self._set_value_names.update(
                {name: member for member, name in column_names.items()}
            )
            oh_dfs.append(oh_df)

        if oh_dfs:
            self._oh_df = pandas.concat(oh_dfs, axis=1)
        else:
            self._oh_df = pandas.DataFrame(index=self._md_df.index, columns=[])

    @cached_property
    def case_response(self) -> pandas.Series:
        """``1`` for the case cohort, ``0`` for controls, missing otherwise."""
        cfg = self._cfg
        md_df = self._md_df

        case = pandas.Series(float("nan"), index=md_df.index)
        case[md_df["cohort"].isin(set(cfg.control_cohorts))] = 0.0
        if cfg.case_cohort:
            case[md_df["cohort"] == cfg.case_cohort] = 1.0
        return case

    @cached_property
    def direct_response_df(self) -> pandas.DataFrame:
        """The gated case and direct responses, plus the raw ``cohort`` column.

        The case response is itself gated (its sum is the case count); a
        ``quantitative`` column is gated on its non-missing sample count and a
        ``binary`` column on its sum. The raw ``cohort`` column rides along so
        ``study_md`` keeps it for ``drop_cohorts`` and the PLINK ``pheno`` --
        skipped when the case response already owns that name.
        """
        cfg = self._cfg
        md_df = self._md_df
        case_name = snake_case(cfg.responses.case)

        response_df = pandas.DataFrame(index=md_df.index, columns=[])
        case = self.case_response
        if case.sum() >= cfg.min_case_count:
            response_df[case_name] = case
        for col in cfg.responses.columns:
            if col.kind == "set":
                continue
            values = md_df[col.field]
            count = (
                values.notna().sum()
                if col.kind == "quantitative"
                else values.sum()
            )
            if count >= cfg.min_case_count:
                response_df[snake_case(col.name)] = values
        if case_name != "cohort" and "cohort" not in response_df.columns:
            response_df["cohort"] = md_df["cohort"]
        return response_df

    @cached_property
    def set_response_df(self) -> pandas.DataFrame:
        """The gated per-value set indicators (each one's sum is its case count)."""
        return self._oh_df.loc[:, self._oh_df.sum() >= self._cfg.min_case_count].copy()

    @cached_property
    def response_df(self) -> pandas.DataFrame:
        """The gated responses plus ``cohort`` -- the frames the stages consume.

        The direct columns (case, gated binary/quantitative, raw ``cohort``)
        come first, then the gated set indicators.
        """
        return pandas.concat([self.direct_response_df, self.set_response_df], axis=1)

    # --- the gated response set ----------------------------------------------

    @property
    def responses(self) -> tuple[str, ...]:
        """The gated response names: every ``response_df`` column but ``cohort``.

        The case response and the direct columns come in config order, then the
        set indicators. ``cohort`` rides along for ``study_md`` (``drop_cohorts``,
        the PLINK ``pheno``) -- it is metadata, not a response; the one
        exception is a case response configured under that name.
        """
        case_name = snake_case(self._cfg.responses.case)
        return tuple(
            name
            for name in self.response_df.columns
            if name != "cohort" or name == case_name
        )

    @property
    def names(self) -> dict[str, str]:
        """The gated ``response -> display name`` map."""
        cfg = self._cfg
        direct_df = self.direct_response_df
        case_name = snake_case(cfg.responses.case)

        names: dict[str, str] = {}
        if case_name in direct_df.columns:
            names[case_name] = cfg.responses.case
        names.update(
            {
                name: member
                for name, member in self._set_value_names.items()
                if name in self.set_response_df.columns
            }
        )
        for col in cfg.responses.columns:
            if col.kind == "set":
                continue
            name = snake_case(col.name)
            if name in direct_df.columns and name != case_name:
                names[name] = col.name
        return names

    @property
    def kinds(self) -> dict[str, str]:
        """The gated ``response -> kind`` map (``binary``/``quantitative``)."""
        cfg = self._cfg
        direct_df = self.direct_response_df
        case_name = snake_case(cfg.responses.case)

        kinds: dict[str, str] = {}
        if case_name in direct_df.columns:
            kinds[case_name] = "binary"
        kinds.update(dict.fromkeys(self.set_response_df.columns, "binary"))
        for col in cfg.responses.columns:
            if col.kind == "set":
                continue
            name = snake_case(col.name)
            if name in direct_df.columns and name != case_name:
                kinds[name] = col.kind
        return kinds

    @property
    def covariate_df(self) -> pandas.DataFrame:
        """The metadata-sourced covariates, keyed by their configured name."""
        return self._covariate_df

    def one_hot_encode(
        self,
        col: str,
        exclude: set[str] | None = None,
        case_only: bool = True,
    ) -> tuple[pandas.DataFrame, dict[str, str]]:
        """One set column as per-value indicator columns.

        Each distinct value in ``col`` (minus ``exclude``) becomes a
        ``{snake_case(col)}_{snake_case(value)}`` column: ``1.0`` when present
        and ``0.0`` when absent. With ``case_only`` (the default) the non-case
        rows stay in the table but are missing -- regenie sees NA for every
        non-case sample, so only cases contribute. Returns the frame (over the
        full ``sample_id`` index) and the ``{value: column name}`` mapping.
        """
        md_df = self._md_df

        col_l: list[set[str]] = md_df[col].tolist()
        assert isinstance(col_l, list) and all(
            isinstance(v, set) for v in col_l
        ), f"type mismatch in set column: col = {col}"

        members = col_l[0].union(*col_l[1:]) if col_l else set()
        if exclude:
            members -= set(exclude)

        base_col_name = snake_case(col)
        col_names = {
            member: f"{base_col_name}_{snake_case(member)}" for member in members
        }
        one_hot_cols = sorted(members)

        oh_df = pandas.DataFrame(
            index=md_df.index,
            columns=[col_names[member] for member in one_hot_cols],
            dtype="float64",
        )
        is_case = self.case_response == 1.0
        for member in one_hot_cols:
            values = md_df[col].apply(
                lambda s, member=member: 1.0 if member in s else 0.0
            )
            if case_only:
                values[~is_case] = float("nan")
            oh_df[col_names[member]] = values
        return oh_df, col_names

    def study_md(self) -> hl.Table:
        """Project the parsed study CSV into a hail table keyed by ``sample_id``.

        The gated responses (including the raw ``cohort`` column) plus the
        covariates. Reuses the frames :meth:`_parse` built, so the CSV is
        touched once. Hail is imported inside the body: importing this module
        never initializes hail.
        """
        import hail as hl

        df = pandas.concat([self.response_df, self.covariate_df], axis=1)
        df = df.reset_index(drop=False, names=["sample_id"])
        return hl.Table.from_pandas(df, key=["sample_id"])


@dataclass
class GwasDatasetMetadata:
    """One exported dataset's GWAS metadata file, as a read/write record.

    The ``{basename}.gwas.md.json`` the export writes next to each dataset: the
    response name/kind maps, the provenance (``source``/``name``), the variant
    count and the filter metadata globals. :meth:`read` parses the file -- every
    field defaulting when it is absent, because standalone exports on an MT that
    never went through the filter stage simply have no metadata file -- and
    :meth:`write` serializes it with the exact on-disk schema the export has
    always written. ``filter_globals`` keeps only present, non-empty values.
    """

    names: dict[str, str]
    kinds: dict[str, str]
    source: str | None = None
    filter_name: str | None = None
    region_name: str | None = None
    n_variants: int | None = None
    filter_globals: dict = field(default_factory=dict)
    effect_mask_info: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.filter_globals = {
            key: value for key, value in self.filter_globals.items() if value
        }

    @classmethod
    def read(cls, dataset: str | Path) -> GwasDatasetMetadata:
        """Read the dataset's GWAS metadata file, defaulting when absent.

        The variant count is the recorded count, else the PLINK ``.bim`` rows --
        the same variant count without a hail session -- else ``0``.
        """
        metadata = cls._read_metadata(dataset)
        value = metadata.get(N_VARIANTS_KEY)
        if not isinstance(value, int | float) or isinstance(value, bool):
            value = cls._bim_n_variants(dataset)
        return cls(
            names=dict(metadata.get(RESPONSE_NAME_MAP_KEY) or {}),
            kinds=dict(metadata.get(RESPONSE_KIND_MAP_KEY) or {}),
            source=metadata.get(TABLE_SOURCE_KEY),
            filter_name=metadata.get(TABLE_NAME_KEY),
            region_name=metadata.get(REGION_NAME_KEY),
            n_variants=int(value),
            filter_globals={
                name: metadata[name]
                for name in FILTER_GLOBAL_NAMES
                if metadata.get(name)
            },
            effect_mask_info=dict(metadata.get(EFFECT_MASK_INFO_KEY) or {}),
        )

    def write(self, dataset_dir: str | Path, *, basename: str | None = None) -> Path:
        """Write the record as the dataset's GWAS metadata file; return its path.

        ``sort_keys`` with :class:`HailStructEncoder`, so the file is
        byte-compatible with the historic export. ``HailStructEncoder`` is
        imported lazily because importing it pulls in hail. ``basename`` names
        the dataset artifact; when omitted, it is recovered from an existing
        file in ``dataset_dir``.
        """
        from gnomad_meb.hail import HailStructEncoder

        metadata: dict = {
            RESPONSE_NAME_MAP_KEY: self.names,
            RESPONSE_KIND_MAP_KEY: self.kinds,
        }
        if self.source is not None:
            metadata[TABLE_SOURCE_KEY] = self.source
        if self.filter_name is not None:
            metadata[TABLE_NAME_KEY] = self.filter_name
        if self.region_name is not None:
            metadata[REGION_NAME_KEY] = self.region_name
        if self.n_variants is not None:
            metadata[N_VARIANTS_KEY] = self.n_variants
        for key, value in self.filter_globals.items():
            metadata[key] = value
        if self.effect_mask_info:
            metadata[EFFECT_MASK_INFO_KEY] = self.effect_mask_info

        dataset_dir = Path(dataset_dir)
        if basename is None:
            basename = self._dataset_basename(dataset_dir)
        path = dataset_dir / f"{basename}{METADATA_SUFFIX}"
        path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True, cls=HailStructEncoder)
        )
        return path

    # --- the GWAS metadata file's private helpers ---------------------------

    @staticmethod
    def _metadata_path(dataset: str | Path) -> Path | None:
        """The dataset directory's GWAS metadata file, or ``None`` when absent."""
        matches = sorted(Path(dataset).glob(f"*{METADATA_SUFFIX}"))
        return matches[0] if matches else None

    @staticmethod
    def _read_metadata(dataset: str | Path) -> dict:
        """The dataset's GWAS metadata as a mapping, or ``{}`` when absent.

        A missing file is not an error: standalone exports on an MT that never
        went through the filter stage simply have no metadata, and every consumer
        falls back to its pre-metadata default.
        """
        path = GwasDatasetMetadata._metadata_path(dataset)
        if path is None:
            return {}
        return json.loads(path.read_text())

    @staticmethod
    def _bim_n_variants(dataset: str | Path) -> int:
        """Count the PLINK ``.bim`` rows, ``0`` without exactly one ``.bim``."""
        bims = sorted(Path(dataset).glob("*.bim"))
        if len(bims) != 1:
            return 0
        with bims[0].open() as handle:
            return sum(1 for _ in handle)

    @staticmethod
    def _dataset_basename(dataset_dir: Path) -> str:
        """Recover the dataset basename from an existing artifact in its directory.

        Used when :meth:`write` is called without an explicit ``basename``; the
        export always writes the PLINK trio and the six TSVs before the metadata
        file, so the name is recovered from one of those. The PLINK
        ``.bim``/``.bed`` come first because the export writes PLINK first; the
        phenotype TSV and an existing metadata file are the fallbacks.
        """
        for pattern, suffix in (
            ("*.bim", ".bim"),
            ("*.bed", ".bed"),
            ("*.phenotype.tsv.bgz", ".phenotype.tsv.bgz"),
            (f"*{METADATA_SUFFIX}", METADATA_SUFFIX),
        ):
            matches = sorted(dataset_dir.glob(pattern))
            if matches:
                return matches[0].name[: -len(suffix)]
        raise GwasMetadataException(
            f"cannot derive the dataset basename from {dataset_dir}: no PLINK, "
            "phenotype or metadata file present"
        )
