"""The dataset key: a ``(table, region)`` identity and every name derived from it.

A GWAS table is paired with a region to form one exported dataset, and every
stage -- export, run, load and the composed workflow -- derives step ids and
checkpoint prefixes from that pair. :class:`DatasetKey` states the identity once
and the empty-table rule once: a raw ``--mt`` (empty table label) reproduces the
single-table names exactly, so it is the same object as the namespaced one.

It is pure Python: importing this module initialises neither hail nor Snakemake.
"""

from __future__ import annotations

from dataclasses import dataclass

from gnomad_meb.lib.gwas.constants import (
    BASE_CHAIN_PREFIXES,
    INTERMEDIATE_PREFIX,
    REGENIE_PREFIX,
)


@dataclass(frozen=True)
class DatasetKey:
    """A ``(table, region)`` identity and every step id / path it derives.

    ``table`` is the table label (empty for a raw ``--mt``); ``region`` is the
    region label (a gene, ``window_10000``, ...). Build one with :meth:`of` from
    the pair, or with :meth:`parse` from a dataset label.
    """

    table: str = ""
    region: str = ""

    @classmethod
    def of(cls, table: str, region: str) -> DatasetKey:
        """The key for a ``(table, region)`` pair (falsy parts normalised to ``""``)."""
        return cls(table=table or "", region=region or "")

    @classmethod
    def parse(cls, label: str) -> DatasetKey:
        """The key a dataset label names (the inverse of :attr:`label`).

        Empty ⇒ ``("", "")``; no dot ⇒ a region-only key; else the table and
        region split on the first dot (table and region labels never contain one).
        """
        if not label:
            return cls()
        if "." not in label:
            return cls(region=label)
        table, _, region = label.partition(".")
        return cls(table=table, region=region)

    @property
    def label(self) -> str:
        """The label: ``"T.R"``, region-only ``"R"``, table-only ``"T"``, or ``""``."""
        if not self.table:
            return self.region
        if not self.region:
            return self.table
        return f"{self.table}.{self.region}"

    @property
    def segment(self) -> str:
        """The ``{table}/{region}`` path segment: ``"T/R"``, ``"R"`` or ``""``."""
        if not self.table:
            return self.region
        if not self.region:
            return self.table
        return f"{self.table}/{self.region}"

    def step(self, kind: str) -> str:
        """The step id for ``kind``: ``kind`` for a raw ``--mt``, else suffixed."""
        label = self.label
        return kind if not label else f"{kind}.{label}"

    def prefix(self, kind: str) -> str:
        """The checkpoint prefix for ``kind`` (without any trailing response).

        The region-independent base chain is table-scoped and ignores the region;
        ``regenie_input`` is the ``{table}/{region}`` dataset directory; and the
        ``load_*`` / ``merge_*`` kinds are the dataset-segment-scoped regenie
        result prefixes.
        """
        if kind in BASE_CHAIN_PREFIXES:
            scope = f"{self.table}/" if self.table else ""
            return f"{INTERMEDIATE_PREFIX}/{scope}{BASE_CHAIN_PREFIXES[kind]}"
        if kind == "regenie_input":
            scope = f"{self.table}/" if self.table else ""
            return f"{INTERMEDIATE_PREFIX}/{scope}{self.region}"

        scope = f"{self.segment}/" if self.segment else ""
        if kind.startswith("load_"):
            return f"{REGENIE_PREFIX}/{scope}parts/{kind[len('load_') :]}"
        if kind.startswith("merge_"):
            return f"{REGENIE_PREFIX}/{scope}{kind[len('merge_') :]}"
        raise ValueError(f"unknown dataset key kind: {kind}")
