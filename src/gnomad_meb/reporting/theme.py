"""Stage-agnostic plotting vocabulary shared by the report builders.

The report builders live one module per pipeline stage (:mod:`gnomad_meb.reporting.vds`,
:mod:`gnomad_meb.reporting.sample_qc`, …). This module holds what they share:
the plotnine theme and the canonical ordering for genomic split labels.

Axis categories that name genomic splits sort **semantically, never
lexicographically**: contigs are ``1``-``22``, ``X``, ``Y``, ``M`` (so ``chr2``
precedes ``chr10``) and a ``VariantDataset``'s task names are ``task-N``, which
sort by integer id (so ``task-2`` precedes ``task-10``).
"""

from __future__ import annotations

import pandas as pd
import plotnine as pn

__all__ = [
    "CONTIG_ORDER",
    "THEME",
    "ordered_splits",
    "split_order_key",
]

#: Canonical contig order for report axes: ``chr1``-``chr22``, then ``chrX``,
#: ``chrY`` and ``chrM``.
CONTIG_ORDER = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM"]

#: The shared minimal theme every report figure carries.
THEME = pn.theme_minimal() + pn.theme(figure_size=(8, 5))

#: Contig rank by the *bare* spelling: ``1``…``22``, ``X``/``Y``/``M`` (with the
#: ``MT`` alias folded onto ``M``). ``chr``-prefixed names are stripped first.
_BARE_CONTIG_RANK: dict[str, int] = {
    str(i): i - 1 for i in range(1, 23)
}
_BARE_CONTIG_RANK.update({"X": 22, "Y": 23, "M": 24, "MT": 24})


def split_order_key(name: str) -> tuple[int, int]:
    """Sort key for a contig or ``task-N`` split name (numeric semantics).

    Contigs rank ``1``-``22``, ``X``, ``Y``, ``M``; a ``task-N`` name ranks by
    its integer id. Unknown names share a final rank and keep their first-seen
    order (``sorted`` is stable), so unrelated spellings never interleave with
    the known ones.
    """
    text = str(name)
    bare = text[3:] if text.startswith("chr") else text
    if bare in _BARE_CONTIG_RANK:
        return (0, _BARE_CONTIG_RANK[bare])
    if text.startswith("task-"):
        suffix = text[len("task-") :]
        if suffix.isdigit():
            return (1, int(suffix))
    return (2, 0)


def ordered_splits(values) -> pd.Categorical:
    """Return ``values`` as an ordered categorical in semantic split order.

    ``task-N`` sorts by integer N; ``chr1``/``1`` by the canonical contig order
    (1-22, X, Y, M); unknown names sort after the known ones in first-seen
    order. The categorical's categories drive the plot's discrete axis order.
    """
    seen = list(dict.fromkeys(str(value) for value in values))
    categories = sorted(seen, key=split_order_key)
    return pd.Categorical(values, categories=categories, ordered=True)
