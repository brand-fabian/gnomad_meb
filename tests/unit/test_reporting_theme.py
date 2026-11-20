"""Hail-free unit tests for the reporting plotting vocabulary."""

from gnomad_meb import reporting
from gnomad_meb.reporting.theme import split_order_key


def test_ordered_splits_canonical_contig_order():
    frame_order = reporting.ordered_splits(["chr10", "chr2", "chr1", "chrX"])
    assert list(frame_order.categories) == ["chr1", "chr2", "chr10", "chrX"]
    assert frame_order.ordered


def test_ordered_splits_bare_contig_spelling():
    order = reporting.ordered_splits(["10", "2", "1", "Y", "X"])
    assert list(order.categories) == ["1", "2", "10", "X", "Y"]


def test_ordered_splits_task_names_sort_by_integer_id():
    order = reporting.ordered_splits(["task-10", "task-2", "task-1"])
    assert list(order.categories) == ["task-1", "task-2", "task-10"]


def test_ordered_splits_unknown_names_last_in_first_seen_order():
    order = reporting.ordered_splits(["chr2", "b_unknown", "a_unknown", "chr1"])
    assert list(order.categories) == [
        "chr1",
        "chr2",
        "b_unknown",
        "a_unknown",
    ]


def test_split_order_key_numeric_semantics():
    assert split_order_key("task-2") < split_order_key("task-10")
    assert split_order_key("chr2") < split_order_key("chr10")
    assert split_order_key("1") < split_order_key("X")
    assert split_order_key("chrX") < split_order_key("chrY")
    assert split_order_key("chrY") < split_order_key("chrM")
