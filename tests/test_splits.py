"""Partições do medium e do hard: caminhos disjuntos."""

from pathlib import Path

from recdev.data import load_records, make_splits

ROOT = Path(__file__).resolve().parents[1]


def _assert_disjoint_cover(split, sizes, total):
    assert (len(split.gallery), len(split.validation), len(split.test)) == sizes
    gallery = {record.path for record in split.gallery}
    validation = {record.path for record in split.validation}
    test = {record.path for record in split.test}
    assert gallery.isdisjoint(validation)
    assert gallery.isdisjoint(test)
    assert validation.isdisjoint(test)
    assert len(gallery | validation | test) == total


def test_medium_splits_are_disjoint():
    records = load_records("medium", ROOT)
    (split,) = make_splits(records, "medium")
    _assert_disjoint_cover(split, (100, 50, 200), 350)


def test_hard_splits_are_disjoint():
    records = load_records("hard", ROOT)
    (split,) = make_splits(records, "hard")
    _assert_disjoint_cover(split, (100, 50, 550), 700)
