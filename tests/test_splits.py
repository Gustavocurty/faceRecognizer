"""Partições do medium: 100/50/200 e caminhos disjuntos."""

from pathlib import Path

from recdev.data import load_records, make_splits

ROOT = Path(__file__).resolve().parents[1]


def test_medium_splits_are_disjoint():
    records = load_records("medium", ROOT)
    (split,) = make_splits(records, "medium")
    assert (len(split.gallery), len(split.validation), len(split.test)) == (100, 50, 200)
    gallery = {record.path for record in split.gallery}
    validation = {record.path for record in split.validation}
    test = {record.path for record in split.test}
    assert gallery.isdisjoint(validation)
    assert gallery.isdisjoint(test)
    assert validation.isdisjoint(test)
    assert len(gallery | validation | test) == 350
