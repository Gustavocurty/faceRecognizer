"""Testes do manifesto e auditoria."""

from pathlib import Path

import pytest

from recdev.manifest import (
    Manifest,
    audit,
    build_manifest,
    check_subset,
    find_duplicates,
    parse_fei_name,
)

ROOT = Path(__file__).resolve().parents[1]


def test_parse_fei_name_variants():
    assert parse_fei_name("11a.jpg") == ("a", "clean")
    assert parse_fei_name("11b-dark.jpg") == ("b", "dark")
    assert parse_fei_name("2a-noise.jpg") == ("a", "noise")
    assert parse_fei_name("87b-noise-dark.jpg") == ("b", "noise-dark")


def test_parse_fei_name_invalid():
    with pytest.raises(ValueError):
        parse_fei_name("xx.jpg")
    with pytest.raises(ValueError):
        parse_fei_name("11a-sepia.jpg")


def test_build_manifest_very_easy():
    manifest = build_manifest("very-easy", ROOT)
    info = audit(manifest)
    assert info["n_images"] == 10
    assert info["n_identities"] == 5
    assert all(v == 2 for v in info["images_per_identity"].values())
    assert info["images_per_condition"] == {"clean": 10}
    assert info["n_split_groups"] == 10


def test_build_manifest_easy():
    manifest = build_manifest("easy", ROOT)
    info = audit(manifest)
    assert info["n_images"] == 40
    assert info["n_identities"] == 5
    assert all(v == 8 for v in info["images_per_identity"].values())
    assert info["images_per_condition"] == {"clean": 10, "dark": 10, "noise": 10, "noise-dark": 10}
    # 10 grupos = 5 identidades x 2 fotos-base
    assert info["n_split_groups"] == 10


def test_identity_indices_contiguous():
    manifest = build_manifest("easy", ROOT)
    indices = {r.identity_index for r in manifest.records}
    assert indices == set(range(5))
    # IDs originais preservados
    assert {r.identity_original for r in manifest.records} == {"2", "11", "15", "16", "87"}


def test_split_groups_group_variants():
    manifest = build_manifest("easy", ROOT)
    groups = {}
    for r in manifest.records:
        groups.setdefault(r.split_group, set()).add(r.condition)
    # Cada grupo deve ter as 4 condições (uma por foto-base a/b).
    for ident in ("2", "11", "15", "16", "87"):
        for src in ("a", "b"):
            group = f"{ident}-{src}"
            assert group in groups
            assert groups[group] == {"clean", "dark", "noise", "noise-dark"}


def test_very_easy_is_subset_of_easy():
    very = build_manifest("very-easy", ROOT)
    easy = build_manifest("easy", ROOT)
    pairs = check_subset(very, easy)
    assert len(pairs) == 10
    # Hashes únicos dentro de cada dataset
    assert not find_duplicates(very)
    assert not find_duplicates(easy)


def test_manifest_csv_roundtrip(tmp_path):
    manifest = build_manifest("very-easy", ROOT)
    path = tmp_path / "manifest.csv"
    manifest.to_csv(path)
    loaded = Manifest.from_csv(path)
    assert loaded.dataset == manifest.dataset
    assert len(loaded.records) == len(manifest.records)
    assert loaded.records[0].sha256 == manifest.records[0].sha256


def test_all_files_have_fei_mapping():
    for dataset in ("very-easy", "easy"):
        manifest = build_manifest(dataset, ROOT)
        unknown = [r for r in manifest.records if not r.fei_name or r.condition == "unknown"]
        assert not unknown, f"Sem mapeamento: {[r.path for r in unknown]}"