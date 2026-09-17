from dataclasses import replace
from pathlib import Path

import pytest

from recdev.manifest import audit, build_manifest, parse_fei_name
from recdev.protocols import Split, make_splits

ROOT = Path(__file__).resolve().parents[1]


def test_medium_manifest_and_conservative_front_group():
    manifest = build_manifest("medium", ROOT)
    info = audit(manifest)
    assert info["n_images"] == 350
    assert info["n_identities"] == 50
    assert info["n_split_groups"] == 200
    assert parse_fei_name("1-04.jpg") == ("04", "fei-04")
    frontal = [r for r in manifest.records if r.identity_original == "1" and r.source_photo in ("11", "12", "13", "14")]
    assert len({r.split_group for r in frontal}) == 1


def test_medium_partitions_disjoint_and_complete():
    manifest = build_manifest("medium", ROOT)
    (split,) = make_splits(manifest)
    assert [len(split.gallery), len(split.validation), len(split.test)] == [100, 50, 200]
    assert {r.source_photo for r in split.gallery} == {"04", "06"}
    assert {r.source_photo for r in split.validation} == {"05"}
    assert {r.source_photo for r in split.test} == {"11", "12", "13", "14"}
    split.validate()
    assert {r.path for part in (split.gallery, split.validation, split.test) for r in part} == {r.path for r in manifest.records}


@pytest.mark.parametrize("attribute", ["path", "split_group", "sha256"])
def test_partition_validator_detects_actual_leakage(attribute):
    (split,) = make_splits(build_manifest("medium", ROOT))
    contaminated = replace(split.test[0], **{attribute: getattr(split.gallery[0], attribute)})
    invalid = Split(split.name, split.gallery, split.validation, [contaminated] + split.test[1:])
    with pytest.raises(ValueError, match="Vazamento"):
        invalid.validate()


def test_incomplete_mapping_is_not_silently_accepted():
    manifest = build_manifest("medium", ROOT)
    manifest.records.pop()
    with pytest.raises(ValueError, match="incompletas"):
        make_splits(manifest)
