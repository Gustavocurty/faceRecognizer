"""Testes de splits e protocolo de avaliação (sem vazamento)."""

from pathlib import Path

import numpy as np

from recdev.evaluation import evaluate_easy, evaluate_very_easy, set_active_method
from recdev.manifest import Record, build_manifest

ROOT = Path(__file__).resolve().parents[1]


class ConstantFeaturizer:
    """Featurizer sintético: vetor determinístico por identidade."""

    def __call__(self, rec: Record) -> np.ndarray:
        rng = np.random.default_rng(int(rec.identity_original))
        vec = rng.normal(size=8).astype(np.float32)
        # Perturba apenas a condição noise-dark para simular degradação real.
        if rec.condition == "noise-dark":
            vec = vec + 0.05 * np.ones(8, dtype=np.float32)
        return vec


def _record(ident: str, photo_id: int, source: str, condition: str) -> Record:
    return Record(
        path=f"easy/{ident}-{photo_id}.jpg",
        dataset="easy",
        identity_original=ident,
        identity_index=sorted({"2", "11", "15", "16", "87"}).index(ident),
        photo_id=photo_id,
        source_photo=source,
        condition=condition,
        sha256="",
        split_group=f"{ident}-{source}",
        fei_name="",
    )


def test_very_easy_folds_disjoint():
    manifest = build_manifest("very-easy", ROOT)
    set_active_method("embedding")
    folds = evaluate_very_easy(manifest, ConstantFeaturizer())
    assert len(folds) == 2
    # Fold A: 5 consultas (uma por identidade). Fold B: idem.
    for fold in folds:
        assert len(fold.predictions) == 5
        assert len(set(fold.true_labels)) == 5  # sem repetição de identidade


def test_easy_folds_by_source_photo():
    manifest = build_manifest("easy", ROOT)
    set_active_method("embedding")
    folds = evaluate_easy(manifest, ConstantFeaturizer())
    # 2 folds x 4 condições
    assert len(folds) == 8
    conditions = [f.condition for f in folds]
    assert conditions.count("clean") == 2
    assert conditions.count("dark") == 2
    assert conditions.count("noise") == 2
    assert conditions.count("noise-dark") == 2


def test_easy_perfect_featurizer_scores():
    manifest = build_manifest("easy", ROOT)
    set_active_method("embedding")

    class PerfectFeaturizer(ConstantFeaturizer):
        def __call__(self, rec: Record) -> np.ndarray:
            # Ignora degradações: mesma identidade -> mesmo vetor.
            return np.random.default_rng(int(rec.identity_original)).normal(size=8).astype(np.float32)

    folds = evaluate_easy(manifest, PerfectFeaturizer())
    for fold in folds:
        assert fold.accuracy == 1.0, f"fold {fold.fold}/{fold.condition}"


def test_no_split_group_crosses_gallery_and_probe():
    """Variantes da mesma foto-base nunca aparecem em galeria e teste juntos."""
    manifest = build_manifest("easy", ROOT)
    by_source: dict[str, list] = {"a": [], "b": []}
    for rec in manifest.records:
        by_source[rec.source_photo].append(rec)

    gallery_groups = {r.split_group for r in by_source["a"]}
    probe_groups = {r.split_group for r in by_source["b"]}
    assert gallery_groups.isdisjoint(probe_groups)


def test_deterministic_folds():
    manifest = build_manifest("very-easy", ROOT)
    set_active_method("embedding")
    f1 = evaluate_very_easy(manifest, ConstantFeaturizer())
    f2 = evaluate_very_easy(manifest, ConstantFeaturizer())
    for a, b in zip(f1, f2):
        assert [p.label for p in a.predictions] == [p.label for p in b.predictions]