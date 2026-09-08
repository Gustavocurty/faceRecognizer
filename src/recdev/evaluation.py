"""Protocolos de avaliação sem vazamento para very-easy e easy.

very-easy: avaliação bidirecional (fold A: foto 1 -> galeria, foto 2 ->
teste; fold B: o inverso).

easy: folds por fotografia-base (a vs b), nunca por arquivo, para que
variantes dark/noise da mesma foto-base nunca atravessem galeria/teste.
A galeria usa apenas a versão clean; as consultas cobrem as 4 condições.

O ``featurizer`` é um callable ``Record -> np.ndarray`` injetado pelo
chamador, permitindo trocar entre baselines de pixels e embeddings.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .classifier import EmbeddingClassifier, PCABaseline, PixelBaseline, Prediction
from .manifest import Manifest, Record

Featurizer = Callable[[Record], np.ndarray]


@dataclass
class FoldResult:
    fold: str
    condition: str
    predictions: list[Prediction]
    true_labels: list[int]
    true_identities: list[str]
    identity_by_index: dict[int, str]
    elapsed_ms: float

    @property
    def accuracy(self) -> float:
        if not self.predictions:
            return 0.0
        hits = sum(1 for p, t in zip(self.predictions, self.true_labels) if p.label == t)
        return hits / len(self.predictions)


@dataclass
class EvalReport:
    dataset: str
    method: str
    folds: list[FoldResult] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        if not self.folds:
            return 0.0
        return float(np.mean([f.accuracy for f in self.folds]))

    def summary(self) -> dict:
        per_fold = [
            {
                "fold": f.fold,
                "condition": f.condition,
                "n_queries": len(f.predictions),
                "accuracy": round(f.accuracy, 4),
                "elapsed_ms": round(f.elapsed_ms, 2),
            }
            for f in self.folds
        ]
        return {
            "dataset": self.dataset,
            "method": self.method,
            "accuracy_mean": round(self.accuracy, 4),
            "n_folds": len(self.folds),
            "folds": per_fold,
        }


def _run_fold(
    fold_name: str,
    condition: str,
    gallery: list[Record],
    queries: list[Record],
    featurizer: Featurizer,
) -> FoldResult:
    gallery_feats = np.stack([featurizer(rec) for rec in gallery])
    gallery_labels = [rec.identity_index for rec in gallery]
    classifier = build_classifier_for(featurizer).fit(gallery_feats, gallery_labels)

    t0 = time.perf_counter()
    query_feats = np.stack([featurizer(rec) for rec in queries])
    predictions = classifier.predict(query_feats)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    identities = {
        rec.identity_index: rec.identity_original for rec in gallery + queries
    }
    return FoldResult(
        fold=fold_name,
        condition=condition,
        predictions=predictions,
        true_labels=[rec.identity_index for rec in queries],
        true_identities=[rec.identity_original for rec in queries],
        identity_by_index=identities,
        elapsed_ms=elapsed_ms,
    )


# O classificador é escolhido conforme a natureza do featurizer.
# Featurizers carregam seu método em ``set_active_method``.
_active_method: str = "embedding"


def set_active_method(method: str) -> None:
    global _active_method
    _active_method = method


def build_classifier_for(_featurizer: Featurizer):
    method = _active_method
    if method == "pixels":
        return PixelBaseline()
    if method == "pca":
        return PCABaseline(n_components=20)
    if method in ("embedding", "embedding-centroid"):
        mode = "centroid" if method == "embedding-centroid" else "nn"
        return EmbeddingClassifier(mode=mode)
    raise ValueError(f"Método desconhecido: {method!r}")


def evaluate_very_easy(manifest: Manifest, featurizer: Featurizer) -> list[FoldResult]:
    """Dois folds: galeria = foto 1 / teste = foto 2, e o inverso."""
    by_identity: dict[str, list[Record]] = {}
    for rec in manifest.records:
        by_identity.setdefault(rec.identity_original, []).append(rec)
    # Ordena por photo_id para determinismo (foto 1 vs foto 2).
    for recs in by_identity.values():
        recs.sort(key=lambda r: r.photo_id)

    gallery_a = [recs[0] for recs in by_identity.values()]
    probe_a = [recs[1] for recs in by_identity.values()]

    return [
        _run_fold("A: photo1-gallery", "all", gallery_a, probe_a, featurizer),
        _run_fold("B: photo2-gallery", "all", probe_a, gallery_a, featurizer),
    ]


def evaluate_easy(manifest: Manifest, featurizer: Featurizer) -> list[FoldResult]:
    """Folds por foto-base a/b; 4 condições de consulta por fold."""
    by_source: dict[str, list[Record]] = {"a": [], "b": []}
    for rec in manifest.records:
        by_source[rec.source_photo].append(rec)

    folds: list[FoldResult] = []
    for fold_name, gallery_source, probe_source in (
        ("A: a-gallery", "a", "b"),
        ("B: b-gallery", "b", "a"),
    ):
        gallery = [r for r in by_source[gallery_source] if r.condition == "clean"]
        if not gallery:  # fallback defensivo
            gallery = by_source[gallery_source]
        by_cond: dict[str, list[Record]] = {}
        for rec in by_source[probe_source]:
            by_cond.setdefault(rec.condition, []).append(rec)
        for cond in ("clean", "dark", "noise", "noise-dark"):
            if cond in by_cond:
                folds.append(
                    _run_fold(fold_name, cond, gallery, by_cond[cond], featurizer)
                )
    return folds