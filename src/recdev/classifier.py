"""Identificação por scores: -distância L2 ou cosseno (maior é melhor)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .reduction import SVDPCA

UNKNOWN_LABEL = -1


@dataclass
class Prediction:
    label: int
    similarity: float
    runner_up: int
    runner_up_similarity: float
    ranking: list[int] = field(default_factory=list)


def feature_matrix(features, dtype="float64") -> np.ndarray:
    x = np.asarray(features, dtype=dtype)
    if x.ndim != 2 or not x.size or not np.isfinite(x).all():
        raise ValueError("Features devem ser matriz 2D não vazia e finita")
    return x


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, np.finfo(x.dtype).eps)


class PixelBaseline:
    def __init__(self, dtype: str = "float64") -> None:
        self.dtype = dtype
        self._features = None
        self._labels = None

    def fit(self, features, labels) -> "PixelBaseline":
        self._features = feature_matrix(features, self.dtype)
        self._labels = np.asarray(labels, dtype=np.int64)
        if self._labels.shape != (len(self._features),) or np.any(self._labels < 0):
            raise ValueError("Rótulos devem ser não negativos e corresponder às amostras")
        return self

    def predict(self, features) -> list[Prediction]:
        q = self._queries(features)
        d2 = ((q ** 2).sum(axis=1)[:, None]
              + (self._features ** 2).sum(axis=1)[None, :]
              - 2 * q @ self._features.T)
        return self._nearest(-np.sqrt(np.maximum(d2, 0)))

    def _queries(self, features):
        if self._features is None:
            raise ValueError("Classificador ainda não ajustado")
        q = feature_matrix(features, self.dtype)
        if q.shape[1] != self._features.shape[1]:
            raise ValueError("Dimensão da consulta incompatível com a galeria")
        return q

    def _nearest(self, scores) -> list[Prediction]:
        predictions = []
        for row in scores:
            order = np.argsort(-row, kind="stable")
            ranked = {}
            for index in order:
                ranked.setdefault(int(self._labels[index]), float(row[index]))
            ranking = list(ranked)
            winner = ranking[0]
            runner = ranking[1] if len(ranking) > 1 else winner
            predictions.append(Prediction(winner, ranked[winner], runner, ranked[runner], ranking))
        return predictions


class PCABaseline(PixelBaseline):
    """Interface simples mantida para uso independente; PCA sem whitening por padrão."""

    def __init__(self, n_components: int | None = None, whiten: bool = False, dtype="float64"):
        super().__init__(dtype)
        self._pca = SVDPCA(n_components, whiten, dtype)

    def fit(self, features, labels):
        return super().fit(self._pca.fit_transform(features), labels)

    def predict(self, features):
        return super().predict(self._pca.transform(features))


class EmbeddingClassifier(PixelBaseline):
    def __init__(self, mode="nn", threshold: float | None = None, dtype="float64"):
        super().__init__(dtype)
        if mode not in ("nn", "centroid"):
            raise ValueError(f"Modo inválido: {mode}")
        if threshold is not None and (not np.isfinite(threshold) or not -1 <= threshold <= 1):
            raise ValueError("Limiar de cosseno deve estar em [-1, 1]")
        self.mode, self.threshold = mode, threshold

    def fit(self, features, labels):
        super().fit(_l2_normalize(feature_matrix(features, self.dtype)), labels)
        if self.mode == "centroid":
            unique = np.unique(self._labels)
            self._features = np.stack([
                _l2_normalize(self._features[self._labels == label].mean(axis=0)) for label in unique
            ])
            self._labels = unique
        return self

    def predict(self, features):
        q = _l2_normalize(self._queries(features))
        predictions = self._nearest(np.clip(q @ self._features.T, -1, 1))
        for prediction in predictions:
            if self.threshold is not None and prediction.similarity < self.threshold:
                prediction.label = UNKNOWN_LABEL
        return predictions


def calibrate_threshold(similarities: list[float], quantile: float = 0.05) -> float:
    """Quantil de scores conhecidos de VALIDAÇÃO; não estima aceitação de impostores."""
    if not similarities or not np.isfinite(similarities).all():
        raise ValueError("Similaridades de calibração vazias ou não finitas")
    if not 0 < quantile < 1:
        raise ValueError("quantile deve estar em (0, 1)")
    return float(np.quantile(similarities, quantile))
