"""Classificadores por similaridade para identificação fechada.

Todos recebem uma galeria (features + rótulos) e classificam consultas
pelo vizinho mais próximo ou centroide, usando distância L2 (baselines)
ou similaridade de cosseno (embeddings).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Rótulo retornado quando a similaridade fica abaixo do limiar (open-set).
UNKNOWN_LABEL = -1


@dataclass
class Prediction:
    label: int
    similarity: float  # maior similaridade (ou -distância) com a galeria
    runner_up: int  # segunda identidade mais provável
    runner_up_similarity: float
    ranking: list[int] = field(default_factory=list)  # identidades por score decrescente (para CMC)


class PixelBaseline:
    """Vizinho mais próximo em vetores de pixels com distância L2."""

    def __init__(self) -> None:
        self._features: np.ndarray | None = None
        self._labels: np.ndarray | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "PixelBaseline":
        self._features = np.asarray(features, dtype=np.float32)
        self._labels = np.asarray(labels, dtype=np.int64)
        return self

    def predict(self, features: np.ndarray) -> list[Prediction]:
        assert self._features is not None
        # Matriz de distâncias L2 entre consultas e galeria.
        q = np.asarray(features, dtype=np.float32)
        d2 = (
            (q ** 2).sum(axis=1)[:, None]
            + (self._features ** 2).sum(axis=1)[None, :]
            - 2 * q @ self._features.T
        )
        np.maximum(d2, 0, out=d2)
        dist = np.sqrt(d2)
        return self._nearest(dist, maximize=False)

    def _nearest(self, scores: np.ndarray, maximize: bool) -> list[Prediction]:
        labels = self._labels
        preds: list[Prediction] = []
        for row in scores:
            order = np.argsort(-row if maximize else row)
            # Ranking de identidades (sem repetição) por score decrescente,
            # usado para CMC (rank-k). Distância: ordem inversa.
            ranking: list[int] = []
            identity_rank: dict[int, float] = {}
            for i in order:
                lab = int(labels[i])
                score = float(row[i])
                if lab not in identity_rank or (maximize and score > identity_rank[lab]):
                    identity_rank[lab] = score
                if lab not in ranking:
                    ranking.append(lab)
            best = order[0]
            label = int(labels[best])
            # Segundo colocado: melhor score com rótulo diferente do vencedor.
            others = [i for i in order if int(labels[i]) != label]
            runner = others[0] if others else best
            preds.append(
                Prediction(
                    label=label,
                    similarity=float(row[best]),
                    runner_up=int(labels[runner]),
                    runner_up_similarity=float(row[runner]),
                    ranking=ranking,
                )
            )
        return preds


class PCABaseline(PixelBaseline):
    """PCA ajustado na galeria + vizinho mais próximo no espaço reduzido."""

    def __init__(self, n_components: int = 20) -> None:
        super().__init__()
        from sklearn.decomposition import PCA

        self._n_components = n_components
        self._pca = PCA(n_components=n_components, whiten=True, random_state=0)

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "PCABaseline":
        n = min(self._n_components, features.shape[0], features.shape[1])
        self._pca.n_components = n
        projected = self._pca.fit_transform(features)
        return super().fit(projected, labels)

    def predict(self, features: np.ndarray) -> list[Prediction]:
        return super().predict(self._pca.transform(features))


class EmbeddingClassifier:
    """Identificação por embeddings faciais com similaridade de cosseno.

    mode="nn": compara com cada imagem da galeria (nearest neighbor).
    mode="centroid": compara com o centroide L2-normalizado por identidade.

    Com threshold > 0 (open-set), consultas com similaridade abaixo do
    limiar são rejeitadas como desconhecidas (label = UNKNOWN_LABEL).
    """

    def __init__(self, mode: str = "nn", threshold: float | None = None) -> None:
        if mode not in ("nn", "centroid"):
            raise ValueError(f"mode inválido: {mode!r}")
        if threshold is not None and threshold <= 0.0:
            raise ValueError(f"threshold deve ser > 0 (ou None): {threshold!r}")
        self.mode = mode
        self.threshold = threshold
        self._features: np.ndarray | None = None
        self._labels: np.ndarray | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "EmbeddingClassifier":
        feats = _l2_normalize(np.asarray(features, dtype=np.float32))
        labels = np.asarray(labels, dtype=np.int64)
        if self.mode == "centroid":
            uniq = np.unique(labels)
            feats = np.stack([_l2_normalize(feats[labels == c].mean(axis=0)) for c in uniq])
            labels = uniq
        self._features = feats
        self._labels = labels
        return self

    def predict(self, features: np.ndarray) -> list[Prediction]:
        assert self._features is not None
        q = _l2_normalize(np.asarray(features, dtype=np.float32))
        sim = q @ self._features.T  # cosseno
        preds = PixelBaseline._nearest(self, sim, maximize=True)
        if self.threshold is not None:
            for p in preds:
                if p.similarity < self.threshold:
                    p.label = UNKNOWN_LABEL
        return preds


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    norm = np.maximum(norm, 1e-12)
    return x / norm


def calibrate_threshold(similarities: list[float], quantile: float = 0.05) -> float:
    """Limiar de rejeição a partir das similaridades de acertos conhecidos.

    Usa o quantil inferior das similaridades observadas (default 5%):
    consultas abaixo desse valor têm alta chance de serem fora-da-galeria.
    """
    if not similarities:
        raise ValueError("Nenhuma similaridade para calibrar")
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile deve estar em (0, 1): {quantile!r}")
    return float(np.quantile(np.asarray(similarities, dtype=np.float32), quantile))