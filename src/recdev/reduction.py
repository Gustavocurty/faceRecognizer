"""PCA explícito por SVD econômica: Xc = U S Vt; Z = Xc V_k.

Não forma X.T @ X (nem uma covariância 30000 x 30000 para pixels).
A rotina de fatoração é do NumPy/LAPACK; centralização, truncamento,
posto numérico, whitening e reconstrução são implementados aqui.
"""

from __future__ import annotations

import numpy as np


class SVDPCA:
    def __init__(self, n_components: int | None = None, whiten: bool = False,
                 dtype: str = "float64"):
        if dtype not in ("float32", "float64"):
            raise ValueError("dtype deve ser float32 ou float64")
        if n_components is not None and (not isinstance(n_components, int) or n_components < 1):
            raise ValueError("n_components deve ser inteiro positivo ou None")
        self.n_components = n_components
        self.whiten = whiten
        self.dtype = dtype

    def _matrix(self, features) -> np.ndarray:
        x = np.asarray(features, dtype=self.dtype)
        if x.ndim != 2 or not x.size or not np.isfinite(x).all():
            raise ValueError("Features devem ser matriz 2D não vazia e finita")
        return x

    def fit(self, features) -> "SVDPCA":
        x = self._matrix(features)
        if len(x) < 2:
            raise ValueError("PCA requer pelo menos duas amostras")
        self.n_samples_, self.n_features_in_ = x.shape
        self.mean_ = x.mean(axis=0)
        centered = x - self.mean_
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
        # Mesmo critério de escala de matrix_rank; posto centralizado <= n-1.
        self.tolerance_ = float(np.finfo(x.dtype).eps * max(x.shape) * singular[0])
        self.rank_ = min(int(np.count_nonzero(singular > self.tolerance_)), len(x) - 1)
        if self.rank_ == 0:
            raise ValueError("PCA indefinido: galeria sem variância (posto numérico zero)")
        self.singular_values_ = singular
        self.explained_variance_ = singular ** 2 / (len(x) - 1)
        self.explained_variance_ratio_ = self.explained_variance_ / self.explained_variance_.sum()
        self._basis = vt[:self.rank_].copy()
        self._select(self.rank_ if self.n_components is None else self.n_components)
        return self

    def _select(self, k: int) -> None:
        if not 1 <= k <= self.rank_:
            raise ValueError(f"k={k} excede o posto numérico disponível ({self.rank_})")
        if k > len(self._basis):
            raise ValueError("Base persistida contém menos componentes; reajuste o PCA para aumentar k")
        self.n_components_ = k
        self.components_ = self._basis[:k].copy()
        self.scale_ = np.sqrt(self.explained_variance_[:k]) if self.whiten else np.ones(k, dtype=self.dtype)

    def truncated(self, k: int | None, whiten: bool | None = None) -> "SVDPCA":
        """Reutiliza uma fatoração na varredura de k, sem reajustar com consultas."""
        from copy import copy

        result = copy(self)
        result.n_components = k
        result.whiten = self.whiten if whiten is None else whiten
        result._select(self.rank_ if k is None else k)
        return result

    def transform(self, features) -> np.ndarray:
        if not hasattr(self, "components_"):
            raise ValueError("PCA ainda não ajustado")
        x = self._matrix(features)
        if x.shape[1] != self.n_features_in_:
            raise ValueError("Dimensão das features difere da dimensão de ajuste")
        return ((x - self.mean_) @ self.components_.T) / self.scale_

    def fit_transform(self, features) -> np.ndarray:
        return self.fit(features).transform(features)

    def inverse_transform(self, projected) -> np.ndarray:
        z = self._matrix(projected)
        if z.shape[1] != self.n_components_:
            raise ValueError("Dimensão da projeção incompatível")
        return (z * self.scale_) @ self.components_ + self.mean_

    def reconstruction_error(self, features) -> float:
        """Erro de Frobenius relativo à energia centralizada das consultas."""
        x = self._matrix(features)
        residual = np.linalg.norm(x - self.inverse_transform(self.transform(x)))
        denominator = np.linalg.norm(x - self.mean_)
        return float(residual / denominator) if denominator else 0.0

    def summary(self) -> dict:
        return {
            "input_dimension": self.n_features_in_,
            "n_components": self.n_components_,
            "numerical_rank": self.rank_,
            "rank_tolerance": self.tolerance_,
            "dtype": self.dtype,
            "whiten": self.whiten,
            "singular_values": self.singular_values_.tolist(),
            "explained_variance_ratio": self.explained_variance_ratio_.tolist(),
            "retained_variance": float(self.explained_variance_ratio_[:self.n_components_].sum()),
            "retained_condition_number": float(self.singular_values_[0] / self.singular_values_[self.n_components_ - 1]),
            "projection_bytes": self.mean_.nbytes + self.components_.nbytes + self.scale_.nbytes,
        }

    def state(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name) for name in
                ("mean_", "components_", "scale_", "singular_values_", "explained_variance_", "explained_variance_ratio_")}

    @classmethod
    def from_state(cls, state: dict, metadata: dict) -> "SVDPCA":
        pca = cls(metadata["n_components"], metadata["whiten"], metadata["dtype"])
        for name, value in state.items():
            setattr(pca, name, np.asarray(value, dtype=pca.dtype))
        pca.n_features_in_ = metadata["input_dimension"]
        pca.n_components_ = metadata["n_components"]
        pca.rank_ = metadata["numerical_rank"]
        pca.tolerance_ = metadata["rank_tolerance"]
        # Galerias persistidas só precisam da base selecionada para inferência.
        pca._basis = pca.components_
        return pca
