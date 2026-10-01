"""PCA explícito por SVD econômica: Xc = U S Vt, Z = Xc V_k.

A fatoração é do NumPy/LAPACK. Centralização, posto e projeção ficam aqui.
Não forma a matriz de covariância.
"""

from __future__ import annotations

import numpy as np


class SVDPCA:
    def fit(self, features) -> "SVDPCA":
        x = self._matrix(features)
        if len(x) < 2:
            raise ValueError("PCA requer pelo menos duas amostras")
        self.n_samples_, self.n_features_ = x.shape
        self.mean_ = x.mean(axis=0)
        _, singular, vt = np.linalg.svd(x - self.mean_, full_matrices=False)
        self.singular_values_ = singular
        self.tolerance_ = float(np.finfo(x.dtype).eps * max(x.shape) * singular[0])
        self.rank_ = min(int(np.count_nonzero(singular > self.tolerance_)), len(x) - 1)
        if self.rank_ == 0:
            raise ValueError("PCA indefinido: galeria sem variância (posto numérico zero)")
        self.basis_ = vt[:self.rank_].copy()
        variance = singular ** 2 / (len(x) - 1)
        self.explained_variance_ratio_ = variance / variance.sum()
        return self

    def transform(self, features, k: int) -> np.ndarray:
        self._check_fitted()
        k = self._check_k(k)
        x = self._matrix(features)
        if x.shape[1] != self.n_features_:
            raise ValueError("Dimensão das features difere da dimensão de ajuste")
        return (x - self.mean_) @ self.basis_[:k].T

    def inverse_transform(self, projected) -> np.ndarray:
        self._check_fitted()
        z = self._matrix(projected)
        k = self._check_k(z.shape[1])
        return z @ self.basis_[:k] + self.mean_

    def reconstruction_error(self, features, k: int) -> float:
        """Erro de Frobenius relativo à energia centralizada."""
        x = self._matrix(features)
        residual = np.linalg.norm(x - self.inverse_transform(self.transform(x, k)))
        denominator = np.linalg.norm(x - self.mean_)
        return float(residual / denominator) if denominator else 0.0

    def retained_variance(self, k: int) -> float:
        self._check_fitted()
        k = self._check_k(k)
        return float(self.explained_variance_ratio_[:k].sum())

    def _check_k(self, k: int) -> int:
        if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or k < 1:
            raise ValueError("k deve ser inteiro positivo")
        if k > self.rank_:
            raise ValueError(f"k={k} excede o posto numérico disponível ({self.rank_})")
        return int(k)

    def _check_fitted(self) -> None:
        if not hasattr(self, "basis_"):
            raise ValueError("PCA ainda não ajustado")

    def _matrix(self, features) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or not x.size or not np.isfinite(x).all():
            raise ValueError("Features devem ser uma matriz 2D finita e não vazia")
        return x
