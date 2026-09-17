"""Invariantes numéricos: posto, projeção, reconstrução e referência externa."""

import numpy as np
import pytest
from sklearn.decomposition import PCA

from recdev.reduction import SVDPCA


@pytest.mark.parametrize("whiten", [False, True])
def test_matches_sklearn_subspace_and_reconstruction(whiten):
    rng = np.random.default_rng(17)
    train = rng.normal(size=(30, 12)) * np.arange(1, 13)
    queries = rng.normal(size=(8, 12))
    ours = SVDPCA(5, whiten).fit(train)
    reference = PCA(n_components=5, whiten=whiten, svd_solver="full").fit(train)
    assert np.allclose(ours.components_.T @ ours.components_, reference.components_.T @ reference.components_)
    assert np.allclose(ours.inverse_transform(ours.transform(queries)),
                       reference.inverse_transform(reference.transform(queries)))
    assert np.allclose(ours.explained_variance_ratio_[:5], reference.explained_variance_ratio_)
    assert np.allclose(ours.components_ @ ours.components_.T, np.eye(5))
    if whiten:
        assert np.allclose(np.var(ours.transform(train), axis=0, ddof=1), 1)


def test_five_gallery_images_have_at_most_four_components():
    x = np.random.default_rng(2).normal(size=(5, 300))
    pca = SVDPCA(whiten=True).fit(x)
    assert pca.rank_ == pca.n_components_ == 4
    assert np.allclose(pca.fit_transform(x), pca.transform(x))
    assert pca.reconstruction_error(x) < 1e-12
    with pytest.raises(ValueError, match="posto numérico"):
        SVDPCA(5).fit(x)


def test_rank_deficient_matrix_and_no_variance():
    t = np.arange(10, dtype=float)
    x = np.column_stack((t, 2 * t + 4, -3 * t))
    assert SVDPCA().fit(x).rank_ == 1
    with pytest.raises(ValueError, match="posto numérico zero"):
        SVDPCA().fit(np.ones((5, 8)))
    with pytest.raises(ValueError):
        SVDPCA().fit(np.ones((1, 8)))


def test_truncation_error_and_energy_are_consistent():
    x = np.random.default_rng(4).normal(size=(15, 25))
    full = SVDPCA().fit(x)
    errors = []
    for k in (1, 3, 7, 14):
        pca = full.truncated(k)
        error = pca.reconstruction_error(x)
        errors.append(error)
        assert np.isclose(error ** 2, 1 - pca.summary()["retained_variance"], atol=1e-12)
    assert errors == sorted(errors, reverse=True)
    assert full.n_components_ == 14


def test_projection_is_frozen_and_rejects_invalid_input():
    rng = np.random.default_rng(0)
    pca = SVDPCA(3).fit(rng.normal(size=(10, 8)))
    mean, basis = pca.mean_.copy(), pca.components_.copy()
    pca.transform(rng.normal(size=(2, 8)) + 100)
    assert np.array_equal(mean, pca.mean_)
    assert np.array_equal(basis, pca.components_)
    with pytest.raises(ValueError):
        pca.transform(np.ones((2, 7)))
    with pytest.raises(ValueError):
        pca.transform(np.full((2, 8), np.nan))
    with pytest.raises(ValueError):
        SVDPCA().transform(np.ones((2, 8)))


def test_precision_changes_numerical_rank_for_nearly_dependent_columns():
    rng = np.random.default_rng(9)
    t = rng.normal(size=30)
    x = np.column_stack((t, t + 1e-8 * rng.normal(size=30)))
    assert SVDPCA(dtype="float64").fit(x).rank_ == 2
    assert SVDPCA(dtype="float32").fit(x).rank_ == 1
