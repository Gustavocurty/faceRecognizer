"""PCA por SVD: projeção, posto e base congelada na consulta."""

import numpy as np
import pytest

from recdev.pca import SVDPCA


def test_matches_svd_rank_limit_and_frozen_basis():
    rng = np.random.default_rng(0)
    train = rng.normal(size=(12, 8))
    pca = SVDPCA().fit(train)
    centered = train - train.mean(axis=0)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    tolerance = np.finfo(np.float64).eps * max(train.shape) * singular[0]
    rank = min(int(np.count_nonzero(singular > tolerance)), len(train) - 1)

    assert pca.rank_ == rank
    assert np.allclose(pca.basis_, vt[:rank])
    assert np.allclose(pca.transform(train, 3), centered @ vt[:3].T)

    mean, basis = pca.mean_.copy(), pca.basis_.copy()
    pca.transform(rng.normal(size=(2, 8)) + 100, 3)
    assert np.array_equal(pca.mean_, mean)
    assert np.array_equal(pca.basis_, basis)

    with pytest.raises(ValueError, match="posto numérico"):
        pca.transform(train, rank + 1)

    five = rng.normal(size=(5, 20))
    small = SVDPCA().fit(five)
    assert small.rank_ == 4
    with pytest.raises(ValueError, match="posto numérico"):
        small.transform(five, 5)
