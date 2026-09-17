from pathlib import Path

import numpy as np
import pytest

from recdev.classifier import PixelBaseline
from recdev.config import MethodConfig
from recdev.pipeline import RecognitionModel


@pytest.mark.parametrize("method", ["pixels", "pca", "embedding", "embedding-pca", "embedding-centroid"])
def test_gallery_roundtrip_predictions_and_projection(method, tmp_path: Path):
    rng = np.random.default_rng(18)
    x = rng.normal(size=(12, 16))
    query = rng.normal(size=(5, 16))
    labels = np.repeat(np.arange(4), 3)
    config = MethodConfig(method, 6 if method in ("pca", "embedding-pca") else None)
    model = RecognitionModel(config).fit(x, labels)
    path = tmp_path / "gallery.npz"
    model.save(path, "synthetic", {i: str(i) for i in range(4)})
    loaded = RecognitionModel.load(path)
    assert model.predict(query) == loaded.predict(query)
    assert model.diagnostics() == loaded.diagnostics()
    if model.reducer:
        assert np.array_equal(model.reducer.transform(query), loaded.reducer.transform(query))
        with pytest.raises(ValueError, match="Base persistida"):
            loaded.reducer.truncated(7)


def test_gallery_tampering_detected(tmp_path):
    model = RecognitionModel(MethodConfig("pixels")).fit(np.eye(3), [0, 1, 2])
    path = tmp_path / "gallery.npz"
    model.save(path, "synthetic", {i: str(i) for i in range(3)})
    with np.load(path, allow_pickle=False) as blob:
        arrays = {key: blob[key].copy() for key in blob.files}
    arrays["features"][0, 0] += 1
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="corrompida"):
        RecognitionModel.load(path)


def test_pixel_scores_have_positive_margin_and_unique_identity_ranking():
    model = PixelBaseline().fit([[0, 0], [0.2, 0], [10, 10]], [0, 0, 1])
    (prediction,) = model.predict([[0.1, 0]])
    assert prediction.similarity <= 0
    assert prediction.similarity - prediction.runner_up_similarity > 0
    assert prediction.ranking == [0, 1]


def test_pca_is_fitted_to_gallery_only():
    rng = np.random.default_rng(8)
    gallery = rng.normal(size=(5, 9))
    model = RecognitionModel(MethodConfig("embedding-pca", 3)).fit(gallery, np.arange(5))
    mean = model.reducer.mean_.copy()
    model.predict(rng.normal(size=(10, 9)) + 1000)
    assert np.array_equal(mean, model.reducer.mean_)
