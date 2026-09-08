"""Testes de pré-processamento e classificadores."""

import numpy as np

from recdev.classifier import EmbeddingClassifier, PCABaseline, PixelBaseline, _l2_normalize
from recdev.preprocessing import load_pixels, load_rgb, load_tensor

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_load_rgb_converts_rgba_to_rgb(tmp_path):
    from PIL import Image

    img = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
    path = tmp_path / "test.png"
    img.save(path)
    out = load_rgb(path)
    assert out.mode == "RGB"
    assert out.size == (100, 100)


def test_pixel_vector_shape_and_range():
    from PIL import Image

    img = Image.new("RGB", (100, 100), (128, 64, 32))
    vec = load_rgb_pathless_vec(img)
    assert vec.shape == (100 * 100 * 3,)
    assert vec.dtype == np.float32
    assert 0.0 <= vec.min() and vec.max() <= 1.0


def load_rgb_pathless_vec(img):
    from recdev.preprocessing import to_pixel_vector

    return to_pixel_vector(img)


def test_tensor_shape_and_norm():
    from PIL import Image

    img = Image.new("RGB", (100, 100), (255, 255, 255))
    t = load_rgb_pathless_tensor(img)
    assert t.shape == (3, 160, 160)
    assert t.dtype == np.float32
    # Branco puro: (255/255 - 0.5)/0.5 = 1.0
    assert np.allclose(t[:, :10, :10], 1.0)


def load_rgb_pathless_tensor(img):
    from recdev.preprocessing import to_tensor

    return to_tensor(img)


def test_l2_normalize():
    x = np.array([[3.0, 4.0], [0.0, 0.0]])
    out = _l2_normalize(x)
    assert np.allclose(np.linalg.norm(out[0]), 1.0)
    assert np.allclose(out[1], 0.0)  # vetor nulo tratado


def test_pixel_baseline_nearest():
    feats = np.array(
        [[0.0, 0.0], [1.0, 1.0], [10.0, 10.0]], dtype=np.float32
    )
    labels = np.array([0, 1, 2])
    clf = PixelBaseline().fit(feats, labels)
    preds = clf.predict(np.array([[0.1, 0.0], [9.0, 9.5]], dtype=np.float32))
    assert preds[0].label == 0
    assert preds[1].label == 2
    assert preds[0].runner_up != 0


def test_pca_baseline():
    rng = np.random.default_rng(0)
    # Três nuvens de pontos bem separadas
    centers = np.array([[0, 0], [50, 0], [0, 50]], dtype=np.float32)
    feats = np.vstack([c + rng.normal(scale=0.5, size=(10, 2)) for c in centers])
    labels = np.repeat([0, 1, 2], 10)
    clf = PCABaseline(n_components=2).fit(feats, labels)
    preds = clf.predict(centers)
    assert [p.label for p in preds] == [0, 1, 2]


def test_embedding_classifier_nn():
    # Embeddings em 3D: identidades em direções distintas
    feats = np.array(
        [
            [1, 0, 0],
            [0.9, 0.1, 0],
            [0, 1, 0],
            [0.1, 0.9, 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 1, 1, 2])
    clf = EmbeddingClassifier(mode="nn").fit(feats, labels)
    preds = clf.predict(np.array([[0.95, 0.05, 0], [0, 0.2, 0.98]], dtype=np.float32))
    assert preds[0].label == 0
    assert preds[1].label == 2
    # Similaridade de cosseno: auto-embedding ~ 1.0
    (self_pred,) = clf.predict(feats[0][None, :])
    assert self_pred.similarity > 0.999


def test_embedding_classifier_centroid():
    rng = np.random.default_rng(1)
    c0 = rng.normal(size=(5, 16)) + np.array([5.0] * 16)
    c1 = rng.normal(size=(5, 16)) - np.array([5.0] * 16)
    feats = np.vstack([c0, c1]).astype(np.float32)
    labels = np.array([0] * 5 + [1] * 5)
    clf = EmbeddingClassifier(mode="centroid").fit(feats, labels)
    preds = clf.predict(np.vstack([c0[:2], c1[:2]]))
    assert [p.label for p in preds] == [0, 0, 1, 1]


def test_classifier_mode_invalid():
    import pytest

    with pytest.raises(ValueError):
        EmbeddingClassifier(mode="invalid")