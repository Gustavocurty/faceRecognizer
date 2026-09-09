"""Testes do cache de embeddings: fingerprint e dispositivo."""

import numpy as np

from recdev.embeddings import EmbeddingCache, auto_device


def _cache() -> EmbeddingCache:
    return EmbeddingCache.build(
        dataset="easy",
        model_name="vggface2",
        paths=["easy/1-1.jpg", "easy/2-1.jpg"],
        labels=[0, 1],
        embeddings=np.eye(2, dtype=np.float32),
    )


def test_verify_ok(tmp_path):
    cache = _cache()
    path = tmp_path / "cache.pt"
    cache.save(path)
    loaded = EmbeddingCache.load(path)
    assert loaded.verify()


def test_verify_fails_on_tampering(tmp_path):
    cache = _cache()
    path = tmp_path / "cache.pt"
    cache.save(path)
    loaded = EmbeddingCache.load(path)
    # Alteração silenciosa no conteúdo deve quebrar o fingerprint.
    loaded.embeddings = loaded.embeddings + 0.5
    assert not loaded.verify()


def test_fingerprint_depends_on_content():
    a = _cache()
    b = EmbeddingCache.build(
        dataset="easy",
        model_name="vggface2",
        paths=["easy/1-1.jpg", "easy/2-1.jpg"],
        labels=[0, 1],
        embeddings=np.array([[1, 0], [0, 2]], dtype=np.float32),
    )
    assert a.fingerprint != b.fingerprint


def test_auto_device_is_valid():
    import torch

    device = auto_device()
    assert device in ("cuda", "mps", "cpu")
    if device == "cuda":
        assert torch.cuda.is_available()