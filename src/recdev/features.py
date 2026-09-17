"""Features reutilizadas entre métodos/folds; rede carregada somente se necessária."""

from __future__ import annotations

import time
import warnings
from zipfile import BadZipFile
from pathlib import Path

import numpy as np

from .manifest import Manifest
from .preprocessing import PREPROCESSING_VERSION, load_pixels


class FeatureStore:
    def __init__(self, root: Path, manifest: Manifest, space="pixels", device=None,
                 batch_size=32, cache_dir: Path | None = None, embedder=None):
        if space not in ("pixels", "embedding"):
            raise ValueError("Espaço deve ser pixels ou embedding")
        self.root, self.manifest, self.space = root, manifest, space
        self.device, self.batch_size = device, batch_size
        self.cache_dir = cache_dir if cache_dir is not None else root / "artifacts" / "cache" / "embeddings"
        self.embedder = embedder
        self.table = {}
        self.computed_images = 0
        self.cache_hits = 0
        self.extraction_ms = 0.0
        self.cache_read_ms = 0.0
        self.decompositions = {}
        self._cache_loaded = False

    def _load_cache(self):
        from .embeddings import EmbeddingCache, model_signature

        self.signature = model_signature()
        self.cache_path = self.cache_dir / f"{self.manifest.dataset}-vggface2.npz"
        start = time.perf_counter()
        if self.cache_path.exists():
            try:
                cache = EmbeddingCache.load(self.cache_path)
                valid = {r.path: (r.sha256, r.identity_index) for r in self.manifest.records}
                if (cache.dataset == self.manifest.dataset and cache.model_name == self.signature
                        and cache.preprocessing == PREPROCESSING_VERSION):
                    for path, label, digest, embedding in zip(cache.paths, cache.labels, cache.source_hashes, cache.embeddings):
                        if valid.get(path) == (digest, label):
                            self.table[path] = embedding
            except (ValueError, OSError, KeyError, EOFError, BadZipFile) as error:
                warnings.warn(f"Cache ignorado; será reconstruído: {error}", stacklevel=2)
        self.cache_read_ms += (time.perf_counter() - start) * 1000
        self._cache_loaded = True

    def get(self, records) -> np.ndarray:
        if not records:
            raise ValueError("Lista de amostras vazia")
        if self.space == "embedding" and not self._cache_loaded:
            self._load_cache()
        missing = list({r.path: r for r in records if r.path not in self.table}.values())
        self.cache_hits += len(records) - len(missing)
        if missing:
            start = time.perf_counter()
            if self.space == "pixels":
                features = np.stack([load_pixels(self.root / r.path) for r in missing])
            else:
                from .embeddings import FaceEmbedder

                if self.embedder is None:
                    self.embedder = FaceEmbedder(device=self.device, batch_size=self.batch_size)
                features = self.embedder.embed_files([r.path for r in missing], self.root)
            self.extraction_ms += (time.perf_counter() - start) * 1000
            self.computed_images += len(missing)
            self.table.update(zip([r.path for r in missing], features))
            if self.space == "embedding":
                self._save_cache()
        return np.stack([self.table[r.path] for r in records])

    def _save_cache(self):
        from .embeddings import EmbeddingCache

        records = [r for r in self.manifest.records if r.path in self.table]
        EmbeddingCache.build(
            self.manifest.dataset, self.signature, [r.path for r in records],
            [r.identity_index for r in records], np.stack([self.table[r.path] for r in records]),
            [r.sha256 for r in records],
        ).save(self.cache_path)

    def __call__(self, record):
        return self.get([record])[0]

    def summary(self):
        return dict(space=self.space, computed_images=self.computed_images, cache_hits=self.cache_hits,
                    extraction_ms=round(self.extraction_ms, 3), cache_read_ms=round(self.cache_read_ms, 3),
                    device=("cpu" if self.space == "pixels" else
                            self.embedder.device if self.embedder is not None else self.device or "cache/not-loaded"),
                    preprocessing=PREPROCESSING_VERSION if self.space == "embedding" else "rgb-100-unit-v1")
