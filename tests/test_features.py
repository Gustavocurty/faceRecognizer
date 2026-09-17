from dataclasses import replace

import numpy as np

from recdev.embeddings import EmbeddingCache
from recdev.features import FeatureStore
from recdev.manifest import Manifest, Record


class FakeEmbedder:
    device = "cpu"

    def __init__(self):
        self.calls = []

    def embed_files(self, paths, root):
        self.calls.extend(paths)
        return np.stack([np.array([1, len(path), 0], dtype=np.float32) for path in paths])


def manifest():
    return Manifest("easy", [Record(f"easy/{i}-1.jpg", "easy", str(i), i, 1, "a", "clean", f"hash-{i}", f"{i}-a") for i in range(2)])


def test_incremental_cache_reused_and_source_change_invalidated(tmp_path):
    data = manifest()
    extractor = FakeEmbedder()
    store = FeatureStore(tmp_path, data, "embedding", embedder=extractor)
    first = store.get(data.records[:1])
    store.get(data.records)
    assert extractor.calls == [r.path for r in data.records]
    other_extractor = FakeEmbedder()
    reloaded = FeatureStore(tmp_path, data, "embedding", embedder=other_extractor)
    assert np.array_equal(reloaded.get(data.records[:1]), first)
    assert not other_extractor.calls

    changed = Manifest(data.dataset, [replace(data.records[0], sha256="new-content"), data.records[1]])
    invalidated = FeatureStore(tmp_path, changed, "embedding", embedder=other_extractor)
    invalidated.get(changed.records)
    assert other_extractor.calls == [data.records[0].path]
    assert invalidated.computed_images == 1


def test_preprocessing_change_invalidates_cache(tmp_path):
    data = manifest()
    first = FeatureStore(tmp_path, data, "embedding", embedder=FakeEmbedder())
    first.get(data.records)
    cache = EmbeddingCache.load(first.cache_path)
    obsolete = EmbeddingCache.build(cache.dataset, cache.model_name, cache.paths, cache.labels,
                                    cache.embeddings, cache.source_hashes, "obsolete-v1")
    obsolete.save(first.cache_path)
    extractor = FakeEmbedder()
    FeatureStore(tmp_path, data, "embedding", embedder=extractor).get(data.records)
    assert len(extractor.calls) == 2
