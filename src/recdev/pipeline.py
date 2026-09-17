"""Pipeline de reconhecimento e galerias portáveis (NPZ sem pickle)."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .classifier import EmbeddingClassifier, PixelBaseline, _l2_normalize, feature_matrix
from .config import MethodConfig
from .preprocessing import PREPROCESSING_VERSION
from .reduction import SVDPCA

GALLERY_SCHEMA = 1


def _digest(metadata, arrays):
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
    for key, value in sorted(arrays.items()):
        array = np.ascontiguousarray(value)
        digest.update(f"{key}:{array.dtype}:{array.shape}".encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


class RecognitionModel:
    def __init__(self, config: MethodConfig):
        self.config = config
        self.reducer = None
        self.classifier = (EmbeddingClassifier(
            mode="centroid" if config.method == "embedding-centroid" else "nn",
            threshold=config.threshold, dtype=config.dtype,
        ) if config.uses_embeddings else PixelBaseline(config.dtype))
        self.metadata = {}

    def prepare(self, features):
        x = feature_matrix(features, self.config.dtype)
        return _l2_normalize(x) if self.config.uses_embeddings else x

    def fit(self, features, labels, decomposition: SVDPCA | None = None):
        x = self.prepare(features)
        self.input_dimension = x.shape[1]
        self.original_gallery_bytes = x.nbytes
        self.timings = {"pca_fit_ms": 0.0, "gallery_projection_ms": 0.0,
                        "classifier_fit_ms": 0.0, "svd_reused": decomposition is not None}
        if self.config.uses_pca:
            start = time.perf_counter()
            self.reducer = (decomposition.truncated(self.config.n_components, self.config.whiten)
                            if decomposition is not None else
                            SVDPCA(self.config.n_components, self.config.whiten, self.config.dtype).fit(x))
            self.timings["pca_fit_ms"] = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            projected = self.reducer.transform(x)
            self.timings["gallery_projection_ms"] = (time.perf_counter() - start) * 1000
        else:
            projected = x
        start = time.perf_counter()
        self.classifier.fit(projected, labels)
        self.timings["classifier_fit_ms"] = (time.perf_counter() - start) * 1000
        return self

    def predict_timed(self, features):
        start = time.perf_counter()
        x = self.prepare(features)
        projected = self.reducer.transform(x) if self.reducer else x
        projection_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        predictions = self.classifier.predict(projected)
        search_ms = (time.perf_counter() - start) * 1000
        return predictions, {"query_projection_ms": projection_ms, "search_ms": search_ms}

    def predict(self, features):
        return self.predict_timed(features)[0]

    def diagnostics(self, queries=None):
        pca = self.reducer.summary() if self.reducer else None
        gallery_bytes = self.classifier._features.nbytes
        projection_bytes = pca["projection_bytes"] if pca else 0
        result = dict(
            input_dimension=self.input_dimension, output_dimension=self.classifier._features.shape[1],
            dtype=self.config.dtype, original_gallery_bytes=self.original_gallery_bytes,
            gallery_bytes=gallery_bytes, projection_bytes=projection_bytes,
            total_feature_bytes=gallery_bytes + projection_bytes,
            feature_storage_ratio=self.original_gallery_bytes / (gallery_bytes + projection_bytes),
            pca=pca,
        )
        if queries is not None and self.reducer:
            result["query_reconstruction_error"] = self.reducer.reconstruction_error(self.prepare(queries))
        return result

    def save(self, path: Path, dataset: str, identity_map: dict[int, str], records=None):
        if self.classifier._features is None:
            raise ValueError("Modelo ainda não ajustado")
        metadata = dict(
            schema=GALLERY_SCHEMA, dataset=dataset, config=self.config.to_dict(),
            identity_map={str(k): v for k, v in identity_map.items()},
            input_dimension=self.input_dimension, original_gallery_bytes=self.original_gallery_bytes,
            pca=self.reducer.summary() if self.reducer else None,
            gallery=[{"path": r.path, "sha256": r.sha256, "group": r.split_group} for r in records or []],
            preprocessing=PREPROCESSING_VERSION if self.config.uses_embeddings else "rgb-100-unit-v1",
        )
        if self.config.uses_embeddings:
            from .embeddings import model_signature

            metadata["model_signature"] = model_signature()
        arrays = {"features": self.classifier._features, "labels": self.classifier._labels}
        if self.reducer:
            arrays.update({f"pca_{key}": value for key, value in self.reducer.state().items()})
        metadata["fingerprint"] = _digest(metadata, arrays)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, metadata=json.dumps(metadata), **arrays)
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path):
        with np.load(path, allow_pickle=False) as blob:
            metadata = json.loads(str(blob["metadata"]))
            arrays = {key: blob[key].copy() for key in blob.files if key != "metadata"}
        fingerprint = metadata.pop("fingerprint")
        if fingerprint != _digest(metadata, arrays) or metadata["schema"] != GALLERY_SCHEMA:
            raise ValueError("Galeria incompatível ou corrompida")
        model = cls(MethodConfig(**metadata["config"]))
        expected_preprocessing = PREPROCESSING_VERSION if model.config.uses_embeddings else "rgb-100-unit-v1"
        if metadata["preprocessing"] != expected_preprocessing:
            raise ValueError("Galeria usa outro pré-processamento; recrie a galeria")
        if model.config.uses_embeddings:
            from .embeddings import model_signature

            if metadata.get("model_signature") != model_signature():
                raise ValueError("Versão da rede difere da galeria; recrie a galeria")
        model.classifier._features = arrays["features"]
        model.classifier._labels = arrays["labels"]
        model.input_dimension = metadata["input_dimension"]
        model.original_gallery_bytes = metadata["original_gallery_bytes"]
        model.metadata = metadata
        if model.config.uses_pca:
            model.reducer = SVDPCA.from_state(
                {key[4:]: value for key, value in arrays.items() if key.startswith("pca_")}, metadata["pca"],
            )
        return model
