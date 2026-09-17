"""Rede congelada, inferência em lotes e cache verificável de embeddings."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .preprocessing import PREPROCESSING_VERSION, load_rgb, to_tensor

MODEL_NAME = "vggface2"


def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def model_signature(model_name=MODEL_NAME) -> str:
    return f"InceptionResnetV1:{model_name}:facenet-pytorch-{version('facenet-pytorch')}"


class FaceEmbedder:
    def __init__(self, model_name=MODEL_NAME, device: str | None = None, batch_size=32):
        from facenet_pytorch import InceptionResnetV1

        if batch_size < 1:
            raise ValueError("batch_size deve ser positivo")
        self.model_name = model_name
        self.device = device or auto_device()
        self.batch_size = batch_size
        self.model = InceptionResnetV1(pretrained=model_name).to(self.device).eval()
        self.model.requires_grad_(False)
        self._mtcnn = None

    @torch.inference_mode()
    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        if not images:
            raise ValueError("Nenhuma imagem para extrair embeddings")
        batches = []
        for start in range(0, len(images), self.batch_size):
            batch = torch.from_numpy(np.stack([to_tensor(img) for img in images[start:start + self.batch_size]]))
            batches.append(self.model(batch.to(self.device)).cpu().numpy().astype(np.float32))
        return np.concatenate(batches)

    def embed_files(self, paths, root: Path | None = None) -> np.ndarray:
        if not paths:
            raise ValueError("Nenhum arquivo para extrair embeddings")
        batches = []
        for start in range(0, len(paths), self.batch_size):
            images = [load_rgb(Path(p) if root is None else root / p) for p in paths[start:start + self.batch_size]]
            batches.append(self.embed_images(images))
        return np.concatenate(batches)

    @torch.inference_mode()
    def embed_image(self, path, detect=False) -> np.ndarray:
        image = load_rgb(Path(path))
        if not detect:
            return self.embed_images([image])[0]
        from facenet_pytorch import MTCNN

        if self._mtcnn is None:
            self._mtcnn = MTCNN(image_size=160, device=self.device, select_largest=True,
                                keep_all=False, post_process=True)
        # Usa o recorte e a padronização da própria MTCNN; não promete alinhamento por landmarks.
        face = self._mtcnn(image)
        if face is None:
            raise ValueError(f"Nenhum rosto detectado em {path}")
        return self.model(face.unsqueeze(0).to(self.device)).cpu().numpy().astype(np.float32)[0]


@dataclass
class EmbeddingCache:
    dataset: str
    model_name: str
    paths: list[str]
    labels: list[int]
    embeddings: np.ndarray
    fingerprint: str
    source_hashes: list[str] = field(default_factory=list)
    preprocessing: str = PREPROCESSING_VERSION

    @classmethod
    def build(cls, dataset, model_name, paths, labels, embeddings,
              source_hashes=None, preprocessing=PREPROCESSING_VERSION):
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        paths, labels = list(paths), [int(label) for label in labels]
        source_hashes = list(source_hashes or [""] * len(paths))
        if (embeddings.ndim != 2 or embeddings.shape[0] != len(paths)
                or len(labels) != len(paths) or len(source_hashes) != len(paths)
                or len(set(paths)) != len(paths) or not np.isfinite(embeddings).all()):
            raise ValueError("Cache com dimensões, caminhos ou valores inválidos")
        metadata = dict(dataset=dataset, model_name=model_name, paths=paths, labels=labels,
                        source_hashes=source_hashes, preprocessing=preprocessing,
                        shape=list(embeddings.shape))
        digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode() + embeddings.tobytes()).hexdigest()
        return cls(dataset, model_name, paths, labels, embeddings, digest, source_hashes, preprocessing)

    def verify(self) -> bool:
        if self.embeddings.dtype != np.float32:
            return False
        try:
            expected = self.build(self.dataset, self.model_name, self.paths, self.labels,
                                  self.embeddings, self.source_hashes, self.preprocessing)
            return expected.fingerprint == self.fingerprint
        except ValueError:
            return False

    def save(self, path: Path):
        if not self.verify():
            raise ValueError("Fingerprint inválido; cache não salvo")
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {key: value for key, value in vars(self).items() if key != "embeddings"}
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, embeddings=self.embeddings, metadata=json.dumps(metadata))
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path):
        with np.load(path, allow_pickle=False) as blob:
            metadata = json.loads(str(blob["metadata"]))
            cache = cls(**metadata, embeddings=blob["embeddings"].copy())
        if not cache.verify():
            raise ValueError("Cache corrompido: fingerprint inválido")
        return cache
