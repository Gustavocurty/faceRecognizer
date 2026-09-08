"""Extração de embeddings faciais com rede pré-treinada (facenet-pytorch).

O backbone permanece congelado; a identificação é feita por similaridade
de cosseno no espaço de embeddings (ver classifier.EmbeddingClassifier).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .preprocessing import load_tensor, load_rgb

MODEL_NAME = "vggface2"
DEVICE = "cpu"


class FaceEmbedder:
    """Wrapper determinístico em torno da InceptionResnetV1 pré-treinada."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        # Import local: falha rápida com mensagem clara se o pacote faltar.
        from facenet_pytorch import InceptionResnetV1

        self.model_name = model_name
        self.model = InceptionResnetV1(pretrained=model_name, device=DEVICE)
        self.model.eval()

    @torch.no_grad()
    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        from .preprocessing import to_tensor

        batch = torch.stack([torch.from_numpy(to_tensor(img)) for img in images])
        emb = self.model(batch)
        return emb.cpu().numpy().astype(np.float32)

    def embed_files(self, paths: list[Path], root: Path | None = None) -> np.ndarray:
        images = [load_rgb(p if root is None else root / p) for p in paths]
        return self.embed_images(images)


@dataclass
class EmbeddingCache:
    dataset: str
    model_name: str
    paths: list[str]
    labels: list[int]
    embeddings: np.ndarray
    fingerprint: str  # hash dos paths + rótulos + versão do modelo

    @classmethod
    def build(
        cls, dataset: str, model_name: str, paths: list[str], labels: list[int],
        embeddings: np.ndarray,
    ) -> "EmbeddingCache":
        payload = json.dumps(
            {"model": model_name, "paths": paths, "labels": labels}, sort_keys=True
        )
        fingerprint = hashlib.sha256(payload.encode()).hexdigest()[:16]
        return cls(dataset, model_name, paths, labels, embeddings, fingerprint)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "dataset": self.dataset,
                "model_name": self.model_name,
                "paths": self.paths,
                "labels": self.labels,
                "embeddings": torch.from_numpy(self.embeddings),
                "fingerprint": self.fingerprint,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "EmbeddingCache":
        blob = torch.load(path, map_location=DEVICE, weights_only=False)
        return cls(
            dataset=blob["dataset"],
            model_name=blob["model_name"],
            paths=blob["paths"],
            labels=blob["labels"],
            embeddings=blob["embeddings"].numpy().astype(np.float32),
            fingerprint=blob["fingerprint"],
        )