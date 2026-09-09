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


def auto_device() -> str:
    """Escolhe o melhor dispositivo disponível: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = auto_device()


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

    @torch.no_grad()
    def embed_image(self, path: Path | str, detect: bool = False) -> np.ndarray:
        """Embedding de uma imagem avulsa.

        Com detect=True, usa MTCNN para detectar e alinhar o rosto antes do
        embedding (útil para fotos arbitrárias fora do padrão 100x100).
        Sem detecção, aplica resize direto.
        """
        from .preprocessing import load_tensor

        img = load_rgb(Path(path))
        if detect:
            from facenet_pytorch import InceptionResnetV1  # noqa: F401

            from .preprocessing import to_tensor

            arr = to_tensor(img)[None, :]  # CHW -> 1CHW
            mtcnn = _get_mtcnn()
            boxes, probs = mtcnn.detect(img)
            if boxes is None or probs[0] is None:
                raise ValueError(
                    f"Nenhum rosto detectado em {path}; rode sem --detect "
                    "ou verifique a imagem"
                )
            x0, y0, x1, y1 = [max(0, int(v)) for v in boxes[0]]
            face = img.crop((x0, y0, x1, y1))
            batch = torch.from_numpy(to_tensor(face))[None, :]
        else:
            batch = torch.from_numpy(load_tensor(Path(path)))[None, :]

        emb = self.model(batch)
        return emb.cpu().numpy().astype(np.float32)[0]


_MTCNN: object | None = None


def _get_mtcnn():
    """Instância MTCNN única (lazy) para detecção/alinhamento facial."""
    global _MTCNN
    if _MTCNN is None:
        from facenet_pytorch import MTCNN

        _MTCNN = MTCNN(device=DEVICE, select_largest=True, keep_all=False)
    return _MTCNN


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
        # Hash cobre paths, rótulos E o conteúdo dos embeddings: qualquer
        # alteração silenciosa no cache quebra o fingerprint.
        emb_digest = hashlib.sha256(
            np.ascontiguousarray(embeddings, dtype=np.float32).tobytes()
        ).hexdigest()
        payload = json.dumps(
            {
                "model": model_name,
                "paths": paths,
                "labels": labels,
                "embeddings": emb_digest,
            },
            sort_keys=True,
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

    def verify(self) -> bool:
        """Confere se o fingerprint corresponde ao conteúdo atual."""
        expected = EmbeddingCache.build(
            dataset=self.dataset,
            model_name=self.model_name,
            paths=self.paths,
            labels=self.labels,
            embeddings=self.embeddings,
        ).fingerprint
        return expected == self.fingerprint