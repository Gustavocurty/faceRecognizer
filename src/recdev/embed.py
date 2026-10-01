"""Rede congelada InceptionResnetV1/VGGFace2 e cache simples de embeddings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

EMBED_SIZE = 160


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def to_tensor(img: Image.Image) -> np.ndarray:
    """CHW float32 com a padronização fixa do facenet-pytorch."""
    img = img.resize((EMBED_SIZE, EMBED_SIZE), Image.Resampling.BILINEAR)
    arr = (np.asarray(img, dtype=np.float32) - 127.5) / 128.0
    return arr.transpose(2, 0, 1)


class Embedder:
    """InceptionResnetV1 pré-treinada. Nenhum peso é atualizado."""

    def __init__(self, batch_size: int = 32):
        if batch_size < 1:
            raise ValueError("batch_size deve ser positivo")
        self.batch_size = batch_size
        self.device = "cpu"
        self._model = None

    def _ensure(self):
        if self._model is not None:
            return
        import torch
        from facenet_pytorch import InceptionResnetV1

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = InceptionResnetV1(pretrained="vggface2").to(self.device).eval()
        self._model.requires_grad_(False)
        self._torch = torch

    def embed_files(self, paths: list[Path]) -> np.ndarray:
        if not paths:
            raise ValueError("Nenhum arquivo para extrair embeddings")
        self._ensure()
        batches = []
        for start in range(0, len(paths), self.batch_size):
            images = [to_tensor(load_rgb(path)) for path in paths[start:start + self.batch_size]]
            batch = self._torch.from_numpy(np.stack(images)).to(self.device)
            with self._torch.inference_mode():
                batches.append(self._model(batch).cpu().numpy().astype(np.float32))
        return np.concatenate(batches)


def load_embeddings(dataset: str, records, root: Path, embedder: Embedder | None = None) -> np.ndarray:
    """Devolve uma linha por registro. Recalcula o cache se a lista de arquivos mudar."""
    paths = [r.path for r in records]
    cache_path = root / "artifacts" / "cache" / f"{dataset}.npz"
    if cache_path.is_file():
        try:
            with np.load(cache_path, allow_pickle=False) as blob:
                cached_paths = blob["paths"].astype(str).tolist()
                embeddings = np.asarray(blob["embeddings"], dtype=np.float32)
            if cached_paths == paths and embeddings.shape == (len(paths), embeddings.shape[1]):
                return embeddings
        except (OSError, ValueError, KeyError):
            pass
    if embedder is None:
        embedder = Embedder()
    embeddings = embedder.embed_files([root / path for path in paths])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, paths=np.array(paths), embeddings=embeddings)
    return embeddings
