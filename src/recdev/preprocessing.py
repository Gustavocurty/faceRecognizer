"""Pré-processamento de imagens para o pipeline.

Carrega imagens 100×100 e as converte para o formato esperado por cada
método: vetores de pixels (baselines) ou tensores normalizados (rede
neural pré-treinada).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

# Entrada esperada pela InceptionResnetV1 (facenet-pytorch).
EMBED_SIZE = 160
PREPROCESSING_VERSION = "rgb-bilinear-160-facenet-fixed-v2"

def load_rgb(path: Path) -> Image.Image:
    """Abre a imagem e garante modo RGB (converte RGBA/tons de cinza)."""
    with Image.open(path) as img:
        return img.convert("RGB")


def to_pixel_vector(img: Image.Image, size: tuple[int, int] = (100, 100)) -> np.ndarray:
    """Vetor de pixels float32 em [0, 1], para baselines (L2/PCA)."""
    img = img.resize(size, Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.reshape(-1)


def to_tensor(img: Image.Image, size: int = EMBED_SIZE) -> np.ndarray:
    """Tensor CHW float32 com normalização usada pelo facenet-pytorch."""
    img = img.resize((size, size), Image.Resampling.BILINEAR)
    # fixed_image_standardization do facenet-pytorch.
    arr = (np.asarray(img, dtype=np.float32) - 127.5) / 128.0
    return arr.transpose(2, 0, 1)  # HWC -> CHW


def load_pixels(path: Path, size: tuple[int, int] = (100, 100)) -> np.ndarray:
    return to_pixel_vector(load_rgb(path), size)


def load_tensor(path: Path, size: int = EMBED_SIZE) -> np.ndarray:
    return to_tensor(load_rgb(path), size)
