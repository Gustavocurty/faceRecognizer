"""Configurações explícitas e serializáveis dos métodos de reconhecimento."""

from dataclasses import asdict, dataclass

METHODS = ("pixels", "pca", "embedding", "embedding-pca", "embedding-centroid")
DATASETS = ("very-easy", "easy", "medium")


@dataclass(frozen=True)
class MethodConfig:
    method: str = "embedding"
    n_components: int | None = None
    whiten: bool = False
    dtype: str = "float64"
    threshold: float | None = None

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f"Método desconhecido: {self.method}")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype deve ser float32 ou float64")
        if self.n_components is not None and (not isinstance(self.n_components, int) or self.n_components < 1):
            raise ValueError("n_components deve ser inteiro positivo")
        if not self.uses_pca and (self.n_components is not None or self.whiten):
            raise ValueError("Componentes/whitening requerem pca ou embedding-pca")
        if self.threshold is not None:
            if not self.uses_embeddings or not -1 <= self.threshold <= 1:
                raise ValueError("Limiar de cosseno deve estar em [-1, 1] em método embedding")

    @property
    def uses_embeddings(self) -> bool:
        return self.method.startswith("embedding")

    @property
    def uses_pca(self) -> bool:
        return self.method in ("pca", "embedding-pca")

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def slug(self) -> str:
        parts = [self.method, self.dtype]
        if self.uses_pca:
            parts += [f"k{self.n_components or 'rank'}", f"w{int(self.whiten)}"]
        if self.threshold is not None:
            parts.append(f"t{self.threshold:g}")
        return "-".join(parts)
