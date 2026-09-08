"""Manifesto e auditoria dos datasets.

Gera registros estruturados a partir dos nomes de arquivo e dos arquivos
``mapping-to-fei-names``, preservando a linhagem de cada imagem
(identidade, foto-base ``a``/``b``, condição fotométrica e hashes).
"""

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path

IMG_SUFFIXES = {".jpg", ".jpeg", ".png"}

# Condições fotométricas derivadas do nome original FEI (ex.: 11a-noise-dark.jpg)
CONDITIONS = ("clean", "dark", "noise", "noise-dark")

_NAME_RE = re.compile(r"^(?P<identity>\d+)-(?P<photo_id>\d+)\.(jpg|jpeg|png)$")


@dataclass
class Record:
    path: str
    dataset: str
    identity_original: str
    identity_index: int
    photo_id: int
    source_photo: str  # "a" ou "b" (fotografia-base FEI)
    condition: str  # clean | dark | noise | noise-dark
    sha256: str
    split_group: str  # agrupa variantes da mesma foto-base
    fei_name: str = ""  # nome original no dataset FEI


@dataclass
class Manifest:
    dataset: str
    records: list[Record] = field(default_factory=list)

    def to_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(self.records[0]).keys()))
            writer.writeheader()
            for rec in self.records:
                writer.writerow(asdict(rec))

    @classmethod
    def from_csv(cls, path: Path) -> "Manifest":
        with Path(path).open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        records = [
            Record(
                path=r["path"],
                dataset=r["dataset"],
                identity_original=r["identity_original"],
                identity_index=int(r["identity_index"]),
                photo_id=int(r["photo_id"]),
                source_photo=r["source_photo"],
                condition=r["condition"],
                sha256=r["sha256"],
                split_group=r["split_group"],
                fei_name=r.get("fei_name", ""),
            )
            for r in rows
        ]
        dataset = records[0].dataset if records else Path(path).stem
        return cls(dataset=dataset, records=records)


def parse_fei_name(fei_name: str) -> tuple[str, str]:
    """Extrai (foto-base, condição) de um nome FEI como ``11a-noise-dark.jpg``."""
    stem = Path(fei_name).stem
    parts = stem.split("-")
    source_photo = parts[0][-1].lower()
    if source_photo not in ("a", "b"):
        raise ValueError(f"Não foi possível extrair foto-base de: {fei_name!r}")
    condition = "-".join(parts[1:]) if len(parts) > 1 else "clean"
    if condition not in CONDITIONS:
        raise ValueError(f"Condição desconhecida em: {fei_name!r}")
    return source_photo, condition


def load_mapping(dataset_dir: Path) -> dict[str, str]:
    """Lê ``mapping-to-fei-names``: {nome-no-dataset: nome-original-FEI}."""
    mapping_path = dataset_dir / "mapping-to-fei-names"
    mapping: dict[str, str] = {}
    if mapping_path.exists():
        for line in mapping_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            fei_name, local_name = line.split()
            mapping[local_name] = fei_name
    return mapping


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(dataset: str, root: Path) -> Manifest:
    """Constrói o manifesto de um dataset a partir dos nomes de arquivo."""
    dataset_dir = root / dataset
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset não encontrado: {dataset_dir}")

    mapping = load_mapping(dataset_dir)
    files = sorted(
        p for p in dataset_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_SUFFIXES
    )
    if not files:
        raise ValueError(f"Nenhuma imagem encontrada em {dataset_dir}")

    identity_by_original: dict[str, int] = {}
    records: list[Record] = []
    for path in files:
        match = _NAME_RE.match(path.name)
        if not match:
            raise ValueError(f"Nome fora do padrão <person>-<photo>.jpg: {path.name}")
        identity_original = match.group("identity")
        photo_id = int(match.group("photo_id"))

        if identity_original not in identity_by_original:
            identity_by_original[identity_original] = len(identity_by_original)
        identity_index = identity_by_original[identity_original]

        fei_name = mapping.get(path.name, "")
        if fei_name:
            source_photo, condition = parse_fei_name(fei_name)
        else:
            source_photo, condition = "?", "unknown"

        digest = sha256_of(path)
        records.append(
            Record(
                path=str(path.relative_to(root)),
                dataset=dataset,
                identity_original=identity_original,
                identity_index=identity_index,
                photo_id=photo_id,
                source_photo=source_photo,
                condition=condition,
                sha256=digest,
                # Agrupa todas as variantes (dark/noise/...) da mesma foto-base.
                split_group=f"{identity_original}-{source_photo}" if source_photo != "?" else f"{identity_original}-?",
                fei_name=fei_name,
            )
        )

    return Manifest(dataset=dataset, records=records)


def find_duplicates(manifest: Manifest) -> dict[str, list[str]]:
    """Agrupa arquivos com conteúdo idêntico (hash SHA-256)."""
    by_hash: dict[str, list[str]] = {}
    for rec in manifest.records:
        by_hash.setdefault(rec.sha256, []).append(rec.path)
    return {h: paths for h, paths in by_hash.items() if len(paths) > 1}


def check_subset(sub: Manifest, sup: Manifest) -> list[tuple[str, str]]:
    """Retorna pares (caminho-em-sub, caminho-em-sup) de duplicatas por hash."""
    by_hash = {r.sha256: r.path for r in sup.records}
    pairs = []
    for rec in sub.records:
        if rec.sha256 in by_hash:
            pairs.append((rec.path, by_hash[rec.sha256]))
    return pairs


def audit(manifest: Manifest) -> dict:
    """Resumo estatístico do dataset para auditoria."""
    records = manifest.records
    identities = sorted({r.identity_original for r in records})
    per_identity = {
        ident: sum(1 for r in records if r.identity_original == ident)
        for ident in identities
    }
    per_condition = {
        cond: sum(1 for r in records if r.condition == cond)
        for cond in sorted({r.condition for r in records})
    }
    groups = {r.split_group for r in records}
    return {
        "dataset": manifest.dataset,
        "n_images": len(records),
        "n_identities": len(identities),
        "identities": identities,
        "images_per_identity": per_identity,
        "images_per_condition": per_condition,
        "n_split_groups": len(groups),
        "duplicates_within": sum(len(v) - 1 for v in find_duplicates(manifest).values()),
    }