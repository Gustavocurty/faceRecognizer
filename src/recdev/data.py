"""Leitura das bases e splits do protocolo source-group-v1, sem hashes."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

IMG_SUFFIXES = {".jpg", ".jpeg", ".png"}
CONDITIONS = ("clean", "dark", "noise", "noise-dark")
DATASETS = (
    "very-easy",
    "easy",
    "medium",
    "hard",
    "fb1-inner",
    "fb1-outer",
    "fb2-inner",
    "fb2-outer",
)
DATASET_ALIASES = {
    "fb1": "fb1-inner",
    "fb2": "fb2-inner",
    "facebook-1": "fb1-inner",
    "facebook-2": "fb2-inner",
    "facebookfaces": "fb1-inner",
    "facebookfaces-2": "fb2-inner",
}
DATASET_REL_DIRS = {
    "very-easy": Path("very-easy"),
    "easy": Path("easy"),
    "medium": Path("medium"),
    "hard": Path("hard"),
    "fb1-inner": Path("extras/facebookfaces/crop-inner"),
    "fb1-outer": Path("extras/facebookfaces/crop-outer"),
    "fb2-inner": Path("extras/facebookfaces-2/crop-inner"),
    "fb2-outer": Path("extras/facebookfaces-2/crop-outer"),
}
HARD_CODES = ("01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14")
HARD_GALLERY = ("04", "06")
HARD_VALIDATION = ("05",)
HARD_TEST = ("01", "02", "03", "07", "08", "09", "10", "11", "12", "13", "14")
FB1_CODES = tuple(str(i) for i in range(1, 10))
FB2_CODES = tuple(str(i) for i in range(1, 11))
FB_GALLERY = ("1", "2")
FB_VALIDATION = ("3",)

_NAME_RE = re.compile(r"^(?P<identity>\d+)-(?P<photo_id>\d+)\.(jpg|jpeg|png)$")


@dataclass
class Record:
    path: str
    identity_original: str
    identity_index: int
    source_photo: str
    condition: str


@dataclass
class Split:
    name: str
    gallery: list[Record]
    validation: list[Record]
    test: list[Record]


def parse_fei_name(fei_name: str) -> tuple[str, str]:
    """Extrai (foto-base, condição) de um nome FEI como ``11a-noise-dark.jpg``."""
    stem = Path(fei_name).stem
    numeric = re.fullmatch(r"\d+-(0[1-9]|1[0-4])", stem)
    if numeric:
        source = numeric.group(1)
        return source, f"fei-{source}"
    parts = stem.split("-")
    source_photo = parts[0][-1].lower()
    if source_photo not in ("a", "b"):
        raise ValueError(f"Não foi possível extrair foto-base de: {fei_name!r}")
    condition = "-".join(parts[1:]) if len(parts) > 1 else "clean"
    if condition not in CONDITIONS:
        raise ValueError(f"Condição desconhecida em: {fei_name!r}")
    return source_photo, condition


def load_mapping(dataset_dir: Path) -> dict[str, str]:
    """Lê ``mapping-to-fei-names``: nome local -> nome original FEI."""
    mapping_path = dataset_dir / "mapping-to-fei-names"
    mapping: dict[str, str] = {}
    if not mapping_path.is_file():
        return mapping
    for line in mapping_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        fei_name, local_name = line.split()
        if local_name in mapping:
            raise ValueError(f"Mapeamento duplicado: {local_name}")
        mapping[local_name] = fei_name
    return mapping


def load_records(dataset: str, root: Path) -> list[Record]:
    resolved = DATASET_ALIASES.get(dataset, dataset)
    if resolved not in DATASET_REL_DIRS:
        raise ValueError(f"Base desconhecida: {dataset}")
    dataset_dir = root / DATASET_REL_DIRS[resolved]
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset não encontrado: {dataset_dir}")
    mapping = load_mapping(dataset_dir)
    files = sorted(p for p in dataset_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_SUFFIXES)
    if not files:
        raise ValueError(f"Nenhuma imagem encontrada em {dataset_dir}")

    identity_by_original: dict[str, int] = {}
    records: list[Record] = []
    for path in files:
        match = _NAME_RE.match(path.name)
        if not match:
            raise ValueError(f"Nome fora do padrão <pessoa>-<foto>.jpg: {path.name}")
        identity_original = match.group("identity")
        if identity_original not in identity_by_original:
            identity_by_original[identity_original] = len(identity_by_original)
        if resolved.startswith("fb"):
            source_photo = match.group("photo_id")
            condition = "crop-inner" if "inner" in resolved else "crop-outer"
        else:
            fei_name = mapping.get(path.name, path.name)
            try:
                source_photo, condition = parse_fei_name(fei_name)
            except ValueError as error:
                if path.name not in mapping:
                    raise ValueError(f"Imagem sem mapeamento FEI: {path.name}") from error
                raise
        records.append(Record(
            path=path.relative_to(root).as_posix(),
            identity_original=identity_original,
            identity_index=identity_by_original[identity_original],
            source_photo=source_photo,
            condition=condition,
        ))
    return records


def make_splits(records: list[Record], dataset: str) -> list[Split]:
    """Galeria, validação e teste sem cruzar a mesma foto-base."""
    resolved = DATASET_ALIASES.get(dataset, dataset)
    if resolved in ("very-easy", "easy"):
        conditions = ("clean",) if resolved == "very-easy" else CONDITIONS
        expected = {(source, condition) for source in ("a", "b") for condition in conditions}
        _check_sources(records, expected)
        pairs = (("a", "b"), ("b", "a"))
        splits = [
            Split(
                name=f"{source}-gallery",
                gallery=[r for r in records if r.source_photo == source and r.condition == "clean"],
                validation=[],
                test=[r for r in records if r.source_photo == other],
            )
            for source, other in pairs
        ]
    elif resolved == "medium":
        codes = ("04", "05", "06", "11", "12", "13", "14")
        _check_sources(records, {(source, f"fei-{source}") for source in codes})
        splits = [Split(
            name="medium-04-06-gallery-05-validation-11-14-test",
            gallery=[r for r in records if r.source_photo in ("04", "06")],
            validation=[r for r in records if r.source_photo == "05"],
            test=[r for r in records if r.source_photo in ("11", "12", "13", "14")],
        )]
    elif resolved == "hard":
        _check_sources(records, {(source, f"fei-{source}") for source in HARD_CODES})
        splits = [Split(
            name="hard-04-06-gallery-05-validation-remaining-test",
            gallery=[r for r in records if r.source_photo in HARD_GALLERY],
            validation=[r for r in records if r.source_photo in HARD_VALIDATION],
            test=[r for r in records if r.source_photo in HARD_TEST],
        )]
    elif resolved.startswith("fb"):
        codes = FB1_CODES if "fb1" in resolved else FB2_CODES
        condition = "crop-inner" if "inner" in resolved else "crop-outer"
        _check_sources(records, {(source, condition) for source in codes})
        splits = [Split(
            name=f"{resolved}-1-2-gallery-3-val-remaining-test",
            gallery=[r for r in records if r.source_photo in FB_GALLERY],
            validation=[r for r in records if r.source_photo in FB_VALIDATION],
            test=[r for r in records if int(r.source_photo) >= 4],
        )]
    else:
        raise ValueError(f"Protocolo não implementado: {dataset}")
    for split in splits:
        _validate(split)
    return splits


def _check_sources(records: list[Record], expected: set[tuple[str, str]]) -> None:
    if not records:
        raise ValueError("Lista de imagens vazia")
    for identity in {r.identity_original for r in records}:
        rows = [r for r in records if r.identity_original == identity]
        actual = {(r.source_photo, r.condition) for r in rows}
        if actual != expected or len(rows) != len(expected):
            raise ValueError(f"Fontes/condições incompletas ou repetidas para identidade {identity}")


def _validate(split: Split) -> None:
    if not split.gallery or not split.test:
        raise ValueError(f"Split {split.name}: galeria ou teste vazio")
    parts = [split.gallery, split.validation, split.test]
    for index, left in enumerate(parts):
        left_paths = {r.path for r in left}
        for right in parts[index + 1:]:
            if left_paths & {r.path for r in right}:
                raise ValueError(f"Vazamento de caminho entre partições em {split.name}")
    gallery_ids = {r.identity_index for r in split.gallery}
    for part in (split.validation, split.test):
        if part and {r.identity_index for r in part} != gallery_ids:
            raise ValueError("Cada partição deve conter as mesmas identidades da galeria")
