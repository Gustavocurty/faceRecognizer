"""Splits determinísticos por origem, compartilhados por todos os métodos.

medium-v1: galeria 04/06, validação 05, teste frontal 11..14.
Os códigos são do mapping FEI, não dos números locais dos arquivos.
"""

from dataclasses import dataclass
from itertools import combinations

from .manifest import Manifest, Record

PROTOCOL_VERSION = "source-group-v1"


@dataclass
class Split:
    name: str
    gallery: list[Record]
    validation: list[Record]
    test: list[Record]

    def validate(self):
        if not self.gallery or not self.test:
            raise ValueError(f"Split {self.name}: galeria/teste vazio")
        partitions = [self.gallery, self.validation, self.test]
        datasets = {r.dataset for part in partitions for r in part}
        if len(datasets) != 1:
            raise ValueError("Não misturar datasets no mesmo split")
        for left, right in combinations(partitions, 2):
            for attr in ("path", "split_group", "sha256"):
                a = {getattr(r, attr) for r in left if getattr(r, attr)}
                b = {getattr(r, attr) for r in right if getattr(r, attr)}
                if a & b:
                    raise ValueError(f"Vazamento entre partições: {attr} em {self.name}")
        gallery_labels = {r.identity_index for r in self.gallery}
        for part in (self.validation, self.test):
            if part and {r.identity_index for r in part} != gallery_labels:
                raise ValueError("Cada partição deve conter as mesmas identidades da galeria")

    def summary(self):
        return {
            "name": self.name,
            **{key: [{"path": r.path, "identity": r.identity_original,
                      "source": r.source_photo, "group": r.split_group, "sha256": r.sha256}
                     for r in getattr(self, key)] for key in ("gallery", "validation", "test")},
        }


def make_splits(manifest: Manifest) -> list[Split]:
    records = manifest.records
    if manifest.dataset in ("very-easy", "easy"):
        expected = {("a", c) for c in ("clean", "dark", "noise", "noise-dark")} | {("b", c) for c in ("clean", "dark", "noise", "noise-dark")}
        if manifest.dataset == "very-easy":
            expected = {("a", "clean"), ("b", "clean")}
        _check_sources(records, expected)
        splits = [Split(
            name=f"{source}-gallery",
            gallery=[r for r in records if r.source_photo == source and r.condition == "clean"],
            validation=[],
            test=[r for r in records if r.source_photo == other],
        ) for source, other in (("a", "b"), ("b", "a"))]
    elif manifest.dataset == "medium":
        _check_sources(records, {(s, f"fei-{s}") for s in ("04", "05", "06", "11", "12", "13", "14")})
        splits = [Split(
            name="medium-04-06-gallery-05-validation-11-14-test",
            gallery=[r for r in records if r.source_photo in ("04", "06")],
            validation=[r for r in records if r.source_photo == "05"],
            test=[r for r in records if r.source_photo in ("11", "12", "13", "14")],
        )]
    else:
        raise ValueError(f"Protocolo não implementado: {manifest.dataset}")
    for split in splits:
        split.validate()
    return splits


def _check_sources(records, expected):
    if not records:
        raise ValueError("Manifesto vazio")
    for identity in {r.identity_original for r in records}:
        rows = [r for r in records if r.identity_original == identity]
        actual = {(r.source_photo, r.condition) for r in rows}
        if actual != expected or len(rows) != len(expected):
            raise ValueError(f"Fontes/condições incompletas ou repetidas para identidade {identity}")
