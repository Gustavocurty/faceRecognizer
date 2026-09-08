"""CLI do recdev: auditoria, avaliação e identificação.

Comandos:
    audit      - constrói o manifesto e audita um dataset
    evaluate   - roda o protocolo de avaliação com um método
    identify   - classifica uma imagem contra a galeria de um dataset
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .classifier import EmbeddingClassifier, PixelBaseline
from .embeddings import MODEL_NAME, FaceEmbedder, EmbeddingCache
from .evaluation import (
    EvalReport,
    build_classifier_for,
    evaluate_easy,
    evaluate_very_easy,
    set_active_method,
)
from .manifest import Manifest, audit, build_manifest, find_duplicates, check_subset
from .preprocessing import load_pixels

ROOT = Path(__file__).resolve().parents[2]


def _save_report(report: EvalReport, extra: dict | None = None) -> Path:
    out_dir = ROOT / "artifacts" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = report.summary()
    if extra:
        payload.update(extra)
    path = out_dir / f"{report.dataset}-{report.method}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _featurizer_for(method: str, embedder: FaceEmbedder | None):
    """Retorna callable Record -> feature vector conforme o método."""
    manifest_cache: dict[str, dict[str, np.ndarray]] = {}

    if method in ("pixels", "pca"):
        return lambda rec: load_pixels(ROOT / rec.path)

    if method in ("embedding", "embedding-centroid"):
        assert embedder is not None
        cache_dir = ROOT / "artifacts" / "embeddings"

        def embed_fn(rec):
            ds = rec.dataset
            key = f"{ds}-{embedder.model_name}"
            if key not in manifest_cache:
                cache_path = cache_dir / f"{ds}-{embedder.model_name}.pt"
                if cache_path.exists():
                    cache = EmbeddingCache.load(cache_path)
                    if cache.model_name == embedder.model_name:
                        manifest_cache[key] = {
                            p: emb for p, emb in zip(cache.paths, cache.embeddings)
                        }
                    else:
                        manifest_cache[key] = {}
                else:
                    manifest_cache[key] = {}

            table = manifest_cache[key]
            if rec.path not in table:
                # Computa e atualiza cache incrementalmente.
                return embedder.embed_files([rec.path], ROOT)[0]
            return table[rec.path]

        return embed_fn

    raise ValueError(f"Método desconhecido: {method!r}")


def cmd_audit(args: argparse.Namespace) -> int:
    manifest = build_manifest(args.dataset, ROOT)
    out = ROOT / "artifacts" / "manifests" / f"{args.dataset}.csv"
    manifest.to_csv(out)
    info = audit(manifest)
    print(json.dumps(info, indent=2, ensure_ascii=False))

    dups = find_duplicates(manifest)
    if dups:
        print(f"\nDuplicatas internas ({len(dups)} grupos):")
        for h, paths in sorted(dups.items())[:10]:
            print("  " + " <-> ".join(paths))

    # Verifica subconjuntos conhecidos entre níveis.
    other = "easy" if args.dataset == "very-easy" else "very-easy"
    other_dir = ROOT / other
    if other_dir.is_dir():
        other_manifest = build_manifest(other, ROOT)
        pairs = check_subset(manifest, other_manifest)
        print(f"\nSubconjunto de '{other}': {len(pairs)}/{len(manifest.records)} imagens")

    print(f"\nManifesto salvo em: {out}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    manifest = build_manifest(args.dataset, ROOT)
    set_active_method(args.method)

    embedder = None
    if args.method.startswith("embedding"):
        embedder = FaceEmbedder()

    if args.dataset == "very-easy":
        folds = evaluate_very_easy(manifest, _featurizer_for(args.method, embedder))
    elif args.dataset == "easy":
        folds = evaluate_easy(manifest, _featurizer_for(args.method, embedder))
    else:
        print(f"Dataset ainda não suportado: {args.dataset}", file=sys.stderr)
        return 2

    report = EvalReport(dataset=args.dataset, method=args.method, folds=folds)
    path = _save_report(report)
    print(json.dumps(report.summary(), indent=2, ensure_ascii=False))
    print(f"\nRelatório salvo em: {path}")
    return 0


def cmd_identify(args: argparse.Namespace) -> int:
    gallery_manifest = build_manifest(args.gallery, ROOT)
    embedder = FaceEmbedder()

    # Galeria: foto 1 de cada identidade (ou clean de 'a' no easy).
    by_identity: dict[str, list] = {}
    for rec in gallery_manifest.records:
        by_identity.setdefault(rec.identity_original, []).append(rec)
    gallery_recs = []
    for ident, recs in sorted(by_identity.items()):
        clean_a = [r for r in recs if r.source_photo == "a" and r.condition == "clean"]
        pick = clean_a[0] if clean_a else sorted(recs, key=lambda r: r.photo_id)[0]
        gallery_recs.append(pick)

    embedder_cache = EmbeddingCache.build(
        dataset=args.gallery,
        model_name=embedder.model_name,
        paths=[r.path for r in gallery_recs],
        labels=[r.identity_index for r in gallery_recs],
        embeddings=embedder.embed_files([r.path for r in gallery_recs], ROOT),
    )
    cache_dir = ROOT / "artifacts" / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{args.gallery}-gallery.pt"
    embedder_cache.save(cache_path)

    clf = EmbeddingClassifier(mode="nn").fit(
        embedder_cache.embeddings,
        np.array([r.identity_index for r in gallery_recs]),
    )

    query_emb = embedder.embed_files([args.image], Path("."))[0]
    (pred,) = clf.predict(query_emb[None, :])
    ident_map = {r.identity_index: r.identity_original for r in gallery_recs}

    print(f"Identidade: {ident_map[pred.label]}")
    print(f"Similaridade: {pred.similarity:.4f}")
    print(f"Segundo colocado: {ident_map.get(pred.runner_up, '?')}")
    print(f"Margem: {pred.similarity - pred.runner_up_similarity:.4f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="recdev", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_audit = sub.add_parser("audit", help="manifesto + auditoria do dataset")
    p_audit.add_argument("--dataset", required=True, choices=["very-easy", "easy"])
    p_audit.set_defaults(func=cmd_audit)

    p_eval = sub.add_parser("evaluate", help="protocolo de avaliação")
    p_eval.add_argument("--dataset", required=True, choices=["very-easy", "easy"])
    p_eval.add_argument(
        "--method",
        default="embedding",
        choices=["pixels", "pca", "embedding", "embedding-centroid"],
    )
    p_eval.set_defaults(func=cmd_evaluate)

    p_ident = sub.add_parser("identify", help="identifica uma imagem")
    p_ident.add_argument("--gallery", required=True, choices=["very-easy", "easy"])
    p_ident.add_argument("--image", required=True)
    p_ident.set_defaults(func=cmd_identify)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())