"""Um comando: grade de k, JSON curto e identificação opcional de uma imagem."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import DATASETS, load_records
from .embed import Embedder, load_embeddings
from .identify import default_components, identify_query, run_experiment

ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="recdev",
        description="Identificação facial: rede pré-treinada, PCA por SVD e cosseno.",
    )
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--root", type=Path, default=ROOT, help="pasta que contém as bases")
    parser.add_argument("--components", nargs="+", type=int, help="valores de k; omissão usa a grade da base")
    parser.add_argument("--image", type=Path, help="consulta avulsa; não grava galeria")
    args = parser.parse_args(argv)
    args.root = args.root.resolve()
    if not args.root.is_dir():
        parser.error(f"Pasta raiz inexistente: {args.root}")
    components = args.components or default_components(args.dataset)
    try:
        records = load_records(args.dataset, args.root)
        embedder = Embedder()
        embeddings = load_embeddings(args.dataset, records, args.root, embedder)
        result = run_experiment(records, embeddings, args.dataset, components)
        _print_table(result)
        directory = args.root / "artifacts" / "runs"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{args.dataset}.json"
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"JSON: {path}")
        if args.image:
            k = _query_k(components, result)
            query = _query_vector(args.image, args.root, records, embeddings, embedder)
            identity, score = identify_query(records, embeddings, args.dataset, query, k)
            print(f"Consulta: identidade {identity}, score {score:.6f}, k={k}")
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    return 0


def _query_k(components, result) -> int:
    if len(components) == 1:
        return components[0]
    if "selected" in result:
        return result["selected"]["k"]
    raise ValueError("Informe um único --components para identificar nesta base")


def _query_vector(image: Path, root: Path, records, embeddings, embedder: Embedder):
    index = {record.path: i for i, record in enumerate(records)}
    candidates = []
    if not image.is_absolute():
        candidates.append((root / image).resolve())
    if image.exists():
        candidates.append(image.resolve())
    for candidate in candidates:
        try:
            relative = candidate.relative_to(root).as_posix()
        except ValueError:
            continue
        if relative in index:
            return embeddings[index[relative]]
    path = image if image.is_file() else root / image
    if not path.is_file():
        raise FileNotFoundError(f"Imagem não encontrada: {image}")
    return embedder.embed_files([path])[0]


def _print_table(result) -> None:
    print(f"{result['dataset']} ({result['mode']})")
    print(f"{'k':>4}  {'partição':<12}  {'acertos':>12}  {'variância':>10}  {'erro':>8}")
    for row in result["grid"]:
        print(
            f"{row['k']:>4}  {row['partition']:<12}  "
            f"{row['hits']:>4}/{row['n_queries']:<4} {row['accuracy']:>6.2%}  "
            f"{row['retained_variance']:>10.4f}  {row['reconstruction_error']:>8.4f}"
        )
    selected = result.get("selected")
    if selected:
        print(
            f"Escolha na validação: k={selected['k']} ({selected['validation_accuracy']:.2%}); "
            f"teste: {selected['test_hits']}/{selected['test_queries']} = {selected['test_accuracy']:.2%}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
