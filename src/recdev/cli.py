"""Um comando: uma base, as quatro de uma vez, ou identificação de uma imagem."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import DATASETS, DATASET_ALIASES, load_records
from .embed import Embedder, load_embeddings
from .identify import default_components, identify_query, run_all, run_experiment

ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="recdev",
        description="Identificação facial: rede pré-treinada, PCA por SVD e cosseno.",
    )
    all_choices = tuple(dict.fromkeys(list(DATASETS) + list(DATASET_ALIASES.keys())))
    parser.add_argument("--all", action="store_true", help="avalia todas as bases cadastradas")
    parser.add_argument("--dataset", choices=all_choices)
    parser.add_argument("--root", type=Path, default=ROOT, help="pasta que contém as bases")
    parser.add_argument("--components", nargs="+", type=int, help="valores de k; omissão usa a grade da base")
    parser.add_argument("--image", type=Path, help="consulta avulsa; não grava galeria")
    args = parser.parse_args(argv)
    args.root = args.root.resolve()
    if not args.root.is_dir():
        parser.error(f"Pasta raiz inexistente: {args.root}")
    if args.all and args.dataset:
        parser.error("--all é incompatível com --dataset")
    if args.all and args.image:
        parser.error("--all é incompatível com --image")
    if args.all and args.components:
        parser.error("--all usa a grade padrão de cada base; omita --components")
    if not args.all and not args.dataset:
        parser.error("informe --dataset ou --all")
    try:
        directory = args.root / "artifacts" / "runs"
        directory.mkdir(parents=True, exist_ok=True)
        if args.all:
            embedder = Embedder()
            results = []
            for dataset in DATASETS:
                result = _run_dataset(dataset, args.root, default_components(dataset), embedder)
                _print_table(result)
                path = _write_json(directory / f"{dataset}.json", result)
                print(f"JSON: {path}")
                results.append(result)
            summary = run_all(results)
            summary_path = _write_json(directory / "summary.json", summary)
            _print_summary(summary)
            print(f"JSON: {summary_path}")
            return 0
        dataset = DATASET_ALIASES.get(args.dataset, args.dataset)
        components = args.components or default_components(dataset)
        embedder = Embedder()
        result = _run_dataset(dataset, args.root, components, embedder)
        _print_table(result)
        path = _write_json(directory / f"{dataset}.json", result)
        print(f"JSON: {path}")
        if args.image:
            k = _query_k(components, result)
            records = load_records(dataset, args.root)
            embeddings = load_embeddings(dataset, records, args.root, embedder)
            query = _query_vector(args.image, args.root, records, embeddings, embedder)
            identity, score = identify_query(records, embeddings, dataset, query, k)
            print(f"Consulta: identidade {identity}, score {score:.6f}, k={k}")
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    return 0


def _run_dataset(dataset, root: Path, components, embedder: Embedder) -> dict:
    records = load_records(dataset, root)
    embeddings = load_embeddings(dataset, records, root, embedder)
    return run_experiment(records, embeddings, dataset, components)


def _write_json(path: Path, payload) -> Path:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


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
    analysis = result.get("detail", {}).get("analysis")
    if analysis:
        print(f"análise: {analysis['summary']}")
        for factor in analysis["factors"]:
            print(f"  - {factor['note']}")


def _print_summary(summary) -> None:
    print("resumo")
    print(f"{'base':<10}  {'acertos':>12}  {'acurácia':>8}  {'k':>4}  {'erros':>5}")
    for row in summary["datasets"]:
        print(
            f"{row['dataset']:<10}  {row['hits']:>4}/{row['n_queries']:<4}  "
            f"{row['accuracy']:>8.2%}  {row['k']:>4}  {row['n_errors']:>5}"
        )
    overall = summary["overall"]
    print(
        f"{'geral':<10}  {overall['hits']:>4}/{overall['n_queries']:<4}  "
        f"{overall['accuracy']:>8.2%}"
    )
    analysis = summary.get("analysis")
    if analysis:
        print(f"análise: {analysis['summary']}")
        for factor in analysis["factors"]:
            if factor["factor"] in ("dataset", "k", "margin", "pose", "condição", "confused_pair"):
                prefix = f"{factor['dataset']}: " if factor.get("dataset") else ""
                print(f"  - {prefix}{factor['note']}")
    errors = summary["errors"]
    print(f"erros ({len(errors)}):")
    if not errors:
        print("  nenhum")
        return
    for error in errors:
        reason = error.get("reason", "")
        extra = f"  [{reason}]" if reason else ""
        matched = error.get("matched_path", "")
        match_bit = f"  foto={matched}" if matched else ""
        print(
            f"  {error['path']}  verdade={error['true']}  "
            f"confundiu={error['predicted']}{match_bit}  "
            f"score={error['score']:.4f}{extra}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
