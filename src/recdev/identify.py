"""Identificação por cosseno depois do PCA ajustado só na galeria."""

from __future__ import annotations

import numpy as np

from .data import make_splits
from .pca import SVDPCA


def l2_normalize(features) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, np.finfo(np.float64).eps)


def default_components(dataset: str) -> list[int]:
    return [8, 16, 32, 64] if dataset == "medium" else [1, 2, 3, 4]


def predict_labels(gallery, gallery_labels, queries, k: int) -> list[int]:
    labels, _ = _predict(gallery, gallery_labels, queries, k)
    return labels


def identify_query(records, embeddings, dataset: str, query, k: int) -> tuple[str, float]:
    """Ajusta o PCA na galeria do primeiro split e devolve identidade e cosseno."""
    split = make_splits(records, dataset)[0]
    gallery, labels = _rows(records, embeddings, split.gallery)
    names = {r.identity_index: r.identity_original for r in split.gallery}
    predicted, scores = _predict(gallery, labels, np.asarray(query, dtype=np.float64).reshape(1, -1), k)
    return names[predicted[0]], scores[0]


def run_experiment(records, embeddings, dataset: str, components: list[int]) -> dict:
    if not components or any(k < 1 for k in components):
        raise ValueError("Componentes devem ser inteiros positivos")
    splits = make_splits(records, dataset)
    fitted = []
    for split in splits:
        gallery, _ = _rows(records, embeddings, split.gallery)
        gallery = l2_normalize(gallery)
        fitted.append((split, gallery, SVDPCA().fit(gallery)))

    has_validation = bool(splits[0].validation)
    partition = "validation" if has_validation else "test"
    grid = [_evaluate(records, embeddings, fitted, k, partition) for k in components]
    result = {
        "dataset": dataset,
        "mode": "validation-selection" if has_validation else "descriptive",
        "components": list(components),
        "grid": grid,
    }
    chosen = min(grid, key=lambda row: (-row["accuracy"], row["k"]))
    if has_validation:
        test = _evaluate(records, embeddings, fitted, chosen["k"], "test", collect_cases=True)
        result["selected"] = {
            "k": chosen["k"],
            "rule": "maior acurácia na validação; empate: menor k",
            "validation_accuracy": chosen["accuracy"],
            "test_accuracy": test["accuracy"],
            "test_hits": test["hits"],
            "test_queries": test["n_queries"],
            "retained_variance": test["retained_variance"],
            "reconstruction_error": test["reconstruction_error"],
        }
        result["detail"] = _make_detail(test)
    else:
        detailed = _evaluate(records, embeddings, fitted, chosen["k"], "test", collect_cases=True)
        result["detail"] = _make_detail(detailed)
    return result


def run_all(results: list[dict]) -> dict:
    """Junta os blocos detail de várias bases num panorama ponderado pelas consultas."""
    datasets = []
    errors = []
    hits = n_queries = 0
    for result in results:
        detail = result["detail"]
        datasets.append({
            "dataset": result["dataset"],
            "mode": result["mode"],
            "k": detail["k"],
            "hits": detail["hits"],
            "n_queries": detail["n_queries"],
            "accuracy": detail["accuracy"],
            "retained_variance": detail["retained_variance"],
            "reconstruction_error": detail["reconstruction_error"],
            "n_errors": len(detail["errors"]),
        })
        hits += detail["hits"]
        n_queries += detail["n_queries"]
        errors.extend({"dataset": result["dataset"], **error} for error in detail["errors"])
    return {
        "datasets": datasets,
        "overall": {
            "hits": hits,
            "n_queries": n_queries,
            "accuracy": hits / n_queries if n_queries else 0.0,
            "n_errors": len(errors),
            "note": "acurácia ponderada pelas consultas de cada protocolo; as bases não são misturadas",
        },
        "errors": errors,
    }


def _rows(records, embeddings, chosen):
    index = {record.path: i for i, record in enumerate(records)}
    matrix = np.asarray(embeddings, dtype=np.float64)[[index[record.path] for record in chosen]]
    return matrix, [record.identity_index for record in chosen]


def _predict(gallery, gallery_labels, queries, k: int):
    gallery = l2_normalize(gallery)
    queries = l2_normalize(queries)
    pca = SVDPCA().fit(gallery)
    projected_gallery = l2_normalize(pca.transform(gallery, k))
    projected_queries = l2_normalize(pca.transform(queries, k))
    return _nearest(projected_queries, projected_gallery, gallery_labels)


def _evaluate(records, embeddings, fitted, k: int, partition: str, collect_cases: bool = False) -> dict:
    hits = n_queries = 0
    variances, errors, weights = [], [], []
    cases = []
    for split, gallery, pca in fitted:
        rows = getattr(split, partition)
        queries, _ = _rows(records, embeddings, rows)
        queries = l2_normalize(queries)
        projected_gallery = l2_normalize(pca.transform(gallery, k))
        projected_queries = l2_normalize(pca.transform(queries, k))
        names = {r.identity_index: r.identity_original for r in split.gallery}
        predicted, scores = _nearest(
            projected_queries, projected_gallery, [r.identity_index for r in split.gallery]
        )
        truth = [r.identity_index for r in rows]
        hits += sum(label == true for label, true in zip(predicted, truth))
        n_queries += len(rows)
        variances.append(pca.retained_variance(k))
        errors.append(pca.reconstruction_error(queries, k))
        weights.append(len(rows))
        if collect_cases:
            for record, label, score in zip(rows, predicted, scores):
                cases.append({
                    "path": record.path,
                    "split": split.name,
                    "true": record.identity_original,
                    "predicted": names[label],
                    "score": score,
                    "condition": record.condition,
                    "correct": label == record.identity_index,
                })
    result = {
        "k": k,
        "partition": partition,
        "hits": hits,
        "n_queries": n_queries,
        "accuracy": hits / n_queries,
        "retained_variance": float(np.mean(variances)),
        "reconstruction_error": float(np.average(errors, weights=weights)),
    }
    if collect_cases:
        result["cases"] = cases
    return result


def _make_detail(row: dict) -> dict:
    cases = row["cases"]
    buckets: dict[str, list[int]] = {}
    for case in cases:
        hits_count, total = buckets.setdefault(case["condition"], [0, 0])
        buckets[case["condition"]] = [hits_count + int(case["correct"]), total + 1]
    return {
        "k": row["k"],
        "partition": row["partition"],
        "hits": row["hits"],
        "n_queries": row["n_queries"],
        "accuracy": row["accuracy"],
        "retained_variance": row["retained_variance"],
        "reconstruction_error": row["reconstruction_error"],
        "by_condition": {
            condition: {
                "hits": hits_count,
                "n_queries": total,
                "accuracy": hits_count / total,
            }
            for condition, (hits_count, total) in sorted(buckets.items())
        },
        "errors": [
            {
                "path": case["path"],
                "split": case["split"],
                "true": case["true"],
                "predicted": case["predicted"],
                "score": case["score"],
                "condition": case["condition"],
            }
            for case in cases
            if not case["correct"]
        ],
    }


def _nearest(queries, gallery, labels):
    """A identidade é a da imagem da galeria com maior cosseno."""
    scores = np.clip(queries @ gallery.T, -1.0, 1.0)
    predicted, chosen_scores = [], []
    for row in scores:
        index = int(np.argsort(-row, kind="stable")[0])
        predicted.append(int(labels[index]))
        chosen_scores.append(float(row[index]))
    return predicted, chosen_scores
