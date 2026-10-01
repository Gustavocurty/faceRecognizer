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
    if has_validation:
        chosen = min(grid, key=lambda row: (-row["accuracy"], row["k"]))
        test = _evaluate(records, embeddings, fitted, chosen["k"], "test")
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
    return result


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


def _evaluate(records, embeddings, fitted, k: int, partition: str) -> dict:
    hits = n_queries = 0
    variances, errors, weights = [], [], []
    for split, gallery, pca in fitted:
        rows = getattr(split, partition)
        queries, _ = _rows(records, embeddings, rows)
        queries = l2_normalize(queries)
        projected_gallery = l2_normalize(pca.transform(gallery, k))
        projected_queries = l2_normalize(pca.transform(queries, k))
        predicted, _ = _nearest(projected_queries, projected_gallery, [r.identity_index for r in split.gallery])
        truth = [r.identity_index for r in rows]
        hits += sum(label == true for label, true in zip(predicted, truth))
        n_queries += len(rows)
        variances.append(pca.retained_variance(k))
        errors.append(pca.reconstruction_error(queries, k))
        weights.append(len(rows))
    return {
        "k": k,
        "partition": partition,
        "hits": hits,
        "n_queries": n_queries,
        "accuracy": hits / n_queries,
        "retained_variance": float(np.mean(variances)),
        "reconstruction_error": float(np.average(errors, weights=weights)),
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
