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
    return [8, 16, 32, 64] if dataset in ("medium", "hard") else [1, 2, 3, 4]


def predict_labels(gallery, gallery_labels, queries, k: int) -> list[int]:
    labels, *_ = _predict(gallery, gallery_labels, queries, k)
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
    analysis = _analyze_all(results, hits, n_queries, errors)
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
        "analysis": analysis,
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
        gallery_labels = [r.identity_index for r in split.gallery]
        score_matrix = np.clip(projected_queries @ projected_gallery.T, -1.0, 1.0)
        predicted, scores = _nearest(projected_queries, projected_gallery, gallery_labels)
        truth = [r.identity_index for r in rows]
        hits += sum(label == true for label, true in zip(predicted, truth))
        n_queries += len(rows)
        variances.append(pca.retained_variance(k))
        errors.append(pca.reconstruction_error(queries, k))
        weights.append(len(rows))
        if collect_cases:
            for record, label, score, row in zip(rows, predicted, scores, score_matrix):
                _, _, second, second_score = _rank_identities(row, gallery_labels)
                true_score = _best_identity_score(row, gallery_labels, record.identity_index)
                margin = None if second_score is None else score - second_score
                cases.append({
                    "path": record.path,
                    "split": split.name,
                    "true": record.identity_original,
                    "predicted": names[label],
                    "score": score,
                    "second": None if second is None else names[second],
                    "second_score": second_score,
                    "true_score": true_score,
                    "margin": margin,
                    "condition": record.condition,
                    "source_photo": record.source_photo,
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
    errors = []
    for case in cases:
        if case["correct"]:
            continue
        errors.append({
            "path": case["path"],
            "split": case["split"],
            "true": case["true"],
            "predicted": case["predicted"],
            "score": case["score"],
            "second": case["second"],
            "second_score": case["second_score"],
            "true_score": case["true_score"],
            "margin": case["margin"],
            "condition": case["condition"],
            "source_photo": case["source_photo"],
            "reason": _justify(case),
        })
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
        "errors": errors,
        "analysis": _analyze_cases(cases, row),
    }


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _justify(case: dict) -> str:
    parts = []
    margin = case["margin"]
    if case["correct"]:
        if margin is None:
            parts.append("acerto")
        elif margin >= 0.15:
            parts.append(f"acerto com margem {margin:.3f} sobre {case['second']}")
        else:
            parts.append(f"acerto apertado: margem {margin:.3f} sobre {case['second']}")
    elif case["true_score"] is not None:
        gap = case["score"] - case["true_score"]
        parts.append(
            f"cosseno {case['score']:.3f} em {case['predicted']}; "
            f"verdadeira {case['true']} em {case['true_score']:.3f} (Δ={gap:.3f})"
        )
    else:
        parts.append(f"cosseno {case['score']:.3f} em {case['predicted']}")
    if case["condition"] in ("dark", "noise", "noise-dark"):
        parts.append(f"condição {case['condition']}")
    if case["source_photo"] in ("01", "02", "03", "07", "08", "09", "10"):
        parts.append(f"pose de perfil {case['source_photo']}")
    elif case["source_photo"] not in ("a", "b"):
        parts.append(f"pose {case['source_photo']}")
    return "; ".join(parts)


def _group_stats(cases: list[dict], key: str) -> dict[str, dict]:
    groups: dict[str, list[int]] = {}
    for case in cases:
        hits_count, total = groups.setdefault(case[key], [0, 0])
        groups[case[key]] = [hits_count + int(case["correct"]), total + 1]
    return {
        value: {
            "hits": hits_count,
            "n_queries": total,
            "errors": total - hits_count,
            "accuracy": hits_count / total,
        }
        for value, (hits_count, total) in sorted(groups.items())
    }


def _worst_group(groups: dict[str, dict], name: str) -> dict | None:
    if not groups:
        return None
    value, stats = min(groups.items(), key=lambda item: (item[1]["accuracy"], -item[1]["errors"]))
    if stats["errors"] == 0:
        return None
    return {"name": name, "value": value, **stats}


def _analyze_cases(cases: list[dict], row: dict) -> dict:
    hits = [case for case in cases if case["correct"]]
    misses = [case for case in cases if not case["correct"]]
    by_condition = _group_stats(cases, "condition")
    by_source = _group_stats(cases, "source_photo")
    pairs: dict[tuple[str, str], int] = {}
    for case in misses:
        pair = (case["true"], case["predicted"])
        pairs[pair] = pairs.get(pair, 0) + 1
    confused = [
        {"true": true, "predicted": predicted, "count": count}
        for (true, predicted), count in sorted(pairs.items(), key=lambda item: (-item[1], item[0]))[:5]
    ]
    candidates = [
        group
        for group in (
            _worst_group(by_source, "pose"),
            _worst_group(by_condition, "condição"),
        )
        if group
    ]
    worst = min(candidates, key=lambda item: (item["accuracy"], -item["errors"])) if candidates else None
    factors = [
        {
            "factor": "k",
            "value": row["k"],
            "note": (
                f"k={row['k']} retém {row['retained_variance']:.1%} da variância "
                f"(erro de reconstrução {row['reconstruction_error']:.3f})"
            ),
        }
    ]
    hit_margin = _mean([case["margin"] for case in hits if case["margin"] is not None])
    miss_margin = _mean([case["margin"] for case in misses if case["margin"] is not None])
    if hit_margin is not None:
        if miss_margin is not None:
            factors.append({
                "factor": "margin",
                "value": miss_margin,
                "note": (
                    f"margem média (1º−2º cosseno): acertos {hit_margin:.3f}, "
                    f"erros {miss_margin:.3f}"
                ),
            })
        else:
            factors.append({
                "factor": "margin",
                "value": hit_margin,
                "note": f"margem média dos acertos: {hit_margin:.3f}",
            })
    if worst:
        factors.append({
            "factor": worst["name"],
            "value": worst["value"],
            "note": (
                f"{worst['name']} {worst['value']} concentrou {worst['errors']}/"
                f"{worst['n_queries']} falhas (acurácia {worst['accuracy']:.1%})"
            ),
        })
    if confused:
        top = confused[0]
        factors.append({
            "factor": "confused_pair",
            "value": f"{top['true']}→{top['predicted']}",
            "note": f"par mais confundido: {top['true']} predito como {top['predicted']} ({top['count']}×)",
        })
    analysis = {
        "k": row["k"],
        "retained_variance": row["retained_variance"],
        "reconstruction_error": row["reconstruction_error"],
        "score": {
            "hits_mean": _mean([case["score"] for case in hits]),
            "errors_mean": _mean([case["score"] for case in misses]),
        },
        "margin": {"hits_mean": hit_margin, "errors_mean": miss_margin},
        "by_condition": by_condition,
        "by_source": by_source,
        "confused_pairs": confused,
        "worst_group": worst,
        "factors": factors,
    }
    analysis["summary"] = _analysis_summary(analysis, len(misses))
    return analysis


def _analysis_summary(analysis: dict, n_errors: int) -> str:
    parts = [
        f"k={analysis['k']} reteve {analysis['retained_variance']:.1%} da variância "
        f"(erro de reconstrução {analysis['reconstruction_error']:.3f})."
    ]
    hit_margin = analysis["margin"]["hits_mean"]
    miss_margin = analysis["margin"]["errors_mean"]
    if hit_margin is not None and miss_margin is not None:
        parts.append(
            f"Margem média (1º−2º cosseno): acertos {hit_margin:.3f}, erros {miss_margin:.3f}."
        )
    elif hit_margin is not None:
        parts.append(f"Margem média dos acertos: {hit_margin:.3f}.")
    if n_errors == 0:
        parts.append("Nenhum erro neste k representativo: os acertos se separam com o cosseno.")
    elif analysis["worst_group"]:
        worst = analysis["worst_group"]
        parts.append(
            f"Maior interferência: {worst['name']} {worst['value']} "
            f"({worst['errors']}/{worst['n_queries']} falhas)."
        )
    return " ".join(parts)


def _analyze_all(results: list[dict], hits: int, n_queries: int, errors: list[dict]) -> dict:
    factors = []
    for result in results:
        analysis = result["detail"]["analysis"]
        factors.append({
            "factor": "dataset",
            "value": result["dataset"],
            "note": analysis["summary"],
        })
        factors.extend(
            {"dataset": result["dataset"], **factor} for factor in analysis["factors"]
        )
    if errors:
        by_dataset = {}
        for error in errors:
            by_dataset[error["dataset"]] = by_dataset.get(error["dataset"], 0) + 1
        worst_dataset, count = max(by_dataset.items(), key=lambda item: (item[1], item[0]))
        factors.insert(0, {
            "factor": "dataset",
            "value": worst_dataset,
            "note": f"{worst_dataset} concentrou {count} dos {len(errors)} erros",
        })
    summary = (
        f"Panorama {hits}/{n_queries}. "
        + (factors[0]["note"] + ". " if factors else "")
        + "Valores que interferem: k (variância retida), margem de cosseno e condição/pose da consulta."
    )
    return {"summary": summary, "factors": factors}


def _rank_identities(row, labels):
    order = np.argsort(-np.asarray(row), kind="stable")
    first = int(order[0])
    predicted = int(labels[first])
    score = float(row[first])
    second = second_score = None
    for index in order[1:]:
        label = int(labels[index])
        if label != predicted:
            second = label
            second_score = float(row[index])
            break
    return predicted, score, second, second_score


def _best_identity_score(row, labels, identity):
    values = np.asarray(row, dtype=np.float64)[np.asarray(labels) == identity]
    return float(values.max()) if values.size else None


def _nearest(queries, gallery, labels):
    """A identidade é a da imagem da galeria com maior cosseno."""
    scores = np.clip(queries @ gallery.T, -1.0, 1.0)
    predicted, chosen_scores = [], []
    for row in scores:
        label, score, _, _ = _rank_identities(row, labels)
        predicted.append(label)
        chosen_scores.append(score)
    return predicted, chosen_scores
