"""Dois grupos separáveis continuam corretos depois da projeção."""

import numpy as np

from recdev.data import Record
from recdev.identify import predict_labels, run_all, run_experiment


def test_separable_groups_survive_projection():
    rng = np.random.default_rng(0)
    person_a = rng.normal(scale=0.15, size=(6, 12))
    person_b = rng.normal(scale=0.15, size=(6, 12))
    person_a[:, 0] += 3
    person_b[:, 1] += 3
    gallery = np.vstack((person_a[:4], person_b[:4]))
    queries = np.vstack((person_a[4:], person_b[4:]))
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    assert predict_labels(gallery, labels, queries, k=2) == [0, 0, 1, 1]


def _easy_with_one_swap():
    """Foto dark nunca entra na galeria; a troca não vaza para a outra dobra."""
    conditions = ("clean", "dark", "noise", "noise-dark")
    records = []
    embeddings = []
    swapped_path = "easy/5-6.jpg"
    for index in range(5):
        identity = str(index + 1)
        for source_index, source in enumerate(("a", "b")):
            for cond_index, condition in enumerate(conditions):
                path = f"easy/{identity}-{source_index * 4 + cond_index + 1}.jpg"
                records.append(Record(
                    path=path,
                    identity_original=identity,
                    identity_index=index,
                    source_photo=source,
                    condition=condition,
                ))
                vector = np.zeros(8, dtype=np.float64)
                vector[index] = 4.0
                vector[6] = 0.01 * source_index
                vector[7] = 0.01 * cond_index
                embeddings.append(vector)
    embeddings[next(i for i, record in enumerate(records) if record.path == swapped_path)] = embeddings[0].copy()
    return records, np.vstack(embeddings), swapped_path


def test_detail_lists_known_misclassification():
    records, embeddings, swapped_path = _easy_with_one_swap()
    result = run_experiment(records, embeddings, "easy", [4])
    assert "cases" not in result["grid"][0]
    detail = result["detail"]
    assert detail["k"] == 4
    assert detail["hits"] == 39
    assert detail["n_queries"] == 40
    assert detail["by_condition"]["dark"]["hits"] == 9
    assert len(detail["errors"]) == 1
    error = detail["errors"][0]
    assert error["path"] == swapped_path
    assert error["true"] == "5"
    assert error["predicted"] == "1"
    assert error["matched_path"] == "easy/1-1.jpg"
    assert error["condition"] == "dark"
    assert "condição dark" in error["reason"]
    assert error["true_score"] is not None
    analysis = detail["analysis"]
    assert analysis["worst_group"]["name"] == "condição"
    assert analysis["worst_group"]["value"] == "dark"
    assert any(factor["factor"] == "k" for factor in analysis["factors"])


def test_run_all_aggregates_errors():
    records, embeddings, swapped_path = _easy_with_one_swap()
    first = run_experiment(records, embeddings, "easy", [4])
    second = dict(first)
    second["dataset"] = "very-easy"
    summary = run_all([first, second])
    assert summary["overall"]["hits"] == 78
    assert summary["overall"]["n_queries"] == 80
    assert summary["overall"]["n_errors"] == 2
    assert {error["dataset"] for error in summary["errors"]} == {"easy", "very-easy"}
    assert summary["errors"][0]["path"] == swapped_path
    assert "summary" in summary["analysis"]
    assert summary["analysis"]["factors"]
