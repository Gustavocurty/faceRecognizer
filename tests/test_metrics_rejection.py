"""Testes de métricas (CMC, confusão, margem) e rejeição open-set."""

import numpy as np
import pytest

from recdev.classifier import UNKNOWN_LABEL, EmbeddingClassifier, calibrate_threshold
from recdev.evaluation import (
    evaluate_easy,
    evaluate_very_easy,
    set_active_method,
)
from recdev.manifest import build_manifest

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ConstantFeaturizer:
    """Featurizer sintético: vetor determinístico por identidade."""

    def __call__(self, rec) -> np.ndarray:
        rng = np.random.default_rng(int(rec.identity_original))
        return rng.normal(size=8).astype(np.float32)


def test_cmc_perfect_and_partial():
    from recdev.classifier import Prediction

    preds = [
        Prediction(label=0, similarity=0.9, runner_up=1, runner_up_similarity=0.5,
                   ranking=[0, 1, 2]),
        # Verdade (1) em 2º lugar: rank-1 erra, rank-5 acerta.
        Prediction(label=2, similarity=0.8, runner_up=1, runner_up_similarity=0.7,
                   ranking=[2, 1, 0]),
    ]
    from recdev.evaluation import FoldResult

    fold = FoldResult(
        fold="t", condition="all", predictions=preds,
        true_labels=[0, 1], true_identities=["a", "b"],
        identity_by_index={0: "a", 1: "b", 2: "c"}, elapsed_ms=0.0,
    )
    assert fold.cmc[1] == 0.5
    assert fold.cmc[5] == 1.0
    assert fold.accuracy == 0.5


def test_confusion_matrix_counts():
    from recdev.classifier import Prediction
    from recdev.evaluation import FoldResult

    preds = [
        Prediction(label=0, similarity=0.9, runner_up=1, runner_up_similarity=0.4,
                   ranking=[0, 1]),
        Prediction(label=1, similarity=0.8, runner_up=0, runner_up_similarity=0.4,
                   ranking=[1, 0]),
        Prediction(label=1, similarity=0.7, runner_up=0, runner_up_similarity=0.4,
                   ranking=[1, 0]),
    ]
    fold = FoldResult(
        fold="t", condition="all", predictions=preds,
        true_labels=[0, 1, 1], true_identities=["a", "b", "b"],
        identity_by_index={0: "a", 1: "b"}, elapsed_ms=0.0,
    )
    m = fold.confusion()
    assert m["a"] == {"a": 1}
    assert m["b"] == {"b": 2}


def test_unknown_rejection_below_threshold():
    feats = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    labels = np.array([0, 1])
    # Consulta equidistante das duas identidades: cosseno ~0.71.
    clf = EmbeddingClassifier(mode="nn", threshold=0.9).fit(feats, labels)
    (pred,) = clf.predict(np.array([[0.9, 0.8, 0.1]], dtype=np.float32))
    assert pred.label == UNKNOWN_LABEL
    # Ranking preservado para CMC/análise.
    assert pred.ranking[0] == 0


def test_unknown_passes_above_threshold():
    feats = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    labels = np.array([0, 1])
    clf = EmbeddingClassifier(mode="nn", threshold=0.5).fit(feats, labels)
    (pred,) = clf.predict(np.array([[1.0, 0.0, 0]], dtype=np.float32))
    assert pred.label == 0


def test_invalid_threshold_raises():
    with pytest.raises(ValueError):
        EmbeddingClassifier(mode="nn", threshold=0.0)
    with pytest.raises(ValueError):
        EmbeddingClassifier(mode="nn", threshold=-0.5)


def test_calibrate_threshold_quantile():
    sims = [0.90, 0.95, 0.97, 0.99, 1.0]
    t = calibrate_threshold(sims, quantile=0.05)
    assert 0.89 <= t <= 0.91
    with pytest.raises(ValueError):
        calibrate_threshold([])
    with pytest.raises(ValueError):
        calibrate_threshold(sims, quantile=1.5)


def test_evaluate_with_threshold_rejects():
    manifest = build_manifest("very-easy", ROOT)
    set_active_method("embedding")
    # Limiar impossível (1.01) rejeita todas as consultas.
    folds = evaluate_very_easy(manifest, ConstantFeaturizer(), threshold=1.01)
    for fold in folds:
        assert all(p.label == UNKNOWN_LABEL for p in fold.predictions)
        assert fold.accuracy == 0.0
        # ranking continua intacto para CMC
        assert all(p.ranking for p in fold.predictions)


def test_report_summary_includes_new_metrics():
    manifest = build_manifest("easy", ROOT)
    set_active_method("embedding")
    folds = evaluate_easy(manifest, ConstantFeaturizer())
    from recdev.evaluation import EvalReport

    report = EvalReport(dataset="easy", method="embedding", folds=folds)
    s = report.summary()
    assert "cmc_mean" in s
    assert s["cmc_mean"]["1"] == pytest.approx(s["accuracy_mean"])
    fold0 = s["folds"][0]
    assert "confusion" in fold0
    assert "margin_mean" in fold0
    assert "cmc" in fold0