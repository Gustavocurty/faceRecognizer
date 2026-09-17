"""Avaliação fechada com splits explícitos, métricas e diagnóstico numérico."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .classifier import UNKNOWN_LABEL, Prediction, calibrate_threshold
from .config import MethodConfig
from .manifest import Manifest, Record
from .pipeline import RecognitionModel
from .protocols import PROTOCOL_VERSION, Split, make_splits
from .reduction import SVDPCA

Featurizer = Callable[[Record], np.ndarray]


@dataclass
class FoldResult:
    fold: str
    condition: str
    predictions: list[Prediction]
    true_labels: list[int]
    true_identities: list[str]
    identity_by_index: dict[int, str]
    elapsed_ms: float
    threshold: float | None = None
    paths: list[str] = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)

    @property
    def hits(self):
        return sum(p.label == truth for p, truth in zip(self.predictions, self.true_labels))

    @property
    def accuracy(self):
        return self.hits / len(self.predictions) if self.predictions else 0.0

    @property
    def margin(self):
        return [p.similarity - p.runner_up_similarity for p in self.predictions]

    @property
    def cmc(self):
        # Ranking é pré-rejeição: rank-1 pode diferir da acurácia com limiar.
        return {k: sum(t in p.ranking[:k] for p, t in zip(self.predictions, self.true_labels)) / (len(self.predictions) or 1)
                for k in (1, 5)}

    def confusion(self):
        matrix = {}
        for prediction, truth in zip(self.predictions, self.true_identities):
            name = self.identity_by_index.get(prediction.label, "unknown")
            matrix.setdefault(truth, {}).setdefault(name, 0)
            matrix[truth][name] += 1
        return matrix

    def summary(self):
        return dict(
            fold=self.fold, condition=self.condition, n_queries=len(self.predictions), hits=self.hits,
            accuracy=self.accuracy, cmc={str(k): value for k, value in self.cmc.items()},
            margin_mean=float(np.mean(self.margin)) if self.margin else 0.0,
            threshold=self.threshold, rejected=sum(p.label == UNKNOWN_LABEL for p in self.predictions),
            elapsed_ms=self.elapsed_ms, timings=self.timings, diagnostics=self.diagnostics,
            confusion=self.confusion(),
            queries=[dict(path=self.paths[i] if self.paths else None, truth=truth,
                          predicted=self.identity_by_index.get(p.label, "unknown"),
                          score=p.similarity, margin=p.similarity - p.runner_up_similarity,
                          ranking=[self.identity_by_index[label] for label in p.ranking])
                     for i, (p, truth) in enumerate(zip(self.predictions, self.true_identities))],
        )


@dataclass
class EvalReport:
    dataset: str
    method: str
    folds: list[FoldResult] = field(default_factory=list)
    config: dict = field(default_factory=dict)
    partition: str = "test"
    fit_metrics: list[dict] = field(default_factory=list)

    @property
    def n_queries(self):
        return sum(len(f.predictions) for f in self.folds)

    @property
    def accuracy(self):
        return sum(f.hits for f in self.folds) / (self.n_queries or 1)

    @property
    def cmc(self):
        return {k: sum(f.cmc[k] * len(f.predictions) for f in self.folds) / (self.n_queries or 1) for k in (1, 5)}

    def summary(self):
        conditions = {}
        confusion = {}
        for fold in self.folds:
            row = conditions.setdefault(fold.condition, dict(hits=0, n_queries=0))
            row["hits"] += fold.hits
            row["n_queries"] += len(fold.predictions)
            for truth, counts in fold.confusion().items():
                for prediction, count in counts.items():
                    confusion.setdefault(truth, {}).setdefault(prediction, 0)
                    confusion[truth][prediction] += count
        for row in conditions.values():
            row["accuracy"] = row["hits"] / row["n_queries"]
        return dict(
            dataset=self.dataset, method=self.method, config=self.config, partition=self.partition,
            protocol=PROTOCOL_VERSION, accuracy_mean=self.accuracy,
            hits=sum(f.hits for f in self.folds), n_queries=self.n_queries,
            cmc_mean={str(k): v for k, v in self.cmc.items()},
            rank5_informative=any(len(f.identity_by_index) > 5 for f in self.folds),
            n_folds=len(self.folds), per_condition=conditions, confusion=confusion,
            fits=self.fit_metrics, folds=[f.summary() for f in self.folds],
        )


def get_features(featurizer, records):
    if hasattr(featurizer, "get"):
        return featurizer.get(records)
    return np.stack([featurizer(record) for record in records])


def fit_split(split: Split, featurizer, config: MethodConfig):
    start = time.perf_counter()
    features = get_features(featurizer, split.gallery)
    gallery_features_ms = (time.perf_counter() - start) * 1000
    model = RecognitionModel(config)
    decomposition = None
    fit_ms = 0.0
    reused = False
    if config.uses_pca and hasattr(featurizer, "decompositions"):
        key = (tuple((r.path, r.sha256) for r in split.gallery), config.dtype)
        decomposition = featurizer.decompositions.get(key)
        reused = decomposition is not None
        if decomposition is None:
            start = time.perf_counter()
            decomposition = SVDPCA(dtype=config.dtype).fit(model.prepare(features))
            fit_ms = (time.perf_counter() - start) * 1000
            featurizer.decompositions[key] = decomposition
    model.fit(features, [r.identity_index for r in split.gallery], decomposition)
    if decomposition is not None:
        model.timings["pca_fit_ms"] += fit_ms
        model.timings["svd_reused"] = reused
    model.timings["gallery_features_ms"] = gallery_features_ms
    return model


def score_records(model, split, records, featurizer, dataset):
    folds = []
    conditions = sorted({r.condition for r in records})
    for condition in conditions:
        queries = [r for r in records if r.condition == condition]
        start = time.perf_counter()
        features = get_features(featurizer, queries)
        features_ms = (time.perf_counter() - start) * 1000
        predictions, timings = model.predict_timed(features)
        timings["query_features_ms"] = features_ms
        identities = {r.identity_index: r.identity_original for r in split.gallery}
        folds.append(FoldResult(
            fold=split.name, condition="all" if dataset == "very-easy" else condition,
            predictions=predictions, true_labels=[r.identity_index for r in queries],
            true_identities=[r.identity_original for r in queries], identity_by_index=identities,
            elapsed_ms=sum(timings.values()), threshold=model.config.threshold,
            paths=[r.path for r in queries], timings=timings, diagnostics=model.diagnostics(features),
        ))
    return folds


def evaluate(manifest: Manifest, featurizer, config: MethodConfig,
             partition="test", splits: list[Split] | None = None) -> EvalReport:
    if partition not in ("test", "validation"):
        raise ValueError("Partição deve ser test ou validation")
    splits = make_splits(manifest) if splits is None else splits
    report = EvalReport(manifest.dataset, config.method, config=config.to_dict(), partition=partition)
    for split in splits:
        split.validate()
        queries = getattr(split, partition)
        if not queries:
            raise ValueError(f"{manifest.dataset} não possui partição {partition} independente")
        model = fit_split(split, featurizer, config)
        report.fit_metrics.append(dict(split=split.name, timings=model.timings, diagnostics=model.diagnostics()))
        report.folds.extend(score_records(model, split, queries, featurizer, manifest.dataset))
    return report


def evaluate_very_easy(manifest, featurizer, threshold=None, config=None):
    return evaluate(manifest, featurizer, config or MethodConfig(threshold=threshold)).folds


def evaluate_easy(manifest, featurizer, threshold=None, config=None):
    return evaluate(manifest, featurizer, config or MethodConfig(threshold=threshold)).folds


def calibrate_threshold_from_folds(folds: list[FoldResult], quantile=0.05):
    """O chamador deve fornecer apenas folds de VALIDAÇÃO independente."""
    return calibrate_threshold([p.similarity for fold in folds
                                for p, truth in zip(fold.predictions, fold.true_labels)
                                if p.label == truth], quantile)
