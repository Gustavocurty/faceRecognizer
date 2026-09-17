from pathlib import Path

import numpy as np

from recdev.classifier import Prediction
from recdev.evaluation import EvalReport, FoldResult
from recdev.experiments import run_experiment
from recdev.manifest import build_manifest

ROOT = Path(__file__).resolve().parents[1]


class InstrumentedStore:
    def __init__(self):
        self.seen = []
        self.decompositions = {}

    def get(self, records):
        self.seen.extend(r.source_photo for r in records)
        return np.stack([np.random.default_rng(int(r.identity_original)).normal(size=10) for r in records])

    def summary(self):
        return {}


def test_medium_selection_only_uses_validation_and_prefers_smallest_k():
    store = InstrumentedStore()
    result = run_experiment(build_manifest("medium", ROOT), {"pixels": store}, [1, 2], methods=["pca"])
    assert result["selections"][0]["config"]["n_components"] == 1
    assert all(r["partition"] == "validation" for r in result["candidates"])
    assert len(result["final_test"]) == 1
    first_test = next(i for i, source in enumerate(store.seen) if source in ("11", "12", "13", "14"))
    assert "05" in store.seen[:first_test]
    assert "05" not in store.seen[first_test:]
    assert result["final_test"][0]["n_queries"] == 200


def test_small_datasets_are_descriptive_and_skip_impossible_k():
    result = run_experiment(build_manifest("very-easy", ROOT), {"pixels": InstrumentedStore()},
                            [1, 4, 5], methods=["pca"])
    assert result["mode"] == "descriptive-fixed-grid"
    assert not result["selections"] and not result["final_test"]
    assert len(result["skipped"]) == 1
    assert len(result["candidates"]) == 2


def test_per_condition_aggregates_both_folds_and_weights_query_counts():
    good = Prediction(0, 1, 1, 0, [0, 1])
    bad = Prediction(1, 1, 0, 0, [1, 0])
    common = dict(condition="clean", identity_by_index={0: "a", 1: "b"}, elapsed_ms=0)
    a = FoldResult("a", predictions=[good], true_labels=[0], true_identities=["a"], **common)
    b = FoldResult("b", predictions=[bad] * 3, true_labels=[0] * 3, true_identities=["a"] * 3, **common)
    report = EvalReport("synthetic", "pixels", [a, b]).summary()
    assert report["per_condition"]["clean"] == dict(hits=1, n_queries=4, accuracy=0.25)
    assert report["accuracy_mean"] == 0.25
