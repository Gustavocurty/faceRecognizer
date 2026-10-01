"""Dois grupos separáveis continuam corretos depois da projeção."""

import numpy as np

from recdev.identify import predict_labels


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
