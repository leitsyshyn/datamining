import numpy as np
import pytest

from bgg_recsys.metrics import (
    coverage,
    intra_list_diversity,
    list_metrics,
    mean_ci,
    novelty,
    paired_ci,
    personalisation,
)


def test_list_metrics_by_hand():
    m = list_metrics({0: np.array([1, 2, 3])}, {0: frozenset({2, 5})}, k=3).loc[0].to_dict()
    assert m["P@3"] == pytest.approx(1 / 3)
    assert m["R@3"] == pytest.approx(1 / 2)
    assert m["HR@3"] == 1
    # one hit at rank 2; the ideal list has two hits, at ranks 1 and 2
    assert m["NDCG@3"] == pytest.approx((1 / np.log2(3)) / (1 + 1 / np.log2(3)))


def test_confidence_intervals():
    assert mean_ci(np.ones(100)) == (1, 1, 1)
    assert paired_ci(np.arange(50) + 2.0, np.arange(50, dtype=float)) == (2, 2, 2)
    mean, low, high = mean_ci(np.random.default_rng(0).normal(size=2000))
    assert low < mean < high and low < 0 < high
    assert high - low == pytest.approx(2 * 1.96 / np.sqrt(2000), rel=0.15)   # close to the normal-theory width


def test_beyond_accuracy_metrics():
    lists = {0: np.array([0, 1]), 1: np.array([0, 2])}
    assert coverage(lists, 4) == 0.75
    share = np.array([0.5, 0.25, 0.25, 0.1])
    assert novelty(lists, share) == pytest.approx((1 + 2 + 1 + 2) / 4)
    features = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]])
    assert intra_list_diversity({0: np.array([0, 1])}, features) == 0
    assert intra_list_diversity({0: np.array([0, 2])}, features) == 1
    assert personalisation({0: np.array([0, 1]), 1: np.array([0, 1])}, n_pairs=50) == 0
    assert personalisation({0: np.array([0, 1]), 1: np.array([2, 3])}, n_pairs=50) == 1
