"""Metric definitions: pure functions of predictions, lists and relevant items. No models, no data loading."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy import stats
from surprise import Prediction, accuracy

Lists = Mapping[int, np.ndarray]          # inner user -> recommended inner items, best first
Relevant = Mapping[int, frozenset]        # inner user -> relevant held-out inner items


# --------------------------------------------------------------------------- rating accuracy
def rating_metrics(predictions: list[Prediction]) -> dict[str, float]:
    return {"RMSE": accuracy.rmse(predictions, verbose=False), "MAE": accuracy.mae(predictions, verbose=False)}


def error_profile(predictions: list[Prediction]) -> pd.DataFrame:
    """Mean prediction and RMSE per true rating (rounded): how far predictions are pulled to the mean."""
    df = pd.DataFrame({"true": [p.r_ui for p in predictions], "pred": [p.est for p in predictions]})
    df["true rating"] = df.true.round().astype(int)
    df["sq"] = (df.pred - df.true) ** 2
    out = df.groupby("true rating").agg(n=("true", "size"), mean_prediction=("pred", "mean"), mse=("sq", "mean"))
    out["RMSE"] = np.sqrt(out.pop("mse"))
    return out


# --------------------------------------------------------------------------- top-N lists, per user
def list_metrics(lists: Lists, relevant: Relevant, k: int) -> pd.DataFrame:
    """Per-user precision, recall, NDCG and hit rate at k (one row per user)."""
    discount = 1 / np.log2(np.arange(2, k + 2))
    rows = {}
    for u, items in lists.items():
        rel = relevant[u]
        gains = np.array([float(i in rel) for i in items[:k]])
        ideal = discount[:min(len(rel), k)].sum()
        rows[u] = (gains.mean(), gains.sum() / len(rel), (gains * discount).sum() / ideal, float(gains.any()))
    return pd.DataFrame.from_dict(rows, orient="index", columns=[f"P@{k}", f"R@{k}", f"NDCG@{k}", f"HR@{k}"])


# --------------------------------------------------------------------------- uncertainty
def mean_ci(x, resamples: int = 2_000, seed: int = 0) -> tuple[float, float, float]:
    """Mean with a 95% percentile-bootstrap interval, resampling users."""
    x = np.asarray(x, dtype=float)
    if np.ptp(x) == 0:
        return float(x.mean()), float(x.mean()), float(x.mean())
    ci = stats.bootstrap((x,), np.mean, n_resamples=resamples, method="percentile", batch=200, rng=np.random.default_rng(seed)).confidence_interval
    return float(x.mean()), float(ci.low), float(ci.high)


def paired_ci(a, b, resamples: int = 2_000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of a - b over the same users, with a 95% paired percentile-bootstrap interval."""
    return mean_ci(np.asarray(a, dtype=float) - np.asarray(b, dtype=float), resamples, seed)


# --------------------------------------------------------------------------- beyond accuracy
def coverage(lists: Lists, n_items: int) -> float:
    """Share of the catalogue that appears in at least one list."""
    return len(np.unique(np.concatenate(list(lists.values())))) / n_items


def novelty(lists: Lists, popularity_share: np.ndarray) -> float:
    """Mean self-information -log2(p_i) of recommended items; p_i = share of users who rated item i."""
    return self_information(np.concatenate(list(lists.values())), popularity_share)


def self_information(items: np.ndarray, popularity_share: np.ndarray) -> float:
    return float(-np.log2(popularity_share[items]).mean())


def intra_list_diversity(lists: Lists, features: np.ndarray) -> float:
    """1 - mean Jaccard similarity of item feature sets within each list, averaged over lists."""
    values = []
    for items in lists.values():
        f = features[items].astype(np.float32)
        inter = f @ f.T
        size = f.sum(axis=1)
        union = size[:, None] + size[None, :] - inter
        jac = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        values.append(1 - jac[np.triu_indices(len(items), 1)].mean())
    return float(np.mean(values))


def personalisation(lists: Lists, n_pairs: int = 20_000, seed: int = 0) -> float:
    """1 - mean Jaccard overlap between the lists of random pairs of users (0 = everyone gets the same list)."""
    sets = [set(v.tolist()) for v in lists.values()]
    rng = np.random.default_rng(seed)
    a, b = rng.integers(0, len(sets), n_pairs), rng.integers(0, len(sets), n_pairs)
    keep = a != b
    overlap = [len(sets[i] & sets[j]) / len(sets[i] | sets[j]) for i, j in zip(a[keep], b[keep])]
    return 1 - float(np.mean(overlap))
