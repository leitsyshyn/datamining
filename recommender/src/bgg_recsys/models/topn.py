"""Models that score every catalogue game for a user, to build top-N lists.

A *scorer* maps an inner user index to a score for every inner item; higher is better. The list
models (`CoRatingKNN`, `EASE`) learn from the frame's sparse ratings matrix. `from_surprise` turns a
fitted Surprise rating model into a scorer by applying the model's own prediction formula to all items.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import scipy.sparse as sp
from surprise import SVD, AlgoBase, BaselineOnly, SVDpp

from ..data import at_least


class Scorer(Protocol):
    def __call__(self, u: int) -> np.ndarray: ...


# --------------------------------------------------------------------------- non-personalised
class Popularity:
    """Recommend the most-rated games."""

    def __init__(self, ratings: sp.csr_matrix):
        self.scores = _n_raters(ratings)
        self.n_items = len(self.scores)

    def __call__(self, u: int) -> np.ndarray:
        return self.scores


class RandomScores:
    """Recommend random games (seeded)."""

    def __init__(self, n_items: int, seed: int):
        self.n_items = n_items
        self.rng = np.random.default_rng(seed)

    def __call__(self, u: int) -> np.ndarray:
        return self.rng.random(size=self.n_items)


# --------------------------------------------------------------------------- item-to-item list models
def _n_raters(ratings: sp.csr_matrix) -> np.ndarray:
    return np.asarray(at_least(ratings, 1.0).sum(axis=0)).ravel()


class ItemToItem:
    """Score = (the user's row of inputs) @ (item x item weights).

    Ties (e.g. a user who has liked nothing yet) are broken by popularity, with a weight far too small
    to change any other ordering.
    """

    inputs: np.ndarray       # users x items: what each user's score is built from
    weights: np.ndarray      # items x items: how much game j speaks for game i
    liked: np.ndarray        # users x items, bool
    tie_break: np.ndarray    # items

    def _finish(self, ratings: sp.csr_matrix, inputs: np.ndarray, weights: np.ndarray, liked: np.ndarray):
        n_raters = _n_raters(ratings)
        self.inputs, self.weights, self.liked = inputs, weights, liked
        self.tie_break = 1e-6 * n_raters / n_raters.max()
        return self

    def __call__(self, u: int) -> np.ndarray:
        return self.inputs[u] @ self.weights + self.tie_break

    def score_items(self, liked_items) -> np.ndarray:
        """Scores for a new user who has only told us which games they like."""
        return self.weights[list(liked_items)].sum(axis=0) + self.tie_break

    def because_of(self, u: int, i: int) -> int | None:
        """The liked game contributing most to `i`'s score: the main reason `i` is recommended to `u`."""
        liked = np.flatnonzero(self.liked[u])
        return int(liked[np.argmax(self.weights[liked, i])]) if len(liked) else None


class CoRatingKNN(ItemToItem):
    """Item-based top-N collaborative filtering on co-ratings (Deshpande & Karypis, 2004).

    Two games are similar when the same users rated both: cosine similarity of their binary "rated"
    vectors. A user's score for a game is the sum of its similarities to the games they liked
    (rating >= `liked_threshold`), plus `unliked_weight` times its similarities to the games they rated
    lower. It learns which games people play together, not how they rate them.
    """

    def __init__(self, liked_threshold: float = 8.0, unliked_weight: float = 0.0):
        self.liked_threshold = liked_threshold
        self.unliked_weight = unliked_weight

    def fit(self, ratings: sp.csr_matrix) -> CoRatingKNN:
        rated = at_least(ratings, 1.0).astype(np.float32)
        self.co_raters = (rated.T @ rated).toarray()
        norm = np.sqrt(np.diag(self.co_raters))
        sim = self.co_raters / np.outer(norm, norm)
        np.fill_diagonal(sim, 0.0)
        liked = at_least(ratings, self.liked_threshold).toarray() > 0
        inputs = np.where(liked, 1.0, self.unliked_weight).astype(np.float32) * rated.toarray()
        return self._finish(ratings, inputs, sim, liked)


class EASE(ItemToItem):
    """EASE, "Embarrassingly Shallow Autoencoders" (Steck, 2019): item-based top-N collaborative filtering
    with *learned* item-item weights.

    The weight matrix B is the ridge-regression solution that best reconstructs every user's row from
    their other games, min ||X - XB||^2 + reg * ||B||^2 with diag(B) = 0, which has a closed form.
    X holds the games each user liked (rating >= `liked_threshold`), or every rated game when it is None.
    """

    def __init__(self, reg: float = 500.0, liked_threshold: float | None = 8.0):
        self.reg = reg
        self.liked_threshold = liked_threshold

    def fit(self, ratings: sp.csr_matrix) -> EASE:
        threshold = 1.0 if self.liked_threshold is None else self.liked_threshold   # ratings are >= 1
        x = at_least(ratings, threshold)
        gram = (x.T @ x).toarray()
        gram[np.diag_indices_from(gram)] += self.reg
        p = np.linalg.inv(gram)
        b = -p / np.diag(p)
        np.fill_diagonal(b, 0.0)
        inputs = x.toarray()
        return self._finish(ratings, inputs, b, liked=inputs > 0)


# --------------------------------------------------------------------------- rating models as scorers
def from_surprise(algo: AlgoBase) -> Scorer:
    """Vectorised scores of a fitted BaselineOnly, SVD or SVD++ for all items, with each model's own
    prediction formula (unclipped). `max_scorer_error` confirms it reproduces Surprise's `estimate`.

    Surprise is a compiled extension without type information, so its fitted attributes are read through `Any`.
    """
    m: Any = algo
    mu = m.trainset.global_mean
    if isinstance(algo, SVDpp):
        def score(u: int) -> np.ndarray:
            items = [j for j, _ in m.trainset.ur[u]]
            implicit = m.yj[items].sum(axis=0) / np.sqrt(len(items))
            return mu + m.bu[u] + m.bi + m.qi @ (m.pu[u] + implicit)
        return score
    if isinstance(algo, SVD):
        if m.biased:
            return lambda u: mu + m.bu[u] + m.bi + m.qi @ m.pu[u]
        return lambda u: m.qi @ m.pu[u]
    if isinstance(algo, BaselineOnly):
        return lambda u: mu + m.bu[u] + m.bi
    raise TypeError(f"no vectorised scorer for {type(algo).__name__}")


def max_scorer_error(algo: AlgoBase, scorer: Scorer, n_pairs: int = 300, seed: int = 0) -> float:
    """Largest absolute difference between `scorer` and Surprise's own `estimate` on random (user, item) pairs."""
    m: Any = algo
    rng = np.random.default_rng(seed)
    users = rng.integers(0, m.trainset.n_users, n_pairs)
    items = rng.integers(0, m.trainset.n_items, n_pairs)
    errors = []
    for u, i in zip(users, items):
        est = m.estimate(int(u), int(i))
        errors.append(abs(scorer(int(u))[i] - (est[0] if isinstance(est, tuple) else est)))
    return float(max(errors))


# --------------------------------------------------------------------------- re-ranking
def _z(x: np.ndarray) -> np.ndarray:
    std = x.std()
    return (x - x.mean()) / std if std > 0 else np.zeros_like(x, dtype=float)


def popularity_penalised(scorer: Scorer, popularity: np.ndarray, beta: float) -> Scorer:
    """Re-rank for novelty: z(score) - beta * z(log popularity)."""
    log_pop = _z(np.log1p(popularity))
    return lambda u: _z(scorer(u)) - beta * log_pop
