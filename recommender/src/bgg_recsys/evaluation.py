"""Evaluation protocols: what is predicted, for whom, and against which held-out ratings.

Metric definitions live in `metrics`; this module decides how they are applied: both test sets, the
full-catalogue ranking task, same-user comparisons, sub-groups, the one-game scenario and robustness.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from surprise import AlgoBase, Prediction
from surprise.model_selection import KFold, cross_validate

from . import metrics
from ._cache import memory
from .config import Config, FrameSpec
from .data import Frame, load_frame, to_dataset
from .models import (
    ItemToItem,
    Name,
    RatingModel,
    Scorer,
    Tuned,
    average_predictions,
    best_item_knn,
    list_scorers,
    popularity_penalised,
    rating_models,
)
from .models.catalogue import Family

SPLITS = ("random", "future")


# --------------------------------------------------------------------------- rating accuracy
@dataclass
class RatingResults:
    """Every rating model fitted once on `frame.train` and scored on both test sets."""

    table: pd.DataFrame                                         # one row per model
    predictions: dict[Name, dict[str, list[Prediction]]]
    fitted: dict[Name, AlgoBase]

    def gain_over(self, reference: Name) -> pd.Series:
        rmse = self.table["RMSE random"].astype(float)
        return 1 - rmse / rmse.to_dict()[reference]


def _rating_row(family: Family, preds: dict[str, list[Prediction]], future_users: set, fit_seconds: float) -> dict:
    random_m, future_m = metrics.rating_metrics(preds["random"]), metrics.rating_metrics(preds["future"])
    same_users = [p for p in preds["random"] if p.uid in future_users]
    return {"family": str(family), "RMSE random": random_m["RMSE"], "MAE random": random_m["MAE"],
            "RMSE random (future users)": metrics.rating_metrics(same_users)["RMSE"],
            "RMSE future": future_m["RMSE"], "MAE future": future_m["MAE"], "fit (s)": fit_seconds}


def evaluate_rating_models(frame: Frame, models: Sequence[RatingModel], seed: int, verbose: bool = True) -> RatingResults:
    """Fit each model on the training ratings and predict the random and the future test set."""
    np.random.seed(seed)   # NormalPredictor samples from numpy's global generator
    testsets, future_users = frame.testsets, set(frame.future.user)
    rows, predictions, fitted = {}, {}, {}
    for model in models:
        algo = model.make()
        start = time.perf_counter()
        algo.fit(frame.trainset)
        fit_seconds = time.perf_counter() - start
        predictions[model.name] = {split: algo.test(testset) for split, testset in testsets.items()}
        rows[model.name] = _rating_row(model.family, predictions[model.name], future_users, fit_seconds)
        if model.keep_fitted:
            fitted[model.name] = algo
        if verbose:
            print(f"{model.name:40s} RMSE random {rows[model.name]['RMSE random']:.4f}   future {rows[model.name]['RMSE future']:.4f}   fit {fit_seconds:4.0f}s")
    return RatingResults(pd.DataFrame(rows).T, predictions, fitted)


def with_average(results: RatingResults, frame: Frame, name: Name, of: tuple[Name, Name]) -> RatingResults:
    """Add the average of two models' predictions as a model of its own (no extra fitting)."""
    preds = {split: average_predictions(*(results.predictions[m][split] for m in of)) for split in SPLITS}
    row = _rating_row(Family.ENSEMBLE, preds, set(frame.future.user), np.nan)
    table = pd.concat([results.table, pd.DataFrame({name: row}).T])
    return RatingResults(table, {**results.predictions, name: preds}, results.fitted)


@dataclass
class CrossValidation:
    folds: pd.DataFrame          # RMSE per fold (rows) and model (columns)
    table: pd.DataFrame          # mean and std of RMSE and MAE per model

    def paired_test(self, a: Name, b: Name) -> dict[str, float]:
        """Paired t-test over identical folds."""
        x, y = self.folds[a].to_numpy(), self.folds[b].to_numpy()
        test: Any = stats.ttest_rel(x, y)
        return {"mean RMSE difference": float(np.mean(x - y)), "t": float(test.statistic), "p": float(test.pvalue),
                f"folds where {a} is better": int(np.sum(x < y))}


def cross_validate_models(frame: Frame, models: Sequence[RatingModel], n_folds: int, seed: int) -> CrossValidation:
    """k-fold cross-validation on all sampled 2022 ratings, with identical folds for every model."""
    data, folds = to_dataset(frame.past), KFold(n_splits=n_folds, random_state=seed)
    results = {m.name: cross_validate(m.make(), data, measures=["rmse", "mae"], cv=folds, n_jobs=-1) for m in models}
    table = pd.DataFrame({name: {"RMSE mean": r["test_rmse"].mean(), "RMSE std": r["test_rmse"].std(),
                                 "MAE mean": r["test_mae"].mean(), "MAE std": r["test_mae"].std()} for name, r in results.items()}).T
    return CrossValidation(pd.DataFrame({name: r["test_rmse"] for name, r in results.items()}), table)


def precision_recall_within_test(predictions: list[Prediction], k: int, threshold: float) -> tuple[float, float]:
    """The Surprise FAQ protocol: rank only each user's own test items. Optimistic, kept for comparison."""
    by_user = defaultdict(list)
    for p in predictions:
        by_user[p.uid].append((p.est, p.r_ui))
    precision, recall = [], []
    for items in by_user.values():
        items.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum(t >= threshold for _, t in items)
        top = items[:k]
        n_rec = sum(e >= threshold for e, _ in top)
        n_hit = sum(e >= threshold and t >= threshold for e, t in top)
        precision.append(n_hit / n_rec if n_rec else 0.0)
        recall.append(n_hit / n_rel if n_rel else 0.0)
    return float(np.mean(precision)), float(np.mean(recall))


# --------------------------------------------------------------------------- ranking over the full catalogue
@dataclass(frozen=True)
class RankingTask:
    """Users to evaluate, the items excluded from their candidates, and their relevant held-out items (inner ids)."""

    users: tuple[int, ...]
    seen: Mapping[int, frozenset]
    relevant: Mapping[int, frozenset]

    def restrict(self, users: Iterable[int]) -> RankingTask:
        keep = set(users)
        users_ = tuple(u for u in self.users if u in keep)
        return RankingTask(users_, {u: self.seen[u] for u in users_}, {u: self.relevant[u] for u in users_})

    def __len__(self):
        return len(self.users)


def ranking_task(frame: Frame, held_out: pd.DataFrame, threshold: float, exclude: tuple[pd.DataFrame, ...] = ()) -> RankingTask:
    """Every user with at least one relevant (rating >= threshold) held-out game known to the frame.

    Candidates are all catalogue games except those the user rated in training and in `exclude`.
    """
    known = held_out.user.isin(frame.user_index.index) & held_out.ID.isin(frame.item_index.index)
    rel = held_out.loc[known & (held_out.rating >= threshold)]
    relevant = {frame.inner_user(u): frozenset(frame.inner_items(ids)) for u, ids in rel.groupby("user").ID}
    ratings = frame.ratings
    seen = {u: set(ratings.indices[ratings.indptr[u]:ratings.indptr[u + 1]].tolist()) for u in relevant}
    for extra in exclude:
        extra = extra.loc[extra.user.isin(rel.user.unique()) & extra.ID.isin(frame.item_index.index)]
        for user, ids in extra.groupby("user").ID:
            seen[frame.inner_user(user)] |= set(frame.inner_items(ids).tolist())
    users = tuple(sorted(relevant))
    return RankingTask(users, {u: frozenset(seen[u]) for u in users}, relevant)


def ranking_tasks(frame: Frame, threshold: float) -> dict[str, RankingTask]:
    """The random test split, and the future (excluding games the user rated in 2022)."""
    return {"random": ranking_task(frame, frame.test, threshold),
            "future": ranking_task(frame, frame.future, threshold, exclude=(frame.test,))}


def top_k(scorer: Scorer, task: RankingTask, k: int) -> dict[int, np.ndarray]:
    """The k highest-scoring unseen items per user, best first."""
    lists = {}
    for u in task.users:
        scores = np.asarray(scorer(u), dtype=float).copy()
        scores[list(task.seen[u])] = -np.inf
        top = np.argpartition(-scores, k)[:k]
        lists[u] = top[np.argsort(-scores[top])]
    return lists


@dataclass
class RankingResults:
    """Top-k lists and per-user metrics for every model on every task."""

    tasks: dict[str, RankingTask]
    lists: dict[tuple[str, Name], dict[int, np.ndarray]]
    per_user: dict[tuple[str, Name], pd.DataFrame]
    k: int
    resamples: int = 2_000
    seed: int = 0
    models: list[Name] = field(default_factory=list)

    @property
    def same_users(self) -> list[int]:
        """Users evaluated in both the random and the future split."""
        return sorted(set(self.tasks["random"].users) & set(self.tasks["future"].users))

    def _metrics(self, split: str, model: Name, users=None) -> pd.DataFrame:
        m = self.per_user[split, model]
        return m if users is None else m.loc[users]

    def summary(self, split: str, users=None, models: Sequence[Name] | None = None) -> pd.DataFrame:
        """Mean of every metric per model, with a 95% interval for precision."""
        rows = {}
        for model in models or self.models:
            m = self._metrics(split, model, users)
            _, low, high = metrics.mean_ci(m[f"P@{self.k}"], self.resamples, self.seed)
            rows[model] = {**m.mean().to_dict(), f"P@{self.k} low": low, f"P@{self.k} high": high}
        return pd.DataFrame(rows).T

    def compare(self, split: str, pairs: Iterable[tuple[Name, Name]], users=None) -> pd.DataFrame:
        """Paired differences in precision between models, over the same users."""
        rows = []
        for a, b in pairs:
            diff, low, high = metrics.paired_ci(self._metrics(split, a, users)[f"P@{self.k}"],
                                                self._metrics(split, b, users)[f"P@{self.k}"], self.resamples, self.seed)
            rows.append({"comparison": f"{a} − {b}", f"Δ P@{self.k}": diff, "95% CI low": low, "95% CI high": high,
                         "significant": "yes" if low > 0 or high < 0 else "no"})
        return pd.DataFrame(rows).set_index("comparison")

    def tidy(self, splits: Sequence[str], models: Sequence[Name], users=None) -> pd.DataFrame:
        """Long format (split, model, user, P@k) for charts."""
        parts = [self._metrics(s, m, users)[[f"P@{self.k}"]].assign(split=s, model=str(m)) for s in splits for m in models]
        return pd.concat(parts).rename_axis("user").reset_index()


def evaluate_rankings(scorers: Mapping[Name, Scorer], tasks: Mapping[str, RankingTask], k: int,
                      resamples: int = 2_000, seed: int = 0) -> RankingResults:
    lists, per_user = {}, {}
    for split, task in tasks.items():
        for name, scorer in scorers.items():
            lists[split, name] = top_k(scorer, task, k)
            per_user[split, name] = metrics.list_metrics(lists[split, name], task.relevant, k)
    return RankingResults(dict(tasks), lists, per_user, k, resamples, seed, list(scorers))


def mean_precision(scorers: Mapping[Name, Scorer], task: RankingTask, k: int) -> dict[Name, float]:
    return {name: float(metrics.list_metrics(top_k(s, task, k), task.relevant, k)[f"P@{k}"].to_numpy().mean()) for name, s in scorers.items()}


# --------------------------------------------------------------------------- sub-groups and other definitions of "liked"
def long_tail_task(task: RankingTask, popularity: np.ndarray, head_share: float) -> tuple[RankingTask, int]:
    """Remove the most popular games, together holding `head_share` of all ratings, from candidates and relevant games."""
    order = np.argsort(-popularity)
    head = frozenset(order[:np.searchsorted(np.cumsum(popularity[order]) / popularity.sum(), head_share) + 1].tolist())
    relevant = {u: task.relevant[u] - head for u in task.users if task.relevant[u] - head}
    return RankingTask(tuple(relevant), {u: task.seen[u] | head for u in relevant}, relevant), len(head)


def precision_by_history(results: RankingResults, frame: Frame, buckets: Sequence[tuple[int, int]], models: Sequence[Name]) -> pd.DataFrame:
    users = np.array(results.tasks["random"].users)
    history = frame.history_length[users]
    rows = {}
    for low, high in buckets:
        group = users[(history >= low) & (history < high)].tolist()
        label = f"history {low}–{high - 1} ratings ({len(group):,} users)" if high < 10**6 else f"history ≥ {low} ratings ({len(group):,} users)"
        rows[label] = {m: results.per_user["random", m].loc[group, f"P@{results.k}"].mean() for m in models}
    return pd.DataFrame(rows).T


def other_thresholds(frame: Frame, fitted: Mapping[Name, AlgoBase], tuned: Tuned, thresholds: Sequence[float],
                     models: Sequence[Name], k: int, seed: int) -> pd.DataFrame:
    """Precision when "liked" means a different rating threshold; list models are rebuilt with that threshold."""
    rows = {}
    for threshold in thresholds:
        ease = {**tuned.ease, "liked_threshold": threshold if tuned.ease["liked_threshold"] is not None else None}
        scorers = list_scorers(frame.ratings, fitted, replace(tuned, ease=ease), threshold, seed)
        task = ranking_task(frame, frame.test, threshold)
        rows[f"liked = rating ≥ {threshold:g} ({len(task):,} users)"] = mean_precision({m: scorers[m] for m in models}, task, k)
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------- if all we know is one game you love
def one_game_lists(scorers: Mapping[Name, ItemToItem], game: int, candidates: np.ndarray, popularity: np.ndarray,
                   excluded: set[int], k: int) -> dict[str, np.ndarray]:
    """Lists built from a single loved game: the most popular candidates, and each list model's suggestions."""
    def top(scores) -> np.ndarray:
        s = np.asarray(scores, dtype=float).copy()
        s[list(excluded | {game})] = -np.inf
        return np.argsort(-s)[:k]
    popular = np.full(len(popularity), -np.inf)
    popular[candidates] = popularity[candidates]
    return {"most popular strategy games": top(popular), **{str(m): top(s.score_items([game])) for m, s in scorers.items()}}


def one_game_test(frame: Frame, scorers: Mapping[Name, ItemToItem], seeds: Mapping[int, int], candidates: np.ndarray,
                  threshold: float, k: int) -> pd.DataFrame:
    """For each user: recommend k games from one game they loved in 2022; count the games they loved later.

    Random picks from `candidates` are scored analytically (expected hits). Games the user rated in 2022 are
    excluded for every method.
    """
    loved_later = _inner_sets(frame, frame.future.loc[frame.future.rating >= threshold])
    played = _inner_sets(frame, frame.past)
    rows = []
    for user, game in seeds.items():
        target, excluded = loved_later.get(user, set()), played[user]
        pool = set(candidates.tolist()) - excluded - {game}
        row = {"random strategy games": k * len(target & pool) / len(pool)}
        lists = one_game_lists(scorers, game, candidates, frame.popularity, excluded, k)
        rows.append(row | {method: len(target & set(items.tolist())) for method, items in lists.items()})
    return pd.DataFrame(rows)


def _inner_sets(frame: Frame, ratings: pd.DataFrame) -> dict[int, set[int]]:
    return {frame.inner_user(u): set(frame.inner_items(ids).tolist()) for u, ids in ratings.groupby("user").ID}


def fans_of(frame: Frame, game_id: int, threshold: float) -> dict[int, int]:
    """Users who loved `game_id` in 2022 and loved at least one catalogue game later -> that game (inner ids)."""
    past, later = frame.past, frame.future.loc[frame.future.rating >= threshold]
    fans = set(past.loc[(past.ID == game_id) & (past.rating >= threshold), "user"]) & set(later.user)
    return {frame.inner_user(u): frame.inner_item(game_id) for u in sorted(fans)}


def favourites_among(frame: Frame, candidates: np.ndarray, threshold: float) -> dict[int, int]:
    """Users who loved later -> their highest-rated 2022 game among `candidates` (inner ids)."""
    past = frame.past.assign(inner=lambda d: frame.item_index.reindex(d.ID).to_numpy())
    later_users = set(frame.future.loc[frame.future.rating >= threshold, "user"])
    loved = past.loc[past.inner.isin(candidates) & (past.rating >= threshold) & past.user.isin(later_users)]
    return {frame.inner_user(u): int(g.sort_values("rating").inner.iloc[-1]) for u, g in loved.groupby("user")}


def within_test_table(results: RatingResults, models: Sequence[Name], k: int, threshold: float) -> pd.DataFrame:
    """Method A for several models (see precision_recall_within_test)."""
    return pd.DataFrame({m: dict(zip([f"P@{k}", f"R@{k}"], precision_recall_within_test(results.predictions[m]["random"], k, threshold)))
                         for m in models}).T


def random_vs_future_table(results: RankingResults, models: Sequence[Name]) -> pd.DataFrame:
    """P@k, NDCG@k and HR@k on the random and the future split, over the users present in both."""
    k, users = results.k, results.same_users
    by_split = {split: results.summary(split, users, models) for split in SPLITS}
    return pd.DataFrame({f"{metric} {split}": by_split[split][f"{metric}@{k}"] for metric in ("P", "NDCG", "HR") for split in SPLITS})


def breakdowns(frame: Frame, results: RankingResults, scorers: Mapping[Name, Scorer], fitted: Mapping[Name, AlgoBase], tuned: Tuned,
               models: Sequence[Name], config: Config) -> pd.DataFrame:
    """Precision on the long tail, by history length, and with other definitions of "liked" (random split)."""
    tail, n_head = long_tail_task(results.tasks["random"], frame.popularity, config.long_tail_share)
    rows = [pd.DataFrame({f"long tail ({n_head} head games removed; {len(tail):,} users)": mean_precision({m: scorers[m] for m in models}, tail, results.k)}).T,
            precision_by_history(results, frame, config.history_buckets, models),
            other_thresholds(frame, fitted, tuned, config.other_thresholds, models, results.k, config.frame.seed)]
    return pd.concat(rows)


def one_game_summary(frame: Frame, scorers: Mapping[Name, ItemToItem], groups: Mapping[str, Mapping[int, int]], candidates: np.ndarray,
                     config: Config) -> pd.DataFrame:
    """Games among the k suggestions that users went on to love (mean and 95% interval), per group of users and method."""
    summary = {}
    for label, seeds in groups.items():
        hits = one_game_test(frame, scorers, seeds, candidates, config.liked, config.k)
        cells = {m: metrics.mean_ci(hits[m], config.bootstrap_resamples, config.frame.seed) for m in hits.columns}
        summary[f"{label} ({len(seeds):,} users)"] = {m: f"{mean:.2f}  [{low:.2f}, {high:.2f}]" for m, (mean, low, high) in cells.items()}
    return pd.DataFrame(summary)


# --------------------------------------------------------------------------- beyond accuracy
def beyond_accuracy(lists: Mapping[int, np.ndarray], precision: float, n_items: int, popularity_share: np.ndarray,
                    features: np.ndarray, seed: int) -> dict[str, float]:
    return {"P@k": precision, "coverage": metrics.coverage(lists, n_items), "novelty (bits)": metrics.novelty(lists, popularity_share),
            "intra-list diversity": metrics.intra_list_diversity(lists, features), "personalisation": metrics.personalisation(lists, seed=seed)}


def beyond_accuracy_table(results: RankingResults, models: Sequence[Name], frame: Frame, features: np.ndarray) -> pd.DataFrame:
    """Coverage, novelty, intra-list diversity and personalisation of each model's random-split lists."""
    share = frame.popularity / frame.n_users
    return pd.DataFrame({m: beyond_accuracy(results.lists["random", m], float(results.per_user["random", m][f"P@{results.k}"].to_numpy().mean()),
                                            frame.n_items, share, features, results.seed) for m in models}).T


def rerank_sweep(scorer: Scorer, frame: Frame, task: RankingTask, betas: Sequence[float], k: int, features: np.ndarray, seed: int) -> pd.DataFrame:
    """Accuracy vs novelty when the scores are penalised by popularity with increasing strength beta."""
    share = frame.popularity / frame.n_users
    rows = {}
    for beta in betas:
        lists = top_k(popularity_penalised(scorer, frame.popularity, beta), task, k)
        precision = float(metrics.list_metrics(lists, task.relevant, k)[f"P@{k}"].to_numpy().mean())
        rows[beta] = beyond_accuracy(lists, precision, frame.n_items, share, features, seed)
    return pd.DataFrame(rows).T.rename_axis("β")


# --------------------------------------------------------------------------- robustness
@memory.cache
def headline(spec: FrameSpec, tuned: Tuned, config: Config) -> pd.Series:
    """The main comparisons on one frame: RMSE of three rating models and P@k of four list models, random vs future.

    The same functions as the main analysis, on a different frame; P@k uses the users present in both test sets.
    """
    frame = load_frame(spec)
    knn = best_item_knn(tuned)
    chosen = [m for m in rating_models(tuned, spec.seed) if m.name in (Name.BASELINE, knn, Name.SVD_TUNED)]
    rated = evaluate_rating_models(frame, chosen, spec.seed, verbose=False)
    row: dict[str, Any] = {"ratings in frame": len(frame.past)}
    for name, rmse in rated.table[["RMSE random", "RMSE future"]].iterrows():
        row[f"RMSE {name}: random"], row[f"RMSE {name}: future"] = rmse.to_numpy()
    scorers = list_scorers(frame.ratings, rated.fitted, tuned, config.liked, spec.seed)
    tasks = ranking_tasks(frame, config.liked)
    ranked = evaluate_rankings({m: scorers[m] for m in (Name.POPULARITY, Name.SVD_TUNED, Name.CO_RATING, Name.EASE)}, tasks, config.k)
    summary = {split: ranked.summary(split, ranked.same_users) for split in SPLITS}
    for model in ranked.models:
        for split in SPLITS:
            row[f"P@{config.k} {model}: {split}"] = summary[split][f"P@{config.k}"].to_dict()[model]
    return pd.Series(row)

