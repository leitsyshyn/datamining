"""The single owner of model names, of how each model is built, and of which models each analysis compares."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import scipy.sparse as sp
from surprise import (
    SVD,
    AlgoBase,
    BaselineOnly,
    KNNBaseline,
    KNNBasic,
    KNNWithMeans,
    KNNWithZScore,
    NormalPredictor,
    SVDpp,
)

from .rating import GlobalMean
from .topn import (
    EASE,
    CoRatingKNN,
    Popularity,
    RandomScores,
    Scorer,
    from_surprise,
    max_scorer_error,
)


class Name(StrEnum):
    # rating prediction (Surprise)
    GLOBAL_MEAN = "Global mean"
    RANDOM_RATINGS = "Random (NormalPredictor)"
    BASELINE = "BaselineOnly"
    KNN_RAW = "Item-KNN: raw ratings (KNNBasic)"
    KNN_MEANS = "Item-KNN: mean-centred (KNNWithMeans)"
    KNN_ZSCORE = "Item-KNN: z-scored (KNNWithZScore)"
    KNN_BASELINE = "Item-KNN: baseline (KNNBaseline)"
    KNN_TUNED = "Item-KNN (tuned)"
    USER_KNN = "User-KNN (KNNBaseline)"
    SVD_DEFAULT = "SVD (default)"
    SVD_TUNED = "SVD (tuned)"
    SVDPP = "SVD++ (default)"
    AVERAGE = "Average of SVD (tuned) and item-KNN"
    # top-N lists
    RANDOM = "Random"
    POPULARITY = "Popularity"
    CO_RATING = "Co-rating KNN"
    CO_RATING_ALL = "Co-rating KNN (all rated games)"
    EASE = "EASE"


class Family(StrEnum):
    BASELINE = "baseline"
    NEIGHBOURHOOD = "neighbourhood"
    FACTORISATION = "matrix factorisation"
    ENSEMBLE = "ensemble"


@dataclass(frozen=True)
class Tuned:
    """Hyperparameters chosen by tuning (section 3)."""

    knn: dict[str, Any]
    svd: dict[str, Any]
    ease: dict[str, Any]


@dataclass(frozen=True)
class RatingModel:
    name: Name
    family: Family
    make: Callable[[], AlgoBase]
    keep_fitted: bool = True     # False for models too large to keep in memory after evaluation


SIMILARITY = {"name": "pearson_baseline", "user_based": False}
DEFAULT_KNN = {"k": 40, "sim_options": SIMILARITY}


def best_item_knn(tuned: Tuned) -> Name:
    """The tuned item-KNN, which is the baseline-adjusted variant when the grid confirms its defaults."""
    same = tuned.knn["k"] == DEFAULT_KNN["k"] and tuned.knn["sim_options"] == SIMILARITY
    return Name.KNN_BASELINE if same else Name.KNN_TUNED


def rating_models(tuned: Tuned, seed: int) -> list[RatingModel]:
    """Every rating-prediction model, in the order they are evaluated."""
    def knn(cls, **kw):
        return lambda: cls(**{**DEFAULT_KNN, **kw}, verbose=False)
    models = [
        RatingModel(Name.GLOBAL_MEAN, Family.BASELINE, GlobalMean),
        RatingModel(Name.RANDOM_RATINGS, Family.BASELINE, NormalPredictor),
        RatingModel(Name.BASELINE, Family.BASELINE, lambda: BaselineOnly(verbose=False)),
        # item-based KNN: same similarity and k, only the normalisation differs
        RatingModel(Name.KNN_RAW, Family.NEIGHBOURHOOD, knn(KNNBasic)),
        RatingModel(Name.KNN_MEANS, Family.NEIGHBOURHOOD, knn(KNNWithMeans)),
        RatingModel(Name.KNN_ZSCORE, Family.NEIGHBOURHOOD, knn(KNNWithZScore)),
        RatingModel(Name.KNN_BASELINE, Family.NEIGHBOURHOOD, knn(KNNBaseline)),
        RatingModel(Name.USER_KNN, Family.NEIGHBOURHOOD, knn(KNNBaseline, sim_options={**SIMILARITY, "user_based": True}),
                    keep_fitted=False),   # a 15,000 x 15,000 similarity matrix
        RatingModel(Name.SVD_DEFAULT, Family.FACTORISATION, lambda: SVD(random_state=seed)),
        RatingModel(Name.SVD_TUNED, Family.FACTORISATION, lambda: SVD(**tuned.svd)),
        RatingModel(Name.SVDPP, Family.FACTORISATION, lambda: SVDpp(random_state=seed, cache_ratings=True)),
    ]
    if best_item_knn(tuned) == Name.KNN_TUNED:
        models.insert(7, RatingModel(Name.KNN_TUNED, Family.NEIGHBOURHOOD, lambda: KNNBaseline(**tuned.knn)))
    return models


def list_scorers(ratings: sp.csr_matrix, fitted: Mapping[Name, AlgoBase], tuned: Tuned, liked: float, seed: int) -> dict[Name, Scorer]:
    """Every model that produces top-N lists: non-personalised references, the fitted rating models among
    RANKED_RATING_MODELS, and the list models."""
    popularity = Popularity(ratings)
    return {
        Name.RANDOM: RandomScores(popularity.n_items, seed),
        Name.POPULARITY: popularity,
        **{name: from_surprise(fitted[name]) for name in RANKED_RATING_MODELS if name in fitted},
        Name.CO_RATING: CoRatingKNN(liked).fit(ratings),
        Name.CO_RATING_ALL: CoRatingKNN(liked, unliked_weight=1.0).fit(ratings),
        Name.EASE: EASE(**tuned.ease).fit(ratings),
    }


def verify_rating_scorers(fitted: Mapping[Name, AlgoBase], scorers: Mapping[Name, Scorer], tolerance: float = 1e-6) -> dict[str, float]:
    """The vectorised scores must reproduce Surprise's own estimates; raises if any model deviates."""
    errors = {f"{name}: max |vectorised − Surprise|": max_scorer_error(fitted[name], scorers[name]) for name in RANKED_RATING_MODELS if name in fitted}
    assert all(e < tolerance for e in errors.values()), errors
    return errors


def list_group(name: Name) -> str:
    """How a model is grouped in the ranking charts."""
    if name in (Name.RANDOM, Name.POPULARITY):
        return "non-personalised"
    if name in (Name.CO_RATING, Name.CO_RATING_ALL, Name.EASE):
        return "top-N CF"
    return "rating-prediction CF"


# --------------------------------------------------------------------------- which models each analysis compares
RANKED_RATING_MODELS = (Name.BASELINE, Name.SVD_DEFAULT, Name.SVD_TUNED, Name.SVDPP)
WITHIN_TEST_RANKING = (Name.RANDOM_RATINGS, Name.BASELINE, Name.SVD_TUNED, Name.SVDPP)     # + best item-KNN
RMSE_CHART = (Name.BASELINE, Name.KNN_RAW, Name.USER_KNN, Name.SVD_DEFAULT, Name.SVD_TUNED, Name.SVDPP, Name.AVERAGE)  # + best item-KNN
COMPARE_RANDOM = (
    (Name.EASE, Name.CO_RATING), (Name.EASE, Name.POPULARITY), (Name.CO_RATING, Name.POPULARITY),
    (Name.CO_RATING, Name.SVDPP), (Name.SVDPP, Name.POPULARITY), (Name.SVD_TUNED, Name.SVD_DEFAULT),
    (Name.CO_RATING, Name.CO_RATING_ALL),
)
COMPARE_FUTURE = (
    (Name.EASE, Name.SVD_TUNED), (Name.EASE, Name.CO_RATING), (Name.SVD_TUNED, Name.CO_RATING),
    (Name.CO_RATING, Name.POPULARITY), (Name.SVD_TUNED, Name.POPULARITY), (Name.SVD_TUNED, Name.BASELINE),
)
WHAT_GETS_RECOMMENDED = (Name.POPULARITY, Name.BASELINE, Name.SVD_TUNED, Name.SVDPP, Name.CO_RATING, Name.EASE)
BREAKDOWNS = (Name.POPULARITY, Name.SVD_TUNED, Name.SVDPP, Name.CO_RATING, Name.EASE)
RANDOM_VS_FUTURE = (Name.RANDOM, Name.POPULARITY, Name.BASELINE, Name.SVD_TUNED, Name.SVDPP, Name.CO_RATING, Name.EASE)
EXAMPLE_LISTS = (Name.SVDPP, Name.EASE)       # best rating model vs best list model, side by side
COMPLEXITY_FIT = (Name.POPULARITY, Name.SVD_TUNED, Name.SVDPP, Name.CO_RATING, Name.EASE)
ONE_GAME = (Name.CO_RATING, Name.EASE)
BEYOND_ACCURACY = (Name.RANDOM, Name.POPULARITY, Name.BASELINE, Name.SVD_TUNED, Name.SVDPP, Name.CO_RATING, Name.EASE)
