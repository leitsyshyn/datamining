"""Every decision of the experiment, in one place.

Nothing here computes anything: the notebook shows `Config.describe()` as its first table, so the
choices that shape the results are visible before any result is.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    ratings_2022: Path = PROJECT / "data/processed/ratings_2022.parquet"
    ratings_2025: Path = PROJECT / "data/processed/ratings.parquet"
    games: Path = PROJECT / "data/raw/games_detailed_info2025.csv"
    figures: Path = PROJECT / "reports/figures"
    cache: Path = PROJECT / "reports/.cache"


@dataclass(frozen=True)
class FrameSpec:
    """Which games and users form an experimental frame. Everything is chosen from the 2022 snapshot."""

    n_games: int = 2_000            # catalogue: the most-rated games
    min_user_ratings: int = 20      # enough history for collaborative filtering
    n_users: int = 15_000           # sampled users; keeps Surprise's dense similarity matrices tractable
    seed: int = 42
    test_size: float = 0.2          # random test split of the 2022 ratings
    validation_size: float = 0.1    # part of training used to tune what GridSearchCV doesn't cover


@dataclass(frozen=True)
class Grids:
    """Hyperparameter grids. GridSearchCV grids use 3-fold CV on training data; the rest use the validation split."""

    knn_k: tuple[int, ...] = (20, 40, 80)
    knn_similarities: tuple[str, ...] = ("msd", "cosine", "pearson_baseline")
    svd_learning_curve_epochs: tuple[int, ...] = (5, 10, 20, 40, 60)
    svd_learning_curve_reg: tuple[float, ...] = (0.02, 0.1)     # Surprise's default and a stronger one
    svd_factors: tuple[int, ...] = (50, 100, 200, 300)
    svd_reg: tuple[float, ...] = (0.05, 0.1, 0.15, 0.2)
    svd_lr: float = 0.005
    ease_reg: tuple[float, ...] = (100, 300, 500, 1000, 3000)
    cv_folds: int = 3


@dataclass(frozen=True)
class Config:
    frame: FrameSpec = field(default_factory=FrameSpec)
    grids: Grids = field(default_factory=Grids)
    paths: Paths = field(default_factory=Paths)

    liked: float = 8.0                       # a rating >= 8 counts as "liked" (relevant) in every ranking metric
    k: int = 10                              # length of a recommendation list
    eval_folds: int = 5                      # cross-validation of the final rating models
    bootstrap_resamples: int = 2_000         # 95% intervals, resampling users

    # robustness checks of the ranking results
    long_tail_share: float = 1 / 3           # the most popular games holding this share of ratings form the "head"
    history_buckets: tuple[tuple[int, int], ...] = ((0, 40), (40, 120), (120, 1_000_000))
    other_thresholds: tuple[float, ...] = (7.0, 9.0)
    rerank_betas: tuple[float, ...] = (0, 0.25, 0.5, 0.75, 1.0)
    robustness: tuple[tuple[str, FrameSpec], ...] = (
        ("user sample seed 7", FrameSpec(seed=7)),
        ("user sample seed 123", FrameSpec(seed=123)),
        ("users with ≥ 50 ratings", FrameSpec(min_user_ratings=50)),
        ("top 4,000 games", FrameSpec(n_games=4_000)),
    )

    # interpretation
    heavy_weight: float = 3.5                # BGG complexity at or above which a game counts as heavy
    light_weight: float = 2.0                # ... and below which as light
    min_user_ratings_for_examples: int = 30
    example_game: int = 169786               # Scythe: the "if all we know is one game you love" scenario
    probe_games: tuple[tuple[str, int], ...] = (
        ("CATAN", 13), ("Pandemic", 30549), ("Codenames", 178900), ("Gloomhaven", 174430),
        ("Twilight Imperium: Fourth Edition", 233078), ("Azul", 230802),
    )
    labelled_games: tuple[int, ...] = (      # labelled in the taste-space figure
        13, 30549, 178900, 174430, 167791, 266192, 68448, 169786, 233078, 822, 230802, 5432, 11901, 224517,
    )

    def quick(self) -> Config:
        """A small configuration for smoke runs: same code path, minutes instead of half an hour."""
        small = FrameSpec(n_games=400, n_users=2_000)
        return replace(self, frame=small, robustness=(("user sample seed 7", replace(small, seed=7)),),
                       paths=replace(self.paths, figures=PROJECT / "reports/quick/figures"),
                       grids=replace(self.grids, svd_factors=(50, 100), knn_k=(20, 40)))

    def describe(self) -> pd.DataFrame:
        f, g = self.frame, self.grids
        rows = {
            "catalogue": f"the {f.n_games:,} most-rated games in Jan 2022",
            "users": f"{f.n_users:,} sampled (seed {f.seed}) among users with ≥ {f.min_user_ratings} ratings on the catalogue",
            "random test split": f"{f.test_size:.0%} of their 2022 ratings",
            "future test set": "their ratings added between Jan 2022 and Feb 2025",
            "validation split": f"{f.validation_size:.0%} of the training ratings (SVD epochs, EASE)",
            "liked / relevant": f"rating ≥ {self.liked:g}",
            "list length": f"top {self.k}",
            "item-KNN grid": f"k ∈ {g.knn_k} × similarity ∈ {g.knn_similarities}, {g.cv_folds}-fold CV",
            "SVD grid": f"factors ∈ {g.svd_factors} × reg ∈ {g.svd_reg}, lr {g.svd_lr}, {g.cv_folds}-fold CV",
            "EASE grid": f"reg ∈ {g.ease_reg} × input ∈ (liked games, all rated games), validation P@{self.k}",
            "intervals": f"95% bootstrap over users, {self.bootstrap_resamples:,} resamples",
        }
        return pd.DataFrame({"decision": rows}).rename_axis("")


CONFIG = Config()
DEFAULT_PATHS = Paths()
