"""Loading and cleaning the BoardGameGeek snapshots, comparing them, and building the experimental frame."""

from __future__ import annotations

import ast
import functools
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from surprise import Dataset, Reader, Trainset

from .config import DEFAULT_PATHS, FrameSpec, Paths

RATING_SCALE = (1, 10)
KEY = ["user", "ID"]
COLUMNS = ["user", "ID", "rating"]


# --------------------------------------------------------------------------- snapshots
@dataclass(frozen=True)
class Snapshot:
    """One cleaned ratings snapshot."""

    ratings: pd.DataFrame          # columns: user, ID, rating
    names: pd.Series               # game ID -> name as listed in the ratings file
    removed: pd.Series             # rows removed per cleaning rule
    corrupt_values: list[float]    # the out-of-scale ratings that were removed


def load_snapshot(path: str | Path) -> Snapshot:
    """Load a ratings parquet and remove missing users, out-of-scale ratings and duplicate (user, game) pairs."""
    raw = pd.read_parquet(path)
    missing_user = raw.user.isna()
    corrupt = ~missing_user & (raw.rating < RATING_SCALE[0])
    kept = raw.loc[~missing_user & ~corrupt]
    duplicate = kept.duplicated(subset=KEY)
    kept = kept.loc[~duplicate]
    removed = pd.Series({
        "missing username": int(missing_user.sum()),
        "rating below 1 (corrupt)": int(corrupt.sum()),
        "duplicate (user, game)": int(duplicate.sum()),
    }, name="rows removed")
    return Snapshot(
        ratings=kept[COLUMNS].reset_index(drop=True),
        names=kept.drop_duplicates(subset="ID").set_index("ID")["name"],
        removed=removed,
        corrupt_values=sorted(raw.loc[corrupt, "rating"].tolist()),
    )


def _parse_list(value) -> list[str]:
    return ast.literal_eval(value) if isinstance(value, str) and value.startswith("[") else []


def load_games(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load game metadata; parse list columns and treat 0 in year / weight / play time as missing.

    Negative years are kept: they are ancient games (Go, Senet) dated BCE.
    Returns the games table and a report of what was fixed.
    """
    raw = pd.read_csv(path, index_col=0).rename(columns={"id": "ID"}).set_index("ID")
    games = raw.loc[:, ["name", "yearpublished", "minplayers", "maxplayers", "playingtime", "averageweight"]].copy()
    games["categories"] = raw["boardgamecategory"].map(_parse_list)
    games["mechanics"] = raw["boardgamemechanic"].map(_parse_list)
    games["strategy_rank"] = pd.to_numeric(raw["Strategy Game Rank"], errors="coerce")   # set for games ranked as strategy games
    report = {}
    for col in ["yearpublished", "averageweight", "playingtime"]:
        report[f"{col} == 0 -> missing"] = int((games[col] == 0).sum())
        games[col] = games[col].replace(0, np.nan)
    report["yearpublished < 0 (ancient games, kept)"] = int((games.yearpublished < 0).sum())
    report["duplicate game IDs"] = int(games.index.duplicated().sum())
    return games, pd.Series(report, name="games")


# --------------------------------------------------------------------------- snapshot comparison
@dataclass(frozen=True)
class SnapshotDiff:
    """What changed between an older and a newer snapshot, per (user, game) pair."""

    new: pd.DataFrame        # pairs only in the newer snapshot: user, ID, rating
    rerated: pd.DataFrame    # pairs in both with a different value: user, ID, rating_old, rating_new
    unchanged: int
    disappeared: int

    def summary(self) -> pd.Series:
        return pd.Series({
            "unchanged": self.unchanged,
            "re-rated (value changed)": len(self.rerated),
            "new in the later snapshot": len(self.new),
            "disappeared": self.disappeared,
        }, name="pairs")


def compare_snapshots(old: pd.DataFrame, new: pd.DataFrame) -> SnapshotDiff:
    merged = old.merge(new, on=KEY, how="outer", suffixes=("_old", "_new"), indicator=True)
    both = merged[merged._merge == "both"]
    rerated = both.loc[both.rating_old != both.rating_new, ["user", "ID", "rating_old", "rating_new"]]
    added = merged.loc[merged._merge == "right_only", ["user", "ID", "rating_new"]].rename(columns={"rating_new": "rating"})
    return SnapshotDiff(
        new=added.reset_index(drop=True),
        rerated=rerated.reset_index(drop=True),
        unchanged=len(both) - len(rerated),
        disappeared=int((merged._merge == "left_only").sum()),
    )


# --------------------------------------------------------------------------- everything on disk
@dataclass(frozen=True, eq=False)
class Data:
    """Both snapshots, the game metadata and what changed between the snapshots."""

    old: Snapshot              # January 2022
    new: Snapshot              # February 2025
    games: pd.DataFrame
    game_fixes: pd.Series
    diff: SnapshotDiff

    @cached_property
    def titles(self) -> pd.Series:
        """Game ID -> name, from the metadata where available, else from the ratings files."""
        names = pd.concat([self.games["name"], self.new.names, self.old.names])
        return cast(pd.Series, names.groupby(level=0).first())


def load(paths: Paths = DEFAULT_PATHS) -> Data:
    """Load and clean everything once per session (about a minute)."""
    return _load(paths.ratings_2022, paths.ratings_2025, paths.games)


@functools.cache  # keyed on the input files only, so output folders (figures, cache) don't cause a reload
def _load(ratings_2022: Path, ratings_2025: Path, games_file: Path) -> Data:
    old, new = load_snapshot(ratings_2022), load_snapshot(ratings_2025)
    games, fixes = load_games(games_file)
    return Data(old=old, new=new, games=games, game_fixes=fixes, diff=compare_snapshots(old.ratings, new.ratings))


# --------------------------------------------------------------------------- Surprise conversions
def to_dataset(ratings: pd.DataFrame) -> Dataset:
    return Dataset.load_from_df(ratings[COLUMNS], Reader(rating_scale=RATING_SCALE))


def to_trainset(ratings: pd.DataFrame) -> Trainset:
    dataset: Any = to_dataset(ratings)   # Surprise is untyped
    return dataset.build_full_trainset()


def as_testset(ratings: pd.DataFrame) -> list[tuple]:
    return list(ratings[COLUMNS].itertuples(index=False, name=None))


# --------------------------------------------------------------------------- the experimental frame
@dataclass(frozen=True, eq=False)
class Frame:
    """One sample of users and games with a random train/test split and a future test set.

    Everything that defines the frame (catalogue, eligible users, the split) comes from the past
    snapshot only; `future` holds the ratings the same users gave later to catalogue games.
    The frame owns the mapping between BGG ids and the dense indices models use ("inner" ids).
    """

    spec: FrameSpec
    train: pd.DataFrame
    test: pd.DataFrame
    future: pd.DataFrame
    catalogue: pd.Index

    # ---- Surprise view and id mapping (the only place that touches Surprise's internals)
    @cached_property
    def trainset(self) -> Trainset:
        return to_trainset(self.train)

    @cached_property
    def items(self) -> np.ndarray:
        """BGG game id of every inner item index."""
        return np.array([self.trainset.to_raw_iid(i) for i in range(self.trainset.n_items)])

    @cached_property
    def item_index(self) -> pd.Series:
        """BGG game id -> inner item index."""
        return pd.Series(self.trainset._raw2inner_id_items, dtype=int)

    @cached_property
    def user_index(self) -> pd.Series:
        """Username -> inner user index."""
        return pd.Series(self.trainset._raw2inner_id_users, dtype=int)

    def inner_items(self, game_ids) -> np.ndarray:
        return self.item_index.loc[list(game_ids)].to_numpy()

    def inner_item(self, game_id) -> int:
        trainset: Any = self.trainset
        return int(trainset.to_inner_iid(game_id))

    def inner_user(self, user) -> int:
        trainset: Any = self.trainset
        return int(trainset.to_inner_uid(user))

    @property
    def n_users(self) -> int:
        return int(self.trainset.n_users)

    @property
    def n_items(self) -> int:
        return int(self.trainset.n_items)

    # ---- matrices
    @cached_property
    def ratings(self) -> sp.csr_matrix:
        """Training ratings as a sparse users x items matrix (0 = not rated; ratings are >= 1)."""
        rows = self.user_index.loc[self.train.user].to_numpy()
        cols = self.item_index.loc[self.train.ID].to_numpy()
        shape = (self.n_users, self.n_items)
        return sp.csr_matrix((self.train.rating.to_numpy(dtype=float), (rows, cols)), shape=shape)

    def liked(self, threshold: float) -> sp.csr_matrix:
        """Sparse 0/1 users x items matrix of training ratings >= threshold."""
        return at_least(self.ratings, threshold)

    @cached_property
    def popularity(self) -> np.ndarray:
        """Number of training ratings per inner item."""
        return np.asarray(at_least(self.ratings, 1.0).sum(axis=0)).ravel()

    @cached_property
    def history_length(self) -> np.ndarray:
        """Number of training ratings per inner user."""
        return np.diff(self.ratings.indptr)

    # ---- test sets
    @cached_property
    def past(self) -> pd.DataFrame:
        """All sampled 2022 ratings (train + random test)."""
        return pd.concat([self.train, self.test])

    @property
    def testsets(self) -> dict[str, list[tuple]]:
        return {"random": as_testset(self.test), "future": as_testset(self.future)}

    @cached_property
    def validation(self) -> Frame:
        """A frame for tuning: the training ratings split again, its `test` being the validation ratings."""
        fit, held_out = cast(list[pd.DataFrame], train_test_split(
            self.train, test_size=self.spec.validation_size, random_state=self.spec.seed))
        return Frame(spec=self.spec, train=fit, test=held_out, future=self.future.iloc[:0], catalogue=self.catalogue)

    # ---- checks
    def check(self) -> pd.Series:
        """Leakage and consistency checks; raises AssertionError on any violation, else returns a short report."""
        def pairs(df):
            return set(zip(df.user, df.ID))
        train, test, future = pairs(self.train), pairs(self.test), pairs(self.future)
        assert not train & test, "a test rating is also in training"
        assert not (train | test) & future, "a future rating already existed in 2022"
        assert set(self.test.user) | set(self.future.user) <= set(self.train.user), "a test user has no training history"
        assert set(self.test.ID) | set(self.future.ID) <= set(self.train.ID), "a test game has no training ratings"
        return pd.Series({
            "no (user, game) pair in more than one set": "✓",
            "every test user and game has training ratings": "✓",
            "min training ratings per game": int(self.popularity.min()),
            "min training ratings per user": int(self.history_length.min()),
            "sampled users who rated catalogue games later": f"{self.future.user.nunique() / self.train.user.nunique():.0%}",
        }, name="check")

    def sizes(self) -> pd.DataFrame:
        rows = {name: {"ratings": len(df), "users": df.user.nunique(), "games": df.ID.nunique()}
                for name, df in [("train", self.train), ("random test", self.test), ("future test", self.future)]}
        return pd.DataFrame(rows).T


def at_least(ratings: sp.csr_matrix, threshold: float) -> sp.csr_matrix:
    """0/1 matrix (float) of the entries >= threshold; all ratings are >= 1, so threshold 1 means "rated"."""
    out = ratings.copy()
    out.data = (out.data >= threshold).astype(np.float64)
    out.eliminate_zeros()
    return out


def build_frame(past: pd.DataFrame, later_ratings: pd.DataFrame, spec: FrameSpec) -> Frame:
    """Catalogue = the `spec.n_games` most-rated games in `past`; users = a sample of those with at least
    `spec.min_user_ratings` ratings on the catalogue; their past ratings are split at random into train/test,
    and their `later_ratings` on catalogue games form the future test set."""
    catalogue = past.ID.value_counts().index[:spec.n_games]
    core = past.loc[past.ID.isin(catalogue)]
    per_user = core.user.value_counts()
    eligible = per_user.index[per_user >= spec.min_user_ratings].to_series()
    users = eligible.sample(min(spec.n_users, len(eligible)), random_state=spec.seed)
    sample = core.loc[core.user.isin(users)]
    train, test = cast(list[pd.DataFrame], train_test_split(sample, test_size=spec.test_size, random_state=spec.seed))
    future = later_ratings.loc[later_ratings.user.isin(users) & later_ratings.ID.isin(catalogue)]
    return Frame(spec=spec, train=train, test=test, future=future, catalogue=catalogue)


def load_frame(spec: FrameSpec, paths: Paths = DEFAULT_PATHS) -> Frame:
    """The frame for `spec`, built once per session from the loaded data."""
    return _load_frame(spec, load(paths))


@functools.cache  # one entry per (spec, data), however the caller passes `paths`
def _load_frame(spec: FrameSpec, data: Data) -> Frame:
    return build_frame(data.old.ratings, data.diff.new, spec)
