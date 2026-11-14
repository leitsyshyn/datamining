# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportReturnType=false, reportIndexIssue=false, reportOperatorIssue=false
# (pandas-stubs cannot follow groupby/agg/pivot chains; the logic here is covered by the notebook's outputs)
"""Descriptive analysis: the tables behind the EDA and the interpretation sections.

Pure functions from loaded data, frames and fitted models to DataFrames; no plotting, no evaluation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from surprise import AlgoBase

from .data import Data, Frame, SnapshotDiff, at_least


# --------------------------------------------------------------------------- the data and its cleaning
def snapshot_overview(data: Data) -> pd.DataFrame:
    def describe(r: pd.DataFrame) -> dict:
        n_u, n_g = r.user.nunique(), r.ID.nunique()
        return {"ratings": len(r), "users": n_u, "games": n_g, "density": len(r) / (n_u * n_g)}
    return pd.DataFrame({"Jan 2022": describe(data.old.ratings), "Feb 2025": describe(data.new.ratings)}).T


def cleaning_report(data: Data) -> pd.DataFrame:
    return pd.DataFrame({"Jan 2022": data.old.removed, "Feb 2025": data.new.removed})


def rating_decimals(ratings: pd.DataFrame) -> pd.Series:
    dec = ratings.rating % 1
    return pd.Series({"integer": (dec == 0).mean(), "half point": (dec == 0.5).mean(), "other decimals": ((dec != 0) & (dec != 0.5)).mean()})


def metadata_coverage(ratings: pd.DataFrame, games: pd.DataFrame) -> pd.Series:
    missing = ~ratings.ID.isin(games.index)
    return pd.Series({"share of ratings on games without metadata": missing.mean(), "games without metadata": ratings.ID[missing].nunique()})


def new_rating_origins(data: Data) -> pd.DataFrame:
    """Share of the ratings added after Jan 2022 by existing vs new users, on existing vs new games."""
    new, old = data.diff.new, data.old.ratings
    users = new.user.isin(set(old.user)).map({True: "user already in 2022", False: "new user"})
    games = new.ID.isin(set(old.ID)).map({True: "game rated in 2022", False: "game new since 2022"})
    return pd.crosstab(users, games, normalize=True)


def core_sizes(ratings: pd.DataFrame, configs: Sequence[tuple[int, int]]) -> pd.DataFrame:
    """Size and density of the core for (number of most-rated games, minimum ratings per user) choices."""
    counts = ratings.ID.value_counts()
    rows = []
    for n_games, min_ratings in configs:
        core = ratings.loc[ratings.ID.isin(counts.index[:n_games])]
        per_user = core.user.value_counts()
        core = core.loc[core.user.isin(per_user.index[per_user >= min_ratings])]
        n_u, n_g = core.user.nunique(), core.ID.nunique()
        rows.append({"games kept": f"top {n_games:,}", "min ratings/user": min_ratings, "ratings": len(core),
                     "users": n_u, "games": n_g, "density": len(core) / (n_u * n_g), "share of all ratings": len(core) / len(ratings)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- EDA
def popularity_facts(games: pd.DataFrame, curve: pd.DataFrame, top_n: int) -> dict[str, float]:
    return {f"share of ratings held by the {top_n:,} most-rated games": float(curve["share of ratings"].iat[top_n - 1]),
            "those games as a share of all games": top_n / len(games),
            "Spearman correlation of popularity and mean rating": float(games.n.corr(games["mean"], method="spearman"))}


def complexity_correlation(games: pd.DataFrame, min_ratings: int = 100) -> float:
    g = games[games.n >= min_ratings].dropna(subset=["averageweight"])
    return float(g.averageweight.corr(g["mean"]))


def drift_summary(delta: pd.Series) -> dict[str, float]:
    return {"re-rated pairs": len(delta), "mean change": float(delta.mean()), "share that went down": float((delta < 0).mean()),
            "share that went up": float((delta > 0).mean())}


def shared_raters(frame: Frame) -> dict[str, float]:
    """How many users rated both games, over all pairs of catalogue games: the evidence behind item similarities."""
    rated = at_least(frame.ratings, 1.0)
    co = (rated.T @ rated).toarray()[np.triu_indices(frame.n_items, 1)]
    return {"median users who rated both games of a pair": float(np.median(co)), "share of pairs with fewer than 20": float((co < 20).mean()),
            "share of pairs with fewer than 5": float((co < 5).mean())}


def game_stats(data: Data) -> pd.DataFrame:
    """Per game: number of ratings, mean and spread (2025 snapshot), with year, complexity and name."""
    stats = data.new.ratings.groupby("ID").rating.agg(n="size", mean="mean", std="std")
    stats = stats.join(data.games[["yearpublished", "averageweight"]])
    stats["name"] = data.titles.reindex(stats.index)
    return stats


def user_stats(ratings: pd.DataFrame) -> pd.DataFrame:
    return ratings.groupby("user").rating.agg(n="size", mean="mean", std="std")


def rating_summary(ratings: pd.DataFrame, liked: float) -> pd.Series:
    r = ratings.rating
    return pd.Series({"mean": r.mean(), "median": r.median(), "std": r.std(), "share 6–8": r.between(6, 8).mean(),
                      f"share ≥ {liked:g}": (r >= liked).mean()})


def user_summary(users: pd.DataFrame, min_ratings: int) -> pd.Series:
    active = users[users.n >= min_ratings]
    return pd.Series({"median ratings per user": users.n.median(), "90th percentile": users.n.quantile(0.9),
                      "share who rated a single game": (users.n == 1).mean(),
                      f"users with ≥ {min_ratings} ratings, all identical": int((active["std"] == 0).sum())})


def generosity_by_activity(users: pd.DataFrame, min_ratings: int, edges: Sequence[float] = (20, 50, 100, 200, 500, 1000, np.inf)) -> pd.DataFrame:
    """Quartiles of the user mean rating, by number of ratings per user."""
    active = users[users.n >= min_ratings]
    bins = pd.cut(active.n, list(edges), right=False)
    return active.groupby(bins, observed=True)["mean"].quantile([0.25, 0.5, 0.75]).unstack()


def concentration(games: pd.DataFrame) -> pd.DataFrame:
    """Cumulative share of ratings held by the most-rated games (a Lorenz-style curve)."""
    counts = np.sort(games.n.to_numpy())[::-1]
    return pd.DataFrame({"share of games": np.arange(1, len(counts) + 1) / len(counts), "share of ratings": np.cumsum(counts) / counts.sum()})


def rating_by_period(games: pd.DataFrame, min_ratings: int = 100, since: int = 1960, width: int = 5) -> pd.Series:
    g = games[(games.n >= min_ratings) & (games.yearpublished >= since)]
    return g.groupby((g.yearpublished // width * width).astype(int))["mean"].median()


def rating_by_era_and_popularity(games: pd.DataFrame, min_ratings: int = 100) -> pd.DataFrame:
    g = games[games.n >= min_ratings]
    era = pd.cut(g.yearpublished, [-5000, 1999, 2014, 2019, 2030], labels=["before 2000", "2000–2014", "2015–2019", "2020+"])
    popularity = pd.qcut(g.n, 4, labels=["least rated quarter", "2nd", "3rd", "most rated quarter"])
    return g.assign(era=era, popularity=popularity).pivot_table(index="era", columns="popularity", values="mean", aggfunc="median", observed=True)


def mechanic_ratings(games: pd.DataFrame, metadata: pd.DataFrame, min_ratings: int = 100, min_games: int = 300, n: int = 6) -> pd.DataFrame:
    """The mechanics with the lowest and highest mean game rating (mechanics used by at least `min_games` games)."""
    g = games[games.n >= min_ratings].join(metadata["mechanics"])
    by = g.explode("mechanics").dropna(subset=["mechanics"]).groupby("mechanics")["mean"].agg(["mean", "size"])
    by = by[by["size"] >= min_games].sort_values("mean")
    return pd.concat([by.head(n).assign(end="lowest"), by.tail(n).assign(end="highest")])


def rerating_drift(diff: SnapshotDiff) -> pd.Series:
    """Change of rating for pairs rated in both snapshots with a different value."""
    return diff.rerated.rating_new - diff.rerated.rating_old


# --------------------------------------------------------------------------- the frame's items
def item_table(frame: Frame, data: Data, games: pd.DataFrame) -> pd.DataFrame:
    """One row per inner item of the frame: BGG id, name, complexity, year, strategy flag, popularity, mean rating."""
    ids = frame.items
    meta = data.games.reindex(ids)
    return pd.DataFrame({
        "ID": ids, "name": data.titles.reindex(ids).to_numpy(), "weight": meta.averageweight.to_numpy(),
        "year": meta.yearpublished.to_numpy(), "strategy": meta.strategy_rank.notna().to_numpy(),
        "training ratings": frame.popularity, "mean rating": games["mean"].reindex(ids).to_numpy(),
    })


def item_features(frame: Frame, data: Data) -> np.ndarray:
    """Binary item x (mechanic or category) matrix, for intra-list diversity."""
    feats = [set(data.games.at[g, "mechanics"]) | set(data.games.at[g, "categories"]) if g in data.games.index else set() for g in frame.items]
    vocab = {f: k for k, f in enumerate(sorted(set().union(*feats)))}
    out = np.zeros((len(feats), len(vocab)), dtype=np.float32)
    for i, fs in enumerate(feats):
        out[i, [vocab[f] for f in fs]] = 1
    return out


def own_novelty(frame: Frame, users: Sequence[int]) -> float:
    """Novelty (bits) of the games the given users actually rated: the reference for recommendation novelty."""
    from .metrics import self_information
    own = np.concatenate([frame.ratings[u].indices for u in users])
    return self_information(own, frame.popularity / frame.n_users)


def name_lists(lists: Mapping[str, np.ndarray], items: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({m: items.name.to_numpy()[idx] for m, idx in lists.items()}, index=range(1, len(next(iter(lists.values()))) + 1))


def loved_weight(frame: Frame, items: pd.DataFrame, liked: float) -> pd.Series:
    """Per inner user: mean complexity of the games they liked in training (NaN if none)."""
    loved = frame.liked(liked).tocoo()
    weights = pd.Series(items.weight.to_numpy()[loved.col], index=loved.row)
    return weights.groupby(level=0).mean().reindex(range(frame.n_users))


# --------------------------------------------------------------------------- what gets recommended
def recommendation_profile(lists: Mapping[str, Mapping[int, np.ndarray]], items: pd.DataFrame, frame: Frame, users: Sequence[int]) -> pd.DataFrame:
    """Median year, mean complexity and mean popularity percentile of recommended games, vs the games users rate."""
    pop_pct = items["training ratings"].rank(pct=True).to_numpy() * 100
    own = np.concatenate([frame.ratings[u].indices for u in users])
    groups = {"(games users actually rate)": own} | {name: np.concatenate(list(lst.values())) for name, lst in lists.items()}
    return pd.DataFrame({name: {"median year": np.nanmedian(items.year.to_numpy()[i]), "mean weight": np.nanmean(items.weight.to_numpy()[i]),
                                "mean popularity percentile": pop_pct[i].mean()} for name, i in groups.items()}).T


def complexity_fit(lists: Mapping[str, Mapping[int, np.ndarray]], items: pd.DataFrame, loved: pd.Series) -> pd.DataFrame:
    """Does the complexity of each user's list follow the complexity of the games they love?"""
    weight = items.weight.to_numpy()
    rows = {}
    for name, lst in lists.items():
        users = [u for u in lst if not np.isnan(loved.get(u, np.nan))]
        rec = pd.Series({u: np.nanmean(weight[lst[u]]) for u in users})
        rows[name] = {"correlation with loved weight": rec.corr(loved[users]), "mean |gap| (weight points)": (rec - loved[users]).abs().mean()}
    return pd.DataFrame(rows).T


def example_users(frame: Frame, users: Sequence[int], loved: pd.Series, min_ratings: int) -> dict[str, int]:
    """A heavy-strategy fan and a light / family gamer (highest / lowest complexity of loved games), and the shortest history."""
    prof = pd.DataFrame({"n": frame.history_length[list(users)], "loved": loved.reindex(users).to_numpy()}, index=list(users)).dropna()
    experienced = prof[prof.n >= min_ratings]
    return {"heavy-strategy fan": int(experienced.loved.idxmax()), "light / family gamer": int(experienced.loved.idxmin()),
            "short history": int(prof.n.idxmin())}


def favourites(frame: Frame, user: int, items: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    row = frame.ratings[user]
    order = np.argsort(-row.data, kind="stable")[:n]
    idx = row.indices[order]
    return pd.DataFrame({"favourite game": items.name.to_numpy()[idx], "their rating": row.data[order], "weight": items.weight.to_numpy()[idx]})


def lists_side_by_side(lists: Mapping[str, np.ndarray], relevant: frozenset, items: pd.DataFrame,
                       explain: tuple[str, Callable[[int], int | None]] | None = None) -> pd.DataFrame:
    """Several models' lists for one user: game, complexity, hit (✓ = rated ≥ the threshold in the held-out set)."""
    cols = {}
    for name, idx in lists.items():
        cols[(name, "game")] = [str(items.name.iat[j])[:36] for j in idx]
        cols[(name, "weight")] = items.weight.to_numpy()[idx]
        cols[(name, "hit")] = ["✓" if j in relevant else "" for j in idx]
    if explain is not None:
        name, because = explain
        cols[(name, "because you liked")] = [str(items.name.iat[because(j)])[:30] if because(j) is not None else "" for j in lists[name]]
    return pd.DataFrame(cols, index=range(1, len(next(iter(lists.values()))) + 1))


# --------------------------------------------------------------------------- the future
def reachability(data: Data, frame: Frame, liked: float) -> pd.Series:
    """How much of what the sampled users rated after Jan 2022 a model trained on the 2022 catalogue could reach."""
    later = data.diff.new.loc[data.diff.new.user.isin(set(frame.train.user))]
    in_catalogue = later.ID.isin(frame.catalogue)
    return pd.Series({"ratings added after Jan 2022": len(later), "share on catalogue games": in_catalogue.mean(),
                      f"share of liked (≥ {liked:g}) on catalogue games": in_catalogue[later.rating >= liked].mean(),
                      "mean rating, their 2022 ratings": frame.past.rating.mean(), "mean rating, their new ratings": later.rating.mean()})


def most_rated_new_games(data: Data, frame: Frame, n: int = 8) -> pd.DataFrame:
    later = data.diff.new.loc[data.diff.new.user.isin(set(frame.train.user))]
    rank_2022 = pd.Series(np.arange(1, data.old.ratings.ID.nunique() + 1), index=data.old.ratings.ID.value_counts().index)
    top = later.ID.value_counts().head(n)
    return pd.DataFrame({"game": data.titles.reindex(top.index), "new ratings": top, "year": data.games.yearpublished.reindex(top.index),
                         "popularity rank in 2022": rank_2022.reindex(top.index), "in catalogue": top.index.isin(frame.catalogue)})


def played_vs_loved(ratings: pd.DataFrame, fan_game: int, suggestions: Mapping[str, Sequence[int]], data: Data, liked: float) -> pd.DataFrame:
    """Among fans of `fan_game` (rated it >= liked): who played each suggested game, and did they love it more than
    everyone else? `suggestions` maps each method to the BGG ids it suggested."""
    fans = set(ratings.loc[(ratings.ID == fan_game) & (ratings.rating >= liked), "user"])
    ids = pd.Index(pd.unique(np.concatenate([np.asarray(v) for v in suggestions.values()])))
    sub = ratings.loc[ratings.ID.isin(ids)].assign(fan=lambda d: d.user.isin(fans), loved=lambda d: d.rating >= liked)
    by_fan, by_all = sub[sub.fan].groupby("ID"), sub.groupby("ID")
    out = pd.DataFrame({
        "game": data.titles.reindex(ids).to_numpy(), "weight": data.games.averageweight.reindex(ids).to_numpy(),
        "fans who played it": by_fan.size().reindex(ids).to_numpy() / len(fans),
        "everyone who played it": by_all.size().reindex(ids).to_numpy() / ratings.user.nunique(),
        "fans who loved it": by_fan.loved.mean().reindex(ids).to_numpy(), "everyone who loved it": by_all.loved.mean().reindex(ids).to_numpy(),
    }, index=ids)
    out["love lift"] = out["fans who loved it"] - out["everyone who loved it"]
    out["suggested by"] = [", ".join(m for m, v in suggestions.items() if g in set(v)) for g in ids]
    return out.sort_values("love lift", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- interpretation
def top_interactions(frame: Frame, n: int = 60) -> pd.DataFrame:
    """Ratings of the n most active users on the n most-rated games (NaN = not rated)."""
    sample = frame.past
    users, games = sample.user.value_counts().head(n).index, sample.ID.value_counts().head(n).index
    sub = sample[sample.user.isin(users) & sample.ID.isin(games)]
    return sub.pivot(index="user", columns="ID", values="rating").reindex(index=users, columns=games)


def fill_rate(frame: Frame, users_per_block: int = 150, games_per_block: int = 20) -> np.ndarray:
    """Share of rated cells in blocks of the user x game matrix, users sorted by activity and games by popularity."""
    sample = frame.past
    u_rank = {u: k for k, u in enumerate(sample.user.value_counts().index)}
    g_rank = {g: k for k, g in enumerate(sample.ID.value_counts().index)}
    ub = (sample.user.map(u_rank) // users_per_block).to_numpy()
    gb = (sample.ID.map(g_rank) // games_per_block).to_numpy()
    blocks = np.zeros((ub.max() + 1, gb.max() + 1))
    np.add.at(blocks, (ub, gb), 1)
    return blocks / (users_per_block * games_per_block)


TAGS = {"Cooperative Game": "mechanics", "Worker Placement": "mechanics", "Dice Rolling": "mechanics", "Deck, Bag, and Pool Building": "mechanics",
        "Area Majority / Influence": "mechanics", "Wargame": "categories", "Party Game": "categories", "Children's Game": "categories",
        "Abstract Strategy": "categories", "Fantasy": "categories", "Economic": "categories"}


def latent_components(svd: AlgoBase, frame: Frame, data: Data, games: pd.DataFrame, n_components: int = 5, seed: int = 0
                      ) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Principal components of SVD's game vectors, and their correlations with metadata the model never saw.

    Returns (games x components with metadata, metadata x component Spearman correlations, explained variance ratios).
    """
    m: Any = svd
    pca = PCA(n_components=n_components, random_state=seed).fit(m.qi)
    emb = pd.DataFrame(pca.transform(m.qi), columns=[f"PC{k + 1}" for k in range(n_components)], index=frame.items)
    emb = emb.join(data.games[["averageweight", "playingtime", "yearpublished", "maxplayers", "mechanics", "categories"]])
    emb["name"] = data.titles.reindex(emb.index)
    emb["mean rating"] = games["mean"].reindex(emb.index)
    emb["log10 ratings"] = np.log10(games["n"].reindex(emb.index))
    for tag, col in TAGS.items():
        emb[tag] = emb[col].map(lambda xs, t=tag: float(t in xs) if isinstance(xs, list) else np.nan)
    features = ["averageweight", "playingtime", "yearpublished", "maxplayers", "mean rating", "log10 ratings", *TAGS]
    pcs = [f"PC{k + 1}" for k in range(n_components)]
    corr = pd.DataFrame({pc: [emb[pc].corr(emb[f], method="spearman") for f in features] for pc in pcs}, index=features)
    return emb, corr, pca.explained_variance_ratio_


def component_extremes(components: pd.DataFrame, component: str, n: int = 6) -> pd.Series:
    s = components.sort_values(component)
    return pd.Series({"low end": "; ".join(s.name.head(n)), "high end": "; ".join(s.name.tail(n)[::-1])}, name=component)


def user_axis_check(svd: AlgoBase, items: pd.DataFrame, loved: pd.Series, heavy: float, light: float) -> float:
    """Correlation between each user's factor-based preference for heavy over light games (q_i·p_u, biases left out)
    and the actual complexity of the games they love."""
    m: Any = svd
    interaction = m.qi @ m.pu.T
    weight = np.nan_to_num(items.weight.to_numpy())
    preference = pd.Series(interaction[weight >= heavy].mean(axis=0) - interaction[(weight > 0) & (weight < light)].mean(axis=0))
    return float(preference.corr(loved))


def neighbours(knn: AlgoBase, frame: Frame, items: pd.DataFrame, probes: Sequence[tuple[str, int]], k: int = 5) -> pd.DataFrame:
    m: Any = knn
    present = [(label, gid) for label, gid in probes if gid in frame.item_index.index]
    return pd.DataFrame({label: [items.name.iat[j] for j in m.get_neighbors(int(frame.item_index[gid]), k=k)] for label, gid in present},
                        index=[f"neighbour {i}" for i in range(1, k + 1)]).T


def biases(svd: AlgoBase, items: pd.DataFrame) -> pd.DataFrame:
    m: Any = svd
    return pd.DataFrame({"game": items.name.to_numpy(), "b_i": m.bi, "year": items.year.to_numpy(),
                         "ratings in training": items["training ratings"].to_numpy()}, index=pd.Index(items.ID.to_numpy(), name="ID"))
