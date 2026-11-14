# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportReturnType=false, reportIndexIssue=false, reportOperatorIssue=false
# (pandas-stubs cannot follow groupby/agg/pivot chains; the logic here is covered by the notebook's outputs)
"""One function per figure of the notebook: prepared data in, figure saved under its name.

Intervals in the ranking charts are 95% bootstrap intervals that seaborn computes from per-user values,
with the same number of resamples and seed as the tables.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .style import (
    DIV,
    GROUP_COLORS,
    INK,
    INK2,
    SEQ,
    SERIES,
    SURFACE,
    group_legend,
    label_points,
    percent_axis,
    save,
)


def _hist(ax, values, **kw):
    sns.histplot(x=np.asarray(values), ax=ax, color=SERIES[0], edgecolor=SURFACE, linewidth=0.5, alpha=1, **kw)


# --------------------------------------------------------------------------- data and EDA
def normalisation(train: pd.DataFrame) -> None:
    user_mean = train.groupby("user").rating.transform("mean")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    _hist(axes[0], train.groupby("user").rating.mean(), bins=40)
    axes[0].set(title="Mean rating per user (training data)", xlabel="user's mean rating", ylabel="users")
    _hist(axes[1], train.rating - user_mean, bins=60)
    axes[1].set(title="Ratings after mean-centring per user", xlabel="rating − user's mean", ylabel="ratings")
    save(fig, "1_normalisation")


def rating_distribution(ratings: pd.DataFrame, games: pd.DataFrame, min_ratings: int = 100) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    shares = ((ratings.rating * 2).round() / 2).value_counts(normalize=True).sort_index()
    axes[0].bar(shares.index, shares.to_numpy(), width=0.4, color=SERIES[0])
    axes[0].set(title="All ratings (rounded to 0.5)", xlabel="rating", ylabel="share of ratings", xticks=range(1, 11))
    percent_axis(axes[0].yaxis)
    _hist(axes[1], games.loc[games.n >= min_ratings, "mean"], bins=50)
    axes[1].set(title=f"Mean rating per game (games with ≥ {min_ratings} ratings)", xlabel="game's mean rating", ylabel="games")
    save(fig, "2_rating_distribution")


def users(users_: pd.DataFrame, generosity: pd.DataFrame, min_ratings: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.4))
    _hist(axes[0], users_.n, bins=np.unique(np.floor(np.logspace(0, np.log10(users_.n.max() + 1), 50))) - 0.5)  # integer edges: no empty bins
    axes[0].set(xscale="log", yscale="log", title="Ratings per user", xlabel="ratings (log)", ylabel="users (log)")
    _hist(axes[1], users_.loc[users_.n >= min_ratings, "mean"], bins=50)
    axes[1].set(title=f"Mean rating per user (≥ {min_ratings} ratings)", xlabel="user's mean rating", ylabel="users")
    x = np.arange(len(generosity))
    axes[2].fill_between(x, generosity[0.25], generosity[0.75], color=SERIES[0], alpha=0.18, linewidth=0)
    axes[2].plot(x, generosity[0.5], marker="o", markersize=5, color=SERIES[0])
    axes[2].set_xticks(x, [str(i).replace(".0", "") for i in generosity.index], rotation=20)
    axes[2].set(title="User mean rating by activity (median, IQR)", xlabel="ratings per user", ylabel="user's mean rating")
    save(fig, "2_users")


def long_tail(curve: pd.DataFrame, games: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].plot(curve["share of games"], curve["share of ratings"], color=SERIES[0])
    for q in (0.01, 0.1):
        i = int(q * len(curve)) - 1
        x, y = curve["share of games"].iat[i], curve["share of ratings"].iat[i]
        axes[0].scatter([x], [y], s=30, color=SERIES[0], zorder=3, edgecolor=SURFACE, linewidth=2)
        axes[0].annotate(f"top {q:.0%} of games → {y:.0%} of ratings", (x, y), xytext=(12, -14), textcoords="offset points", color=INK2, fontsize=9)
    axes[0].set(title="Concentration of ratings across games", xlabel="share of games (most rated first)", ylabel="cumulative share of ratings")
    percent_axis(axes[0].xaxis)
    percent_axis(axes[0].yaxis)
    hb = axes[1].hexbin(np.log10(games.n), games["mean"], gridsize=45, cmap=SEQ, mincnt=1, bins="log", linewidths=0)
    axes[1].set(title="Popularity vs mean rating (per game)", xlabel="number of ratings (log10)", ylabel="mean rating")
    fig.colorbar(hb, ax=axes[1], label="games (log)")
    save(fig, "2_long_tail")


def item_characteristics(games: pd.DataFrame, by_period: pd.Series, mechanics: pd.DataFrame, min_ratings: int = 100) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    g = games[games.n >= min_ratings].dropna(subset=["averageweight"])
    axes[0].hexbin(g.averageweight, g["mean"], gridsize=40, cmap=SEQ, mincnt=1, bins="log", linewidths=0)
    axes[0].set(title="Complexity vs mean rating", xlabel="BGG weight (1 = light, 5 = heavy)", ylabel="game's mean rating")
    axes[1].plot(by_period.index, by_period.to_numpy(), marker="o", markersize=4, color=SERIES[0])
    axes[1].set(title="Median game rating by release period (5-year bins)", xlabel="year published", ylabel="median of game means")
    colors = [SERIES[1] if end == "lowest" else SERIES[0] for end in mechanics.end]
    axes[2].barh(mechanics.index, mechanics["mean"], color=colors, height=0.6)
    axes[2].set_xlim(mechanics["mean"].min() - 0.3, mechanics["mean"].max() + 0.1)
    axes[2].set(title="Mechanics: lowest vs highest mean rating", xlabel="mean of game means")
    axes[2].grid(axis="y", visible=False)
    save(fig, "2_item_characteristics")


def drift(delta: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.2))
    _hist(ax, delta.clip(-5, 5), bins=np.arange(-5.25, 5.5, 0.5))
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set(title="Change of re-rated games, 2022 → 2025 (clipped at ±5)", xlabel="new rating − old rating", ylabel="pairs")
    save(fig, "2_drift")


# --------------------------------------------------------------------------- tuning
def learning_curve(curve: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.6))
    data = curve.assign(regularisation=curve.reg_all.map(lambda r: f"reg_all = {r:g}"))
    sns.lineplot(data, x="epochs", y="RMSE", hue="regularisation", style="split", markers=True, dashes={"validation": "", "training": (4, 2)}, ax=ax)
    ax.set(title="SVD learning curve (100 factors)", xticks=sorted(curve.epochs.unique()))
    ax.legend(fontsize=8.5)
    save(fig, "3_learning_curve")


def tuning(svd_grid: pd.DataFrame, knn_grid: pd.DataFrame, epochs: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    sns.lineplot(svd_grid.astype({"n_factors": str}), x="reg_all", y="CV RMSE", hue="n_factors", marker="o", ax=axes[0])
    axes[0].set(title=f"SVD: CV RMSE vs regularisation ({epochs} epochs)")
    axes[0].legend(title="factors")
    sns.lineplot(knn_grid, x="k", y="CV RMSE", hue="similarity", marker="o", ax=axes[1])
    axes[1].set(title="Item-KNN: CV RMSE vs k", xlabel="k (neighbours)", xticks=sorted(knn_grid.k.unique()))
    sns.move_legend(axes[1], "upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    save(fig, "3_tuning")


# --------------------------------------------------------------------------- evaluation
def _wrap(name: str) -> str:
    return name.replace(": ", "\n").replace(" (", "\n(").replace("Average of ", "Average of\n")


def rmse(table: pd.DataFrame, models: Sequence[str]) -> None:
    """RMSE on the random split (same users as the future set) and on the future split."""
    rows = table.loc[list(models)]
    tidy = pd.concat([pd.DataFrame({"model": [_wrap(m) for m in models], "split": split, "RMSE": rows[col].to_numpy()})
                      for split, col in (("random split (same users)", "RMSE random (future users)"), ("future split", "RMSE future"))])
    fig, ax = plt.subplots(figsize=(14, 4.2))
    sns.barplot(tidy, x="model", y="RMSE", hue="split", ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=8, color=INK2)
    ax.set_ylim(tidy.RMSE.min() - 0.06, tidy.RMSE.max() + 0.04)
    ax.set(title="RMSE on the random and the future test set (same users; lower is better)", xlabel="")
    ax.grid(axis="x", visible=False)
    save(fig, "4_rmse")


def error_profile(profile: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].plot(profile.index, profile.index, color=INK2, linestyle="--", linewidth=1, label="perfect prediction")
    axes[0].plot(profile.index, profile.mean_prediction, marker="o", markersize=5, color=SERIES[0], label="mean prediction")
    axes[0].set(title="Mean prediction by true rating (SVD tuned)", xlabel="true rating (rounded)", ylabel="predicted rating", xticks=range(1, 11))
    axes[0].legend()
    axes[1].bar(profile.index, profile.RMSE, color=SERIES[0], width=0.6)
    share = axes[1].twinx()
    share.grid(False)
    share.plot(profile.index, profile.n / profile.n.sum(), color=SERIES[1], marker="o", markersize=4)
    share.set_ylabel("share of test ratings", color=SERIES[1])
    percent_axis(share.yaxis)
    axes[1].set(title="RMSE by true rating", xlabel="true rating (rounded)", ylabel="RMSE", xticks=range(1, 11))
    save(fig, "4_error_profile")


def precision(tidy: pd.DataFrame, groups: Mapping[str, str], k: int, resamples: int, seed: int) -> None:
    """Mean P@k per model with 95% intervals (random split), coloured by model group."""
    order = tidy.groupby("model")[f"P@{k}"].mean().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(9, 4.4))
    sns.barplot(tidy.assign(group=tidy.model.map(groups)), x=f"P@{k}", y="model", hue="group", palette=GROUP_COLORS, order=order,
                errorbar=("ci", 95), n_boot=resamples, seed=seed, err_kws={"color": INK2, "linewidth": 1}, capsize=0.2, legend=False, ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=18, fontsize=9, color=INK2)
    ax.set(title=f"Precision@{k} over the full catalogue, random split (95% CI)", ylabel="")
    ax.grid(axis="y", visible=False)
    group_legend(ax, loc="lower right")
    save(fig, "4_ranking")


def random_vs_future(tidy: pd.DataFrame, models: Sequence[str], k: int, resamples: int, seed: int, n_users: int) -> None:
    fig, ax = plt.subplots(figsize=(13, 4.2))
    sns.barplot(tidy, x="model", y=f"P@{k}", hue="split", order=list(models), errorbar=("ci", 95), n_boot=resamples, seed=seed,
                err_kws={"color": INK2, "linewidth": 1}, capsize=0.15, ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=14, fontsize=8, color=INK2)
    ax.set_xticks(range(len(models)), [_wrap(m) for m in models])
    ax.set(title=f"Precision@{k}: random vs future split, same {n_users:,} users (95% CI)", xlabel="")
    ax.grid(axis="x", visible=False)
    save(fig, "4_future")


# --------------------------------------------------------------------------- recommendations
def beyond_accuracy(models: pd.DataFrame, rerank: pd.DataFrame, own_novelty: float, groups: Mapping[str, str], reranked: str) -> None:
    """Accuracy vs novelty and vs coverage for every model, with the re-ranking path of one model."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, x, label in ((axes[0], "novelty (bits)", "novelty (bits, higher = less obvious)"), (axes[1], "coverage", "catalogue coverage")):
        ax.plot(rerank[x], rerank["P@k"], color=SERIES[2], linewidth=1.5, linestyle=":", zorder=2)
        ax.scatter(rerank[x].iloc[1:], rerank["P@k"].iloc[1:], s=22, color=SERIES[2], zorder=3)
        labelled = rerank.iloc[1:] if x == "coverage" else rerank.iloc[-1:]  # on the novelty axis the β points nearly coincide
        for k, (beta, row) in enumerate(labelled.iterrows()):
            ax.annotate(f"β={beta:g}", (row[x], row["P@k"]), xytext=(0, 7 if k % 2 == 0 else -13), textcoords="offset points",
                        fontsize=8, color=SERIES[2], ha="center")
        ax.scatter(models[x], models["P@k"], s=60, color=[GROUP_COLORS[groups[m]] for m in models.index], edgecolor=SURFACE, linewidth=2, zorder=4)
        label_points(ax, models[x].to_numpy(), models["P@k"].to_numpy(), [str(m) for m in models.index])
        ax.set(xlabel=label, ylabel="precision (accuracy)")
    axes[0].axvline(own_novelty, color=INK2, linestyle="--", linewidth=1)
    axes[0].annotate("novelty of what users actually rate", (own_novelty, models["P@k"].max()), xytext=(4, -4), textcoords="offset points", fontsize=8.5, color=INK2)
    percent_axis(axes[1].xaxis)
    axes[0].set_title(f"Accuracy vs novelty (dotted: {reranked} re-ranked)")
    axes[1].set_title("Accuracy vs catalogue coverage")
    group_legend(axes[1], marker=True, loc="center right")
    save(fig, "5_beyond_accuracy")


# --------------------------------------------------------------------------- interpretation
def interactions(matrix: pd.DataFrame, blocks: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.1, 1]})
    sns.heatmap(matrix.to_numpy(), cmap=SEQ, vmin=1, vmax=10, ax=axes[0], cbar_kws={"label": "rating", "shrink": 0.8})
    axes[0].set(title=f"Ratings: {len(matrix)} most active users × {matrix.shape[1]} most-rated games", xlabel="games (most rated →)", ylabel="users (most active ↓)")
    sns.heatmap(blocks, cmap=SEQ, ax=axes[1], cbar_kws={"label": "share of cells rated", "shrink": 0.8, "format": "{x:.0%}"})
    axes[1].set(title="Fill rate of the whole frame (blocks of users × games)", xlabel="games, by popularity →", ylabel="users, by activity ↓")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    save(fig, "6_interactions")


def taste_space(components: pd.DataFrame, correlations: pd.DataFrame, labelled: Sequence[int]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw={"width_ratios": [1.25, 1]})
    points = axes[0].scatter(components.PC1, components.PC2, c=components.averageweight, cmap=SEQ, s=10, linewidths=0, alpha=0.9)
    shown = [g for g in labelled if g in components.index]
    axes[0].set_xlim(components.PC1.min() - 0.05, components.PC1.max() + 0.45)
    label_points(axes[0], components.loc[shown, "PC1"].to_numpy(), components.loc[shown, "PC2"].to_numpy(),
                 [str(components.at[g, "name"]).split(":")[0] for g in shown], min_gap_px=15)
    fig.colorbar(points, ax=axes[0], label="complexity (BGG weight)")
    axes[0].set(title="Games in the SVD taste space (PC1 × PC2 of q_i)", xlabel="PC1", ylabel="PC2")
    annotations = correlations.map(lambda v: f"{v:.2f}" if abs(v) >= 0.25 else "")
    sns.heatmap(correlations, cmap=DIV, vmin=-0.8, vmax=0.8, annot=annotations, fmt="", annot_kws={"fontsize": 7.5, "color": INK},
                cbar_kws={"label": "Spearman correlation", "shrink": 0.8}, ax=axes[1])
    axes[1].set_title("What the components correlate with (metadata unseen by the model)")
    save(fig, "6_latent_factors")


def biases(user_bias: np.ndarray, game_bias: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    _hist(axes[0], user_bias, bins=60)
    axes[0].set(title="User bias b_u (rating generosity)", xlabel="b_u (rating points above / below the global mean)", ylabel="users")
    _hist(axes[1], game_bias, bins=60)
    axes[1].set(title="Game bias b_i (quality after removing user habits)", xlabel="b_i", ylabel="games")
    save(fig, "6_biases")
