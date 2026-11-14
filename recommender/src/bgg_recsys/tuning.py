"""Hyperparameter search. Each function returns the chosen settings and the table they were chosen from.

Surprise's GridSearchCV (3-fold CV on the training ratings) tunes item-KNN and SVD. What it doesn't cover,
the number of SVD epochs and EASE, is chosen on the frame's validation split. The test sets are never used.
Results are cached on disk, keyed by the frame spec and the grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from surprise import SVD, KNNBaseline
from surprise.model_selection import GridSearchCV

from . import metrics
from ._cache import memory
from .config import FrameSpec, Grids
from .data import as_testset, load_frame, to_dataset
from .evaluation import mean_precision, ranking_task
from .models import EASE, Name


@dataclass(frozen=True)
class TuningResult:
    best: dict[str, Any]
    table: pd.DataFrame


def _cv_table(search: GridSearchCV, params: list[str]) -> pd.DataFrame:
    res = pd.DataFrame(search.cv_results)
    out = pd.DataFrame({p: res[f"param_{p}"] for p in params})
    out["CV RMSE"], out["CV RMSE std"] = res.mean_test_rmse, res.std_test_rmse
    return out.sort_values("CV RMSE")


@memory.cache
def tune_item_knn(spec: FrameSpec, grids: Grids) -> TuningResult:
    """Neighbourhood size and similarity measure of item-based KNNBaseline."""
    grid = {"k": list(grids.knn_k), "verbose": [False],
            "sim_options": {"name": list(grids.knn_similarities), "user_based": [False]}}
    search = GridSearchCV(KNNBaseline, grid, measures=["rmse"], cv=grids.cv_folds, n_jobs=-1)
    search.fit(to_dataset(load_frame(spec).train))
    table = _cv_table(search, ["k", "sim_options"])
    table.insert(1, "similarity", [options["name"] for options in table.pop("sim_options")])
    return TuningResult(search.best_params["rmse"], table)


@memory.cache
def svd_learning_curve(spec: FrameSpec, grids: Grids) -> TuningResult:
    """Training and validation RMSE of SVD (100 factors) by epochs, for each regularisation strength.
    Best = the epoch count with the lowest validation RMSE under the strongest regularisation."""
    validation = load_frame(spec).validation
    held_out = as_testset(validation.test)
    train_sample = as_testset(validation.train.sample(min(100_000, len(validation.train)), random_state=spec.seed))
    rows = []
    for reg in grids.svd_learning_curve_reg:
        for epochs in grids.svd_learning_curve_epochs:
            algo = SVD(n_factors=100, n_epochs=epochs, reg_all=reg, random_state=spec.seed).fit(validation.trainset)
            for split, testset in (("training", train_sample), ("validation", held_out)):
                rows.append({"reg_all": reg, "epochs": epochs, "split": split, "RMSE": metrics.rating_metrics(algo.test(testset))["RMSE"]})
    curve = pd.DataFrame(rows)
    strongest = curve.loc[(curve.reg_all == max(grids.svd_learning_curve_reg)) & (curve.split == "validation")]
    return TuningResult({"n_epochs": int(strongest.sort_values("RMSE", kind="stable").epochs.to_numpy()[0])}, curve)


@memory.cache
def tune_svd(spec: FrameSpec, grids: Grids, n_epochs: int) -> TuningResult:
    """Number of factors and regularisation of SVD at a fixed number of epochs."""
    grid = {"n_factors": list(grids.svd_factors), "reg_all": list(grids.svd_reg), "n_epochs": [n_epochs],
            "lr_all": [grids.svd_lr], "random_state": [spec.seed]}
    search = GridSearchCV(SVD, grid, measures=["rmse"], cv=grids.cv_folds, n_jobs=-1)
    search.fit(to_dataset(load_frame(spec).train))
    return TuningResult(search.best_params["rmse"], _cv_table(search, ["n_factors", "reg_all"]))


@memory.cache
def tune_ease(spec: FrameSpec, grids: Grids, liked: float, k: int) -> TuningResult:
    """EASE regularisation, and whether its input is the games a user liked or every game they rated,
    by precision@k on the validation split."""
    validation = load_frame(spec).validation
    task = ranking_task(validation, validation.test, liked)
    rows = []
    for reg in grids.ease_reg:
        for source, threshold in (("liked games", liked), ("all rated games", None)):
            model = EASE(reg, threshold).fit(validation.ratings)
            rows.append({"reg": reg, "input": source, f"validation P@{k}": mean_precision({Name.EASE: model}, task, k)[Name.EASE]})
    table = pd.DataFrame(rows)
    best = table.sort_values(f"validation P@{k}", ascending=False, kind="stable").iloc[0]
    return TuningResult({"reg": float(best.reg), "liked_threshold": liked if best.input == "liked games" else None}, table)
