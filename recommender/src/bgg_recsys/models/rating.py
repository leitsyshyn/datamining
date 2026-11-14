"""Rating-prediction pieces Surprise doesn't provide: a global-mean baseline and averaging two models."""

from __future__ import annotations

import numpy as np
from surprise import AlgoBase, Prediction, Trainset


class GlobalMean(AlgoBase):
    """Predict the training mean for every (user, game) pair."""

    def fit(self, trainset: Trainset):
        AlgoBase.fit(self, trainset)
        self.mu = trainset.global_mean
        return self

    def estimate(self, u, i):
        return self.mu


def average_predictions(*prediction_lists: list[Prediction]) -> list[Prediction]:
    """Average the estimates of several models made on the same test set."""
    first = prediction_lists[0]
    for other in prediction_lists[1:]:
        assert all(a.uid == b.uid and a.iid == b.iid for a, b in zip(first, other)), "predictions are not aligned"
    est = np.mean([[p.est for p in preds] for preds in prediction_lists], axis=0)
    return [Prediction(p.uid, p.iid, p.r_ui, e, {}) for p, e in zip(first, est)]
