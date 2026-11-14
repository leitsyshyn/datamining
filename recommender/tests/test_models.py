import numpy as np
import pandas as pd
import pytest
from surprise import SVD, BaselineOnly, SVDpp

from bgg_recsys.config import FrameSpec
from bgg_recsys.data import Frame
from bgg_recsys.models import (
    EASE,
    CoRatingKNN,
    average_predictions,
    from_surprise,
    max_scorer_error,
    popularity_penalised,
)


@pytest.mark.parametrize("make", [lambda: BaselineOnly(verbose=False), lambda: SVD(random_state=0), lambda: SVD(biased=False, random_state=0),
                                  lambda: SVDpp(random_state=0, n_epochs=5)])
def test_vectorised_scorer_reproduces_surprise(frame, make):
    algo = make().fit(frame.trainset)
    assert max_scorer_error(algo, from_surprise(algo)) < 1e-9


def test_corating_scores_by_hand():
    # u0 liked games 0 and 1; u1 rated 1 and 2 (liked neither); u2 rated 0 and 2
    ratings = pd.DataFrame([("u0", 0, 9.0), ("u0", 1, 8.0), ("u1", 1, 5.0), ("u1", 2, 4.0), ("u2", 0, 6.0), ("u2", 2, 7.0)],
                           columns=["user", "ID", "rating"])
    frame = Frame(spec=FrameSpec(), train=ratings, test=ratings.iloc[:0], future=ratings.iloc[:0], catalogue=pd.Index([0, 1, 2]))
    model = CoRatingKNN().fit(frame.ratings)
    i = {raw: frame.inner_item(raw) for raw in (0, 1, 2)}
    u0, u1 = frame.inner_user("u0"), frame.inner_user("u1")
    assert model.weights[i[0], i[2]] == pytest.approx(0.5)      # every pair shares one rater out of two each
    assert model(u0)[i[2]] == pytest.approx(1.0, abs=1e-5)       # 0.5 via game 0 + 0.5 via game 1, plus the tiny tie-break
    np.testing.assert_allclose(model(u1), model.tie_break)       # liked nothing -> only the popularity tie-break
    assert model.because_of(u0, i[2]) in (i[0], i[1])


def test_ease_matches_constrained_ridge_regression(frame):
    """EASE's closed form must equal, column by column, a ridge regression of each game on all *other* games."""
    model = EASE(reg=5.0, liked_threshold=None).fit(frame.ratings)
    x = model.inputs
    assert np.allclose(np.diag(model.weights), 0)
    for j in (0, 7, 19):
        others = np.delete(np.arange(x.shape[1]), j)
        a = x[:, others]
        b = np.linalg.solve(a.T @ a + 5.0 * np.eye(len(others)), a.T @ x[:, j])
        np.testing.assert_allclose(model.weights[others, j], b, atol=1e-8)


def test_score_items_equals_scoring_a_user_with_those_likes(frame):
    for model in (CoRatingKNN().fit(frame.ratings), EASE(reg=5.0).fit(frame.ratings)):
        u = int(np.argmax(model.liked.sum(axis=1)))
        np.testing.assert_allclose(model(u), model.score_items(np.flatnonzero(model.liked[u])), rtol=1e-5, atol=1e-9)


def test_rerank_handles_constant_scores(frame):
    scores = popularity_penalised(lambda u: np.zeros(frame.ratings.shape[1]), frame.popularity, beta=0.5)(0)
    assert np.isfinite(scores).all()


def test_average_predictions(frame):
    testset = [(frame.trainset.to_raw_uid(0), frame.trainset.to_raw_iid(0), 5.0)]
    a = BaselineOnly(verbose=False).fit(frame.trainset).test(testset)
    b = SVD(random_state=0).fit(frame.trainset).test(testset)
    assert average_predictions(a, b)[0].est == pytest.approx((a[0].est + b[0].est) / 2)
