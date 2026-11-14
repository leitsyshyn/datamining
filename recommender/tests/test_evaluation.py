import numpy as np
import pytest

from bgg_recsys.evaluation import (
    RankingTask,
    evaluate_rankings,
    long_tail_task,
    ranking_task,
    ranking_tasks,
    top_k,
)
from bgg_recsys.models import Name, Popularity


def task(relevant, seen=None):
    users = tuple(relevant)
    seen = seen or {u: frozenset() for u in users}
    return RankingTask(users, {u: frozenset(seen[u]) for u in users}, {u: frozenset(r) for u, r in relevant.items()})


def test_top_k_excludes_seen_and_orders_by_score():
    scores = np.array([0.1, 0.5, 0.2, 0.9, 0.3])
    assert top_k(lambda u: scores, task({0: {1}}, seen={0: {3}}), k=2)[0].tolist() == [1, 4]


def test_restrict_keeps_only_given_users():
    t = task({0: {1}, 1: {2}, 2: {3}}).restrict([0, 2])
    assert t.users == (0, 2) and set(t.relevant) == {0, 2}


def test_ranking_task_excludes_training_and_extra_games(frame):
    t = ranking_task(frame, frame.test, threshold=8)
    for u in t.users:
        assert set(frame.ratings[u].indices) <= t.seen[u]
        assert not t.relevant[u] & set(frame.ratings[u].indices)
    future = ranking_tasks(frame, threshold=8)["future"]
    test_items = {(frame.inner_user(u), frame.inner_item(i)) for u, i in zip(frame.test.user, frame.test.ID)}
    assert all(i in future.seen[u] for u, i in test_items if u in future.seen)


def test_long_tail_removes_head_from_candidates_and_relevant():
    popularity = np.array([100.0, 50, 10, 5, 1])
    t, n_head = long_tail_task(task({0: {0, 3}, 1: {0}}), popularity, head_share=0.5)
    assert n_head == 1 and t.users == (0,)
    assert t.relevant[0] == {3} and 0 in t.seen[0]


def test_ranking_results_summary_and_compare(frame):
    tasks = ranking_tasks(frame, threshold=8)
    pop = Popularity(frame.ratings)
    results = evaluate_rankings({Name.POPULARITY: pop, Name.RANDOM: pop}, tasks, k=3)
    summary = results.summary("random")
    assert summary.loc[Name.POPULARITY, "P@3"] == pytest.approx(summary.loc[Name.RANDOM, "P@3"])
    diff = results.compare("random", [(Name.POPULARITY, Name.RANDOM)])
    assert diff.iloc[0]["Δ P@3"] == 0 and diff.iloc[0]["significant"] == "no"
    assert set(results.tidy(["random"], [Name.POPULARITY]).columns) >= {"user", "model", "split", "P@3"}
