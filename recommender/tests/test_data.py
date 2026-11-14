import numpy as np
import pytest

from bgg_recsys.data import Frame


def test_sparse_ratings_match_surprise_trainset(frame):
    ts = frame.trainset
    for u in range(ts.n_users):
        row = frame.ratings[u]
        assert dict(zip(row.indices.tolist(), row.data.tolist())) == {j: r for j, r in ts.ur[u]}


def test_id_mapping_round_trips(frame):
    assert (frame.inner_items(frame.items) == np.arange(len(frame.items))).all()
    assert frame.popularity.sum() == len(frame.train)
    assert frame.history_length.sum() == len(frame.train)


def test_check_passes_on_a_clean_frame(frame):
    assert frame.check()["no (user, game) pair in more than one set"] == "✓"


def test_check_catches_leakage(frame):
    leaky = Frame(spec=frame.spec, train=frame.train, test=frame.train.head(3), future=frame.future, catalogue=frame.catalogue)
    with pytest.raises(AssertionError, match="also in training"):
        leaky.check()


def test_validation_frame_splits_training_only(frame):
    v = frame.validation
    assert len(v.train) + len(v.test) == len(frame.train)
    assert set(zip(v.test.user, v.test.ID)) <= set(zip(frame.train.user, frame.train.ID))
