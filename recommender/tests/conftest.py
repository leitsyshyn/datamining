import numpy as np
import pandas as pd
import pytest

from bgg_recsys.config import FrameSpec
from bgg_recsys.data import Frame, build_frame


def random_ratings(n_users: int = 40, n_items: int = 30, density: float = 0.4, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = [(f"u{u}", 1000 + i, float(rng.integers(1, 11))) for u in range(n_users) for i in range(n_items) if rng.random() < density]
    return pd.DataFrame(rows, columns=["user", "ID", "rating"])


@pytest.fixture(scope="session")
def frame() -> Frame:
    past = random_ratings(seed=0)
    later = random_ratings(seed=1).merge(past[["user", "ID"]], on=["user", "ID"], how="left", indicator=True)
    later = later.loc[later._merge == "left_only", ["user", "ID", "rating"]]
    return build_frame(past, later, FrameSpec(n_games=30, min_user_ratings=5, n_users=40, seed=0))
