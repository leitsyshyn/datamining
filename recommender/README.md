# Board Game Recommender (Data Mining lab)

Collaborative-filtering recommender for **BoardGameGeek** ratings, built for the recommender-system lab
of a Data Mining course.

Models learn from what was known in **January 2022** and are tested twice: on a random 20% of
those 2022 ratings (the classic split) and on the ratings the same users actually gave
**later, by February 2025** (the future).

## Results

Frame: the 2,000 most-rated games of 2022 and 15,000 sampled users with ≥ 20 ratings on them.
P@10 on the random split is over all 14,432 test users; on the future, over the 6,876 users present in
both test sets. Those users score higher than average on the random split too (EASE 0.170 instead of 0.145);
the notebook compares the two splits on the same users.

| | Random split | Future |
|---|---|---|
| **Rating prediction (RMSE)**: `BaselineOnly` | 1.217 | 1.266 |
| Item-KNN (`KNNBaseline`, k = 40, `pearson_baseline`) | 1.175 | 1.278 |
| SVD (tuned: 300 factors, `reg_all` 0.1, 60 epochs) | **1.171** | **1.247** |
| Average of SVD and item-KNN | **1.167** | 1.255 |
| **Top-10 lists over the full catalogue (P@10)**: popularity | 0.077 | 0.039 |
| SVD (tuned) | 0.026 | 0.063 |
| Co-rating item-KNN | 0.118 | 0.061 |
| EASE | **0.145** | **0.087** |

- Collaborative filtering improves RMSE by 3–4% over the bias baseline; normalisation is essential
  (item-KNN on raw ratings is worse than `BaselineOnly`).
- Lower RMSE does not give better lists: every rating model loses to popularity on top-10 lists.
  EASE wins on both test sets (hit rate 69% on the random split) and on the long tail.
- A random split overstates popularity-like models: on the future, popularity collapses and the
  co-rating model's lead over SVD disappears. 55% of what users rated next was outside the 2022 catalogue.
- The conclusions hold on four other frames (other user samples, users with ≥ 50 ratings, 4,000 games).

The notebook has the full results, confidence intervals and interpretation.

## Layout

```
recommender/
├── notebooks/recommender.py     ← the notebook source (jupytext, percent format)
├── notebooks/recommender.ipynb  ← the same notebook with all outputs (the deliverable)
├── src/bgg_recsys/              ← the code the notebook calls
│   ├── config.py                every decision in one place: frame, grids, thresholds, paths
│   ├── data.py                  loading and cleaning snapshots, comparing them, the experimental frame
│   ├── models/                  rating models (Surprise), top-N scorers (popularity, co-rating KNN, EASE), the model catalogue
│   ├── tuning.py                grid searches and the SVD learning curve (cached on disk)
│   ├── evaluation.py            rating and ranking evaluation, breakdowns, one-game test, robustness
│   ├── metrics.py               RMSE/MAE, ranking metrics, bootstrap intervals, beyond-accuracy metrics
│   ├── analysis.py              EDA and interpretation tables
│   └── report/                  chart style, figures and table formatting
├── tests/                       data, model, metric and evaluation tests (pytest)
├── scripts/download_data.py     Kaggle download + conversion to Parquet
├── reports/figures/             all figures (written by the notebook)
└── data/                        raw CSVs and Parquet files (not in git)
```

## Running it

```bash
cd recommender
uv sync                                   # Python environment with everything below
uv run python scripts/download_data.py    # needs a Kaggle API token in ~/.kaggle; ~4 GB download
uv run pytest                             # 22 tests, under a second
uv run jupyter lab                        # open notebooks/recommender.ipynb
```

A full run takes about 40 minutes on a laptop (M1 Max), mostly tuning and the robustness appendix.
Their results are cached in `reports/.cache`, so later runs take about 20 minutes; delete that folder
to recompute from scratch. Headless: `cd notebooks && uv run papermill recommender.ipynb recommender.ipynb`;
`-p QUICK True` runs the same code on a small frame in a few minutes.

To edit the notebook as plain Python, change `notebooks/recommender.py` and sync it with
`uv run jupytext --sync notebooks/recommender.py`; outputs of unchanged cells are kept.

## Data

[BoardGameGeek Reviews](https://www.kaggle.com/datasets/jvanelteren/boardgamegeek-reviews) on Kaggle
(jvanelteren): `bgg-26m-reviews.csv` (Feb 2025), `bgg-19m-reviews.csv` (Jan 2022) and
`games_detailed_info2025.csv`. BoardGameGeek data is for non-commercial use with credit to BoardGameGeek.
