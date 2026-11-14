"""Download the BoardGameGeek Reviews dataset (Kaggle: jvanelteren/boardgamegeek-reviews).

Fetches the ratings snapshots of February 2025 (26M) and January 2022 (19M) plus the game
metadata into data/raw/, then converts each ratings CSV (1.6-2.2 GB, with free-text comments)
into a slim Parquet file holding only the columns the recommender needs.

Usage: uv run python scripts/download_data.py
Requires a Kaggle API token (~/.kaggle/access_token or ~/.kaggle/kaggle.json).
"""

import subprocess
import zipfile
from pathlib import Path

import pandas as pd

DATASET = "jvanelteren/boardgamegeek-reviews"
FILES = ["games_detailed_info2025.csv", "bgg-26m-reviews.csv", "bgg-19m-reviews.csv"]
# ratings CSV -> slim parquet. The 2022 snapshot (19m) is used for the temporal split:
# pairs present in 2025 but not in 2022 are ratings added between Jan 2022 and Feb 2025.
PARQUETS = {"bgg-26m-reviews.csv": "ratings.parquet", "bgg-19m-reviews.csv": "ratings_2022.parquet"}

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


def download(name: str) -> Path:
    dest = RAW / name
    if dest.exists():
        print(f"skip   {name} (already present)")
        return dest
    print(f"fetch  {name}")
    subprocess.run(["kaggle", "datasets", "download", DATASET, "-f", name, "-p", str(RAW)], check=True)
    archive = RAW / f"{name}.zip"
    if archive.exists():  # Kaggle zips large single-file downloads
        with zipfile.ZipFile(archive) as zf:
            zf.extract(name, RAW)
        archive.unlink()
    return dest


def to_parquet(csv_path: Path, out_name: str) -> None:
    out = PROCESSED / out_name
    if out.exists():
        print(f"skip   {out.name} (already present)")
        return
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    keep = [c for c in header if c.lower() != "comment" and not c.startswith("Unnamed")]
    print(f"convert {csv_path.name} -> {out.name} (columns: {keep})")
    df = pd.read_csv(csv_path, usecols=keep, engine="pyarrow")
    df.to_parquet(out, index=False)
    print(f"wrote  {out} ({len(df):,} rows)")


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    paths = {name: download(name) for name in FILES}
    for csv_name, out_name in PARQUETS.items():
        to_parquet(paths[csv_name], out_name)
    print("done")


if __name__ == "__main__":
    main()
