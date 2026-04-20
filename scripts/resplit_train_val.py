"""Merge ISIC2018 train+val, then re-split into stratified, lesion-grouped train/val.

Test set is left untouched. Original train.csv and val.csv are backed up with a .bak suffix.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", default="Data/csv")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    train_csv = csv_dir / "train.csv"
    val_csv = csv_dir / "val.csv"
    groups_csv = csv_dir / "lesion_groupings.csv"

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    groups_df = pd.read_csv(groups_csv)[["image", "lesion_id"]]

    pool = pd.concat([train_df, val_df], ignore_index=True)
    pool = pool.drop_duplicates(subset="image", keep="first").reset_index(drop=True)

    pool = pool.merge(groups_df, on="image", how="left")
    # Val images have no lesion mapping — give each its own unique group id.
    missing = pool["lesion_id"].isna()
    pool.loc[missing, "lesion_id"] = "SOLO_" + pool.loc[missing, "image"].astype(str)

    labels = pool[CLASSES].to_numpy().argmax(axis=1)
    groups = pool["lesion_id"].to_numpy()

    n_splits = max(2, round(1 / args.val_fraction))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    train_idx, val_idx = next(sgkf.split(pool, labels, groups))

    new_train = pool.iloc[train_idx][["image", *CLASSES]].reset_index(drop=True)
    new_val = pool.iloc[val_idx][["image", *CLASSES]].reset_index(drop=True)

    # Safety check: no lesion overlap.
    overlap = set(pool.iloc[train_idx]["lesion_id"]) & set(pool.iloc[val_idx]["lesion_id"])
    assert not overlap, f"Lesion leakage detected: {len(overlap)} shared lesions"

    shutil.copy2(train_csv, train_csv.with_suffix(".csv.bak"))
    shutil.copy2(val_csv, val_csv.with_suffix(".csv.bak"))
    new_train.to_csv(train_csv, index=False)
    new_val.to_csv(val_csv, index=False)

    print(f"Pool size: {len(pool)} | Train: {len(new_train)} | Val: {len(new_val)}")
    print(f"Val fraction: {len(new_val) / len(pool):.3f}")
    print("\nPer-class counts:")
    counts = pd.DataFrame(
        {
            "train": new_train[CLASSES].sum().astype(int),
            "val": new_val[CLASSES].sum().astype(int),
        }
    )
    counts["val_pct"] = (counts["val"] / (counts["train"] + counts["val"]) * 100).round(1)
    print(counts)
    print(f"\nBackups: {train_csv.with_suffix('.csv.bak').name}, {val_csv.with_suffix('.csv.bak').name}")


if __name__ == "__main__":
    main()
