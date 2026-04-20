"""Move image files so images/train/ and images/val/ match train.csv and val.csv."""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "Data" / "images"
CSV_DIR = ROOT / "Data" / "csv"
EXT = ".jpg"


def index_existing() -> dict[str, Path]:
    """Map image_id -> current folder path, across train/ and val/."""
    index: dict[str, Path] = {}
    for split in ("train", "val"):
        for p in (IMG_DIR / split).glob(f"*{EXT}"):
            index[p.stem] = p
    return index


def main() -> None:
    train_ids = set(pd.read_csv(CSV_DIR / "train.csv")["image"])
    val_ids = set(pd.read_csv(CSV_DIR / "val.csv")["image"])

    overlap = train_ids & val_ids
    assert not overlap, f"IDs in both train.csv and val.csv: {len(overlap)}"

    existing = index_existing()
    moved_to_train = moved_to_val = 0
    missing: list[str] = []

    for img_id in train_ids:
        src = existing.get(img_id)
        if src is None:
            missing.append(img_id)
            continue
        dst = IMG_DIR / "train" / f"{img_id}{EXT}"
        if src != dst:
            shutil.move(str(src), str(dst))
            moved_to_train += 1

    for img_id in val_ids:
        src = existing.get(img_id)
        if src is None:
            missing.append(img_id)
            continue
        dst = IMG_DIR / "val" / f"{img_id}{EXT}"
        if src != dst:
            shutil.move(str(src), str(dst))
            moved_to_val += 1

    train_files = {p.stem for p in (IMG_DIR / "train").glob(f"*{EXT}")}
    val_files = {p.stem for p in (IMG_DIR / "val").glob(f"*{EXT}")}

    print(f"Moved -> train/: {moved_to_train}")
    print(f"Moved -> val/  : {moved_to_val}")
    print(f"train/ now has : {len(train_files)} files (csv: {len(train_ids)})")
    print(f"val/   now has : {len(val_files)} files (csv: {len(val_ids)})")

    extra_train = train_files - train_ids
    extra_val = val_files - val_ids
    if extra_train:
        print(f"WARNING: {len(extra_train)} extra files in train/ not in csv")
    if extra_val:
        print(f"WARNING: {len(extra_val)} extra files in val/ not in csv")
    if missing:
        print(f"WARNING: {len(missing)} CSV ids with no image file")

    assert train_files == train_ids, "train/ folder != train.csv"
    assert val_files == val_ids, "val/ folder != val.csv"
    print("\nOK: folders match CSVs exactly.")


if __name__ == "__main__":
    main()
