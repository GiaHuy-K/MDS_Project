"""
Step 1 (Option B - ALTERNATIVE): Build 5 folds from a pre-split 70/15/15 dataset.
Input : data/dataset_final_70_15_15/train|val|test/Level_0..3/
Output: data/acne04_folds/fold_1..5/train|val|test/Level_0..3/

NOTE - there are two "01_" scripts, each a different entry point for Step 1:
  * 01_split_dataset_kfold.py         - use when you have the RAW ACNE04 release
                                       (JPEGImages/ + NNEW_*.txt fold files).
  * 01_split_dataset_from_existing.py (THIS FILE) - use when you only have a pre-split
                                       70/15/15 dataset; it pools all images and
                                       regenerates 5 stratified folds.
Run only ONE of them; both produce the same output layout.

⚠️ DATA LEAKAGE WARNING:
This script pools all images from train/val/test at the IMAGE level (not patient level)
and re-splits them using StratifiedKFold.  If the original dataset contains near-duplicate
images or multiple images from the same patient across different splits, pooling and
re-splitting may place near-duplicate / same-patient images into both the train and test
sets of a new fold — leading to data leakage and inflated evaluation metrics.

Recommendation: prefer 01_split_dataset_kfold.py (which uses the official ACNE04 fold
files) as the default unless you have a specific reason to pool and re-split.
"""

import os
import shutil
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from core.paths_config import DATA_DIR, KFOLD_DATASET_DIR
from core.model_utils import CLASS_NAMES

SOURCE_DIR  = DATA_DIR / "dataset_final_70_15_15"
OUTPUT_DIR  = KFOLD_DATASET_DIR
N_FOLDS     = 5
VAL_RATIO   = 0.15
RANDOM_SEED = 42


def collect_all_images():
    all_images = []

    for split in ["train", "val", "test"]:
        for cls in CLASS_NAMES:
            cls_dir = SOURCE_DIR / split / cls
            if not cls_dir.exists():
                print(f"  [SKIP] Not found: {cls_dir}")
                continue
            for img_file in sorted(cls_dir.iterdir()):
                if img_file.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                    all_images.append((str(img_file), cls))

    return all_images


def create_kfold_splits(all_images):
    paths  = np.array([x[0] for x in all_images])
    labels = np.array([x[1] for x in all_images])

    # Label-encode for StratifiedKFold
    label_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
    encoded      = np.array([label_to_idx[l] for l in labels])

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    folds = {}
    for fold_idx, (trainval_indices, test_indices) in enumerate(skf.split(paths, encoded), start=1):
        fold_name = f"fold_{fold_idx}"

        # Carve val out of trainval
        trainval_paths   = paths[trainval_indices]
        trainval_encoded = encoded[trainval_indices]

        # Stratified split trainval -> train + val
        train_idx, val_idx = train_test_split(
            np.arange(len(trainval_paths)),
            test_size=VAL_RATIO,
            random_state=RANDOM_SEED,
            stratify=trainval_encoded,
        )

        folds[fold_name] = {
            "train": [(trainval_paths[i], labels[trainval_indices[i]]) for i in train_idx],
            "val":   [(trainval_paths[i], labels[trainval_indices[i]]) for i in val_idx],
            "test":  [(paths[i], labels[i]) for i in test_indices],
        }

    return folds


def copy_fold_data(folds):
    if OUTPUT_DIR.exists():
        print(f"  Removing old folder: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    total_copied = 0
    for fold_name, splits in folds.items():
        print(f"\n--- {fold_name} ---")
        for split_name, items in splits.items():
            counts = defaultdict(int)
            for src_path, cls_name in items:
                dst_dir = OUTPUT_DIR / fold_name / split_name / cls_name
                dst_dir.mkdir(parents=True, exist_ok=True)

                dst_path = dst_dir / Path(src_path).name
                # Handle duplicate file names (an image may appear in both train/ and test/)
                if dst_path.exists():
                    stem = dst_path.stem
                    suffix = dst_path.suffix
                    counter = 1
                    while dst_path.exists():
                        dst_path = dst_dir / f"{stem}_dup{counter}{suffix}"
                        counter += 1

                shutil.copy2(src_path, dst_path)
                counts[cls_name] += 1
                total_copied += 1

            count_str = " | ".join(f"{c}: {counts[c]}" for c in CLASS_NAMES)
            print(f"  {split_name:5s}: {sum(counts.values()):4d} images ({count_str})")

    return total_copied


def main():
    print("=" * 60)
    print(" BUILD 5-FOLD CROSS-VALIDATION FROM EXISTING DATASET")
    print("=" * 60)
    print()
    print("⚠️  WARNING: This script pools images at the IMAGE level (not patient")
    print("   level). If the original dataset contains near-duplicate or same-patient")
    print("   images across splits, re-splitting may cause data leakage.")
    print("   Consider using 01_split_dataset_kfold.py (official ACNE04 folds) instead.")
    print()
    print(f"Source: {SOURCE_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Folds : {N_FOLDS}")

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Dataset not found: {SOURCE_DIR}")

    print("\n--- Collecting all images ---")
    all_images = collect_all_images()
    print(f"  Total: {len(all_images)} images")

    # Statistics
    class_counts = defaultdict(int)
    for _, cls in all_images:
        class_counts[cls] += 1
    for cls in CLASS_NAMES:
        print(f"  {cls}: {class_counts[cls]} images")

    print(f"\n--- Creating {N_FOLDS} stratified folds ---")
    folds = create_kfold_splits(all_images)

    print(f"\n--- Copying images into fold structure ---")
    total = copy_fold_data(folds)

    print(f"\n{'=' * 60}")
    print(f" DONE! Copied {total} images into {N_FOLDS} folds.")
    print(f" Output folder: {OUTPUT_DIR}")
    print(f" Layout: {OUTPUT_DIR}/fold_X/train|val|test/Level_0..3/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
