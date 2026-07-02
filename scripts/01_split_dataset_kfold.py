"""
Step 1 (Option A - PRIMARY): Split the ACNE04 dataset into 5 folds using the
original .txt fold files (NNEW_trainval_<fold>.txt / NNEW_test_<fold>.txt).

Output: data/acne04_folds/fold_1..5/train|val|test/Level_0..3/

NOTE - there are two "01_" scripts, each a different entry point for Step 1:
  * 01_split_dataset_kfold.py         (THIS FILE) - use when you have the RAW ACNE04
                                       release (JPEGImages/ + NNEW_*.txt fold files).
  * 01_split_dataset_from_existing.py - use when you only have a pre-split
                                       70/15/15 dataset and want to rebuild 5 folds.
Run only ONE of them; both produce the same output layout.
"""

import os
import re
import glob
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.model_selection import train_test_split

from core.paths_config import IMAGE_SOURCE_DIR, TXT_DIR, KFOLD_DATASET_DIR

TRAINVAL_PATTERN      = "NNEW_trainval_*.txt"
TEST_PATTERN          = "NNEW_test_*.txt"
VAL_RATIO_OF_TRAINVAL = 0.15
RANDOM_SEED           = 42
VALID_LEVELS          = {0, 1, 2, 3}


def parse_line(line):
    """Parse one line of a .txt file -> (img_name, level) or None."""
    line = line.strip()
    if not line:
        return None

    parts = line.replace(",", " ").split()
    if len(parts) < 2:
        return None

    img_name = os.path.basename(parts[0])
    label_raw = parts[1]  # column 2 = severity (0-3); do NOT use parts[-1] (lesion count)

    match = re.search(r"\d+", label_raw)
    if not match:
        return None
    level = int(match.group())

    if level not in VALID_LEVELS:
        return None

    return img_name, level



def load_fold_file(txt_path):
    """Read a .txt file and return list[(img_name, level)]."""
    items = []
    skipped = 0
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parsed = parse_line(line)
            if parsed:
                items.append(parsed)
            else:
                skipped += 1
    if skipped:
        print(f"    Warning: skipped {skipped} unparseable line(s) in {os.path.basename(txt_path)}")
    return items


def copy_images(items, split_name, fold_dir, src_dir, out_dir):
    count = 0
    for img_name, level in items:
        src_path = os.path.join(src_dir, img_name)
        if not os.path.exists(src_path):
            print(f"    Skipping (image not found): {img_name}")
            continue
        dst_dir = os.path.join(out_dir, fold_dir, split_name, f"Level_{level}")
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy(src_path, os.path.join(dst_dir, img_name))
        count += 1
    return count


def find_fold_files(pattern):
    files = sorted(glob.glob(os.path.join(TXT_DIR, pattern)))
    return files


def extract_fold_id(filename):
    match = re.search(r"(\d+)", os.path.basename(filename))
    return match.group(1) if match else os.path.splitext(os.path.basename(filename))[0]


def main():
    print("--- Splitting dataset by ACNE04's original K-FOLD files ---")
    print(f"Image source: {IMAGE_SOURCE_DIR}")
    print(f"Txt folder  : {TXT_DIR}")
    print(f"Output      : {KFOLD_DATASET_DIR}")

    if not os.path.exists(IMAGE_SOURCE_DIR):
        raise FileNotFoundError(f"Image folder not found: {IMAGE_SOURCE_DIR}")

    trainval_files = find_fold_files(TRAINVAL_PATTERN)
    test_files = find_fold_files(TEST_PATTERN)

    if not trainval_files or not test_files:
        raise FileNotFoundError(
            f"No fold files found in {TXT_DIR}.\n"
            f"  Searched with patterns: '{TRAINVAL_PATTERN}' and '{TEST_PATTERN}'.\n"
            f"  If your file names differ, edit TRAINVAL_PATTERN / TEST_PATTERN above."
        )

    if os.path.exists(KFOLD_DATASET_DIR):
        print("Cleaning up old output folder...")
        shutil.rmtree(KFOLD_DATASET_DIR)

    print(f"\nFound {len(trainval_files)} trainval file(s), {len(test_files)} test file(s).")

    for trainval_path in trainval_files:
        fold_id = extract_fold_id(trainval_path)
        fold_dir = f"fold_{fold_id}"

        matching_test = [t for t in test_files if extract_fold_id(t) == fold_id]
        if not matching_test:
            print(f"  Skipping fold {fold_id}: no matching test file found.")
            continue
        test_path = matching_test[0]

        print(f"\n--- Fold {fold_id} ---")
        trainval_items = load_fold_file(trainval_path)
        test_items = load_fold_file(test_path)

        labels = [lv for _, lv in trainval_items]
        train_items, val_items = train_test_split(
            trainval_items,
            test_size=VAL_RATIO_OF_TRAINVAL,
            random_state=RANDOM_SEED,
            stratify=labels,
        )

        n_train = copy_images(train_items, "train", fold_dir, IMAGE_SOURCE_DIR, KFOLD_DATASET_DIR)
        n_val = copy_images(val_items, "val", fold_dir, IMAGE_SOURCE_DIR, KFOLD_DATASET_DIR)
        n_test = copy_images(test_items, "test", fold_dir, IMAGE_SOURCE_DIR, KFOLD_DATASET_DIR)

        print(f"  Train: {n_train} | Val: {n_val} | Test: {n_test}")

    print(f"\nDone! Output: {KFOLD_DATASET_DIR}")


if __name__ == "__main__":
    main()
