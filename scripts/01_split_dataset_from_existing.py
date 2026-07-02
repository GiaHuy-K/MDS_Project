"""
Buoc 1 (phuong an thay the): Tao 5-fold tu dataset da chia san 70/15/15.
Input: dataset_final_70_15_15/train|val|test/Level_0..3/
Output: dataset_acne04_folds/fold_1..5/train|val|test/Level_0..3/
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from paths_config import PROJECT_ROOT, KFOLD_DATASET_DIR
from model_utils import CLASS_NAMES

SOURCE_DIR  = PROJECT_ROOT / "dataset_final_70_15_15"
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
                print(f"  [SKIP] Khong tim thay: {cls_dir}")
                continue
            for img_file in sorted(cls_dir.iterdir()):
                if img_file.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                    all_images.append((str(img_file), cls))

    return all_images


def create_kfold_splits(all_images):
    paths  = np.array([x[0] for x in all_images])
    labels = np.array([x[1] for x in all_images])

    # Label encode cho StratifiedKFold
    label_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
    encoded      = np.array([label_to_idx[l] for l in labels])

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    folds = {}
    for fold_idx, (trainval_indices, test_indices) in enumerate(skf.split(paths, encoded), start=1):
        fold_name = f"fold_{fold_idx}"

        # Tach val tu trainval
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
        print(f"  Dang xoa thu muc cu: {OUTPUT_DIR}")
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
                # Xu ly trung ten file (co the anh tu train/ va test/ trung ten)
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
            print(f"  {split_name:5s}: {sum(counts.values()):4d} anh ({count_str})")

    return total_copied


def main():
    print("=" * 60)
    print(" TAO 5-FOLD CROSS-VALIDATION TU DATASET HIEN CO")
    print("=" * 60)
    print(f"Nguon : {SOURCE_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Folds : {N_FOLDS}")

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Khong tim thay dataset: {SOURCE_DIR}")

    print("\n--- Thu thap tat ca anh ---")
    all_images = collect_all_images()
    print(f"  Tong cong: {len(all_images)} anh")

    # Thong ke
    class_counts = defaultdict(int)
    for _, cls in all_images:
        class_counts[cls] += 1
    for cls in CLASS_NAMES:
        print(f"  {cls}: {class_counts[cls]} anh")

    print(f"\n--- Tao {N_FOLDS} stratified fold ---")
    folds = create_kfold_splits(all_images)

    print(f"\n--- Copy anh vao cau truc fold ---")
    total = copy_fold_data(folds)

    print(f"\n{'=' * 60}")
    print(f" HOAN TAT! Da copy {total} anh vao {N_FOLDS} fold.")
    print(f" Thu muc output: {OUTPUT_DIR}")
    print(f" Cau truc: {OUTPUT_DIR}/fold_X/train|val|test/Level_0..3/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
