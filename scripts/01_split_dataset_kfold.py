"""
Buoc 1: Chia dataset ACNE04 thanh 5 fold dua tren file .txt goc
(NNEW_trainval_<fold>.txt / NNEW_test_<fold>.txt).

Output: dataset_acne04_folds/fold_1..5/train|val|test/Level_0..3/
"""

import os
import re
import glob
import shutil

from sklearn.model_selection import train_test_split

from paths_config import IMAGE_SOURCE_DIR, TXT_DIR, KFOLD_DATASET_DIR

TRAINVAL_PATTERN      = "NNEW_trainval_*.txt"
TEST_PATTERN          = "NNEW_test_*.txt"
VAL_RATIO_OF_TRAINVAL = 0.15
RANDOM_SEED           = 42
VALID_LEVELS          = {0, 1, 2, 3}


def parse_line(line):
    """Parse 1 dong file .txt -> (img_name, level) hoac None."""
    line = line.strip()
    if not line:
        return None

    parts = line.replace(",", " ").split()
    if len(parts) < 2:
        return None

    img_name = os.path.basename(parts[0])
    label_raw = parts[1]  # cot 2 = do nang (0-3), khong dung parts[-1] (lesion count)

    match = re.search(r"\d+", label_raw)
    if not match:
        return None
    level = int(match.group())

    if level not in VALID_LEVELS:
        return None

    return img_name, level



def load_fold_file(txt_path):
    """Doc file .txt, tra ve list[(img_name, level)]."""
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
        print(f"    Canh bao: bo qua {skipped} dong khong parse duoc trong {os.path.basename(txt_path)}")
    return items


def copy_images(items, split_name, fold_dir, src_dir, out_dir):
    count = 0
    for img_name, level in items:
        src_path = os.path.join(src_dir, img_name)
        if not os.path.exists(src_path):
            print(f"    Bo qua (khong tim thay anh): {img_name}")
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
    print("--- Dang chia dataset theo K-FOLD goc cua ACNE04 ---")
    print(f"Nguon anh : {IMAGE_SOURCE_DIR}")
    print(f"Thu muc txt: {TXT_DIR}")
    print(f"Output     : {KFOLD_DATASET_DIR}")

    if not os.path.exists(IMAGE_SOURCE_DIR):
        raise FileNotFoundError(f"Khong tim thay thu muc anh: {IMAGE_SOURCE_DIR}")

    trainval_files = find_fold_files(TRAINVAL_PATTERN)
    test_files = find_fold_files(TEST_PATTERN)

    if not trainval_files or not test_files:
        raise FileNotFoundError(
            f"Khong tim thay file fold trong {TXT_DIR}.\n"
            f"  Tim theo mau: '{TRAINVAL_PATTERN}' va '{TEST_PATTERN}'.\n"
            f"  Neu ten file cua ban khac, sua TRAINVAL_PATTERN / TEST_PATTERN o tren."
        )

    if os.path.exists(KFOLD_DATASET_DIR):
        print("Dang don dep folder cu...")
        shutil.rmtree(KFOLD_DATASET_DIR)

    print(f"\nTim thay {len(trainval_files)} file trainval, {len(test_files)} file test.")

    for trainval_path in trainval_files:
        fold_id = extract_fold_id(trainval_path)
        fold_dir = f"fold_{fold_id}"

        matching_test = [t for t in test_files if extract_fold_id(t) == fold_id]
        if not matching_test:
            print(f"  Bo qua fold {fold_id}: khong tim thay file test tuong ung.")
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

    print(f"\nHoan tat! Output: {KFOLD_DATASET_DIR}")


if __name__ == "__main__":
    main()