"""
Buoc 2: EDA — kiem tra phan bo du lieu, phat hien anh loi, ve bieu do.
Output: eda_outputs/class_distribution.png, sample_images.png
"""

import os

import matplotlib.pyplot as plt
from PIL import Image

from paths_config import KFOLD_DATASET_DIR, EDA_OUTPUT_DIR, ensure_dirs
from model_utils import CLASS_NAMES

FOLD_NAME   = "fold_1"
DATA_DIR    = str(KFOLD_DATASET_DIR / FOLD_NAME)
SPLITS      = ["train", "val", "test"]
OUTPUT_DIR  = str(EDA_OUTPUT_DIR)


def count_images_per_class(data_dir):
    counts = {}
    for split in SPLITS:
        split_dir = os.path.join(data_dir, split)
        counts[split] = {}
        for cls in CLASS_NAMES:
            cls_dir = os.path.join(split_dir, cls)
            if os.path.exists(cls_dir):
                counts[split][cls] = len(os.listdir(cls_dir))
            else:
                counts[split][cls] = 0
    return counts


def check_broken_images(data_dir):
    broken = []
    sizes = []

    for split in SPLITS:
        for cls in CLASS_NAMES:
            cls_dir = os.path.join(data_dir, split, cls)
            if not os.path.exists(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                fpath = os.path.join(cls_dir, fname)
                try:
                    with Image.open(fpath) as img:
                        img.verify()
                    with Image.open(fpath) as img:
                        sizes.append(img.size)
                except Exception as e:
                    broken.append((fpath, str(e)))

    return broken, sizes


def plot_class_distribution(counts, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))

    x = range(len(CLASS_NAMES))
    width = 0.25

    for i, split in enumerate(SPLITS):
        values = [counts[split][cls] for cls in CLASS_NAMES]
        positions = [pos + i * width for pos in x]
        ax.bar(positions, values, width, label=split)

    ax.set_xticks([pos + width for pos in x])
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylabel("So luong anh")
    ax.set_title("Phan bo so luong anh theo class va split")
    ax.legend()

    plt.tight_layout()
    save_path = os.path.join(output_dir, "class_distribution.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Da luu bieu do phan bo lop -> {save_path}")


def plot_sample_images(data_dir, output_dir, n_samples=4):
    fig, axes = plt.subplots(len(CLASS_NAMES), n_samples, figsize=(n_samples * 2.5, len(CLASS_NAMES) * 2.5))

    for row, cls in enumerate(CLASS_NAMES):
        cls_dir = os.path.join(data_dir, "train", cls)
        if not os.path.exists(cls_dir):
            continue
        files = os.listdir(cls_dir)[:n_samples]

        for col, fname in enumerate(files):
            img_path = os.path.join(cls_dir, fname)
            img = Image.open(img_path)
            ax = axes[row, col] if len(CLASS_NAMES) > 1 else axes[col]
            ax.imshow(img)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(cls, fontsize=12)
                ax.text(-0.1, 0.5, cls, transform=ax.transAxes,
                        rotation=90, va="center", ha="right", fontsize=12)

    plt.suptitle("Anh mau moi Level (tu tap train)")
    plt.tight_layout()
    save_path = os.path.join(output_dir, "sample_images.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Da luu anh mau -> {save_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("--- EDA: Dem so luong anh ---")
    counts = count_images_per_class(DATA_DIR)
    for split in SPLITS:
        total = sum(counts[split].values())
        print(f"\n{split.upper()} (tong {total} anh):")
        for cls in CLASS_NAMES:
            pct = counts[split][cls] / total * 100 if total > 0 else 0
            print(f"  {cls}: {counts[split][cls]} anh ({pct:.1f}%)")

    print("\n--- EDA: Kiem tra anh loi va kich thuoc ---")
    broken, sizes = check_broken_images(DATA_DIR)

    if broken:
        print(f"  Phat hien {len(broken)} anh loi:")
        for fpath, err in broken[:10]:
            print(f"    {fpath} -> {err}")
        if len(broken) > 10:
            print(f"    ... va {len(broken) - 10} file khac")
    else:
        print("  Khong phat hien anh loi.")

    if sizes:
        widths  = [s[0] for s in sizes]
        heights = [s[1] for s in sizes]
        print(f"  Kich thuoc anh: rong {min(widths)}-{max(widths)}px, "
              f"cao {min(heights)}-{max(heights)}px, "
              f"trung binh {sum(widths)//len(widths)}x{sum(heights)//len(heights)}px")

    print("\n--- EDA: Ve bieu do ---")
    plot_class_distribution(counts, OUTPUT_DIR)
    plot_sample_images(DATA_DIR, OUTPUT_DIR)

    print(f"\nHoan tat! Xem ket qua trong thu muc '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()
