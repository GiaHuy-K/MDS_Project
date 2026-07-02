"""
Step 2: EDA - inspect data distribution, detect corrupt images, plot charts.
Output: outputs/eda/class_distribution.png, sample_images.png
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
from PIL import Image

from core.paths_config import KFOLD_DATASET_DIR, EDA_OUTPUT_DIR, ensure_dirs
from core.model_utils import CLASS_NAMES

SPLITS      = ["train", "val", "test"]


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
    ax.set_ylabel("Number of images")
    ax.set_title("Image count distribution by class and split")
    ax.legend()

    plt.tight_layout()
    save_path = os.path.join(output_dir, "class_distribution.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved class distribution chart -> {save_path}")


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

    plt.suptitle("Sample images per Level (from the train set)")
    plt.tight_layout()
    save_path = os.path.join(output_dir, "sample_images.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved sample images -> {save_path}")


def main():
    # Ensure base EDA output dir exists
    ensure_dirs(EDA_OUTPUT_DIR)

    # Discover folds (folders named like 'fold_1', 'fold_2', ...)
    fold_dirs = sorted([p.name for p in KFOLD_DATASET_DIR.iterdir() if p.is_dir() and p.name.startswith('fold_')])
    if not fold_dirs:
        print(f"No fold directories found under {KFOLD_DATASET_DIR}")
        return

    for fold in fold_dirs:
        data_dir = str(KFOLD_DATASET_DIR / fold)
        output_dir = str(EDA_OUTPUT_DIR / fold)
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n=== EDA for {fold} ===")
        print("--- Counting images ---")
        counts = count_images_per_class(data_dir)
        for split in SPLITS:
            total = sum(counts[split].values())
            print(f"\n{split.upper()} (total {total} images):")
            for cls in CLASS_NAMES:
                pct = counts[split][cls] / total * 100 if total > 0 else 0
                print(f"  {cls}: {counts[split][cls]} images ({pct:.1f}%)")

        print("\n--- Checking for corrupt images and sizes ---")
        broken, sizes = check_broken_images(data_dir)

        if broken:
            print(f"  Found {len(broken)} corrupt image(s):")
            for fpath, err in broken[:10]:
                print(f"    {fpath} -> {err}")
            if len(broken) > 10:
                print(f"    ... and {len(broken) - 10} more file(s)")
        else:
            print("  No corrupt images found.")

        if sizes:
            widths  = [s[0] for s in sizes]
            heights = [s[1] for s in sizes]
            print(f"  Image sizes: width {min(widths)}-{max(widths)}px, "
                  f"height {min(heights)}-{max(heights)}px, "
                  f"mean {sum(widths)//len(widths)}x{sum(heights)//len(heights)}px")

        print("\n--- Plotting charts ---")
        plot_class_distribution(counts, output_dir)
        plot_sample_images(data_dir, output_dir)

        print(f"Done for {fold}. See results in the '{output_dir}/' folder")


if __name__ == "__main__":
    main()
