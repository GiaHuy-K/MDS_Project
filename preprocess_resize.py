"""
Pre-resize toan bo anh trong dataset xuong kich thuoc can thiet.
Chay 1 LAN DUY NHAT truoc khi train -> giam thoi gian load anh tu ~2800ms/batch xuong ~50ms/batch.

Cach dung:
    python preprocess_resize.py

Ket qua: tao thu muc moi ben canh kfold_dataset voi anh da resize san
    kfold_dataset_resized/
        fold_1/train/Level_0/xxx.jpg
        fold_1/val/...
        ...
"""

import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from model_utils import MODEL_CONFIGS
from paths_config import KFOLD_DATASET_DIR, PROJECT_ROOT

# ==========================================
# CAU HINH
# ==========================================
# Lay img_size lon nhat trong 3 model de resize 1 lan duy nhat
# EfficientNet-Lite3=280, MobileNetV3-Small=224, ShuffleNetV2=224
TARGET_SIZE  = max(cfg["img_size"] for cfg in MODEL_CONFIGS.values())
INPUT_DIR    = KFOLD_DATASET_DIR
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "kfold_dataset_resized")
QUALITY      = 95   # JPEG quality (95 = rat tot, file nho hon PNG)

print(f"Input  : {INPUT_DIR}")
print(f"Output : {OUTPUT_DIR}")
print(f"Target size: {TARGET_SIZE}x{TARGET_SIZE}")


def resize_dataset():
    input_root  = Path(INPUT_DIR)
    output_root = Path(OUTPUT_DIR)

    # Tim tat ca file anh
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    all_images = [
        p for p in input_root.rglob("*")
        if p.suffix.lower() in extensions
    ]

    if not all_images:
        print("Khong tim thay anh nao! Kiem tra lai INPUT_DIR.")
        return

    print(f"\nTim thay {len(all_images)} anh. Bat dau resize...")

    skipped = 0
    for img_path in tqdm(all_images, desc="Resizing"):
        # Tinh duong dan output tuong ung
        rel_path    = img_path.relative_to(input_root)
        out_path    = output_root / rel_path.with_suffix(".jpg")   # luu thanh JPEG
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Skip neu da ton tai (resume-friendly)
        if out_path.exists():
            skipped += 1
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.BILINEAR)
            img.save(out_path, "JPEG", quality=QUALITY)
        except Exception as e:
            print(f"\n[WARN] Loi voi {img_path}: {e}")

    total    = len(all_images)
    processed = total - skipped
    print(f"\nHoan thanh: {processed} anh resize | {skipped} anh da co san (skip)")
    print(f"Anh da resize luu tai: {OUTPUT_DIR}")
    print(f"\nTiep theo: cap nhat paths_config.py:")
    print(f'  KFOLD_DATASET_DIR = r"{OUTPUT_DIR}"')
    print(f"Sau do chay lai 03_train_kfold.py binh thuong.")


if __name__ == "__main__":
    resize_dataset()