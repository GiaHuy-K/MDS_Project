"""
Buoc 5: Grad-CAM++ Explainability (XAI).
Output: gradcam_outputs/ — heatmap tung model va anh grid so sanh 3 model.
"""

import os

import torch
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from model_utils import MODEL_CONFIGS, CLASS_NAMES, get_model, get_target_layers
from paths_config import KFOLD_DATASET_DIR, CHECKPOINT_DIR, GRADCAM_OUTPUT_DIR, ensure_dirs

# ==========================================
# CAU HINH
# ==========================================
OUTPUT_DIR = GRADCAM_OUTPUT_DIR
FOLD_NAME  = "fold_1"    # Dong bo voi 04_evaluate.py

# Danh sach anh muon xem Grad-CAM++ - de trong [] de TU DONG lay 1 anh dai dien
# moi Level tu thu muc test.
# Muon chon tay thi dien duong dan tuyet doi vao day, vi du:
#   SAMPLE_IMAGES = [r"C:\duong\dan\anh1.jpg", r"C:\duong\dan\anh2.jpg"]
SAMPLE_IMAGES = []

# So anh tu dong lay moi Level (chi dung khi SAMPLE_IMAGES = [])
N_PER_LEVEL = 2

# Cac model muon so sanh
MODELS_TO_RUN = ["efficientnet_lite3", "mobilenet_small", "shufflenet"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Chay tren thiet bi: {DEVICE}")


# ==========================================
# TU DONG CHON ANH MAU TU TEST SET
# ==========================================
def _auto_pick_samples(n_per_level=N_PER_LEVEL):
    """Neu SAMPLE_IMAGES de trong, tu dong lay anh dai dien tu fold_1/test/Level_X."""
    test_dir = os.path.join(KFOLD_DATASET_DIR, "fold_1", "test")
    picked = []
    if not os.path.isdir(test_dir):
        print(f"  Khong tim thay {test_dir} de tu dong chon anh mau.")
        return picked

    for level in sorted(os.listdir(test_dir)):
        level_dir = os.path.join(test_dir, level)
        if not os.path.isdir(level_dir):
            continue
        imgs = sorted([
            f for f in os.listdir(level_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        for img in imgs[:n_per_level]:
            picked.append(os.path.join(level_dir, img))
            print(f"  Auto pick: {level}/{img}")
    return picked


# ==========================================
# TRANSFORM THEO TUNG MODEL
# ==========================================
def build_eval_transform(model_name):
    cfg = MODEL_CONFIGS[model_name]
    transform = transforms.Compose([
        transforms.Resize(
            (cfg["img_size"], cfg["img_size"]),
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        ),
        transforms.ToTensor(),
        transforms.Normalize(cfg["mean"], cfg["std"]),
    ])
    return transform, cfg["img_size"]


# ==========================================
# GRAD-CAM++ CHO 1 MODEL / 1 ANH
# ==========================================
def apply_gradcam_plusplus(model_name, img_path, output_dir):
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"best_{model_name}_{FOLD_NAME}.pth")
    if not os.path.exists(checkpoint_path):
        print(f"  Bo qua {model_name}: khong tim thay checkpoint {checkpoint_path}")
        return None

    if not os.path.exists(img_path):
        print(f"  Bo qua: khong tim thay anh {img_path}")
        return None

    # Load model
    model = get_model(model_name, pretrained=False)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    )
    model.eval().to(DEVICE)

    # Chuan bi anh
    eval_transform, img_size = build_eval_transform(model_name)

    # img_np: float32 [0,1] RGB, dung de overlay heatmap
    img_pil = Image.open(img_path).convert("RGB").resize((img_size, img_size))
    img_np  = np.array(img_pil).astype(np.float32) / 255.0

    # tensor: normalized, dung de forward qua model
    tensor = eval_transform(
        Image.open(img_path).convert("RGB")
    ).unsqueeze(0).to(DEVICE)

    # Du doan (predicted class + confidence)
    with torch.no_grad():
        output     = model(tensor)
        pred_idx   = output.argmax(dim=1).item()
        probs      = torch.softmax(output, dim=1)[0]
        confidence = probs[pred_idx].item()

    # In xac suat tung class de tham khao
    prob_str = " | ".join(
        f"{CLASS_NAMES[i]}:{probs[i]*100:.1f}%"
        for i in range(len(CLASS_NAMES))
    )
    print(f"    {model_name}: [{prob_str}]")
    print(f"    -> Du doan: {CLASS_NAMES[pred_idx]} ({confidence*100:.1f}%)")

    target_layers = get_target_layers(model, model_name)
    targets       = [ClassifierOutputTarget(pred_idx)]

    with GradCAMPlusPlus(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]

    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    img_basename = os.path.splitext(os.path.basename(img_path))[0]
    save_name    = f"{img_basename}_{model_name}_pred-{CLASS_NAMES[pred_idx]}.jpg"
    save_path    = os.path.join(output_dir, save_name)
    cv2.imwrite(save_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    print(f"    Luu: {save_path}")

    del model
    torch.cuda.empty_cache()

    return {
        "vis"       : visualization,   # RGB uint8
        "pred_idx"  : pred_idx,
        "pred_label": CLASS_NAMES[pred_idx],
        "confidence": confidence,
        "img_size"  : img_size,
    }


# ==========================================
# GHEP ANH SO SANH (GRID)
# ==========================================
def save_comparison_grid(img_path, results, output_dir, grid_size=280):
    """
    Ghep [Anh goc | EfficientNet-Lite3 | MobileNetV3 | ShuffleNetV2] thanh 1 hang.
    Moi o co nhan ten model + du doan + confidence phia tren.
    """
    orig     = Image.open(img_path).convert("RGB").resize((grid_size, grid_size))
    orig_np  = np.array(orig)
    orig_bgr = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)

    tiles  = [orig_bgr]
    labels = ["Original"]

    for model_name in MODELS_TO_RUN:
        res = results.get(model_name)
        if res is None:
            continue
        vis_resized = cv2.resize(
            cv2.cvtColor(res["vis"], cv2.COLOR_RGB2BGR),
            (grid_size, grid_size)
        )
        tiles.append(vis_resized)
        label = (
            f"{MODEL_CONFIGS[model_name]['display_name']}\n"
            f"{res['pred_label']} {res['confidence']*100:.1f}%"
        )
        labels.append(label)

    if len(tiles) <= 1:
        print("  Khong du ket qua de tao grid.")
        return

    grid = np.concatenate(tiles, axis=1)

    # Ve nhan len moi o (2 dong: ten model + du doan)
    for i, label in enumerate(labels):
        x    = i * grid_size + 8
        lines = label.split("\n")
        for j, line in enumerate(lines):
            y = 22 + j * 22
            # Shadow trang truoc de de doc tren moi nen
            cv2.putText(grid, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(grid, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 0), 1, cv2.LINE_AA)

    img_basename = os.path.splitext(os.path.basename(img_path))[0]
    save_path    = os.path.join(output_dir, f"{img_basename}_gradcampp_grid.jpg")
    cv2.imwrite(save_path, grid)
    print(f"  Da luu grid so sanh: {save_path}")


# ==========================================
# MAIN
# ==========================================
def main():
    ensure_dirs(OUTPUT_DIR)

    sample_images = SAMPLE_IMAGES if SAMPLE_IMAGES else _auto_pick_samples()
    if not sample_images:
        print(
            "Khong co anh mau nao de chay Grad-CAM++.\n"
            "Hay dien duong dan vao SAMPLE_IMAGES hoac kiem tra thu muc test."
        )
        return

    print(f"\nSe chay Grad-CAM++ cho {len(sample_images)} anh x {len(MODELS_TO_RUN)} model")
    print(f"Output: {OUTPUT_DIR}\n")

    for img_path in sample_images:
        print(f"\n{'='*60}")
        print(f"Anh: {os.path.basename(img_path)}")
        print(f"{'='*60}")

        results = {}
        for model_name in MODELS_TO_RUN:
            res = apply_gradcam_plusplus(model_name, img_path, OUTPUT_DIR)
            results[model_name] = res

        # Tao anh ghep so sanh ca 3 model
        save_comparison_grid(img_path, results, OUTPUT_DIR)

    print(f"\nHoan tat! Xem ket qua trong: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()