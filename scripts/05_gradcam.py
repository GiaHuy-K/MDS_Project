"""
Step 5: Grad-CAM++ Explainability (XAI) on all 5 folds.
Output: outputs/gradcam/fold_X/ - per-model heatmaps and a side-by-side grid comparing the 3 models.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
import cv2
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from core.model_utils import (
    MODEL_CONFIGS,
    CLASS_NAMES,
    build_image_transform,
    build_pil_transform,
    get_model,
    get_target_layers,
)
from core.paths_config import KFOLD_DATASET_DIR, CHECKPOINT_DIR, GRADCAM_OUTPUT_DIR, ensure_dirs

# ==========================================
# CONFIGURATION
# ==========================================
FOLDS = [f"fold_{i}" for i in range(1, 6)]
BASE_OUTPUT_DIR = GRADCAM_OUTPUT_DIR

# Images to run Grad-CAM++ on. Leave [] to AUTOMATICALLY pick one representative
# image per Level from the test folder.
SAMPLE_IMAGES = []
N_PER_LEVEL = 2
MODELS_TO_RUN = ["efficientnet_lite3", "mobilenet_small", "shufflenet"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {DEVICE}")


# ==========================================
# AUTO-PICK SAMPLE IMAGES FROM THE TEST SET
# ==========================================
def _auto_pick_samples(fold_name, n_per_level=N_PER_LEVEL):
    """When SAMPLE_IMAGES is empty, auto-pick representative images from fold_X/test/Level_X."""
    test_dir = os.path.join(KFOLD_DATASET_DIR, fold_name, "test")
    picked = []
    if not os.path.isdir(test_dir):
        print(f"  {test_dir} not found for auto-picking sample images.")
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
# PER-MODEL TRANSFORM
# ==========================================
def build_eval_transform(model_name):
    cfg = MODEL_CONFIGS[model_name]
    transform = build_image_transform(model_name, train=False)
    return transform, cfg["img_size"]


# ==========================================
# GRAD-CAM++ FOR ONE MODEL / ONE IMAGE
# ==========================================
def apply_gradcam_plusplus(model_name, fold_name, img_path, output_dir):
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"best_{model_name}_{fold_name}.pth")
    if not os.path.exists(checkpoint_path):
        print(f"  Skipping {model_name}: checkpoint not found {checkpoint_path}")
        return None

    if not os.path.exists(img_path):
        print(f"  Skipping: image not found {img_path}")
        return None

    # Load model
    model = get_model(model_name, pretrained=False)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    )
    model.eval().to(DEVICE)

    # Prepare the image
    eval_transform, img_size = build_eval_transform(model_name)
    pil_transform = build_pil_transform(model_name)

    # img_np: float32 [0,1] RGB, used to overlay the heatmap
    img_pil = pil_transform(Image.open(img_path).convert("RGB"))
    img_np  = np.array(img_pil).astype(np.float32) / 255.0

    # tensor: normalized, used for the forward pass
    tensor = eval_transform(
        Image.open(img_path).convert("RGB")
    ).unsqueeze(0).to(DEVICE)

    # Prediction (predicted class + confidence)
    with torch.no_grad():
        output     = model(tensor)
        pred_idx   = output.argmax(dim=1).item()
        probs      = torch.softmax(output, dim=1)[0]
        confidence = probs[pred_idx].item()

    # Print per-class probabilities for reference
    prob_str = " | ".join(
        f"{CLASS_NAMES[i]}:{probs[i]*100:.1f}%"
        for i in range(len(CLASS_NAMES))
    )
    print(f"    {model_name}: [{prob_str}]")
    print(f"    -> Prediction: {CLASS_NAMES[pred_idx]} ({confidence*100:.1f}%)")

    target_layers = get_target_layers(model, model_name)
    targets       = [ClassifierOutputTarget(pred_idx)]

    with GradCAMPlusPlus(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]

    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    img_basename = os.path.splitext(os.path.basename(img_path))[0]
    save_name    = f"{img_basename}_{model_name}_pred-{CLASS_NAMES[pred_idx]}.jpg"
    save_path    = os.path.join(output_dir, save_name)
    cv2.imwrite(save_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    print(f"    Saved: {save_path}")

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
# BUILD COMPARISON GRID
# ==========================================
def save_comparison_grid(img_path, results, output_dir, grid_size=280):
    """
    Stitch [Original | EfficientNet-Lite3 | MobileNetV3 | ShuffleNetV2] into a single row.
    Each cell is labeled with the model name + prediction + confidence at the top.
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
        print("  Not enough results to build a grid.")
        return

    grid = np.concatenate(tiles, axis=1)

    # Draw labels on each cell (2 lines: model name + prediction)
    for i, label in enumerate(labels):
        x    = i * grid_size + 8
        lines = label.split("\n")
        for j, line in enumerate(lines):
            y = 22 + j * 22
            # White shadow first so text is readable on any background
            cv2.putText(grid, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(grid, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 0), 1, cv2.LINE_AA)

    img_basename = os.path.splitext(os.path.basename(img_path))[0]
    save_path    = os.path.join(output_dir, f"{img_basename}_gradcampp_grid.jpg")
    cv2.imwrite(save_path, grid)
    print(f"  Saved comparison grid: {save_path}")


# ==========================================
# MAIN
# ==========================================
def main():
    print(f"\nRunning Grad-CAM++ for {len(MODELS_TO_RUN)} model(s) across ALL 5 FOLDS")
    print(f"Output: {BASE_OUTPUT_DIR}\n")

    for fold_name in FOLDS:
        print(f"\n" + "#"*60)
        print(f" RUNNING GRAD-CAM FOR: {fold_name}")
        print("#"*60)

        output_dir = os.path.join(BASE_OUTPUT_DIR, fold_name)
        ensure_dirs(output_dir)

        sample_images = SAMPLE_IMAGES if SAMPLE_IMAGES else _auto_pick_samples(fold_name)
        if not sample_images:
            print(f"  [SKIP] No sample images found in {fold_name}.")
            continue

        for img_path in sample_images:
            print(f"\n  Image: {os.path.basename(img_path)}")
            results = {}
            for model_name in MODELS_TO_RUN:
                res = apply_gradcam_plusplus(model_name, fold_name, img_path, output_dir)
                results[model_name] = res

            # Build the side-by-side comparison of all 3 models
            save_comparison_grid(img_path, results, output_dir)

    print(f"\nDone! See results in: {BASE_OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
