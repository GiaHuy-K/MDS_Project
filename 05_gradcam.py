"""
Buoc 5: Grad-CAM Explainability (XAI).

Tao heatmap cho thay model "nhin" vao vung nao cua anh khi du doan,
phuc vu phan "Discussion ve kha nang ung dung cho Asian skin tones".

Chay cho 1 hoac nhieu anh, voi 1 hoac ca 3 model nhe de so sanh.
Ngoai cac anh heatmap rieng le, script con tao 1 anh ghep (grid) so sanh
goc / Lite3 / MobileNetV3-Small / ShuffleNetV2 canh nhau cho moi anh mau.
"""

import os

import torch
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from model_utils import MODEL_CONFIGS, CLASS_NAMES, get_model, get_target_layers
from paths_config import SPLIT_DATASET_DIR, CHECKPOINT_DIR, GRADCAM_OUTPUT_DIR, ensure_dirs

# ==========================================
# CAU HINH
# ==========================================
OUTPUT_DIR = GRADCAM_OUTPUT_DIR

# Danh sach anh muon xem Grad-CAM - de trong [] de TU DONG lay 1 anh dai dien
# moi Level tu thu muc test (xem ham _auto_pick_samples ben duoi).
# Muon chon tay thi dien duong dan tuyet doi/tuong doi vao day, vi du:
#   SAMPLE_IMAGES = [r"C:\duong\dan\anh1.jpg", r"C:\duong\dan\anh2.jpg"]
SAMPLE_IMAGES = []

# Cac model muon so sanh
MODELS_TO_RUN = ["efficientnet_lite3", "mobilenet_small", "shufflenet"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _auto_pick_samples(n_per_level=1):
    """Neu SAMPLE_IMAGES de trong, tu dong lay vai anh dai dien tu test/Level_X."""
    test_dir = os.path.join(SPLIT_DATASET_DIR, "test")
    picked = []
    if not os.path.isdir(test_dir):
        print(f"  Khong tim thay {test_dir} de tu dong chon anh mau.")
        return picked

    for level in sorted(os.listdir(test_dir)):
        level_dir = os.path.join(test_dir, level)
        if not os.path.isdir(level_dir):
            continue
        imgs = [f for f in os.listdir(level_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        for img in imgs[:n_per_level]:
            picked.append(os.path.join(level_dir, img))
    return picked


def build_eval_transform(model_name):
    cfg = MODEL_CONFIGS[model_name]
    return transforms.Compose([
        transforms.Resize((cfg["img_size"], cfg["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(cfg["mean"], cfg["std"]),
    ]), cfg["img_size"]


def apply_gradcam(model_name, img_path, output_dir):
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"best_{model_name}.pth")
    if not os.path.exists(checkpoint_path):
        print(f"  Bo qua {model_name}: khong tim thay checkpoint {checkpoint_path}")
        return None

    if not os.path.exists(img_path):
        print(f"  Bo qua: khong tim thay anh {img_path}")
        return None

    # Load model
    model = get_model(model_name, pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval().to(DEVICE)

    # Chuan bi anh (resize theo dung do phan giai cua tung model)
    eval_transform, img_size = build_eval_transform(model_name)
    img_pil = Image.open(img_path).convert("RGB").resize((img_size, img_size))
    img_np = np.array(img_pil) / 255.0
    tensor = eval_transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(DEVICE)

    # Du doan
    with torch.no_grad():
        output = model(tensor)
        pred_idx = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1)[0, pred_idx].item()

    # Grad-CAM
    target_layers = get_target_layers(model, model_name)
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=tensor)[0]
        visualization = show_cam_on_image(img_np.astype(np.float32), grayscale_cam, use_rgb=True)

    # Luu ket qua
    img_basename = os.path.splitext(os.path.basename(img_path))[0]
    save_name = f"{img_basename}_{model_name}_pred-{CLASS_NAMES[pred_idx]}.jpg"
    save_path = os.path.join(output_dir, save_name)
    cv2.imwrite(save_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    print(f"  {model_name}: du doan={CLASS_NAMES[pred_idx]} ({confidence*100:.1f}%) -> {save_path}")

    del model
    torch.cuda.empty_cache()

    return visualization  # RGB uint8, kich thuoc img_size x img_size


def save_comparison_grid(img_path, results, output_dir, grid_size=224):
    """Ghep [Anh goc | model1 | model2 | model3] thanh 1 hang cho de so sanh."""
    orig = Image.open(img_path).convert("RGB").resize((grid_size, grid_size))
    orig_np = np.array(orig)

    tiles = [orig_np]
    labels = ["Original"]
    for model_name, vis in results.items():
        if vis is None:
            continue
        vis_resized = cv2.resize(vis, (grid_size, grid_size))
        tiles.append(vis_resized)
        labels.append(MODEL_CONFIGS[model_name]["display_name"])

    if len(tiles) <= 1:
        return

    grid = np.concatenate(tiles, axis=1)
    grid_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)

    # Ve nhan ten model len tren moi o
    for i, label in enumerate(labels):
        x = i * grid_size + 8
        cv2.putText(grid_bgr, label, (x, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(grid_bgr, label, (x, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 1, cv2.LINE_AA)

    img_basename = os.path.splitext(os.path.basename(img_path))[0]
    save_path = os.path.join(output_dir, f"{img_basename}_comparison_grid.jpg")
    cv2.imwrite(save_path, grid_bgr)
    print(f"  Da luu anh so sanh: {save_path}")


def main():
    ensure_dirs(OUTPUT_DIR)

    sample_images = SAMPLE_IMAGES if SAMPLE_IMAGES else _auto_pick_samples()
    if not sample_images:
        print("Khong co anh mau nao de chay Grad-CAM. Hay dien SAMPLE_IMAGES thu cong.")
        return

    for img_path in sample_images:
        print(f"\n--- Anh: {os.path.basename(img_path)} ---")
        results = {}
        for model_name in MODELS_TO_RUN:
            vis = apply_gradcam(model_name, img_path, OUTPUT_DIR)
            results[model_name] = vis

        if os.path.exists(img_path):
            save_comparison_grid(img_path, results, OUTPUT_DIR)

    print(f"\nHoan tat! Xem ket qua trong thu muc '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()
