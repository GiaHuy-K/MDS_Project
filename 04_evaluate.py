"""
Buoc 4: Danh gia chi tiet 1 model tren TEST SET.

Sua MODEL_NAME ben duoi de chon model muon xem chi tiet:
  "efficientnet_lite3" / "mobilenet_small" / "shufflenet"

Ket qua xuat ra trong eval_outputs/:
  - cm_<model>.png            Confusion matrix dang heatmap
  - roc_<model>.png           ROC curves One-vs-Rest cho 4 class + AUC
  - misclassified_<model>.csv Danh sach anh bi du doan sai (duong dan, nhan that, nhan du doan)
  - In ra Precision/Recall/F1 (tung class + macro/weighted), AUC-ROC (macro/weighted)
"""

import os
import csv

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

from model_utils import MODEL_CONFIGS, NUM_CLASSES, CLASS_NAMES, get_model
from paths_config import SPLIT_DATASET_DIR, CHECKPOINT_DIR, EVAL_OUTPUT_DIR, ensure_dirs

# ==========================================
# CAU HINH - SUA O DAY
# ==========================================
MODEL_NAME = "efficientnet_lite3"   # hoac "mobilenet_small" / "shufflenet"

# Duong dan tu dong lay tu paths_config.py (khong can sua tay khi doi may).
DATA_DIR = SPLIT_DATASET_DIR
TEST_DIR = os.path.join(DATA_DIR, "test")

BATCH_SIZE = 32
OUTPUT_DIR = EVAL_OUTPUT_DIR
ensure_dirs(OUTPUT_DIR)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_test_loader(model_name):
    cfg = MODEL_CONFIGS[model_name]
    img_size = cfg["img_size"]
    eval_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(cfg["mean"], cfg["std"]),
    ])
    test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=eval_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return test_dataset, test_loader


@torch.no_grad()
def run_inference(model, loader):
    all_probs, all_preds, all_labels = [], [], []
    for inputs, labels in tqdm(loader, desc="Evaluating", leave=False):
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)

        all_probs.append(probs)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    return np.concatenate(all_probs, axis=0), np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {MODEL_CONFIGS[model_name]['display_name']}")
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"cm_{model_name}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Da luu confusion matrix: {save_path}")


def plot_roc_curves(y_true, y_probs, model_name):
    y_true_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(NUM_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Macro-average AUC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(NUM_CLASSES)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(NUM_CLASSES):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= NUM_CLASSES
    macro_auc = auc(all_fpr, mean_tpr)

    # Weighted-average AUC
    class_counts = y_true_bin.sum(axis=0)
    weighted_auc = sum(roc_auc[i] * class_counts[i] for i in range(NUM_CLASSES)) / class_counts.sum()

    plt.figure(figsize=(7, 6))
    for i in range(NUM_CLASSES):
        plt.plot(fpr[i], tpr[i], label=f"{CLASS_NAMES[i]} (AUC = {roc_auc[i]:.3f})")
    plt.plot(all_fpr, mean_tpr, "k--", label=f"Macro-avg (AUC = {macro_auc:.3f})")
    plt.plot([0, 1], [0, 1], "gray", linestyle=":", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves (One-vs-Rest) - {MODEL_CONFIGS[model_name]['display_name']}")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"roc_{model_name}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Da luu ROC curves: {save_path}")

    print(f"\nAUC-ROC macro-average:    {macro_auc:.4f}")
    print(f"AUC-ROC weighted-average: {weighted_auc:.4f}")
    for i in range(NUM_CLASSES):
        print(f"  AUC {CLASS_NAMES[i]}: {roc_auc[i]:.4f}")


def save_misclassified(test_dataset, y_true, y_pred, model_name):
    save_path = os.path.join(OUTPUT_DIR, f"misclassified_{model_name}.csv")
    samples = test_dataset.samples  # list of (path, label_idx)

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "true_label", "predicted_label"])
        count = 0
        for (img_path, _), true_idx, pred_idx in zip(samples, y_true, y_pred):
            if true_idx != pred_idx:
                writer.writerow([img_path, CLASS_NAMES[true_idx], CLASS_NAMES[pred_idx]])
                count += 1

    print(f"\nSo anh du doan sai: {count}/{len(samples)} -> {save_path}")


def main():
    print(f"Danh gia model: {MODEL_CONFIGS[MODEL_NAME]['display_name']} ({MODEL_NAME})")
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"best_{MODEL_NAME}.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Khong tim thay checkpoint: {checkpoint_path}. Hay chay 03_train.py truoc."
        )

    test_dataset, test_loader = build_test_loader(MODEL_NAME)
    print(f"Test set: {len(test_dataset)} anh")

    model = get_model(MODEL_NAME, pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval().to(DEVICE)

    y_probs, y_pred, y_true = run_inference(model, test_loader)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    plot_confusion_matrix(cm, MODEL_NAME)

    plot_roc_curves(y_true, y_probs, MODEL_NAME)

    save_misclassified(test_dataset, y_true, y_pred, MODEL_NAME)


if __name__ == "__main__":
    main()
