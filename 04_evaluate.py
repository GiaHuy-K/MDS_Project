"""
Buoc 4: Danh gia chi tiet CA 3 MODEL tren TEST SET cua fold_1.

Ket qua xuat ra trong eval_outputs/:
  - cm_<model>.png              Confusion matrix dang heatmap
  - roc_<model>.png             ROC curves One-vs-Rest cho 4 class + AUC
  - misclassified_<model>.csv   Danh sach anh bi du doan sai
  - summary_eval.csv            Bang tong hop Acc + F1 + AUC ca 3 model
  - In ra Precision/Recall/F1 (tung class + macro/weighted), AUC-ROC
"""

import os
import csv
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    f1_score,
)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

from model_utils import ALL_MODELS, MODEL_CONFIGS, NUM_CLASSES, CLASS_NAMES, get_model
from paths_config import KFOLD_DATASET_DIR, CHECKPOINT_DIR, EVAL_OUTPUT_DIR, ensure_dirs

# ==========================================
# CAU HINH
# ==========================================
FOLD_NAME  = "fold_1"    # Dong bo voi 03_train.py
TEST_DIR   = os.path.join(KFOLD_DATASET_DIR, FOLD_NAME, "test")
BATCH_SIZE = 32
OUTPUT_DIR = EVAL_OUTPUT_DIR
SUMMARY_CSV = os.path.join(OUTPUT_DIR, "summary_eval.csv")

ensure_dirs(OUTPUT_DIR)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Chay tren thiet bi : {DEVICE}")
print(f"Test set           : {TEST_DIR}")
print(f"Ket qua luu tai    : {OUTPUT_DIR}")
print(f"Bat dau            : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


# ==========================================
# 1. DATALOADER
# ==========================================
def build_test_loader(model_name):
    cfg = MODEL_CONFIGS[model_name]
    img_size = cfg["img_size"]
    eval_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(cfg["mean"], cfg["std"]),
    ])
    test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=eval_transforms)
    test_loader  = DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0, pin_memory=True
    )
    print(f"[{model_name}] img_size={img_size} | Test: {len(test_dataset)} anh")
    return test_dataset, test_loader


# ==========================================
# 2. INFERENCE
# ==========================================
@torch.no_grad()
def run_inference(model, loader):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []

    for inputs, labels in tqdm(loader, desc="  Inferencing", leave=False):
        inputs  = inputs.to(DEVICE)
        outputs = model(inputs)
        probs   = torch.softmax(outputs, dim=1).cpu().numpy()
        preds   = probs.argmax(axis=1)

        all_probs.append(probs)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    return (
        np.concatenate(all_probs, axis=0),
        np.array(all_preds),
        np.array(all_labels),
    )


# ==========================================
# 3. CONFUSION MATRIX
# ==========================================
def plot_confusion_matrix(cm, model_name):
    display_name = MODEL_CONFIGS[model_name]["display_name"]

    # Tinh % de hien thi ca so luong lan ti le
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Plot 1: So luong ---
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=axes[0], linewidths=0.5
    )
    axes[0].set_xlabel("Predicted", fontsize=11)
    axes[0].set_ylabel("True",      fontsize=11)
    axes[0].set_title(f"Confusion Matrix (Count)\n{display_name}", fontsize=12)

    # --- Plot 2: Ti le % ---
    sns.heatmap(
        cm_pct, annot=True, fmt=".1f", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=axes[1], linewidths=0.5,
        vmin=0, vmax=100
    )
    axes[1].set_xlabel("Predicted", fontsize=11)
    axes[1].set_ylabel("True",      fontsize=11)
    axes[1].set_title(f"Confusion Matrix (% per True class)\n{display_name}", fontsize=12)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"cm_{model_name}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  -> Confusion matrix : {save_path}")


# ==========================================
# 4. ROC CURVES
# ==========================================
def plot_roc_curves(y_true, y_probs, model_name):
    display_name = MODEL_CONFIGS[model_name]["display_name"]
    y_true_bin   = label_binarize(y_true, classes=list(range(NUM_CLASSES)))

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(NUM_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc[i]        = auc(fpr[i], tpr[i])

    # Macro-average
    all_fpr  = np.unique(np.concatenate([fpr[i] for i in range(NUM_CLASSES)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(NUM_CLASSES):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr  /= NUM_CLASSES
    macro_auc  = auc(all_fpr, mean_tpr)

    # Weighted-average
    class_counts = y_true_bin.sum(axis=0)
    weighted_auc = (
        sum(roc_auc[i] * class_counts[i] for i in range(NUM_CLASSES))
        / class_counts.sum()
    )

    # Plot
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
    plt.figure(figsize=(7, 6))
    for i in range(NUM_CLASSES):
        plt.plot(
            fpr[i], tpr[i],
            color=colors[i], lw=2,
            label=f"{CLASS_NAMES[i]} (AUC={roc_auc[i]:.3f})"
        )
    plt.plot(
        all_fpr, mean_tpr, "k--", lw=2,
        label=f"Macro-avg (AUC={macro_auc:.3f})"
    )
    plt.plot([0, 1], [0, 1], color="gray", linestyle=":", lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate",  fontsize=11)
    plt.title(f"ROC Curves (One-vs-Rest)\n{display_name}", fontsize=12)
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, f"roc_{model_name}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  -> ROC curves       : {save_path}")
    print(f"     AUC macro        : {macro_auc:.4f}")
    print(f"     AUC weighted     : {weighted_auc:.4f}")
    for i in range(NUM_CLASSES):
        print(f"     AUC {CLASS_NAMES[i]:<12}: {roc_auc[i]:.4f}")

    return macro_auc, weighted_auc


# ==========================================
# 5. MISCLASSIFIED CSV
# ==========================================
def save_misclassified(test_dataset, y_true, y_pred, model_name):
    save_path = os.path.join(OUTPUT_DIR, f"misclassified_{model_name}.csv")
    samples   = test_dataset.samples

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "true_label", "predicted_label"])
        count = 0
        for (img_path, _), true_idx, pred_idx in zip(samples, y_true, y_pred):
            if true_idx != pred_idx:
                writer.writerow([
                    img_path,
                    CLASS_NAMES[true_idx],
                    CLASS_NAMES[pred_idx],
                ])
                count += 1

    total = len(samples)
    print(f"  -> Misclassified    : {count}/{total} anh ({count/total*100:.1f}%) -> {save_path}")
    return count


# ==========================================
# 6. DANH GIA 1 MODEL
# ==========================================
def evaluate_model(model_name):
    display_name     = MODEL_CONFIGS[model_name]["display_name"]
    checkpoint_path  = os.path.join(CHECKPOINT_DIR, f"best_{model_name}.pth")

    print(f"\n{'='*60}")
    print(f" DANH GIA: {display_name} ({model_name})")
    print(f"{'='*60}")

    if not os.path.exists(checkpoint_path):
        print(f"  [SKIP] Khong tim thay checkpoint: {checkpoint_path}")
        print(f"         Hay chay 03_train.py truoc.")
        return None

    # Load model
    model = get_model(model_name, pretrained=False)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    )
    model.to(DEVICE).eval()

    # Inference
    test_dataset, test_loader = build_test_loader(model_name)
    y_probs, y_pred, y_true   = run_inference(model, test_loader)

    # Metrics
    acc      = (y_pred == y_true).mean()
    macro_f1 = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    wt_f1    = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n  Test Accuracy       : {acc*100:.2f}%")
    print(f"  Macro F1            : {macro_f1:.4f}")
    print(f"  Weighted F1         : {wt_f1:.4f}")
    print("\n  Classification Report:")
    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, digits=4
    )
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("  Confusion Matrix:")
    print(cm)
    plot_confusion_matrix(cm, model_name)

    # ROC
    macro_auc, weighted_auc = plot_roc_curves(y_true, y_probs, model_name)

    # Misclassified
    save_misclassified(test_dataset, y_true, y_pred, model_name)

    # Luu report chi tiet ra .txt
    report_path = os.path.join(OUTPUT_DIR, f"report_{model_name}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Model         : {display_name}\n")
        f.write(f"Checkpoint    : {checkpoint_path}\n")
        f.write(f"Test set      : {TEST_DIR}\n")
        f.write(f"Timestamp     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test Accuracy : {acc*100:.2f}%\n")
        f.write(f"Macro F1      : {macro_f1:.4f}\n")
        f.write(f"Weighted F1   : {wt_f1:.4f}\n")
        f.write(f"AUC macro     : {macro_auc:.4f}\n")
        f.write(f"AUC weighted  : {weighted_auc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))
    print(f"  -> Report chi tiet  : {report_path}")

    del model
    torch.cuda.empty_cache()

    return {
        "model"       : display_name,
        "test_acc"    : round(acc * 100, 2),
        "macro_f1"    : round(macro_f1, 4),
        "weighted_f1" : round(wt_f1, 4),
        "auc_macro"   : round(macro_auc, 4),
        "auc_weighted": round(weighted_auc, 4),
    }


# ==========================================
# 7. CHAY CA 3 MODEL & XUAT BANG TONG HOP
# ==========================================
if __name__ == "__main__":
    all_results = []

    for model_name in ALL_MODELS:
        result = evaluate_model(model_name)
        if result:
            all_results.append(result)

    if not all_results:
        print("\nKhong co model nao duoc danh gia. Kiem tra lai checkpoint.")
    else:
        # Bang tong hop
        print("\n" + "=" * 75)
        print(" BANG TONG HOP - CA 3 MODEL")
        print(f" Test set: {TEST_DIR}")
        print("=" * 75)
        header = (
            f"{'Model':<22}{'Test Acc':>10}{'Macro F1':>10}"
            f"{'Wtd F1':>10}{'AUC macro':>12}{'AUC wtd':>10}"
        )
        print(header)
        print("-" * 75)
        for r in all_results:
            print(
                f"{r['model']:<22}"
                f"{r['test_acc']:>9.2f}%"
                f"{r['macro_f1']:>10.4f}"
                f"{r['weighted_f1']:>10.4f}"
                f"{r['auc_macro']:>12.4f}"
                f"{r['auc_weighted']:>10.4f}"
            )

        # Luu summary CSV
        with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            writer.writeheader()
            writer.writerows(all_results)

        print(f"\nDa luu summary CSV : {SUMMARY_CSV}")
        print(f"Ket thuc           : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")