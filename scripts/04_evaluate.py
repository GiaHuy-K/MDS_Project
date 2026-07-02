"""
Step 4: Detailed evaluation of the 3 models on all 5 folds.
Output: outputs/eval/fold_X/cm_*.png, roc_*.png, misclassified_*.csv, summary_eval.csv
"""

import os
import csv
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    f1_score,
    cohen_kappa_score,
)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

from core.model_utils import (
    ALL_MODELS,
    MODEL_CONFIGS,
    NUM_CLASSES,
    CLASS_NAMES,
    ordinal_logits_to_class_indices,
    ordinal_logits_to_class_probabilities,
    build_image_transform,
    get_model,
)
from core.paths_config import KFOLD_DATASET_DIR, CHECKPOINT_DIR, EVAL_OUTPUT_DIR, ensure_dirs

# ==========================================
# CONFIGURATION
# ==========================================
FOLDS      = [f"fold_{i}" for i in range(1, 6)]
BATCH_SIZE = 32
BASE_OUTPUT_DIR = EVAL_OUTPUT_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device  : {DEVICE}")
print(f"Results saved to   : {BASE_OUTPUT_DIR}")
print(f"Started            : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


# ==========================================
# 1. DATALOADER
# ==========================================
def build_test_loader(model_name, test_dir):
    img_size = MODEL_CONFIGS[model_name]["img_size"]
    eval_transforms = build_image_transform(model_name, train=False)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=eval_transforms)
    test_loader  = DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0, pin_memory=True
    )
    print(f"[{model_name}] img_size={img_size} | Test: {len(test_dataset)} images")
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
        probs   = ordinal_logits_to_class_probabilities(outputs).cpu().numpy()
        preds   = ordinal_logits_to_class_indices(outputs).cpu().numpy()

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
def plot_confusion_matrix(cm, model_name, output_dir):
    display_name = MODEL_CONFIGS[model_name]["display_name"]

    # Compute percentages so we can show both counts and ratios
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Plot 1: Counts ---
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=axes[0], linewidths=0.5
    )
    axes[0].set_xlabel("Predicted", fontsize=11)
    axes[0].set_ylabel("True",      fontsize=11)
    axes[0].set_title(f"Confusion Matrix (Count)\n{display_name}", fontsize=12)

    # --- Plot 2: Percentages ---
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
    save_path = os.path.join(output_dir, f"cm_{model_name}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  -> Confusion matrix : {save_path}")


# ==========================================
# 4. ROC CURVES
# ==========================================
def plot_roc_curves(y_true, y_probs, model_name, output_dir):
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

    save_path = os.path.join(output_dir, f"roc_{model_name}.png")
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
def save_misclassified(test_dataset, y_true, y_pred, model_name, output_dir):
    save_path = os.path.join(output_dir, f"misclassified_{model_name}.csv")
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
    print(f"  -> Misclassified    : {count}/{total} images ({count/total*100:.1f}%) -> {save_path}")
    return count


# ==========================================
# 6. EVALUATE ONE MODEL
# ==========================================
def evaluate_model(model_name, fold_name, test_dir, output_dir):
    display_name     = MODEL_CONFIGS[model_name]["display_name"]
    checkpoint_path  = os.path.join(CHECKPOINT_DIR, f"best_{model_name}_{fold_name}.pth")

    print(f"\n{'='*60}")
    print(f" EVALUATING: {display_name} ({model_name}) on {fold_name}")
    print(f"{'='*60}")

    if not os.path.exists(checkpoint_path):
        print(f"  [SKIP] Checkpoint not found: {checkpoint_path}")
        print(f"         Run 03_train.py first.")
        return None

    # Load model
    model = get_model(model_name, pretrained=False, ordinal=True)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    )
    model.to(DEVICE).eval()

    # Inference
    test_dataset, test_loader = build_test_loader(model_name, test_dir)
    y_probs, y_pred, y_true   = run_inference(model, test_loader)

    # Metrics
    acc      = (y_pred == y_true).mean()
    macro_f1 = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    wt_f1    = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    qwk      = cohen_kappa_score(y_true, y_pred, weights="quadratic")

    print(f"\n  Test Accuracy       : {acc*100:.2f}%")
    print(f"  Macro F1            : {macro_f1:.4f}")
    print(f"  Weighted F1         : {wt_f1:.4f}")
    print(f"  Quadratic W. Kappa  : {qwk:.4f}")
    print("\n  Classification Report:")
    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, digits=4
    )
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("  Confusion Matrix:")
    print(cm)
    plot_confusion_matrix(cm, model_name, output_dir)

    # ROC
    macro_auc, weighted_auc = plot_roc_curves(y_true, y_probs, model_name, output_dir)

    # Misclassified
    save_misclassified(test_dataset, y_true, y_pred, model_name, output_dir)

    # Save the detailed report to .txt
    report_path = os.path.join(output_dir, f"report_{model_name}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Model         : {display_name}\n")
        f.write(f"Fold          : {fold_name}\n")
        f.write(f"Checkpoint    : {checkpoint_path}\n")
        f.write(f"Test set      : {test_dir}\n")
        f.write(f"Timestamp     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test Accuracy : {acc*100:.2f}%\n")
        f.write(f"Macro F1      : {macro_f1:.4f}\n")
        f.write(f"Weighted F1   : {wt_f1:.4f}\n")
        f.write(f"QWK           : {qwk:.4f}\n")
        f.write(f"AUC macro     : {macro_auc:.4f}\n")
        f.write(f"AUC weighted  : {weighted_auc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))
    print(f"  -> Detailed report  : {report_path}")

    del model
    torch.cuda.empty_cache()

    return {
        "model"       : display_name,
        "fold"        : fold_name,
        "test_acc"    : round(acc * 100, 2),
        "macro_f1"    : round(macro_f1, 4),
        "weighted_f1" : round(wt_f1, 4),
        "qwk"         : round(qwk, 4),
        "auc_macro"   : round(macro_auc, 4),
        "auc_weighted": round(weighted_auc, 4),
    }


# ==========================================
# 7. RUN ALL 3 MODELS ON ALL FOLDS & EXPORT SUMMARY TABLE
# ==========================================
if __name__ == "__main__":
    global_results = []

    # Discover available fold directories under KFOLD_DATASET_DIR (e.g. fold_1..fold_5)
    available_folds = []
    try:
        available_folds = sorted([
            p.name for p in Path(KFOLD_DATASET_DIR).iterdir()
            if p.is_dir() and p.name.startswith("fold_")
        ])
    except Exception:
        available_folds = []

    if not available_folds:
        # Fall back to configured FOLDS constant
        available_folds = FOLDS

    for fold_name in available_folds:
        print(f"\n" + "#"*60)
        print(f" RUNNING EVALUATION FOR: {fold_name}")
        print("#"*60)

        test_dir = os.path.join(str(KFOLD_DATASET_DIR), fold_name, "test")
        if not os.path.exists(test_dir):
            print(f"  [SKIP] Test directory not found: {test_dir}")
            continue

        output_dir = os.path.join(BASE_OUTPUT_DIR, fold_name)
        ensure_dirs(output_dir)

        fold_results = []
        for model_name in ALL_MODELS:
            result = evaluate_model(model_name, fold_name, test_dir, output_dir)
            if result:
                fold_results.append(result)
                global_results.append(result)

        if fold_results:
            # Save summary CSV per fold
            summary_csv = os.path.join(output_dir, f"summary_eval_{fold_name}.csv")
            with open(summary_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(fold_results[0].keys()))
                writer.writeheader()
                writer.writerows(fold_results)
            print(f"\nSaved fold summary CSV : {summary_csv}")

    if not global_results:
        print("\nNo model was evaluated. Check the checkpoints.")
    else:
        # Save global summary CSV
        global_summary_csv = os.path.join(BASE_OUTPUT_DIR, "summary_eval_all_folds.csv")
        with open(global_summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(global_results[0].keys()))
            writer.writeheader()
            writer.writerows(global_results)
        print(f"\nSaved global summary CSV : {global_summary_csv}")
        print(f"Finished           : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
