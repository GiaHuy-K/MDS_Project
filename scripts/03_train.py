"""
Step 3: Training pipeline - 5-Fold Cross Validation with 3 lightweight models.

Models: EfficientNet-Lite3 (main), MobileNetV3-Small, ShuffleNetV2-x1.0
Best checkpoint is selected by Val Macro F1 (robust to class imbalance).
Output: checkpoints/best_<model>_<fold>.pth, results/kfold_results_detail.csv, results/kfold_results_summary.csv
"""

import os
import csv
import json
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt

from core.model_utils import (
    ALL_MODELS,
    MODEL_CONFIGS,
    NUM_CLASSES,
    CLASS_NAMES,
    build_image_transform,
    get_model,
    get_param_groups,
    count_params,
    measure_inference_time,
)
from core.paths_config import (
    KFOLD_DATASET_DIR, CHECKPOINT_DIR, EVAL_OUTPUT_DIR, RESULTS_DIR, ensure_dirs
)

# ==========================================
# 1. CONFIGURATION
# ==========================================
BATCH_SIZE        = 16
EPOCHS            = 100
LR_BACKBONE       = 1e-5   # small LR for the pretrained backbone - curbs overfitting when fine-tuning
LR_HEAD           = 1e-4   # larger LR for the freshly initialized classifier head
WARMUP_EPOCHS     = 3      # first few epochs ramp LR from 0 -> target (stabilizes training, esp. ShuffleNet)
PATIENCE          = 10
WEIGHT_DECAY      = 5e-4
USE_WEIGHTED_SAMPLER = True  # keep current class-balancing baseline unless class_weight is enabled
USE_CLASS_WEIGHT      = False # use per-class loss weights instead of sampler
USE_FOCAL_LOSS        = False # focal loss replaces plain CE for harder-example emphasis
FOCAL_GAMMA           = 2.0
USE_LABEL_SMOOTHING   = True  # toggle label smoothing for CE-based runs
LABEL_SMOOTHING       = 0.1
GRAD_CLIP_NORM    = 1.0    # guards against exploding gradients / early-training instability
USE_MIXUP          = True   # toggle mixup for ablation
MIXUP_ALPHA        = 0.2    # 0 = disable mixup even if USE_MIXUP is True
MIXUP_PROB         = 0.5    # probability of applying mixup to a given train batch
USE_COLOR_JITTER   = True   # toggle for ablation
RANDOM_SEED       = 42

RESULTS_CSV  = os.path.join(RESULTS_DIR, "kfold_results_detail.csv")    # per-fold results
SUMMARY_CSV  = os.path.join(RESULTS_DIR, "kfold_results_summary.csv")   # final averages
ensure_dirs(CHECKPOINT_DIR, RESULTS_DIR)

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# ==========================================
# 2. DEVICE
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device  : {DEVICE}")
print(f"Started            : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ==========================================
# 3. AUTO-DISCOVER FOLDS
# ==========================================
def discover_folds():
    if not os.path.isdir(KFOLD_DATASET_DIR):
        raise FileNotFoundError(
            f"Not found: {KFOLD_DATASET_DIR}\n"
            f"Run scripts/01_split_dataset_kfold.py first."
        )
    folds = sorted(
        [
            d for d in os.listdir(KFOLD_DATASET_DIR)
            if d.startswith("fold_")
            and os.path.isdir(os.path.join(KFOLD_DATASET_DIR, d))
        ],
        key=lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else x,
    )
    if not folds:
        raise FileNotFoundError(f"No folds found in {KFOLD_DATASET_DIR}")
    print(f"Found {len(folds)} fold(s): {folds}\n")
    return folds




# ==========================================
# 4. DATALOADERS
# ==========================================
def build_loaders(model_name, fold_dir):
    train_transforms = build_image_transform(
        model_name,
        train=True,
        enable_color_jitter=USE_COLOR_JITTER,
    )
    eval_transforms = build_image_transform(model_name, train=False)

    train_dataset = datasets.ImageFolder(
        root=os.path.join(fold_dir, "train"), transform=train_transforms)
    val_dataset   = datasets.ImageFolder(
        root=os.path.join(fold_dir, "val"),   transform=eval_transforms)
    test_dataset  = datasets.ImageFolder(
        root=os.path.join(fold_dir, "test"),  transform=eval_transforms)

    # Guard: ImageFolder sorts class folders alphabetically, so its label order must
    # match CLASS_NAMES - otherwise reports/plots would show mislabeled classes.
    for split_name, ds in (("train", train_dataset), ("val", val_dataset), ("test", test_dataset)):
        assert ds.classes == CLASS_NAMES, (
            f"ImageFolder class order ({split_name}) does not match CLASS_NAMES: "
            f"{ds.classes} != {CLASS_NAMES}"
        )

    # Class weights and WeightedRandomSampler are two alternative imbalance strategies.
    # Keep only one active at a time to avoid double-correction during ablation.
    labels         = [lbl for _, lbl in train_dataset.samples]
    class_counts   = Counter(labels)
    class_weights = build_class_weights(class_counts)

    use_weighted_sampler = USE_WEIGHTED_SAMPLER and not (USE_CLASS_WEIGHT or USE_FOCAL_LOSS)
    if USE_WEIGHTED_SAMPLER and not use_weighted_sampler:
        print("  [Ablation] Weighted sampler disabled because class_weight/focal_loss is enabled.")

    sampler = None
    shuffle = True
    if use_weighted_sampler:
        sample_weights = [1.0 / class_counts[lbl] for lbl in labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )
        shuffle = False

    # num_workers=4 on Windows requires the if __name__=="__main__" guard (present at file end)
    kw = dict(num_workers=4, pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        shuffle=shuffle,
        **kw,
    )
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,   **kw)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,   **kw)

    return train_loader, val_loader, test_loader, train_dataset, class_weights


def build_class_weights(class_counts):
    """Return normalized per-class weights for loss reweighting."""
    total = sum(class_counts.values())
    weights = []
    for class_idx in range(NUM_CLASSES):
        count = class_counts.get(class_idx, 0)
        if count == 0:
            raise ValueError(f"Missing samples for class index {class_idx} ({CLASS_NAMES[class_idx]})")
        weights.append(total / (NUM_CLASSES * count))
    return torch.tensor(weights, dtype=torch.float32)


def log_class_distribution(train_dataset):
    """Print the train-set class distribution for monitoring - NOT used to compute a
    loss class weight, since WeightedRandomSampler already balances each batch."""
    labels = [lbl for _, lbl in train_dataset.samples]
    counts = Counter(labels)
    print(f"  Class counts : {dict(sorted(counts.items()))}")


def mixup_data(x, y, alpha):
    """Blend two images within a batch by a factor lam ~ Beta(alpha, alpha)."""
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


class FocalLoss(nn.Module):
    """Multi-class focal loss for class-imbalance ablation."""

    def __init__(self, gamma=2.0, weight=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction="none",
        )
        pt = torch.exp(-ce_loss)
        loss = (1 - pt).pow(self.gamma) * ce_loss

        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


def build_train_criterion(class_weights):
    """Build the training loss from the current ablation flags."""
    loss_weight = class_weights.to(DEVICE) if USE_CLASS_WEIGHT or USE_FOCAL_LOSS else None
    use_label_smoothing = USE_LABEL_SMOOTHING and not USE_FOCAL_LOSS
    if USE_FOCAL_LOSS and USE_LABEL_SMOOTHING:
        print("  [Ablation] Label smoothing disabled because focal loss is enabled.")

    if USE_FOCAL_LOSS:
        criterion_train = FocalLoss(gamma=FOCAL_GAMMA, weight=loss_weight)
    else:
        criterion_train = nn.CrossEntropyLoss(
            weight=loss_weight,
            label_smoothing=LABEL_SMOOTHING if use_label_smoothing else 0.0,
        )

    criterion_eval = nn.CrossEntropyLoss()
    return criterion_train.to(DEVICE), criterion_eval.to(DEVICE)

def plot_learning_curve(history, model_name, fold_name):
    ensure_dirs(EVAL_OUTPUT_DIR)
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax1.set_title(f'Learning Curve (Loss) - {model_name} {fold_name}')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # F1 plot
    ax2.plot(epochs, history['train_f1'], 'b-', label='Train Macro F1')
    ax2.plot(epochs, history['val_f1'], 'r-', label='Val Macro F1')
    ax2.set_title(f'Learning Curve (F1) - {model_name} {fold_name}')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Macro F1')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    save_path = os.path.join(EVAL_OUTPUT_DIR, f"learning_curve_{model_name}_{fold_name}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    -> Saved learning curve: {save_path}")


# ==========================================
# 5. TRAIN / EVAL LOOP FOR ONE EPOCH
# ==========================================
def run_epoch(model, loader, criterion, optimizer=None, scaler=None, desc=""):
    """
    Return: (avg_loss, accuracy, macro_f1, all_preds, all_labels)
    optimizer=None -> eval mode
    """
    is_train = optimizer is not None
    amp_enabled = DEVICE.type == "cuda"
    total_loss, total_correct = 0.0, 0
    all_preds, all_labels = [], []

    if is_train:
        model.train()
        for inputs, labels in tqdm(loader, desc=desc, leave=False):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            use_mixup = USE_MIXUP and MIXUP_ALPHA > 0 and torch.rand(1).item() < MIXUP_PROB
            if use_mixup:
                inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, MIXUP_ALPHA)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=DEVICE.type, enabled=amp_enabled):
                outputs = model(inputs)
                if use_mixup:
                    loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
                else:
                    loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()

            total_loss    += loss.item() * inputs.size(0)
            total_correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    else:
        model.eval()
        with torch.no_grad():
            for inputs, labels in tqdm(loader, desc=desc, leave=False):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                with torch.amp.autocast(device_type=DEVICE.type, enabled=amp_enabled):
                    outputs = model(inputs)
                    loss    = criterion(outputs, labels)
                preds   = outputs.argmax(dim=1)

                total_loss    += loss.item() * inputs.size(0)
                total_correct += (preds == labels).sum().item()
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = total_correct / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, accuracy, macro_f1, all_preds, all_labels


# ==========================================
# 6. TRAIN ONE FOLD
# ==========================================
def train_one_fold(model_name, fold_name, fold_dir):
    print(f"\n  --- {MODEL_CONFIGS[model_name]['display_name']} | {fold_name} ---")

    train_loader, val_loader, test_loader, train_ds, class_weights = build_loaders(model_name, fold_dir)
    log_class_distribution(train_ds)

    model = get_model(model_name).to(DEVICE)

    criterion_train, criterion_eval = build_train_criterion(class_weights)

    optimizer = optim.AdamW(
        get_param_groups(model, model_name, LR_BACKBONE, LR_HEAD),
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-7
    )
    scaler = torch.amp.GradScaler(device=DEVICE.type, enabled=(DEVICE.type == "cuda"))

    best_val_f1    = 0.0
    patience_count = 0
    save_path = os.path.join(CHECKPOINT_DIR, f"best_{model_name}_{fold_name}.pth")

    history = {'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': []}

    for epoch in range(1, EPOCHS + 1):
        # Linear LR warmup for the first WARMUP_EPOCHS epochs (both param groups)
        if epoch <= WARMUP_EPOCHS:
            warmup_factor = epoch / WARMUP_EPOCHS
            optimizer.param_groups[0]["lr"] = LR_BACKBONE * warmup_factor
            optimizer.param_groups[1]["lr"] = LR_HEAD * warmup_factor

        current_lr_bb   = optimizer.param_groups[0]["lr"]
        current_lr_head = optimizer.param_groups[1]["lr"]

        train_loss, train_acc, train_f1, _, _ = run_epoch(
            model, train_loader, criterion_train, optimizer, scaler,
            desc=f"  [{fold_name}] E{epoch:02d}/{EPOCHS} Train"
        )
        val_loss, val_acc, val_f1, _, _ = run_epoch(
            model, val_loader, criterion_eval,
            desc=f"  [{fold_name}] E{epoch:02d}/{EPOCHS} Val"
        )
        # Only let the plateau scheduler act after warmup is finished
        if epoch > WARMUP_EPOCHS:
            scheduler.step(val_f1)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)

        print(
            f"  Epoch {epoch:02d}/{EPOCHS} | LR(bb)={current_lr_bb:.2e} LR(head)={current_lr_head:.2e} | "
            f"Train Loss={train_loss:.4f} Acc={train_acc*100:.1f}% F1={train_f1:.4f} | "
            f"Val Loss={val_loss:.4f} Acc={val_acc*100:.1f}% F1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1    = val_f1
            patience_count = 0
            torch.save(model.state_dict(), save_path)
            print(f"    -> Saved best (val_f1={val_f1:.4f} | val_acc={val_acc*100:.2f}%)")
        else:
            patience_count += 1
            new_lr_head = optimizer.param_groups[1]["lr"]
            if new_lr_head < current_lr_head:
                print(f"    [Scheduler] LR head: {current_lr_head:.2e} -> {new_lr_head:.2e}")
            if patience_count >= PATIENCE:
                print(f"    Early stopping at epoch {epoch}.")
                break

    plot_learning_curve(history, model_name, fold_name)

    # Evaluate on the test set (exactly once)
    model.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))
    test_loss, test_acc, test_f1, test_preds, test_labels = run_epoch(
        model, test_loader, criterion_eval,
        desc=f"  [{fold_name}] Test"
    )
    print(f"\n  >> {fold_name} | Test Acc={test_acc*100:.2f}% | Macro F1={test_f1:.4f}")
    print(f"\n  Classification Report ({fold_name}):")
    print(classification_report(test_labels, test_preds, target_names=CLASS_NAMES))
    print(f"  Confusion Matrix ({fold_name}):")
    print(confusion_matrix(test_labels, test_preds))

    del model
    torch.cuda.empty_cache()

    return {
        "fold"    : fold_name,
        "test_acc": round(test_acc * 100, 2),
        "test_f1" : round(test_f1, 4),
        "best_val_f1": round(best_val_f1, 4),
    }


# ==========================================
# 7. RESUME HELPERS - SAVE / LOAD PER-FOLD RESULTS
# ==========================================
def _progress_path(model_name):
    """JSON file storing the results of already-completed folds."""
    return os.path.join(CHECKPOINT_DIR, f"progress_{model_name}.json")

def load_progress(model_name):
    """Read results of previously completed folds. Returns dict {fold_name: result}."""
    path = _progress_path(model_name)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  [Resume] Found {len(data)} completed fold(s): {list(data.keys())}")
        return data
    return {}

def save_progress(model_name, fold_name, result):
    """Save one fold's result to JSON right after that fold finishes."""
    path  = _progress_path(model_name)
    data  = load_progress(model_name) if os.path.exists(path) else {}
    data[fold_name] = result
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  [Resume] Saved progress -> {path}")


# ==========================================
# 8. TRAIN ONE MODEL ACROSS ALL FOLDS
# ==========================================
def train_pipeline(model_name, folds):
    display_name = MODEL_CONFIGS[model_name]["display_name"]
    print(f"\n{'='*65}")
    print(f" TRAINING: {display_name} ({model_name}) | {len(folds)} fold(s)")
    print(f"{'='*65}")

    # Load results of previously completed folds (if any)
    done = load_progress(model_name)

    fold_results = []
    for fold_name in folds:
        if fold_name in done:
            # SKIP: this fold already ran, reuse its stored result
            r = done[fold_name]
            print(f"\n  [SKIP] {fold_name} already has a result: "
                  f"Test Acc={r['test_acc']}% | F1={r['test_f1']}")
            fold_results.append(r)
            continue

        # TRAIN: this fold has no result yet
        fold_dir = os.path.join(KFOLD_DATASET_DIR, fold_name)
        result   = train_one_fold(model_name, fold_name, fold_dir)
        fold_results.append(result)

        # Save immediately after the fold finishes -> survives a Colab crash
        save_progress(model_name, fold_name, result)

    accs = [r["test_acc"] for r in fold_results]
    f1s  = [r["test_f1"]  for r in fold_results]

    mean_acc = statistics.mean(accs)
    std_acc  = statistics.stdev(accs) if len(accs) > 1 else 0.0
    mean_f1  = statistics.mean(f1s)
    std_f1   = statistics.stdev(f1s)  if len(f1s)  > 1 else 0.0

    print(f"\n  [{display_name}] SUMMARY OVER {len(folds)} FOLD(S):")
    print(f"  Test Acc : {mean_acc:.2f}% +/- {std_acc:.2f}%")
    print(f"  Macro F1 : {mean_f1:.4f} +/- {std_f1:.4f}")

    # Compute lightweight metrics (measured on fold_1's checkpoint, no extra training)
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"best_{model_name}_fold_1.pth")
    total_params, _, size_mb, cpu_ms, gpu_ms = 0, 0, 0.0, 0.0, None
    if os.path.exists(ckpt_path):
        tmp_model = get_model(model_name).to(DEVICE)
        tmp_model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
        total_params, _, size_mb = count_params(tmp_model)
        cpu_ms = measure_inference_time(tmp_model, model_name, torch.device("cpu"))
        if torch.cuda.is_available():
            gpu_ms = measure_inference_time(tmp_model, model_name, torch.device("cuda"))
        del tmp_model
        torch.cuda.empty_cache()
        print(f"  Params: {total_params/1e6:.2f}M | Size: {size_mb:.2f}MB | "
              f"CPU: {cpu_ms:.2f}ms" + (f" | GPU: {gpu_ms:.2f}ms" if gpu_ms else ""))

    return {
        "model"    : display_name,
        "mean_acc" : round(mean_acc, 2),
        "std_acc"  : round(std_acc, 2),
        "mean_f1"  : round(mean_f1, 4),
        "std_f1"   : round(std_f1, 4),
        "params_M" : round(total_params / 1e6, 2),
        "size_MB"  : round(size_mb, 2),
        "cpu_ms"   : round(cpu_ms, 2),
        "gpu_ms"   : round(gpu_ms, 2) if gpu_ms else "",
        "fold_detail": fold_results,
    }


# ==========================================
# 9. MAIN
# ==========================================
if __name__ == "__main__":
    folds         = discover_folds()
    models_to_run = ALL_MODELS
    final_results = []
    all_fold_rows = []

    for m in models_to_run:
        result = train_pipeline(m, folds)
        final_results.append(result)

        for fd in result["fold_detail"]:
            all_fold_rows.append({
                "model"      : result["model"],
                "fold"       : fd["fold"],
                "test_acc"   : fd["test_acc"],
                "test_f1"    : fd["test_f1"],
                "best_val_f1": fd["best_val_f1"],
            })

    # ---- Summary table ----
    print("\n" + "=" * 80)
    print(" SUMMARY TABLE - 5-FOLD CROSS VALIDATION")
    print("=" * 80)
    header = (
        f"{'Model':<22}{'Acc mean':>10}{'Acc std':>9}"
        f"{'F1 mean':>10}{'F1 std':>9}{'Params(M)':>11}{'Size(MB)':>10}"
        f"{'CPU(ms)':>10}{'GPU(ms)':>10}"
    )
    print(header)
    print("-" * 80)
    for r in final_results:
        gpu_str = f"{r['gpu_ms']:.2f}" if r["gpu_ms"] != "" else "   -"
        print(
            f"{r['model']:<22}"
            f"{r['mean_acc']:>9.2f}%"
            f"{r['std_acc']:>8.2f}%"
            f"{r['mean_f1']:>10.4f}"
            f"{r['std_f1']:>9.4f}"
            f"{r['params_M']:>11.2f}"
            f"{r['size_MB']:>10.2f}"
            f"{r['cpu_ms']:>10.2f}"
            f"{gpu_str:>10}"
        )

    # ---- Save per-fold detail CSV ----
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["model", "fold", "test_acc", "test_f1", "best_val_f1"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_fold_rows)
    print(f"\nSaved per-fold detail : {RESULTS_CSV}")

    # ---- Save summary CSV ----
    summary_rows = [
        {k: v for k, v in r.items() if k != "fold_detail"}
        for r in final_results
    ]
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved 5-fold summary  : {SUMMARY_CSV}")
    print(f"Finished              : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
