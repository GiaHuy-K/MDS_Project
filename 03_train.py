"""
Buoc 3: Training pipeline cho 3 model NHE (Lightweight) - 5-FOLD CROSS VALIDATION
  - EfficientNet-Lite3   (model chinh cua de tai)
  - MobileNetV3-Small
  - ShuffleNetV2-x1.0

Quy trinh:
- Lap qua tat ca fold_1 -> fold_5
- Moi fold: train toan bo model (KHONG freeze) -> chon best model theo VAL MACRO F1
- Sau khi het 5 fold: tinh mean +/- std cua Accuracy va Macro F1
- Xuat CSV tong hop tat ca fold + bang trung binh cuoi cung

=== CAC THAY DOI SO VOI BAN GOC ===
[BUG FIX] torch.enable_grad() / torch.no_grad(): tach 2 nhanh if/else rieng biet
[BUG FIX] torch.load them weights_only=True (tranh warning PyTorch >= 2.0)
[FIX]     Best model chon theo MACRO F1, khong phai accuracy (phu hop class imbalance)
[FIX]     Augmentation nhe hon: bo RandomErasing + RandomAffine, giam rotation 15->10
[FIX]     Scheduler doi sang ReduceLROnPlateau(mode=max, metric=val_f1)
[FIX]     Bo freeze/unfreeze: train toan bo model ngay tu dau voi LR thong nhat
[ADD]     WeightedRandomSampler: oversample minority class
[ADD]     run_epoch tra ve macro_f1
[ADD]     K-fold discovery tu dong (khong can hardcode so fold)
[ADD]     CSV ket qua chi tiet tung fold + bang trung binh
[ADD]     cudnn.deterministic = True
"""

import os
import csv
import statistics
from collections import Counter
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm

from model_utils import (
    ALL_MODELS,
    MODEL_CONFIGS,
    NUM_CLASSES,
    CLASS_NAMES,
    get_model,
    count_params,
    measure_inference_time,
)
from paths_config import KFOLD_DATASET_DIR, CHECKPOINT_DIR, PROJECT_ROOT, ensure_dirs

# ==========================================
# 1. CAU HINH
# ==========================================
BATCH_SIZE      = 32
EPOCHS          = 40
LR              = 1e-4    # Train toan bo model voi 1 LR thong nhat
PATIENCE        = 7
WEIGHT_DECAY    = 5e-4
LABEL_SMOOTHING = 0.1
RANDOM_SEED     = 42

RESULTS_CSV  = os.path.join(PROJECT_ROOT, "kfold_results_detail.csv")    # ket qua tung fold
SUMMARY_CSV  = os.path.join(PROJECT_ROOT, "kfold_results_summary.csv")   # trung binh cuoi
ensure_dirs(CHECKPOINT_DIR)

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Chay tren thiet bi : {DEVICE}")
print(f"Bat dau            : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ==========================================
# 2. TIM FOLD TU DONG
# ==========================================
def discover_folds():
    if not os.path.isdir(KFOLD_DATASET_DIR):
        raise FileNotFoundError(
            f"Khong tim thay: {KFOLD_DATASET_DIR}\n"
            f"Hay chay 01b_split_dataset_kfold.py truoc."
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
        raise FileNotFoundError(f"Khong tim thay fold nao trong {KFOLD_DATASET_DIR}")
    print(f"Tim thay {len(folds)} fold: {folds}\n")
    return folds





# ==========================================
# 4. DATALOADERS
# ==========================================
def build_loaders(model_name, fold_dir):
    cfg      = MODEL_CONFIGS[model_name]
    img_size = cfg["img_size"]
    mean, std = cfg["mean"], cfg["std"]

    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size),
                          interpolation=transforms.InterpolationMode.BILINEAR,
                          antialias=True),   # Bo Resize+Crop: nhanh hon, it CPU hon
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size),
                          interpolation=transforms.InterpolationMode.BILINEAR,
                          antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dataset = datasets.ImageFolder(
        root=os.path.join(fold_dir, "train"), transform=train_transforms)
    val_dataset   = datasets.ImageFolder(
        root=os.path.join(fold_dir, "val"),   transform=eval_transforms)
    test_dataset  = datasets.ImageFolder(
        root=os.path.join(fold_dir, "test"),  transform=eval_transforms)

    # WeightedRandomSampler: oversample minority class
    labels         = [lbl for _, lbl in train_dataset.samples]
    class_counts   = Counter(labels)
    sample_weights = [1.0 / class_counts[lbl] for lbl in labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    # Windows: num_workers > 0 can dung if __name__ == "__main__" guard
    # RTX 3050 Ti: num_workers=4 giam CPU bottleneck dang ke
    kw = dict(num_workers=4, pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, **kw)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,   **kw)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,   **kw)

    return train_loader, val_loader, test_loader, train_dataset


def build_class_weights(train_dataset):
    labels  = [lbl for _, lbl in train_dataset.samples]
    counts  = Counter(labels)
    total   = sum(counts.values())
    weights = torch.tensor(
        [total / counts[i] for i in range(NUM_CLASSES)], dtype=torch.float
    )
    weights = weights / weights.mean()
    print(f"  Class counts : {dict(sorted(counts.items()))}")
    print(f"  Class weights: {[round(w, 3) for w in weights.tolist()]}")
    return weights


# ==========================================
# 5. VONG LAP TRAIN / EVAL MOT EPOCH
# ==========================================
def run_epoch(model, loader, criterion, optimizer=None, desc=""):
    """
    Return: (avg_loss, accuracy, macro_f1, all_preds, all_labels)
    optimizer=None -> eval mode
    """
    is_train = optimizer is not None
    total_loss, total_correct = 0.0, 0
    all_preds, all_labels = [], []

    if is_train:
        model.train()
        for inputs, labels in tqdm(loader, desc=desc, leave=False):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss    = criterion(outputs, labels)
            preds   = outputs.argmax(dim=1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss    += loss.item() * inputs.size(0)
            total_correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    else:
        model.eval()
        with torch.no_grad():
            for inputs, labels in tqdm(loader, desc=desc, leave=False):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
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
# 6. TRAIN 1 FOLD
# ==========================================
def train_one_fold(model_name, fold_name, fold_dir):
    print(f"\n  --- {MODEL_CONFIGS[model_name]['display_name']} | {fold_name} ---")

    train_loader, val_loader, test_loader, train_ds = build_loaders(model_name, fold_dir)
    class_weights = build_class_weights(train_ds).to(DEVICE)

    model = get_model(model_name).to(DEVICE)

    criterion_train = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    criterion_eval  = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-7
    )

    best_val_f1    = 0.0
    patience_count = 0
    save_path = os.path.join(CHECKPOINT_DIR, f"best_{model_name}_{fold_name}.pth")

    for epoch in range(1, EPOCHS + 1):
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc, train_f1, _, _ = run_epoch(
            model, train_loader, criterion_train, optimizer,
            desc=f"  [{fold_name}] E{epoch:02d}/{EPOCHS} Train"
        )
        val_loss, val_acc, val_f1, _, _ = run_epoch(
            model, val_loader, criterion_eval,
            desc=f"  [{fold_name}] E{epoch:02d}/{EPOCHS} Val"
        )
        scheduler.step(val_f1)

        print(
            f"  Epoch {epoch:02d}/{EPOCHS} | LR={current_lr:.2e} | "
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
            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr < current_lr:
                print(f"    [Scheduler] LR: {current_lr:.2e} -> {new_lr:.2e}")
            if patience_count >= PATIENCE:
                print(f"    Early stopping tai epoch {epoch}.")
                break

    # Danh gia test set (1 lan duy nhat)
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
# 7. RESUME HELPERS - LUU / LOAD KET QUA TUNG FOLD
# ==========================================
import json

def _progress_path(model_name):
    """File JSON luu ket qua cac fold da chay xong."""
    return os.path.join(CHECKPOINT_DIR, f"progress_{model_name}.json")

def load_progress(model_name):
    """Doc ket qua cac fold da chay truoc do. Tra ve dict {fold_name: result}."""
    path = _progress_path(model_name)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  [Resume] Tim thay {len(data)} fold da chay: {list(data.keys())}")
        return data
    return {}

def save_progress(model_name, fold_name, result):
    """Luu ket qua 1 fold vao JSON ngay sau khi fold do ket thuc."""
    path  = _progress_path(model_name)
    data  = load_progress(model_name) if os.path.exists(path) else {}
    data[fold_name] = result
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  [Resume] Da luu progress -> {path}")


# ==========================================
# 8. TRAIN PIPELINE 1 MODEL QUA TAT CA FOLD
# ==========================================
def train_pipeline(model_name, folds):
    display_name = MODEL_CONFIGS[model_name]["display_name"]
    print(f"\n{'='*65}")
    print(f" HUAN LUYEN: {display_name} ({model_name}) | {len(folds)} fold")
    print(f"{'='*65}")

    # Load ket qua cac fold da chay truoc do (neu co)
    done = load_progress(model_name)

    fold_results = []
    for fold_name in folds:
        if fold_name in done:
            # SKIP: fold nay da chay roi, lay ket qua cu
            r = done[fold_name]
            print(f"\n  [SKIP] {fold_name} da co ket qua: "
                  f"Test Acc={r['test_acc']}% | F1={r['test_f1']}")
            fold_results.append(r)
            continue

        # TRAIN: fold nay chua co ket qua
        fold_dir = os.path.join(KFOLD_DATASET_DIR, fold_name)
        result   = train_one_fold(model_name, fold_name, fold_dir)
        fold_results.append(result)

        # Luu ngay sau khi fold xong -> Colab crash van khong mat
        save_progress(model_name, fold_name, result)

    accs = [r["test_acc"] for r in fold_results]
    f1s  = [r["test_f1"]  for r in fold_results]

    mean_acc = statistics.mean(accs)
    std_acc  = statistics.pstdev(accs) if len(accs) > 1 else 0.0
    mean_f1  = statistics.mean(f1s)
    std_f1   = statistics.pstdev(f1s)  if len(f1s)  > 1 else 0.0

    print(f"\n  [{display_name}] TONG KET {len(folds)} FOLD:")
    print(f"  Test Acc : {mean_acc:.2f}% +/- {std_acc:.2f}%")
    print(f"  Macro F1 : {mean_f1:.4f} +/- {std_f1:.4f}")

    # Tinh lightweight metrics (dung fold_1 de do, khong train them)
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

    # ---- Bang tong ket ----
    print("\n" + "=" * 80)
    print(" BANG TONG KET - 5-FOLD CROSS VALIDATION")
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

    # ---- Luu CSV chi tiet tung fold ----
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["model", "fold", "test_acc", "test_f1", "best_val_f1"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_fold_rows)
    print(f"\nDa luu chi tiet tung fold : {RESULTS_CSV}")

    # ---- Luu CSV tong ket ----
    summary_rows = [
        {k: v for k, v in r.items() if k != "fold_detail"}
        for r in final_results
    ]
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Da luu tong ket 5-fold    : {SUMMARY_CSV}")
    print(f"Ket thuc                  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")