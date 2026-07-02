"""
Buoc 3: Training pipeline — 5-Fold Cross Validation voi 3 model nhe.

Model: EfficientNet-Lite3 (chinh), MobileNetV3-Small, ShuffleNetV2-x1.0
Chon best checkpoint theo Val Macro F1 (phu hop class imbalance).
Output: checkpoints/best_<model>_<fold>.pth, kfold_results_detail.csv, kfold_results_summary.csv
"""

import os
import csv
import json
import statistics
from collections import Counter
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt

from model_utils import (
    ALL_MODELS,
    MODEL_CONFIGS,
    NUM_CLASSES,
    CLASS_NAMES,
    get_model,
    get_param_groups,
    count_params,
    measure_inference_time,
)
from paths_config import KFOLD_DATASET_DIR, CHECKPOINT_DIR, EVAL_OUTPUT_DIR, PROJECT_ROOT, ensure_dirs

# ==========================================
# 1. CAU HINH
# ==========================================
BATCH_SIZE        = 16
EPOCHS            = 100
LR_BACKBONE       = 1e-5   # LR nho cho backbone da pretrained - chong overfit khi fine-tune
LR_HEAD           = 1e-4   # LR lon hon cho classifier head moi khoi tao
WARMUP_EPOCHS     = 3      # so epoch dau tang dan LR tu 0 -> LR muc tieu (on dinh, dac biet cho ShuffleNet)
PATIENCE          = 10
WEIGHT_DECAY      = 5e-4
LABEL_SMOOTHING   = 0.1
GRAD_CLIP_NORM    = 1.0    # chong no gradient / dao dong dau training
MIXUP_ALPHA       = 0.2    # 0 = tat mixup
MIXUP_PROB        = 0.5    # xac suat ap dung mixup cho 1 batch train
ENABLE_COLOR_JITTER = True # bat/tat de ablation
RANDOM_SEED       = 42

RESULTS_CSV  = os.path.join(PROJECT_ROOT, "kfold_results_detail.csv")    # ket qua tung fold
SUMMARY_CSV  = os.path.join(PROJECT_ROOT, "kfold_results_summary.csv")   # trung binh cuoi
ensure_dirs(CHECKPOINT_DIR)

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# ==========================================
# 2. THIET BI
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Chay tren thiet bi : {DEVICE}")
print(f"Bat dau            : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ==========================================
# 3. TIM FOLD TU DONG
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

    train_transform_list = [
        transforms.Resize((img_size, img_size),
                          interpolation=transforms.InterpolationMode.BILINEAR,
                          antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
    ]
    if ENABLE_COLOR_JITTER:
        train_transform_list.append(
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1)
        )
    train_transform_list += [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ]
    train_transforms = transforms.Compose(train_transform_list)
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

    for split_name, ds in (("train", train_dataset), ("val", val_dataset), ("test", test_dataset)):
        assert ds.classes == CLASS_NAMES, (
            f"Thu tu class cua ImageFolder ({split_name}) khong khop CLASS_NAMES: "
            f"{ds.classes} != {CLASS_NAMES}"
        )

    # WeightedRandomSampler: oversample minority class (KHONG dung them class_weights
    # trong loss cung luc - tranh double-correction, xem log.md Round 3)
    labels         = [lbl for _, lbl in train_dataset.samples]
    class_counts   = Counter(labels)
    sample_weights = [1.0 / class_counts[lbl] for lbl in labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    # num_workers=4: Windows yeu cau if __name__=="__main__" guard (da co o cuoi file)
    kw = dict(num_workers=4, pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, **kw)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,   **kw)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,   **kw)

    return train_loader, val_loader, test_loader, train_dataset


def log_class_distribution(train_dataset):
    """In phan bo lop cua tap train de theo doi - KHONG dung de tinh class weight
    cho loss vi da co WeightedRandomSampler can bang batch roi."""
    labels = [lbl for _, lbl in train_dataset.samples]
    counts = Counter(labels)
    print(f"  Class counts : {dict(sorted(counts.items()))}")


def mixup_data(x, y, alpha):
    """Tron 2 anh trong batch theo he so lam ~ Beta(alpha, alpha)."""
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam

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
# 5. VONG LAP TRAIN / EVAL MOT EPOCH
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

            use_mixup = MIXUP_ALPHA > 0 and torch.rand(1).item() < MIXUP_PROB
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
# 6. TRAIN 1 FOLD
# ==========================================
def train_one_fold(model_name, fold_name, fold_dir):
    print(f"\n  --- {MODEL_CONFIGS[model_name]['display_name']} | {fold_name} ---")

    train_loader, val_loader, test_loader, train_ds = build_loaders(model_name, fold_dir)
    log_class_distribution(train_ds)

    model = get_model(model_name).to(DEVICE)

    criterion_train = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    criterion_eval  = nn.CrossEntropyLoss()

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
                print(f"    Early stopping tai epoch {epoch}.")
                break

    plot_learning_curve(history, model_name, fold_name)

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
    std_acc  = statistics.stdev(accs) if len(accs) > 1 else 0.0
    mean_f1  = statistics.mean(f1s)
    std_f1   = statistics.stdev(f1s)  if len(f1s)  > 1 else 0.0

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