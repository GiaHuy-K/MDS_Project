"""
Buoc 3: Training pipeline cho 3 model NHE (Lightweight) - 1 FOLD DUY NHAT
  - EfficientNet-Lite3   (model chinh cua de tai)
  - MobileNetV3-Small
  - ShuffleNetV2-x1.0

=== TOAN BO THAY DOI SO VOI BAN GOC ===
[BUG FIX] torch.enable_grad() / torch.no_grad() dung dung: tach 2 nhanh if/else rieng
[BUG FIX] Scheduler unfreeze: 2 param_group LR rieng biet (head vs backbone), khong reset chung
[FIX]     Best model chon theo MACRO F1 (khong phai accuracy) -> phu hop class imbalance
[FIX]     Augmentation nhe hon: bo RandomErasing + RandomAffine, giam rotation 20->10
[FIX]     torch.load them weights_only=True (tranh warning PyTorch >= 2.0)
[ADD]     WeightedRandomSampler: oversample minority class + class weights trong loss
[ADD]     Gradual Unfreezing dung: head train voi LR cao, backbone unfreeze voi LR rieng thap
[ADD]     run_epoch tra ve macro_f1 -> dung lam metric chinh thay accuracy
[ADD]     Luu classification report + confusion matrix ra file .txt
[ADD]     CSV ket qua them cot macro_f1, best_val_f1
[ADD]     cudnn.deterministic = True dam bao ket qua co the tai tao
"""

import os
import csv
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
FOLD_NAME = "fold_1"   # Chi train tren 1 fold nay
DATA_DIR  = os.path.join(KFOLD_DATASET_DIR, FOLD_NAME)
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR   = os.path.join(DATA_DIR, "val")
TEST_DIR  = os.path.join(DATA_DIR, "test")

BATCH_SIZE      = 32
EPOCHS          = 40
HEAD_LR         = 3e-4   # LR cho classifier head (epoch 1 -> FREEZE_EPOCHS)
BACKBONE_LR     = 1e-5   # LR rieng cho backbone sau khi unfreeze (THAP hon nhieu)
PATIENCE        = 7       # So epoch cho truoc khi early stopping
WEIGHT_DECAY    = 5e-4
LABEL_SMOOTHING = 0.1     # Chi dung trong criterion_train
FREEZE_EPOCHS   = 5       # So epoch chi train head truoc khi unfreeze backbone
RANDOM_SEED     = 42

RESULTS_CSV  = os.path.join(PROJECT_ROOT, "lightweight_comparison_results.csv")
REPORT_DIR   = os.path.join(PROJECT_ROOT, "reports")
ensure_dirs(CHECKPOINT_DIR)
ensure_dirs(REPORT_DIR)

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Chay tren thiet bi: {DEVICE}")
print(f"Training tren: {FOLD_NAME}")
print(f"Thoi gian bat dau: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ==========================================
# 2. FREEZE / UNFREEZE HELPERS
# ==========================================
HEAD_KEYWORDS = {
    "efficientnet_lite3": ("classifier",),
    "mobilenet_small"   : ("classifier",),
    "shufflenet"        : ("fc",),
}

def _is_head_param(name, model_name):
    keywords = HEAD_KEYWORDS.get(model_name, ("classifier", "fc", "head"))
    return any(kw in name for kw in keywords)

def freeze_backbone(model, model_name):
    """Freeze toan bo backbone, chi de head trainable."""
    frozen, trainable = 0, 0
    for name, param in model.named_parameters():
        if _is_head_param(name, model_name):
            param.requires_grad = True
            trainable += 1
        else:
            param.requires_grad = False
            frozen += 1
    print(f"  [Freeze] Frozen: {frozen} params | Trainable (head only): {trainable} params")

def unfreeze_backbone_group(model, model_name, optimizer):
    """
    [BUG FIX] Them backbone vao optimizer nhu 1 param_group rieng voi BACKBONE_LR.
    Head giu nguyen LR hien tai cua no -> KHONG reset scheduler.
    """
    backbone_params = [
        p for n, p in model.named_parameters()
        if not _is_head_param(n, model_name)
    ]
    for p in backbone_params:
        p.requires_grad = True

    # Them backbone nhu 1 param_group moi voi LR rieng
    optimizer.add_param_group({
        "params"      : backbone_params,
        "lr"          : BACKBONE_LR,
        "weight_decay": WEIGHT_DECAY,
    })
    print(f"  [Unfreeze] Backbone added as param_group[1] | LR={BACKBONE_LR:.1e}")
    print(f"  [Unfreeze] Head LR giu nguyen: {optimizer.param_groups[0]['lr']:.2e}")


# ==========================================
# 3. DATALOADERS
# ==========================================
def build_loaders(model_name):
    cfg      = MODEL_CONFIGS[model_name]
    img_size = cfg["img_size"]
    mean, std = cfg["mean"], cfg["std"]

    # [FIX] Augmentation nhe hon: phu hop dataset nho (~990 anh)
    # Bo RandomErasing va RandomAffine: qua aggressive, co the xoa vung mun quan trong
    train_transforms = transforms.Compose([
        transforms.Resize((img_size + 20, img_size + 20)),
        transforms.RandomCrop(img_size),          # them diversity vi tri
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),            # giam tu 20 -> 10
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transforms)
    val_dataset   = datasets.ImageFolder(root=VAL_DIR,   transform=eval_transforms)
    test_dataset  = datasets.ImageFolder(root=TEST_DIR,  transform=eval_transforms)

    # [ADD] WeightedRandomSampler: oversample minority class khi lay batch
    # Ket hop voi class weights trong loss -> danh manh vao Level_2, Level_3 (it anh)
    labels         = [lbl for _, lbl in train_dataset.samples]
    class_counts   = Counter(labels)
    sample_weights = [1.0 / class_counts[lbl] for lbl in labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              sampler=sampler, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                              shuffle=False,  num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                              shuffle=False,  num_workers=0, pin_memory=True)

    print(
        f"[{model_name}] img_size={img_size} | "
        f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)} anh"
    )
    return train_loader, val_loader, test_loader, train_dataset


def build_class_weights(train_dataset):
    """Class weight ti le nghich voi so luong anh moi class."""
    labels  = [lbl for _, lbl in train_dataset.samples]
    counts  = Counter(labels)
    total   = sum(counts.values())
    weights = torch.tensor([total / counts[i] for i in range(NUM_CLASSES)], dtype=torch.float)
    weights = weights / weights.mean()   # chuan hoa quanh 1.0
    print(f"  Class counts : {dict(sorted(counts.items()))}")
    print(f"  Class weights: {[round(w, 3) for w in weights.tolist()]}")
    return weights


# ==========================================
# 4. VONG LAP TRAIN / EVAL MOT EPOCH
# ==========================================
def run_epoch(model, loader, criterion, optimizer=None, desc=""):
    """
    [BUG FIX] Tach thanh 2 nhanh if/else rieng biet.
    torch.enable_grad() KHONG the dung nhu context manager truc tiep.

    Return: (avg_loss, accuracy, macro_f1, all_preds, all_labels)
    """
    is_train = optimizer is not None
    total_loss, total_correct = 0.0, 0
    all_preds, all_labels = [], []

    if is_train:
        # ---- TRAIN BRANCH: gradient bat ON ----
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
        # ---- EVAL BRANCH: gradient tat OFF ----
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
# 5. TRAIN PIPELINE DAY DU CHO 1 MODEL
# ==========================================
def train_pipeline(model_name):
    print(f"\n{'='*65}")
    print(f" HUAN LUYEN: {MODEL_CONFIGS[model_name]['display_name']} ({model_name})")
    print(f"{'='*65}")

    train_loader, val_loader, test_loader, train_dataset = build_loaders(model_name)
    class_weights = build_class_weights(train_dataset).to(DEVICE)

    model = get_model(model_name).to(DEVICE)

    # Giai doan 1: chi train head voi HEAD_LR
    freeze_backbone(model, model_name)

    criterion_train = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    criterion_eval  = nn.CrossEntropyLoss()

    # Optimizer khoi tao voi head params
    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer   = optim.AdamW(head_params, lr=HEAD_LR, weight_decay=WEIGHT_DECAY)

    # ReduceLROnPlateau: adaptive, tot hon CosineAnnealing voi dataset nho
    # Giam LR khi val_f1 khong cai thien sau 3 epoch
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-7
    )

    best_val_f1    = 0.0
    patience_count = 0
    backbone_unfrozen = False
    save_path = os.path.join(CHECKPOINT_DIR, f"best_{model_name}.pth")

    for epoch in range(1, EPOCHS + 1):

        # [BUG FIX] Giai doan 2: unfreeze backbone vao dung 1 param_group rieng
        # Khong reset scheduler -> head giu duoc LR hien tai
        if not backbone_unfrozen and epoch == FREEZE_EPOCHS + 1:
            unfreeze_backbone_group(model, model_name, optimizer)
            backbone_unfrozen = True
            patience_count = 0   # Reset patience sau unfreeze

        current_lr_head = optimizer.param_groups[0]["lr"]

        train_loss, train_acc, train_f1, _, _ = run_epoch(
            model, train_loader, criterion_train, optimizer,
            desc=f"[{model_name}] E{epoch:02d}/{EPOCHS} Train"
        )
        val_loss, val_acc, val_f1, _, _ = run_epoch(
            model, val_loader, criterion_eval,
            desc=f"[{model_name}] E{epoch:02d}/{EPOCHS} Val"
        )

        # Scheduler step dua tren val_f1 (metric chinh)
        scheduler.step(val_f1)

        # Log LR backbone neu da unfreeze
        lr_log = f"LR_head={current_lr_head:.2e}"
        if backbone_unfrozen and len(optimizer.param_groups) > 1:
            lr_log += f" | LR_backbone={optimizer.param_groups[1]['lr']:.1e}"

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | {lr_log} | "
            f"Train Loss={train_loss:.4f} Acc={train_acc*100:.1f}% F1={train_f1:.4f} | "
            f"Val   Loss={val_loss:.4f}  Acc={val_acc*100:.1f}%  F1={val_f1:.4f}"
        )

        # [FIX] Chon best model theo MACRO F1, khong phai accuracy
        if val_f1 > best_val_f1:
            best_val_f1    = val_f1
            patience_count = 0
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved best model  (val_f1={val_f1:.4f} | val_acc={val_acc*100:.2f}%)")
        else:
            patience_count += 1
            # Theo doi LR co giam khong (thay verbose=True da bi remove)
            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr < current_lr_head:
                print(f"  [Scheduler] LR_head giam: {current_lr_head:.2e} -> {new_lr:.2e}")
            if patience_count >= PATIENCE:
                print(f"  Early stopping sau {epoch} epochs (patience={PATIENCE}).")
                break

    # ==========================================
    # DANH GIA CUOI CUNG TREN TEST SET (1 LAN DUY NHAT)
    # ==========================================
    print(f"\n--- Danh gia tren TEST SET (chay 1 lan duy nhat) ---")
    model.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))
    test_loss, test_acc, test_f1, test_preds, test_labels = run_epoch(
        model, test_loader, criterion_eval,
        desc=f"[{model_name}] Test"
    )

    print(f"Test Loss : {test_loss:.4f}")
    print(f"Test Acc  : {test_acc*100:.2f}%")
    print(f"Macro F1  : {test_f1:.4f}")

    report_str = classification_report(test_labels, test_preds, target_names=CLASS_NAMES)
    cm         = confusion_matrix(test_labels, test_preds)
    print("\nClassification Report:")
    print(report_str)
    print("Confusion Matrix:")
    print(cm)

    # [ADD] Luu classification report + confusion matrix ra file .txt
    report_path = os.path.join(REPORT_DIR, f"report_{model_name}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Model     : {MODEL_CONFIGS[model_name]['display_name']}\n")
        f.write(f"Fold      : {FOLD_NAME}\n")
        f.write(f"Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test Acc  : {test_acc*100:.2f}%\n")
        f.write(f"Macro F1  : {test_f1:.4f}\n")
        f.write(f"Best Val F1: {best_val_f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report_str)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))
    print(f"  -> Report luu tai: {report_path}")

    # ==========================================
    # THONG SO LIGHTWEIGHT: params, size, inference time
    # ==========================================
    total_params, trainable_params, size_mb = count_params(model)
    cpu_ms = measure_inference_time(model, model_name, torch.device("cpu"))
    gpu_ms = None
    if torch.cuda.is_available():
        gpu_ms = measure_inference_time(model, model_name, torch.device("cuda"))

    print(f"\nTham so   : {total_params/1e6:.2f}M | Dung luong: {size_mb:.2f} MB")
    print(f"Inference CPU: {cpu_ms:.2f} ms/anh"
          + (f" | GPU: {gpu_ms:.2f} ms/anh" if gpu_ms else ""))

    del model
    torch.cuda.empty_cache()

    return {
        "model"        : MODEL_CONFIGS[model_name]["display_name"],
        "fold"         : FOLD_NAME,
        "test_acc"     : round(test_acc * 100, 2),
        "test_macro_f1": round(test_f1, 4),
        "best_val_f1"  : round(best_val_f1, 4),
        "params_M"     : round(total_params / 1e6, 2),
        "size_MB"      : round(size_mb, 2),
        "cpu_ms"       : round(cpu_ms, 2),
        "gpu_ms"       : round(gpu_ms, 2) if gpu_ms else "",
    }


# ==========================================
# 6. CHAY TAT CA MODEL & XUAT BANG TONG KET
# ==========================================
if __name__ == "__main__":
    models_to_run = ALL_MODELS  # ["efficientnet_lite3", "mobilenet_small", "shufflenet"]
    final_results = []

    for m in models_to_run:
        final_results.append(train_pipeline(m))

    # In bang tong ket
    print("\n" + "=" * 85)
    print(" BANG TONG KET - LIGHTWEIGHT MODEL COMPARISON")
    print(f" Fold: {FOLD_NAME}")
    print("=" * 85)
    header = (
        f"{'Model':<22}{'Test Acc':>10}{'Test F1':>10}"
        f"{'BestValF1':>12}{'Params(M)':>11}{'Size(MB)':>10}{'CPU(ms)':>10}{'GPU(ms)':>10}"
    )
    print(header)
    print("-" * 85)
    for r in final_results:
        gpu_str = f"{r['gpu_ms']:.2f}" if r["gpu_ms"] != "" else "   -"
        print(
            f"{r['model']:<22}"
            f"{r['test_acc']:>9.2f}%"
            f"{r['test_macro_f1']:>10.4f}"
            f"{r['best_val_f1']:>12.4f}"
            f"{r['params_M']:>11.2f}"
            f"{r['size_MB']:>10.2f}"
            f"{r['cpu_ms']:>10.2f}"
            f"{gpu_str:>10}"
        )

    # Luu CSV
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(final_results[0].keys()))
        writer.writeheader()
        writer.writerows(final_results)
    print(f"\nDa luu CSV : {RESULTS_CSV}")
    print(f"Da luu reports: {REPORT_DIR}/")
    print(f"Ket thuc   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")