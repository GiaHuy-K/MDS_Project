"""
Buoc 3: Training pipeline cho 3 model NHE (Lightweight):
  - EfficientNet-Lite3   (model chinh cua de tai)
  - MobileNetV3-Small
  - ShuffleNetV2-x1.0

Quy trinh:
- Train tren tap train (co augmentation)
- Chon best model dua tren VAL accuracy (KHONG dung test)
- Early stopping neu val khong cai thien
- Sau khi train xong, danh gia 1 LAN DUY NHAT tren test set
- In bang tong ket: Accuracy, so tham so, dung luong, thoi gian inference
"""

import os
import csv

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
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
DATA_DIR  = os.path.join(KFOLD_DATASET_DIR, "fold_1")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR   = os.path.join(DATA_DIR, "val")
TEST_DIR  = os.path.join(DATA_DIR, "test")

BATCH_SIZE   = 32
EPOCHS       = 30
LR           = 1e-4
PATIENCE     = 7
WEIGHT_DECAY = 1e-4
RANDOM_SEED  = 42

RESULTS_CSV = os.path.join(PROJECT_ROOT, "lightweight_comparison_results.csv")
ensure_dirs(CHECKPOINT_DIR)

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Chay tren thiet bi: {DEVICE}")


# ==========================================
# 2. TRANSFORMS THEO TUNG MODEL
# ==========================================
def build_loaders(model_name):
    cfg      = MODEL_CONFIGS[model_name]
    img_size = cfg["img_size"]
    mean, std = cfg["mean"], cfg["std"]

    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
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

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=True)

    print(
        f"[{model_name}] img_size={img_size} | "
        f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)} anh"
    )
    return train_loader, val_loader, test_loader


# ==========================================
# 3. VONG LAP TRAIN / EVAL MOT EPOCH
# ==========================================
def run_epoch(model, loader, criterion, optimizer=None, desc=""):
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
    return avg_loss, accuracy, all_preds, all_labels


# ==========================================
# 4. TRAIN PIPELINE DAY DU CHO 1 MODEL
# ==========================================
def train_pipeline(model_name):
    print(f"\n{'='*60}")
    print(f" HUAN LUYEN: {MODEL_CONFIGS[model_name]['display_name']} ({model_name})")
    print(f"{'='*60}")

    train_loader, val_loader, test_loader = build_loaders(model_name)

    model     = get_model(model_name).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc   = 0.0
    patience_count = 0
    save_path = os.path.join(CHECKPOINT_DIR, f"best_{model_name}.pth")

    for epoch in range(1, EPOCHS + 1):
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc, _, _ = run_epoch(
            model, train_loader, criterion, optimizer,
            desc=f"[{model_name}] Epoch {epoch}/{EPOCHS} Train"
        )
        val_loss, val_acc, _, _ = run_epoch(
            model, val_loader, criterion,
            desc=f"[{model_name}] Epoch {epoch}/{EPOCHS} Val"
        )
        scheduler.step()

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            patience_count = 0
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved best model (val_acc={val_acc*100:.2f}%)")
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"  Early stopping sau {epoch} epochs.")
                break

    # ==========================================
    # DANH GIA CUOI CUNG TREN TEST SET (1 LAN DUY NHAT)
    # ==========================================
    print(f"\n--- Danh gia tren TEST SET (chay 1 lan duy nhat) ---")
    model.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))
    test_loss, test_acc, test_preds, test_labels = run_epoch(
        model, test_loader, criterion,
        desc=f"[{model_name}] Test"
    )
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=CLASS_NAMES))
    print("Confusion Matrix:")
    print(confusion_matrix(test_labels, test_preds))

    # ==========================================
    # THONG SO LIGHTWEIGHT
    # ==========================================
    total_params, trainable_params, size_mb = count_params(model)
    cpu_ms = measure_inference_time(model, model_name, torch.device("cpu"))
    gpu_ms = None
    if torch.cuda.is_available():
        gpu_ms = measure_inference_time(model, model_name, torch.device("cuda"))

    print(f"\nTham so: {total_params/1e6:.2f}M | Dung luong: {size_mb:.2f} MB")
    print(f"Inference CPU: {cpu_ms:.2f} ms/anh"
          + (f" | GPU: {gpu_ms:.2f} ms/anh" if gpu_ms else ""))

    del model
    torch.cuda.empty_cache()

    return {
        "model"   : MODEL_CONFIGS[model_name]["display_name"],
        "test_acc": test_acc,
        "params_M": total_params / 1e6,
        "size_MB" : size_mb,
        "cpu_ms"  : cpu_ms,
        "gpu_ms"  : gpu_ms if gpu_ms else "",
    }


# ==========================================
# 5. CHAY TAT CA MODEL & XUAT BANG TONG KET
# ==========================================
if __name__ == "__main__":
    models_to_run = ALL_MODELS  # ["efficientnet_lite3", "mobilenet_small", "shufflenet"]
    final_results = []

    for m in models_to_run:
        final_results.append(train_pipeline(m))

    print("\n" + "=" * 70)
    print(" BANG TONG KET - LIGHTWEIGHT MODEL COMPARISON")
    print("=" * 70)
    header = (
        f"{'Model':<22}{'Test Acc':>10}{'Params(M)':>12}"
        f"{'Size(MB)':>10}{'CPU(ms)':>10}{'GPU(ms)':>10}"
    )
    print(header)
    print("-" * 70)
    for r in final_results:
        gpu_str = f"{r['gpu_ms']:.2f}" if r["gpu_ms"] != "" else "-"
        print(
            f"{r['model']:<22}{r['test_acc']*100:>9.2f}%"
            f"{r['params_M']:>12.2f}{r['size_MB']:>10.2f}"
            f"{r['cpu_ms']:>10.2f}{gpu_str:>10}"
        )

    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(final_results[0].keys()))
        writer.writeheader()
        writer.writerows(final_results)
    print(f"\nDa luu bang so sanh vao: {RESULTS_CSV}")