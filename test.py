"""
Test nhanh pipeline voi:
  - 1 fold (fold_1)
  - 1 model (efficientnet_lite3)
  - 5 epoch
  - In thoi gian moi epoch de uoc tinh tong thoi gian train that

Chay: python test_speed.py
"""

import os
import time
from collections import Counter
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.metrics import f1_score
from tqdm import tqdm

from model_utils import MODEL_CONFIGS, NUM_CLASSES, get_model
from paths_config import KFOLD_DATASET_DIR, CHECKPOINT_DIR, ensure_dirs

# ==========================================
# CAU HINH TEST
# ==========================================
MODEL_NAME  = "efficientnet_lite3"
FOLD_NAME   = "fold_1"
TEST_EPOCHS = 5       # Chi chay 5 epoch de do toc do
BATCH_SIZE  = 32
LR          = 1e-4
WEIGHT_DECAY = 5e-4

FOLD_DIR = os.path.join(KFOLD_DATASET_DIR, FOLD_NAME)
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 55)
print(" SPEED TEST - 1 fold / 1 model / 5 epochs")
print("=" * 55)
print(f"Device    : {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
print(f"Model     : {MODEL_NAME}")
print(f"Fold      : {FOLD_NAME}")
print(f"Bat dau   : {datetime.now().strftime('%H:%M:%S')}")
print()

# ==========================================
# DATALOADER
# ==========================================
cfg      = MODEL_CONFIGS[MODEL_NAME]
img_size = cfg["img_size"]
mean, std = cfg["mean"], cfg["std"]

train_transforms = transforms.Compose([
    transforms.Resize((img_size, img_size),
                      interpolation=transforms.InterpolationMode.BILINEAR,
                      antialias=True),
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
    root=os.path.join(FOLD_DIR, "train"), transform=train_transforms)
val_dataset   = datasets.ImageFolder(
    root=os.path.join(FOLD_DIR, "val"),   transform=eval_transforms)

labels         = [lbl for _, lbl in train_dataset.samples]
class_counts   = Counter(labels)
sample_weights = [1.0 / class_counts[lbl] for lbl in labels]
sampler = WeightedRandomSampler(
    weights=sample_weights, num_samples=len(sample_weights), replacement=True
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          sampler=sampler, num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                          shuffle=False,  num_workers=0, pin_memory=True)

print(f"Train: {len(train_dataset)} anh | Val: {len(val_dataset)} anh")
print(f"Batches/epoch: train={len(train_loader)} | val={len(val_loader)}")
print()

# ==========================================
# MODEL & OPTIMIZER
# ==========================================
model     = get_model(MODEL_NAME).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# ==========================================
# TRAIN LOOP - DO THOI GIAN
# ==========================================
epoch_times = []

for epoch in range(1, TEST_EPOCHS + 1):
    t_epoch_start = time.perf_counter()

    # ---- Train ----
    model.train()
    t_train_start = time.perf_counter()
    total_correct = 0
    for inputs, labels_b in tqdm(train_loader,
                                 desc=f"Epoch {epoch}/{TEST_EPOCHS} Train",
                                 leave=False):
        inputs, labels_b = inputs.to(DEVICE), labels_b.to(DEVICE)
        outputs = model(inputs)
        loss    = criterion(outputs, labels_b)
        preds   = outputs.argmax(dim=1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_correct += (preds == labels_b).sum().item()
    train_acc  = total_correct / len(train_dataset)
    t_train    = time.perf_counter() - t_train_start

    # ---- Val ----
    model.eval()
    t_val_start = time.perf_counter()
    all_preds, all_labels_v = [], []
    with torch.no_grad():
        for inputs, labels_b in tqdm(val_loader,
                                     desc=f"Epoch {epoch}/{TEST_EPOCHS} Val  ",
                                     leave=False):
            inputs, labels_b = inputs.to(DEVICE), labels_b.to(DEVICE)
            outputs = model(inputs)
            preds   = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels_v.extend(labels_b.cpu().tolist())
    val_acc  = sum(p == l for p, l in zip(all_preds, all_labels_v)) / len(val_dataset)
    val_f1   = f1_score(all_labels_v, all_preds, average="macro", zero_division=0)
    t_val    = time.perf_counter() - t_val_start

    t_epoch  = time.perf_counter() - t_epoch_start
    epoch_times.append(t_epoch)

    print(
        f"Epoch {epoch:02d}/{TEST_EPOCHS} | "
        f"Train={t_train:.1f}s | Val={t_val:.1f}s | Total={t_epoch:.1f}s | "
        f"TrainAcc={train_acc*100:.1f}% | ValAcc={val_acc*100:.1f}% | ValF1={val_f1:.4f}"
    )

# ==========================================
# UOC TINH TONG THOI GIAN
# ==========================================
avg_epoch   = sum(epoch_times) / len(epoch_times)
# Bo epoch 1 (warm up chậm hơn)
stable_avg  = sum(epoch_times[1:]) / max(len(epoch_times) - 1, 1)

REAL_EPOCHS = 40   # so epoch thuc te trong 03_train_kfold.py
NUM_FOLDS   = 5
NUM_MODELS  = 3

est_per_fold  = stable_avg * REAL_EPOCHS          # giay
est_per_model = est_per_fold * NUM_FOLDS           # giay
est_total     = est_per_model * NUM_MODELS         # giay

def fmt(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h {m}m {s}s"

print()
print("=" * 55)
print(" UOC TINH THOI GIAN TRAIN THAT")
print("=" * 55)
print(f"Trung binh moi epoch (on dinh): {stable_avg:.1f}s")
print(f"1 fold  (40 epoch)             : {fmt(est_per_fold)}")
print(f"1 model (5 fold)               : {fmt(est_per_model)}")
print(f"3 model (15 fold tong cong)    : {fmt(est_total)}")
print()
print("GPU utilization: mo Task Manager -> GPU 1 (RTX 3050 Ti)")
print("Neu < 50% -> van con CPU bottleneck")
print(f"Ket thuc: {datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    pass