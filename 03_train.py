"""
Buoc 3 (PHIEN BAN 5-FOLD CV): Training pipeline cho 3 model NHE
tren du 5-fold cua ACNE04 (dataset_acne04_folds/fold_* tu 01b_split_dataset_kfold.py).

=== CAC THAY DOI SO VOI BAN FOLD DON ===
Anti-overfit:
  - Augmentation manh: RandomAffine, RandomErasing
  - weight_decay tang len 5e-4, label_smoothing=0.1 (chi khi train)
  - Class weighting tu dong tu so luong anh moi class
  - criterion_train / criterion_eval tach rieng (val+test dung CE thuan)

Gradual Unfreezing (sua 4 loi):
  [1] add_param_group thay vi tao moi optimizer -> giu trang thai scheduler
  [2] Log trainable params sau freeze_backbone -> phat hien freeze sai ten layer
  [3] Reset patience_count sau unfreeze -> tranh dung som khi backbone vua mo
  [4] cudnn.benchmark=False kem deterministic=True -> cu phap chinh xac, khong lam cham them

5-Fold CV:
  - Tu dong tim tat ca fold_* trong KFOLD_DATASET_DIR
  - Luu checkpoint rieng: checkpoints/best_<model>_<fold>.pth (KHONG ghi de)
  - Tong hop mean +/- std accuracy qua 5 fold cho bang so sanh trong bao cao
  - Params/size/inference chi do 1 lan/model (dac trung kien truc, khong phu thuoc fold)
"""

import os
import csv
import statistics
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from model_utils import (
    ALL_MODELS, MODEL_CONFIGS, NUM_CLASSES, CLASS_NAMES,
    get_model, count_params, measure_inference_time,
)
from paths_config import KFOLD_DATASET_DIR, CHECKPOINT_DIR, PROJECT_ROOT, ensure_dirs

# ==========================================
# 1. CAU HINH
# ==========================================
BATCH_SIZE      = 32
EPOCHS          = 40
LR              = 1e-4
PATIENCE        = 5       # giam tu 7->5: dung som hon
WEIGHT_DECAY    = 5e-4    # tang tu 1e-4->5e-4
LABEL_SMOOTHING = 0.1     # chi ap dung khi train, KHONG dung khi eval

# Gradual Unfreezing: freeze backbone trong N epoch dau, sau do unfreeze voi LR nho
FREEZE_EPOCHS = 5         # so epoch freeze backbone (0 = tat, train toan bo ngay tu dau)
UNFREEZE_LR   = LR * 0.1  # LR cho backbone khi unfreeze (nho hon LR cua head)

RANDOM_SEED = 42
RESULTS_CSV = os.path.join(PROJECT_ROOT, "lightweight_comparison_results_kfold.csv")
ensure_dirs(CHECKPOINT_DIR)

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)

# [SUA 4] benchmark=False phai di kem voi deterministic=True
# Neu khong can reproducibility tuyet doi -> co the dat ca 2 ve False de tang toc
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Chay tren thiet bi: {DEVICE}")


# ==========================================
# 2. TIM FOLD CO SAN
# ==========================================
def discover_folds():
    if not os.path.isdir(KFOLD_DATASET_DIR):
        raise FileNotFoundError(
            f"Khong tim thay {KFOLD_DATASET_DIR}.\n"
            f"-> Hay chay 01b_split_dataset_kfold.py truoc."
        )
    folds = sorted(
        [d for d in os.listdir(KFOLD_DATASET_DIR)
         if d.startswith("fold_") and os.path.isdir(os.path.join(KFOLD_DATASET_DIR, d))],
        key=lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else x,
    )
    if not folds:
        raise FileNotFoundError(f"Khong tim thay thu muc fold_* nao trong {KFOLD_DATASET_DIR}.")
    return folds


# ==========================================
# 3. FREEZE / UNFREEZE BACKBONE
# ==========================================
# Ten layer cuoi (head/classifier) theo tung model, dung de xac dinh
# params NAO duoc giu lai khi freeze va NAO duoc them vao optimizer khi unfreeze.
HEAD_KEYWORDS = {
    "efficientnet_lite3": ("classifier",),
    "mobilenet_small"   : ("classifier",),
    "shufflenet"        : ("fc",),
}

def _is_head_param(name, model_name):
    return any(kw in name for kw in HEAD_KEYWORDS.get(model_name, ("classifier", "fc", "head")))


def freeze_backbone(model, model_name):
    """Freeze tat ca tru head/classifier. In log de kiem tra ten layer."""
    trainable = []
    for name, param in model.named_parameters():
        param.requires_grad = _is_head_param(name, model_name)
        if param.requires_grad:
            trainable.append(name)

    # [SUA 2] Log de phat hien ngay neu freeze nham (trainable rong = toan bo bi freeze)
    if trainable:
        print(f"  [Freeze] Trainable params ({len(trainable)}): {trainable}")
    else:
        print("  [CANH BAO] Khong co param nao trainable sau freeze_backbone!")
        print("  -> Kiem tra lai HEAD_KEYWORDS cho model nay.")


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True


# ==========================================
# 4. TRANSFORMS & DATALOADER
# ==========================================
def build_loaders(model_name, fold_dir):
    cfg = MODEL_CONFIGS[model_name]
    img_size, mean, std = cfg["img_size"], cfg["mean"], cfg["std"]

    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.2),
    ])
    eval_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.ImageFolder(root=os.path.join(fold_dir, "train"), transform=train_transforms)
    val_ds   = datasets.ImageFolder(root=os.path.join(fold_dir, "val"),   transform=eval_transforms)
    test_ds  = datasets.ImageFolder(root=os.path.join(fold_dir, "test"),  transform=eval_transforms)

    kw = dict(num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, **kw)

    print(f"  img_size={img_size} | Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader, train_ds


def build_class_weights(train_dataset):
    """Class weight ti le nghich voi so luong anh (giam bien lech lop)."""
    labels  = [lbl for _, lbl in train_dataset.samples]
    counts  = Counter(labels)
    total   = sum(counts.values())
    weights = torch.tensor(
        [total / counts[i] for i in range(NUM_CLASSES)], dtype=torch.float
    )
    weights = weights / weights.mean()   # chuan hoa ~ 1 de on dinh LR
    print(f"  Class counts : {dict(sorted(counts.items()))}")
    print(f"  Class weights: {[round(w, 3) for w in weights.tolist()]}")
    return weights


# ==========================================
# 5. VONG LAP TRAIN/VAL MOT EPOCH
# ==========================================
def run_epoch(model, loader, criterion, optimizer=None, desc=""):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total_correct = 0.0, 0
    all_preds, all_labels = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for inputs, labels in tqdm(loader, desc=desc, leave=False):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss    = criterion(outputs, labels)
            preds   = outputs.argmax(dim=1)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss    += loss.item() * inputs.size(0)
            total_correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return total_loss / len(loader.dataset), total_correct / len(loader.dataset), all_preds, all_labels


# ==========================================
# 6. TRAIN + DANH GIA 1 MODEL TREN 1 FOLD
# ==========================================
def train_one_fold(model_name, fold_name, fold_dir):
    print(f"\n  --- {MODEL_CONFIGS[model_name]['display_name']} | {fold_name} ---")

    train_loader, val_loader, test_loader, train_ds = build_loaders(model_name, fold_dir)
    class_weights = build_class_weights(train_ds).to(DEVICE)

    model = get_model(model_name).to(DEVICE)

    # Criterion tach rieng: train co weight+smoothing, eval dung CE thuan
    criterion_train = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    criterion_eval  = nn.CrossEntropyLoss()

    # Luc dau chi toi uu head (backbone frozen)
    if FREEZE_EPOCHS > 0:
        freeze_backbone(model, model_name)
        head_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(head_params, lr=LR, weight_decay=WEIGHT_DECAY)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    scheduler      = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    best_val_acc   = 0.0
    patience_count = 0
    save_path      = os.path.join(CHECKPOINT_DIR, f"best_{model_name}_{fold_name}.pth")

    for epoch in range(1, EPOCHS + 1):

        # [SUA 1 + 3] Unfreeze bang add_param_group, giu scheduler, reset patience
        if FREEZE_EPOCHS > 0 and epoch == FREEZE_EPOCHS + 1:
            unfreeze_all(model)
            backbone_params = [
                p for n, p in model.named_parameters()
                if not _is_head_param(n, model_name)
            ]
            optimizer.add_param_group({"params": backbone_params, "lr": UNFREEZE_LR})
            patience_count = 0   # [SUA 3] reset de tranh dung ngay sau unfreeze
            print(
                f"  [Epoch {epoch}] Unfreeze backbone -> them vao optimizer "
                f"voi LR={UNFREEZE_LR:.2e} | Scheduler giu nguyen trang thai."
            )

        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc, _, _ = run_epoch(
            model, train_loader, criterion_train, optimizer,
            desc=f"[{model_name}|{fold_name}] E{epoch}/{EPOCHS} Train"
        )
        val_loss, val_acc, _, _ = run_epoch(
            model, val_loader, criterion_eval,
            desc=f"[{model_name}|{fold_name}] E{epoch}/{EPOCHS} Val"
        )
        scheduler.step()

        print(
            f"  Epoch {epoch:02d}/{EPOCHS} | LR: {current_lr:.2e} | "
            f"Train {train_loss:.4f}/{train_acc*100:.2f}% | "
            f"Val {val_loss:.4f}/{val_acc*100:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            patience_count = 0
            torch.save(model.state_dict(), save_path)
            print(f"    -> Best saved (val={val_acc*100:.2f}%)")
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"    Early stopping tai epoch {epoch}.")
                break

    # Danh gia 1 lan tren test set cua dung fold nay
    model.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))
    test_loss, test_acc, test_preds, test_labels = run_epoch(
        model, test_loader, criterion_eval, desc=f"[{model_name}|{fold_name}] Test"
    )
    print(f"\n  >> {fold_name} Test Accuracy: {test_acc*100:.2f}%")
    print(classification_report(test_labels, test_preds, target_names=CLASS_NAMES))
    print(confusion_matrix(test_labels, test_preds))

    del model
    torch.cuda.empty_cache()
    return test_acc


# ==========================================
# 7. TRAIN 1 MODEL TREN TAT CA FOLD
# ==========================================
def train_pipeline(model_name, folds):
    display_name = MODEL_CONFIGS[model_name]["display_name"]
    print(f"\n{'='*70}")
    print(f" HUAN LUYEN: {display_name} ({model_name}) - {len(folds)} fold")
    print(f"{'='*70}")

    fold_accs = {}
    for fold_name in folds:
        fold_dir = os.path.join(KFOLD_DATASET_DIR, fold_name)
        fold_accs[fold_name] = train_one_fold(model_name, fold_name, fold_dir)

    accs      = list(fold_accs.values())
    mean_acc  = statistics.mean(accs)
    std_acc   = statistics.pstdev(accs) if len(accs) > 1 else 0.0

    # Params/size/inference: dac trung kien truc, do 1 lan
    probe = get_model(model_name).to(DEVICE)
    total_params, _, size_mb = count_params(probe)
    cpu_ms = measure_inference_time(probe, model_name, torch.device("cpu"))
    gpu_ms = measure_inference_time(probe, model_name, torch.device("cuda")) \
             if torch.cuda.is_available() else None
    del probe
    torch.cuda.empty_cache()

    print(f"\n{display_name} - TONG KET {len(folds)}-FOLD:")
    for fn, acc in fold_accs.items():
        print(f"  {fn}: {acc*100:.2f}%")
    print(f"  Mean: {mean_acc*100:.2f}%  Std: {std_acc*100:.2f}%")
    print(f"  Params: {total_params/1e6:.2f}M | Size: {size_mb:.2f} MB")
    print(f"  CPU: {cpu_ms:.2f} ms/anh" + (f" | GPU: {gpu_ms:.2f} ms/anh" if gpu_ms else ""))

    return {
        "model"          : display_name,
        "fold_accuracies": fold_accs,
        "mean_acc"       : mean_acc,
        "std_acc"        : std_acc,
        "params_M"       : total_params / 1e6,
        "size_MB"        : size_mb,
        "cpu_ms"         : cpu_ms,
        "gpu_ms"         : gpu_ms if gpu_ms else "",
    }


# ==========================================
# 8. CHAY TAT CA & XUAT BANG TONG KET
# ==========================================
if __name__ == "__main__":
    folds = discover_folds()
    print(f"Tim thay {len(folds)} fold: {folds}")

    models_to_run = ALL_MODELS   # ["efficientnet_lite3", "mobilenet_small", "shufflenet"]
    final_results = []
    for m in models_to_run:
        final_results.append(train_pipeline(m, folds))

    print("\n" + "=" * 82)
    print(f" BANG TONG KET - LIGHTWEIGHT MODEL COMPARISON ({len(folds)}-FOLD CV)")
    print("=" * 82)
    print(f"{'Model':<22}{'Mean Acc':>12}{'Std':>8}{'Params(M)':>12}{'Size(MB)':>10}{'CPU(ms)':>10}{'GPU(ms)':>10}")
    for r in final_results:
        gpu_str = f"{r['gpu_ms']:.2f}" if r["gpu_ms"] != "" else "-"
        print(
            f"{r['model']:<22}{r['mean_acc']*100:>11.2f}%"
            f"{r['std_acc']*100:>7.2f}%"
            f"{r['params_M']:>12.2f}{r['size_MB']:>10.2f}"
            f"{r['cpu_ms']:>10.2f}{gpu_str:>10}"
        )

    # CSV: 1 dong / (model x fold) + cot tong hop mean/std/params/...
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "fold", "test_acc",
            "mean_acc", "std_acc",
            "params_M", "size_MB", "cpu_ms", "gpu_ms",
        ])
        for r in final_results:
            for fold_name, acc in r["fold_accuracies"].items():
                writer.writerow([
                    r["model"], fold_name, f"{acc:.4f}",
                    f"{r['mean_acc']:.4f}", f"{r['std_acc']:.4f}",
                    f"{r['params_M']:.2f}", f"{r['size_MB']:.2f}",
                    f"{r['cpu_ms']:.2f}", r["gpu_ms"],
                ])
    print(f"\nDa luu CSV chi tiet vao: {RESULTS_CSV}")