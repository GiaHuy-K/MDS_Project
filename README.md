# Lightweight Acne Vulgaris Severity Grading via Transfer Learning with Grad-CAM Explainability for Asian Skin Tones

Pipeline phan loai muc do mun (Acne Vulgaris) tu anh (Acne04, 4 class: Level_0..3),
su dung Transfer Learning voi 3 model **NHE (lightweight)**:

| Model | Nguon | Vai tro |
|---|---|---|
| **EfficientNet-Lite3** | `timm` (`tf_efficientnet_lite3`) | Model chinh cua de tai |
| MobileNetV3-Small | `torchvision` | Doi chung lightweight |
| ShuffleNetV2-x1.0 | `torchvision` | Doi chung lightweight |

Kem **Grad-CAM** de giai thich vung anh model dua vao de ra quyet dinh (XAI),
phuc vu phan tich tinh phu hop tren da chau A (Asian skin tones).

## Cau truc thu muc

```
acne_project/
├── 01_split_dataset.py   
├── 02_eda.py              
├── model_utils.py         
├── 03_train.py            
├── 04_evaluate.py         
├── 05_gradcam.py          
├── requirements.txt       
└── README.md
```

Sau khi chay xong, cac thu muc duoc tu dong tao:

```
checkpoints/                          # best_efficientnet_lite3.pth, best_mobilenet_small.pth, best_shufflenet.pth
eda_outputs/                          # Bieu do phan bo lop, anh mau
eval_outputs/                         # Confusion matrix, ROC curves, classification report, misclassified.csv
gradcam_outputs/                      # Anh heatmap Grad-CAM + anh grid so sanh 3 model
lightweight_comparison_results.csv    # Bang tong ket: accuracy, params(M), size(MB), inference time (CPU/GPU)
```

## Vi sao co model_utils.py moi?

EfficientNet-Lite3 (ban TF goc, port qua `timm`) duoc huan luyen voi do phan giai
**280x280** va chuan hoa **mean/std = [0.5,0.5,0.5]**, khac voi chuan ImageNet
(224x224, mean/std ImageNet) ma `torchvision` dung cho MobileNetV3 / ShuffleNetV2.
`model_utils.py` luu cau hinh rieng cho tung model (`MODEL_CONFIGS`) de transfer
learning khai thac dung pretrained weights, tranh giam accuracy do tien xu ly sai.

## Cai dat moi truong

```
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

pip install -r requirements.txt
```

Neu may co GPU NVIDIA, cai PyTorch ban CUDA theo huong dan tai: <https://pytorch.org/get-started/locally/>

`timm` se tu dong tai pretrained weights cua `tf_efficientnet_lite3` tu Hugging Face Hub
trong lan chay dau tien (can mang).

## Cac buoc chay

### Buoc 1 — Chia dataset (giu nguyen nhu cu)

Mo `01_split_dataset.py`, sua 2 duong dan, roi chay `python 01_split_dataset.py`.
Ket qua: `dataset_final_70_15_15/` voi cau truc `train/val/test/Level_0..3`.

### Buoc 2 — EDA (giu nguyen nhu cu)

`python 02_eda.py` — kiem tra mat can bang class, anh loi.

### Buoc 3 — Training 3 model nhe

Mo `03_train.py`, sua `DATA_DIR`. Co the chinh:

```python
EPOCHS = 30
LR = 1e-4
PATIENCE = 7
```

Chay:

```
python 03_train.py
```

Script tu dong train lan luot **EfficientNet-Lite3 → MobileNetV3-Small → ShuffleNetV2-x1.0**,
luu best checkpoint vao `checkpoints/best_<model>.pth`, va cuoi cung in + luu CSV
bang so sanh: Test Accuracy, so tham so (M), dung luong (MB), thoi gian inference
tren CPU (va GPU neu co) — day la so lieu trung tam cho phan "lightweight" cua bao cao.

**Luu y:** Neu chi muon test thu pipeline truoc, giam `EPOCHS` va sua
`models_to_run = ["efficientnet_lite3"]` trong `03_train.py` truoc khi chay full.

### Buoc 4 — Danh gia chi tiet

Mo `04_evaluate.py`, doi:

```python
MODEL_NAME = "efficientnet_lite3"   # hoac "mobilenet_small" / "shufflenet"
```

Chay `python 04_evaluate.py`. Ket qua trong `eval_outputs/`:
- `cm_<model>.png` — confusion matrix heatmap
- `roc_<model>.png` — ROC One-vs-Rest 4 class + AUC macro/weighted
- `misclassified_<model>.csv` — danh sach anh du doan sai (duong dan, nhan that, nhan du doan)
- Classification report (Precision/Recall/F1 tung class + macro/weighted) in ra console

### Buoc 5 — Grad-CAM (XAI)

Mo `05_gradcam.py`, sua `SAMPLE_IMAGES` (nen chon vai anh dai dien cho moi Level,
uu tien da nhieu tong mau khac nhau de phan tich tinh phu hop voi da chau A). Chay:

```
python 05_gradcam.py
```

Ket qua trong `gradcam_outputs/`:
- Heatmap rieng cho moi (anh, model)
- `<ten_anh>_comparison_grid.jpg` — anh ghep [Goc | Lite3 | MobileNetV3-Small | ShuffleNetV2]
  de dua thang vao bao cao, so sanh truc quan vung "focus" cua tung model.

## Quy trinh khoa hoc — luu y quan trong

- **Test set chi duoc dung 1 lan cuoi cung** sau khi da chon xong best model qua validation set.
- **Augmentation** (flip, rotate, color jitter) chi ap dung cho tap train; val/test giu nguyen.
- `RANDOM_SEED` dong nhat o 01 va 03 de dam bao tai lap.
- Moi model dung dung do phan giai/chuan hoa rieng (xem `model_utils.py`) — KHONG tron lan
  transform giua cac model khi so sanh.

## Cac buoc tiep theo cho bao cao

1. Dien `lightweight_comparison_results.csv` (Buoc 3) vao bang so sanh 3 model
   (params, accuracy, size MB, inference time) — luan diem chinh cho tu khoa "Lightweight".
2. Dien classification report + AUC-ROC tu Buoc 4 cho EfficientNet-Lite3 (model chinh).
3. Chon 2–3 anh grid Grad-CAM dien hinh cho moi Level dua vao phan Discussion.
4. Phan tich confusion matrix: Lite3 hay nham Level nao voi Level nao?
5. Viet phan Discussion ve "Asian skin tones": Grad-CAM co focus dung vung mun tren
   nen da/sac to khac nhau khong, co bi nham lan voi vet tham/da toi mau khong.
6. Neu can them tinh thuyet phuc, co the do FLOPs bang `thop`/`fvcore` de bo sung
   bang so sanh ben canh params va inference time.

## Tai lieu tham khao

- Transfer Learning (PyTorch): <https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html>
- Grad-CAM library: <https://github.com/jacobgil/pytorch-grad-cam>
- EfficientNet paper: <https://arxiv.org/abs/1905.11946>
- EfficientNet-Lite (timm docs): <https://huggingface.co/docs/timm/en/models/tf-efficientnet-lite>
- MobileNetV3 paper: <https://arxiv.org/abs/1905.02244>
- ShuffleNetV2 paper: <https://arxiv.org/abs/1807.11164>
