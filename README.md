# Phân loại mức độ mụn trứng cá (Acne Vulgaris Severity Grading)
## Sử dụng Transfer Learning với mô hình nhẹ và Grad-CAM++ Explainability

**Đề tài NCKH — Nhóm 02**

Pipeline Deep Learning phân loại ảnh mụn trứng cá thành 4 mức độ nặng (Level 0–3) theo dataset ACNE04, so sánh 3 mô hình nhẹ (lightweight) bằng 5-Fold Cross Validation, kèm trực quan hóa Grad-CAM++.

---

## Mô hình sử dụng

| Model | Nguồn | Vai trò |
|---|---|---|
| **EfficientNet-Lite3** | `timm` (`tf_efficientnet_lite3`) | Mô hình chính của đề tài |
| MobileNetV3-Small | `torchvision` | Mô hình đối chiếu |
| ShuffleNetV2-x1.0 | `torchvision` | Mô hình đối chiếu |

---

## Cấu trúc thư mục

```
MDS_Project/
├── paths_config.py               # Quản lý đường dẫn tập trung
├── model_utils.py                # Định nghĩa 3 model + config (img_size, mean, std)
├── 01b_split_dataset_kfold.py    # Chia dataset theo file .txt gốc của ACNE04
├── generate_kfold_from_existing.py  # Tạo 5-fold từ dataset đã chia sẵn 70/15/15
├── 02_eda.py                     # Phân tích khám phá dữ liệu (EDA)
├── 03_train.py                   # Training 3 model × 5 fold
├── 04_evaluate.py                # Đánh giá chi tiết: Confusion Matrix, ROC, AUC
├── 05_gradcam.py                 # Grad-CAM++ heatmap + grid so sánh 3 model
├── requirements.txt
└── README.md
```

Sau khi chạy, các thư mục sau được tạo tự động:

```
Classification/                   # Dataset gốc ACNE04 (không tracked bởi git)
dataset_acne04_folds/             # Output k-fold (không tracked)
checkpoints/                      # Model weights (không tracked)
eda_outputs/                      # Biểu đồ phân bố, ảnh mẫu
eval_outputs/                     # Confusion matrix, ROC, classification report
gradcam_outputs/                  # Heatmap Grad-CAM++
kfold_results_detail.csv          # Kết quả từng fold của từng model
kfold_results_summary.csv         # Tổng hợp mean ± std qua 5 fold
```

---

## Cài đặt môi trường

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

Nếu máy có GPU NVIDIA, cài PyTorch bản CUDA theo hướng dẫn tại: https://pytorch.org/get-started/locally/

> `timm` tự động tải pretrained weights của `tf_efficientnet_lite3` từ Hugging Face Hub ở lần chạy đầu tiên (cần mạng).

---

## Các bước chạy

### Bước 1 — Chuẩn bị dataset k-fold

**Trường hợp A:** Có sẵn file `.txt` gốc của ACNE04 (`NNEW_trainval_*.txt`, `NNEW_test_*.txt`) trong thư mục `Classification/`:

```bash
python 01b_split_dataset_kfold.py
```

**Trường hợp B:** Đã có dataset chia sẵn dạng `dataset_final_70_15_15/train|val|test/Level_X/`:

```bash
python generate_kfold_from_existing.py
```

Cả hai script đều xuất ra cùng cấu trúc: `dataset_acne04_folds/fold_1..5/train|val|test/Level_0..3/`

### Bước 2 — EDA

```bash
python 02_eda.py
```

Kiểm tra số lượng ảnh, phát hiện ảnh lỗi, vẽ biểu đồ phân bố lớp. Kết quả trong `eda_outputs/`.

### Bước 3 — Training

```bash
python 03_train.py
```

Train lần lượt 3 model qua 5 fold. Tham số chính có thể chỉnh trong `03_train.py`:

```python
BATCH_SIZE    = 16
EPOCHS        = 100
LR_BACKBONE   = 1e-5   # LR nho cho backbone pretrained
LR_HEAD       = 1e-4   # LR lon hon cho classifier head moi
WARMUP_EPOCHS = 3
PATIENCE      = 10
```

Kết quả: checkpoint `checkpoints/best_<model>_<fold>.pth`, CSV tổng hợp tại root.

> **Resume:** Nếu bị ngắt giữa chừng, chạy lại `03_train.py` — các fold đã hoàn thành sẽ được bỏ qua tự động (dựa trên file `checkpoints/progress_<model>.json`).

### Bước 4 — Đánh giá chi tiết

```bash
python 04_evaluate.py
```

Đánh giá cả 3 model trên test set của `fold_1`. Kết quả trong `eval_outputs/`:

- `cm_<model>.png` — Confusion Matrix (số lượng + tỷ lệ %)
- `roc_<model>.png` — ROC curves One-vs-Rest, AUC macro/weighted
- `misclassified_<model>.csv` — Danh sách ảnh dự đoán sai
- `report_<model>.txt` — Classification report đầy đủ
- `summary_eval.csv` — Bảng tổng hợp 3 model

### Bước 5 — Grad-CAM++ (XAI)

```bash
python 05_gradcam.py
```

Mặc định tự động chọn 2 ảnh đại diện mỗi Level từ `fold_1/test/`. Để chọn ảnh thủ công:

```python
# trong 05_gradcam.py
SAMPLE_IMAGES = [r"duong\dan\anh1.jpg", r"duong\dan\anh2.jpg"]
```

Kết quả trong `gradcam_outputs/`: heatmap riêng lẻ và ảnh ghép grid so sánh 3 model.

---

## Lưu ý kỹ thuật

- **Test set chỉ dùng 1 lần** — sau khi chọn xong cấu hình qua validation.
- **Augmentation** chỉ áp dụng cho tập train; val/test giữ nguyên.
- Mỗi model dùng đúng độ phân giải và chuẩn hóa riêng (xem `model_utils.py`) — không trộn lẫn transform giữa các model.
- `RANDOM_SEED = 42` nhất quán toàn pipeline để đảm bảo tái lập.

---

## Tài liệu tham khảo

- ACNE04 Dataset: Wu, X. et al. (2019). *Joint Acne Image Grading and Counting via Label Distribution Learning.*
- EfficientNet: Tan, M. & Le, Q. (2019). *EfficientNet: Rethinking Model Scaling for CNNs.* [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)
- MobileNetV3: Howard, A. et al. (2019). *Searching for MobileNetV3.* [arXiv:1905.02244](https://arxiv.org/abs/1905.02244)
- ShuffleNetV2: Ma, N. et al. (2018). *ShuffleNet V2.* [arXiv:1807.11164](https://arxiv.org/abs/1807.11164)
- Grad-CAM++: Chattopadhay, A. et al. (2018). *Grad-CAM++.* [arXiv:1710.11063](https://arxiv.org/abs/1710.11063)
- pytorch-grad-cam: https://github.com/jacobgil/pytorch-grad-cam
