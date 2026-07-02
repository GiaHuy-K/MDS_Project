# Code Review Log — MDS_Project

**Ngày review:** 2026-06-29
**Ngày polish:** 2026-06-29  
**Reviewer:** Claude Sonnet 4.6  
**Phạm vi:** Toàn bộ pipeline (paths_config, model_utils, 01b, generate_kfold, 02_eda, 03_train, 04_evaluate, 05_gradcam)

---

## Tóm tắt

Phát hiện 12 vấn đề: 1 lỗi nghiêm trọng (sai data path), 5 cần sửa trước khi chạy thật, 4 dọn dẹp code, 2 gợi ý nhỏ. Tất cả đã được áp dụng trực tiếp vào code.

---

## Chi tiết các thay đổi

### 🔴 #1 — `02_eda.py`: Sai đường dẫn data `[CRITICAL]`
- **Vấn đề:** `DATA_DIR` hardcode sang `dataset_final_70_15_15/` — EDA phân tích sai data, không khớp với data train thực tế (`dataset_acne04_folds/`).
- **Sửa:** Đổi `DATA_DIR` trỏ vào `KFOLD_DATASET_DIR / "fold_1"`, dùng `KFOLD_DATASET_DIR` từ `paths_config`.

---

### 🟡 #3 — `03_train.py`: `import json` nằm giữa file
- **Vấn đề:** `import json` đặt tại section 7 thay vì đầu file — vi phạm PEP 8, gây khó đọc.
- **Sửa:** Di chuyển lên khối import đầu file, xóa khai báo thừa ở section 7.

---

### 🟡 #4 — `generate_kfold_from_existing.py`: `import train_test_split` trong function
- **Vấn đề:** `from sklearn.model_selection import train_test_split` nằm bên trong `create_kfold_splits()` — import lại mỗi lần gọi hàm.
- **Sửa:** Di chuyển lên đầu file, gộp vào dòng import `StratifiedKFold`.

---

### 🟡 #5 — `03_train.py`: Nhảy số section (thiếu section 3)
- **Vấn đề:** Sections đánh số 1, 2, 4, 5... — thiếu section 3.
- **Sửa:** Thêm section `# 3. THIET BI` bao quanh phần khai báo `DEVICE`.

---

### 🟡 #6 — `03_train.py`: Smoketest values chưa đổi lại
- **Vấn đề:** Các giá trị smoke test còn giữ nguyên — nếu chạy thật sẽ cho kết quả sai hoàn toàn.
- **Sửa:**

| Tham số | Smoketest | Khôi phục |
|---------|-----------|-----------|
| `EPOCHS` | 5 | **40** |
| `PATIENCE` | 3 | **7** |
| `num_workers` | 0 | **4** |
| `cudnn.deterministic` | False | **True** |
| `cudnn.benchmark` | True | **False** |

---

### 🟡 #8 — `02_eda.py`: Import không dùng
- **Vấn đề:** `from collections import defaultdict` và `ensure_dirs` import nhưng không sử dụng.
- **Sửa:** Xóa `defaultdict` khỏi import. `ensure_dirs` cũng được xóa cùng lúc khi refactor import (#1).

---

### 🟡 #9 — `CLASS_NAMES` khai báo trùng lặp ở 3 nơi
- **Vấn đề:** `CLASS_NAMES` định nghĩa lại trong `02_eda.py` và `generate_kfold_from_existing.py` thay vì import từ nguồn duy nhất `model_utils.py`.
- **Sửa:** Cả hai file đổi sang `from model_utils import CLASS_NAMES`.

---

### 🟡 #10 — `transforms.Resize()` không nhất quán giữa train và evaluate
- **Vấn đề:** `03_train.py` dùng `interpolation=BILINEAR, antialias=True` nhưng `04_evaluate.py` và `05_gradcam.py` thì không — gây sai lệch nhỏ ở preprocessing giữa train và inference.
- **Sửa:** Thêm `interpolation=BILINEAR, antialias=True` vào `Resize` trong `build_test_loader()` (`04_evaluate.py`) và `build_eval_transform()` (`05_gradcam.py`).

---

### 🟢 #11 — `03_train.py`: `pstdev` vs `stdev`
- **Vấn đề:** `statistics.pstdev` (chia N) thay vì `statistics.stdev` (chia N-1) — convention ML papers thường dùng sample std.
- **Sửa:** Đổi sang `statistics.stdev` cho cả `std_acc` và `std_f1`.

---

---

## Round 2 — Project Polish (2026-06-29)

### Mục tiêu
Đưa project lên chuẩn NCKH sinh viên: xóa noise, chuẩn hóa README, dọn .gitignore.

### Thay đổi

| File | Thay đổi |
|------|----------|
| `03_train.py` | Xóa `=== CAC THAY DOI SO VOI BAN GOC ===` khỏi docstring (changelog thuộc về git history, không phải code). Sửa thứ tự section: 3→2 (THIET BI), 2→3 (TIM FOLD). Xóa các inline comment `[SMOKETEST]`, `[FIX]`, `# Bo Resize+Crop`. |
| `05_gradcam.py` | Xóa các comment `[FIX]`, `[ADD]` trong import và function body. |
| `README.md` | Viết lại hoàn toàn: đúng tên file, đúng pipeline k-fold, đúng output CSV, thêm hướng dẫn resume, thêm tài liệu tham khảo học thuật đầy đủ. |
| `.gitignore` | Thêm `dataset_final_70_15_15/`, `kfold_results*.csv`, `*.docx`. |

### Không thay đổi
- `Nhom_02_Draft.docx` — giữ trong thư mục, gitignore để không commit lên repo.
- Các file source logic — không có thay đổi logic, chỉ dọn presentation.

---

## Không sửa (giải thích)

| # | Vấn đề | Lý do không sửa |
|---|--------|-----------------|
| #2 | README lỗi thời | File `.md` — cần cập nhật thủ công theo nội dung thực tế sau khi chạy xong |
| #7 | `04_evaluate.py` chỉ evaluate fold_1 | Intentional design: dùng để visualize (CM, ROC, misclassified); kết quả 5-fold aggregate đã có trong `kfold_results_summary.csv` |
| #12 | `requirements.txt` không pin version | Nên chạy `pip freeze > requirements.txt` sau khi môi trường ổn định, không pin cứng trước |

---

## Round 3 — Điều tra nguyên nhân Accuracy thấp & sửa (2026-07-01)

### Bối cảnh
Kết quả 5-fold ban đầu: EfficientNet-Lite3 61.91%, MobileNetV3-Small 67.06%, ShuffleNetV2-x1.0 68.02% (`kfold_results_summary.csv`), std_acc ShuffleNetV2 lên tới 9.97%. Đối chiếu log huấn luyện đầy đủ (`toan_bo_log_chay_qua_dem.txt`) phát hiện train acc đạt 90-96% trong khi val acc chỉ 50-65% (gap ~30 điểm %) → **overfitting nghiêm trọng** là nguyên nhân chính, cộng thêm double-correction mất cân bằng lớp và mất ổn định training (đặc biệt ShuffleNetV2 ở các epoch đầu).

### Thay đổi — theo thứ tự ưu tiên

| Ưu tiên | File | Thay đổi | Vị trí |
|---|---|---|---|
| 🔴 Cao | `model_utils.py` | Thêm hàm `get_param_groups()` — tách tham số backbone/head để dùng LR khác nhau | `model_utils.py` (trước `count_params`) |
| 🔴 Cao | `03_train.py` | Bỏ `LR` đơn nhất, thay bằng `LR_BACKBONE=1e-5` (backbone pretrained) và `LR_HEAD=1e-4` (classifier head mới); optimizer dùng `get_param_groups()` thay vì `model.parameters()` | Cấu hình §1, `train_one_fold()` |
| 🔴 Cao | `03_train.py` | Bỏ double-correction mất cân bằng lớp: xóa `build_class_weights()` + `weight=class_weights` trong loss, chỉ giữ `WeightedRandomSampler`. Thay bằng `log_class_distribution()` chỉ để log, không tính weight | `build_loaders()`, `train_one_fold()` |
| 🔴 Cao | `03_train.py` | Thêm augmentation chống overfit: `RandomErasing(p=0.25)` sau `Normalize`; thêm cờ `ENABLE_COLOR_JITTER` để bật/tắt ablation ColorJitter | `build_loaders()` |
| 🟡 Trung bình | `03_train.py` | Thêm LR warmup tuyến tính `WARMUP_EPOCHS=3` epoch đầu (tăng dần từ 0 → LR mục tiêu) trước khi `ReduceLROnPlateau` hoạt động — giảm dao động đầu training (quan sát thấy rõ nhất ở ShuffleNetV2) | `train_one_fold()` vòng lặp epoch |
| 🟡 Trung bình | `03_train.py` | Thêm gradient clipping `GRAD_CLIP_NORM=1.0` (`clip_grad_norm_`) trước `optimizer.step()` | `run_epoch()` |
| 🟢 Thấp | `03_train.py` | Thêm `assert ds.classes == CLASS_NAMES` cho cả train/val/test — phòng ngừa lệch thứ tự nhãn giữa `ImageFolder.class_to_idx` và `CLASS_NAMES` (hiện tại đang khớp đúng, nhưng không có assertion để phát hiện sớm nếu lệch) | `build_loaders()` |
| ⚪ Hiệu năng | `03_train.py` | Thêm AMP (`torch.amp.autocast` + `GradScaler`) cho vòng train/eval — giảm thời gian/VRAM, không ảnh hưởng logic accuracy | `run_epoch()`, `train_one_fold()` |
| ⚪ Hiệu năng | `03_train.py` | Thêm `persistent_workers=True` cho DataLoader — giảm overhead spawn worker mỗi epoch | `build_loaders()` |
| — | `03_train.py` | Cập nhật log epoch: in riêng `LR(bb)` và `LR(head)` vì optimizer giờ có 2 param groups | `train_one_fold()` |
| — | `README.md` | Cập nhật bảng tham số ở mục "Bước 3 — Training" cho khớp `LR_BACKBONE`/`LR_HEAD`/`WARMUP_EPOCHS` | Mục "Các bước chạy" |

### Chưa làm (cần chạy thực nghiệm để đánh giá)
- Ablation ColorJitter bật/tắt (đã có cờ `ENABLE_COLOR_JITTER`, cần chạy 2 lượt để so sánh).
- Mixup đã thêm vào `run_epoch()` (`MIXUP_ALPHA=0.2`, `MIXUP_PROB=0.5`) cùng đợt sửa augmentation — cần chạy lại 5-fold để đo tác động thực tế lên accuracy/F1.

### Kỳ vọng
Chưa chạy lại pipeline để đo kết quả mới. Đã xóa toàn bộ `checkpoints/best_*.pth` và `checkpoints/progress_*.json` (kết quả cũ, trước khi sửa) để lần chạy `03_train.py` tiếp theo train lại từ đầu với code mới — cần kiểm chứng gap train-val có thu hẹp và `std_acc` giữa các fold có giảm hay không.

---

## Round 4 — Tái cấu trúc thư mục cho chuyên nghiệp (2026-07-01)

### Mục tiêu
Project trước đó có file/folder tên lẫn lộn nhiều kiểu (`01b_...py`, `generate_...py` không theo số thứ tự, output nằm rải rác ở root). Tổ chức lại theo cấu trúc chuẩn: tách source code (`core/`, `scripts/`), tài liệu (`docs/`), dữ liệu (`data/`), và output có thể tái tạo (`outputs/`, `results/`).

### Đổi tên / di chuyển (dùng `git mv` cho file tracked)

| Cũ | Mới |
|---|---|
| `paths_config.py` | `core/paths_config.py` |
| `model_utils.py` | `core/model_utils.py` |
| `01b_split_dataset_kfold.py` | `scripts/01_split_dataset_kfold.py` |
| `generate_kfold_from_existing.py` | `scripts/01_split_dataset_from_existing.py` |
| `02_eda.py` | `scripts/02_eda.py` |
| `03_train.py` | `scripts/03_train.py` |
| `04_evaluate.py` | `scripts/04_evaluate.py` |
| `05_gradcam.py` | `scripts/05_gradcam.py` |
| `Nhom_02_Draft.docx` | `docs/Nhom_02_Draft.docx` |
| `dataset_acne04_folds/` (5.3GB) | `data/acne04_folds/` |
| `eda_outputs/` | `outputs/eda/` (giữ nội dung cũ — không phụ thuộc hyperparameter training) |
| `kfold_results_*.csv` (khi chạy lại) | `results/kfold_results_*.csv` |
| Mới thêm | `core/__init__.py` (rỗng, để `from core.xxx import ...` hoạt động) |

### Xóa (output/log lỗi thời, sẽ tái tạo khi chạy lại pipeline với code đã sửa)

| File/folder | Lý do xóa |
|---|---|
| `toan_bo_log_chay_qua_dem.txt` (705KB, tracked bởi git) | Log overnight của lần chạy CŨ, trước khi sửa overfitting — giữ lại sẽ gây hiểu nhầm. Xóa bằng `git rm` |
| `eval_outputs/` (learning curves, confusion matrix, ROC, report — 1.9MB) | Toàn bộ gắn với checkpoint CŨ đã xóa ở Round 3 — sẽ được `04_evaluate.py` tạo lại |
| `gradcam_outputs/` (heatmap — 1.4MB) | Gắn với checkpoint CŨ đã xóa — sẽ được `05_gradcam.py` tạo lại |
| `kfold_results_detail.csv`, `kfold_results_summary.csv` | Số liệu accuracy CŨ (61-68%), không còn phản ánh code hiện tại |
| `__pycache__/` | File biên dịch tạm, luôn tự tạo lại |

### Cập nhật code theo cấu trúc mới
- `core/paths_config.py`: `PROJECT_ROOT` đổi từ `Path(__file__).parent` → `Path(__file__).parent.parent` (vì file giờ nằm trong `core/`). Thêm `DATA_DIRNAME="data"`, gom `Classification/` và `acne04_folds/` vào trong `data/`. Đổi `REPORT_DIRNAME`/`REPORT_DIR` (chưa từng dùng) thành `RESULTS_DIRNAME`/`RESULTS_DIR = "results"`. Output folders gom vào `OUTPUT_ROOT_DIR = "outputs"` (`outputs/eda`, `outputs/eval`, `outputs/gradcam`).
- Cả 6 script trong `scripts/`: thêm `sys.path.insert(0, str(Path(__file__).resolve().parent.parent))` trước import, đổi `from paths_config/model_utils import` → `from core.paths_config/model_utils import`.
- `scripts/03_train.py`: `RESULTS_CSV`/`SUMMARY_CSV` đổi từ ghi tại `PROJECT_ROOT` sang `RESULTS_DIR`.
- `scripts/01_split_dataset_from_existing.py`: `SOURCE_DIR` đổi từ `PROJECT_ROOT / "dataset_final_70_15_15"` → `DATA_DIR / "dataset_final_70_15_15"`.
- `run_all.bat`: cập nhật toàn bộ lệnh `python xxx.py` → `python scripts\xxx.py`, kiểm tra dataset tại `data\acne04_folds\...`.
- `README.md`: viết lại cây thư mục và toàn bộ lệnh chạy (`python scripts/...`), cập nhật đường dẫn output.
- `.gitignore`: gọn lại — ignore theo thư mục cha (`data/`, `outputs/`, `results/`, `docs/*.docx`) thay vì liệt kê từng thư mục con lẻ tẻ; thêm `*_log_*.txt` để chặn log overnight kiểu cũ lọt vào git lần sau.

### Kiểm chứng
- `python core/paths_config.py` — toàn bộ đường dẫn resolve đúng.
- `python scripts/02_eda.py` — chạy thật thành công end-to-end với cấu trúc mới (990/175/292 ảnh train/val/test đúng như trước, ghi đúng vào `outputs/eda/`).
- Import smoke-test cả 6 script (`01_split_dataset_kfold`, `01_split_dataset_from_existing`, `03_train`, `04_evaluate`, `05_gradcam`, `02_eda`) — tất cả `IMPORT OK`.
- `python -m py_compile` toàn bộ `core/` + `scripts/` — không lỗi cú pháp.

### Lưu ý
- Các thay đổi rename/xóa file tracked đã được `git mv`/`git rm` (staged), nhưng **chưa commit** — người dùng tự quyết định khi nào commit.
- `checkpoints/`, `data/acne04_folds/` (5.3GB dataset đã split, giữ nguyên không đổi nội dung) không bị xóa.

---

## Round 5 — Preprocessing pipeline (2026-07-02)

### Mục tiêu
Cải thiện chất lượng ảnh đầu vào trước khi training: phát hiện và crop vùng mặt (thay vì resize toàn bộ ảnh gốc), tự động cân bằng contrast, và giữ tỷ lệ khung hình (không méo).

### Thay đổi

| Ưu tiên | File | Thay đổi | Vị trí |
|---|---|---|---|
| 🔴 Cao | `core/model_utils.py` | Thêm class `FaceROICrop` — dùng Haar cascade (`haarcascade_frontalface_default.xml`) để detect mặt, crop vùng mặt với padding 15%, fallback sang `_center_square_crop` (crop vuông từ tâm) nếu không detect được | class `FaceROICrop` |
| 🟡 Trung bình | `core/model_utils.py` | Thêm `ImageOps.autocontrast` vào PIL transform pipeline — tự động kéo dãn histogram contrast cho mỗi ảnh | `build_pil_transform()` |
| 🟡 Trung bình | `core/model_utils.py` | Đổi logic resize: thay vì `Resize((H, W))` trực tiếp (méo tỷ lệ), giờ crop vuông trước (qua `FaceROICrop`) rồi `Resize(img_size)` — giữ tỷ lệ 1:1 | `build_pil_transform()` |
| 🟡 Trung bình | `core/model_utils.py` | Tách `build_pil_transform()` ra riêng khỏi `build_image_transform()` để `05_gradcam.py` có thể dùng PIL transform (trước `ToTensor`/`Normalize`) để lấy ảnh overlay | `build_pil_transform()` |

---

## Round 6 — Ordinal reformulation (threshold-based binary decomposition) (2026-07-02)

### Mục tiêu
Bài toán phân loại mức độ mụn là **ordinal** (Level_0 < Level_1 < Level_2 < Level_3). Chuyển output head và loss function từ softmax multi-class sang ordinal binary decomposition (K-1 threshold logits + BCE loss), giúp model học được quan hệ thứ tự giữa các lớp.

> ⚠️ **CẢNH BÁO:** Checkpoint `.pth` cũ (trước Round 6) không còn tương thích với head mới (output K-1 = 3 logits thay vì K = 4) — cần train lại từ đầu.

### Thay đổi

| Ưu tiên | File | Thay đổi | Vị trí |
|---|---|---|---|
| 🔴 Cao | `core/model_utils.py` | Thêm `ORDINAL_OUTPUTS = NUM_CLASSES - 1`; thêm hàm `encode_ordinal_targets()` — mã hóa label k thành vector nhị phân [1]*k + [0]*(K-1-k) | module-level |
| 🔴 Cao | `core/model_utils.py` | Thêm `ordinal_logits_to_class_indices()` — decode ordinal logits thành class index bằng threshold 0.5 trên sigmoid | module-level |
| 🔴 Cao | `core/model_utils.py` | Thêm `ordinal_logits_to_class_probabilities()` — chuyển ordinal logits thành phân phối xác suất K lớp (có clamp + renormalize do thiếu ràng buộc đơn điệu) | module-level |
| 🔴 Cao | `core/model_utils.py` | Sửa `get_model()`: thêm tham số `ordinal=False`; khi `ordinal=True`, output head là `nn.Linear(in_features, K-1)` thay vì `nn.Linear(in_features, K)` | `get_model()` |
| 🔴 Cao | `scripts/03_train.py` | Thêm `OrdinalBCEWithLogitsLoss` (ordinal BCE với pos_weight + label smoothing tùy chọn) và `OrdinalFocalLoss` (focal loss variant cho ablation) | class definitions |
| 🔴 Cao | `scripts/03_train.py` | Thêm cờ ablation `USE_ORDINAL`, sửa `run_epoch()` để encode target bằng `encode_ordinal_targets()` và decode prediction bằng `ordinal_logits_to_class_indices()` | `run_epoch()`, cấu hình §1 |
| 🟡 Trung bình | `scripts/04_evaluate.py` | Sửa inference: dùng `ordinal_logits_to_class_probabilities()` cho ROC/AUC và `ordinal_logits_to_class_indices()` cho accuracy/F1 | `run_inference()` |
| 🟡 Trung bình | `core/model_utils.py` | Thêm class `OrdinalClassTarget` — Grad-CAM target scoring cho ordinal probability của một class cụ thể | class `OrdinalClassTarget` |
| 🟡 Trung bình | `scripts/05_gradcam.py` | Sửa Grad-CAM target: dùng `OrdinalClassTarget(pred_idx)` thay vì `ClassifierOutputTarget` | `apply_gradcam_plusplus()` |

### Ghi chú kỹ thuật
- Kiến trúc sử dụng `nn.Linear(in_features, K-1)` với trọng số **độc lập** cho mỗi threshold — đây là ordinal binary decomposition kiểu Frank & Hall / Niu et al., **KHÔNG phải** CORAL chuẩn (Cao, Mirjalili & Raschka, 2020) vốn yêu cầu shared weight vector giữa các threshold để đảm bảo rank monotonicity.
- Do thiếu ràng buộc shared weight, sigmoid outputs không đảm bảo đơn điệu → `ordinal_logits_to_class_probabilities()` có bước `clamp(min=0)` + renormalize để xử lý trường hợp xác suất âm.
