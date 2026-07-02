"""
paths_config.py
Tu dong xac dinh duong dan dua tren VI TRI CUA FILE NAY, khong can sua tay
khi doi may / clone sang thu muc khac / doi ten project.

Quy uoc: file nay luon nam o THU MUC GOC cua project, cung cap voi
01_xxx.py, 02_xxx.py, 03_xxx.py... Moi script khac se "from paths_config import ..."
thay vi tu khai bao duong dan tuyet doi.

Cau truc thu muc project (mac dinh):
project_root/                      <- noi chua file paths_config.py nay
├── Classification/
│   ├── JPEGImages/                <- anh goc ACNE04
│   └── NNEW_trainval_*.txt, NNEW_test_*.txt   <- file fold goc cua ACNE04
├── dataset_acne04_folds/          <- output chia k-fold theo fold goc ACNE04
├── checkpoints/
├── eda_outputs/
├── eval_outputs/
├── gradcam_outputs/
└── *.py

Neu cau truc may ban KHAC mac dinh tren, chi can sua 1 LAN DUY NHAT
o phan "TUY CHINH" ben duoi - moi script khac dung chung file nay
se tu dong cap nhat theo, khong phai sua tung file rieng le.
"""

import os
from pathlib import Path

# ==========================================
# 1. GOC PROJECT - TU DONG NHAN DIEN
# ==========================================
# Mac dinh: thu muc chua chinh file paths_config.py nay.
PROJECT_ROOT = Path(__file__).resolve().parent

# Cho phep override TOAN BO goc project bang bien moi truong, khong can sua code.
#   Windows (PowerShell): $env:ACNE_PROJECT_ROOT = "D:\du-an-khac"
#   Windows (cmd):         set ACNE_PROJECT_ROOT=D:\du-an-khac
#   Linux / Mac:           export ACNE_PROJECT_ROOT=/duong/dan/khac
_env_root = os.environ.get("ACNE_PROJECT_ROOT")
if _env_root:
    PROJECT_ROOT = Path(_env_root).resolve()

# ==========================================
# 2. TUY CHINH (sua o day neu cau truc thu muc cua ban khac mac dinh)
# ==========================================
CLASSIFICATION_DIRNAME  = "Classification"
IMAGE_SUBDIR            = "JPEGImages"

KFOLD_DATASET_DIRNAME   = "dataset_acne04_folds"   # output chia k-fold theo fold goc ACNE04

CHECKPOINT_DIRNAME      = "checkpoints"
EDA_OUTPUT_DIRNAME      = "eda_outputs"
EVAL_OUTPUT_DIRNAME     = "eval_outputs"
GRADCAM_OUTPUT_DIRNAME  = "gradcam_outputs"
REPORT_DIRNAME          = "reports"

# ==========================================
# 3. CAC DUONG DAN SUY RA - DUNG TRUC TIEP TRONG CAC SCRIPT KHAC
# ==========================================
CLASSIFICATION_DIR  = PROJECT_ROOT / CLASSIFICATION_DIRNAME
IMAGE_SOURCE_DIR    = CLASSIFICATION_DIR / IMAGE_SUBDIR
TXT_DIR             = CLASSIFICATION_DIR   # noi chua NNEW_trainval_X.txt / NNEW_test_X.txt

KFOLD_DATASET_DIR   = PROJECT_ROOT / KFOLD_DATASET_DIRNAME

CHECKPOINT_DIR      = PROJECT_ROOT / CHECKPOINT_DIRNAME
EDA_OUTPUT_DIR      = PROJECT_ROOT / EDA_OUTPUT_DIRNAME
EVAL_OUTPUT_DIR     = PROJECT_ROOT / EVAL_OUTPUT_DIRNAME
GRADCAM_OUTPUT_DIR  = PROJECT_ROOT / GRADCAM_OUTPUT_DIRNAME
REPORT_DIR          = PROJECT_ROOT / REPORT_DIRNAME


def ensure_dirs(*dirs):
    """Tao cac thu muc neu chua ton tai (khong loi neu da co)."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def _check(path: Path) -> str:
    return "OK" if path.exists() else "KHONG TIM THAY"


if __name__ == "__main__":
    # Chay: python paths_config.py  -> in ra de tu kiem tra duong dan da dung chua
    print(f"PROJECT_ROOT      = {PROJECT_ROOT}")
    print(f"IMAGE_SOURCE_DIR  = {IMAGE_SOURCE_DIR}   [{_check(IMAGE_SOURCE_DIR)}]")
    print(f"TXT_DIR           = {TXT_DIR}             [{_check(TXT_DIR)}]")
    print(f"KFOLD_DATASET_DIR = {KFOLD_DATASET_DIR}   [{_check(KFOLD_DATASET_DIR)}]")
    print(f"CHECKPOINT_DIR    = {CHECKPOINT_DIR}       [{_check(CHECKPOINT_DIR)}]")
    print(f"EDA_OUTPUT_DIR    = {EDA_OUTPUT_DIR}       [{_check(EDA_OUTPUT_DIR)}]")
    print(f"EVAL_OUTPUT_DIR   = {EVAL_OUTPUT_DIR}      [{_check(EVAL_OUTPUT_DIR)}]")
    print(f"GRADCAM_OUTPUT_DIR= {GRADCAM_OUTPUT_DIR}   [{_check(GRADCAM_OUTPUT_DIR)}]")
    print(f"REPORT_DIR        = {REPORT_DIR}           [{_check(REPORT_DIR)}]")
    print()
    print("Neu duong dan nao bao 'KHONG TIM THAY' va sai mac dinh,")
    print("sua phan '2. TUY CHINH' o tren, hoac set bien moi truong ACNE_PROJECT_ROOT.")