"""
paths_config.py
Resolves all project paths automatically based on THIS FILE'S LOCATION, so nothing needs
to be edited by hand when switching machines / cloning to a different folder / renaming
the project.

Convention: this file lives inside core/, and the project root is the PARENT of core/.
Scripts in scripts/ add the project root to sys.path and then do
"from core.paths_config import ..." instead of hard-coding absolute paths.

Default project layout:
project_root/
├── core/                           <- paths_config.py, model_utils.py
├── scripts/                        <- 01_xxx.py, 02_xxx.py, ...
├── docs/                           <- documents (docx, ...)
├── data/
│   ├── Classification/             <- raw ACNE04 images + fold .txt files
│   │   └── JPEGImages/
│   └── acne04_folds/               <- k-fold split output
├── checkpoints/                    <- model weights (not tracked)
├── outputs/
│   ├── eda/
│   ├── eval/
│   └── gradcam/
├── results/                        <- kfold_results_*.csv
└── requirements.txt, README.md, ...

If your layout differs from the default, edit ONLY ONCE in the "CUSTOMIZE" section below -
every other script sharing this file updates automatically, no need to edit each file.
"""

import os
from pathlib import Path

# ==========================================
# 1. PROJECT ROOT - AUTO-DETECTED
# ==========================================
# Default: the PARENT of core/ (the folder that contains this paths_config.py file).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Allow overriding the entire project root via an environment variable, no code change needed.
#   Windows (PowerShell): $env:ACNE_PROJECT_ROOT = "D:\another-project"
#   Windows (cmd):         set ACNE_PROJECT_ROOT=D:\another-project
#   Linux / Mac:           export ACNE_PROJECT_ROOT=/some/other/path
_env_root = os.environ.get("ACNE_PROJECT_ROOT")
if _env_root:
    PROJECT_ROOT = Path(_env_root).resolve()

# ==========================================
# 2. CUSTOMIZE (edit here if your folder layout differs from the default)
# ==========================================
DATA_DIRNAME            = "data"
CLASSIFICATION_DIRNAME  = "Classification"
IMAGE_SUBDIR            = "JPEGImages"

KFOLD_DATASET_DIRNAME   = "acne04_folds"   # k-fold split output based on ACNE04's original folds

CHECKPOINT_DIRNAME      = "checkpoints"

OUTPUT_ROOT_DIRNAME     = "outputs"
EDA_OUTPUT_DIRNAME      = "eda"
EVAL_OUTPUT_DIRNAME     = "eval"
GRADCAM_OUTPUT_DIRNAME  = "gradcam"

RESULTS_DIRNAME         = "results"        # kfold_results_detail.csv / summary.csv

# ==========================================
# 3. DERIVED PATHS - USE THESE DIRECTLY IN OTHER SCRIPTS
# ==========================================
DATA_DIR            = PROJECT_ROOT / DATA_DIRNAME
CLASSIFICATION_DIR  = DATA_DIR / CLASSIFICATION_DIRNAME
IMAGE_SOURCE_DIR    = CLASSIFICATION_DIR / IMAGE_SUBDIR
TXT_DIR             = CLASSIFICATION_DIR   # holds NNEW_trainval_X.txt / NNEW_test_X.txt

KFOLD_DATASET_DIR   = DATA_DIR / KFOLD_DATASET_DIRNAME

CHECKPOINT_DIR      = PROJECT_ROOT / CHECKPOINT_DIRNAME

OUTPUT_ROOT_DIR     = PROJECT_ROOT / OUTPUT_ROOT_DIRNAME
EDA_OUTPUT_DIR      = OUTPUT_ROOT_DIR / EDA_OUTPUT_DIRNAME
EVAL_OUTPUT_DIR     = OUTPUT_ROOT_DIR / EVAL_OUTPUT_DIRNAME
GRADCAM_OUTPUT_DIR  = OUTPUT_ROOT_DIR / GRADCAM_OUTPUT_DIRNAME

RESULTS_DIR         = PROJECT_ROOT / RESULTS_DIRNAME


def ensure_dirs(*dirs):
    """Create directories if they don't exist yet (no error if they already do)."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def _check(path: Path) -> str:
    return "OK" if path.exists() else "NOT FOUND"


if __name__ == "__main__":
    # Run: python core/paths_config.py  -> prints resolved paths so you can verify them
    print(f"PROJECT_ROOT      = {PROJECT_ROOT}")
    print(f"IMAGE_SOURCE_DIR  = {IMAGE_SOURCE_DIR}   [{_check(IMAGE_SOURCE_DIR)}]")
    print(f"TXT_DIR           = {TXT_DIR}             [{_check(TXT_DIR)}]")
    print(f"KFOLD_DATASET_DIR = {KFOLD_DATASET_DIR}   [{_check(KFOLD_DATASET_DIR)}]")
    print(f"CHECKPOINT_DIR    = {CHECKPOINT_DIR}       [{_check(CHECKPOINT_DIR)}]")
    print(f"EDA_OUTPUT_DIR    = {EDA_OUTPUT_DIR}       [{_check(EDA_OUTPUT_DIR)}]")
    print(f"EVAL_OUTPUT_DIR   = {EVAL_OUTPUT_DIR}      [{_check(EVAL_OUTPUT_DIR)}]")
    print(f"GRADCAM_OUTPUT_DIR= {GRADCAM_OUTPUT_DIR}   [{_check(GRADCAM_OUTPUT_DIR)}]")
    print(f"RESULTS_DIR       = {RESULTS_DIR}          [{_check(RESULTS_DIR)}]")
    print()
    print("If any path shows 'NOT FOUND' and is wrong for your setup,")
    print("edit section '2. CUSTOMIZE' above, or set the ACNE_PROJECT_ROOT env variable.")
