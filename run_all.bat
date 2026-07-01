@echo off
echo ========================================================
echo       MDS PROJECT - AUTOMATED PIPELINE RUNNER
echo ========================================================

REM Kiem tra xem co thu muc venv khong de activate
if exist venv\Scripts\activate.bat (
    echo [INFO] Kich hoat moi truong ao - virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo [WARNING] Khong tim thay thu muc venv. Dang su dung Python cua he thong.
)

echo.
echo [1/5] CHUAN BI DU LIEU (Data Splitting)
if exist dataset_acne04_folds\fold_1\train\ (
    echo [INFO] Da tim thay dataset chia san. Tu dong bo qua buoc 1.
) else (
    echo [INFO] Khong tim thay dataset k-fold. Tu dong tien hanh chia dataset tu file txt goc...
    python 01b_split_dataset_kfold.py
)

echo.
echo --------------------------------------------------------
echo [2/5] CHAY EDA (Phan tich kham pha du lieu)...
python 02_eda.py

echo.
echo --------------------------------------------------------
echo [3/5] CHAY TRAINING (Huan luyen mo hinh 5-Fold)...
echo LUU Y: Buoc nay ton nhieu thoi gian nhat. 
echo (Neu ban tat di giua chung, lan sau chay lai no se tu dong resume tu cac fold chua hoan thanh)
python 03_train.py

echo.
echo --------------------------------------------------------
echo [4/5] CHAY EVALUATE (Danh gia mo hinh chi tiet)...
python 04_evaluate.py

echo.
echo --------------------------------------------------------
echo [5/5] CHAY GRAD-CAM (Sinh anh truc quan AI - XAI)...
python 05_gradcam.py

echo.
echo ========================================================
echo       PIPELINE DA HOAN TAT!
echo Vui long kiem tra cac thu muc *_outputs va file .csv
echo ========================================================
pause
