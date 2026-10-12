#!/bin/bash

# =============================================================================
# TUAB (TUH ABNORMAL) PREPROCESSING LAUNCHER
# =============================================================================
# INSTRUCTIONS:
# 1. Activate your python environment (e.g., conda activate mage).
# 2. Update 'ROOT_DIR' to the folder containing the TUAB dataset (v3.0.1).
# 3. Update 'OUT_DIR' to your desired output location.
# =============================================================================

# ------------------ Configuration ----------------
# NOTE: Replace these paths with your actual directories
ROOT_DIR="/path/to/dataset/tuh_eeg_abnormal/v3.0.1/edf"
OUT_DIR="/path/to/output/TUAB_eegpt"

# Settings
FS=200
WIN_SEC=10
NOTCH=60.0  

SCRIPT_PATH="preprocessing/downstream/prepare_tuab.py"

# ------------------ Optimization -----------------
# Limit threads to avoid locking issues in multiprocessing
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ------------------ Run Script -------------------
echo "Starting TUAB Preprocessing..."
echo "Script  : $SCRIPT_PATH"
echo "Input   : $ROOT_DIR"
echo "Output  : $OUT_DIR"
echo "Settings: $FS Hz, $WIN_SEC sec windows, Notch $NOTCH Hz"

python "$SCRIPT_PATH" \
  --root_dir "$ROOT_DIR" \
  --out_dir "$OUT_DIR" \
  --fs $FS \
  --win_sec $WIN_SEC \
  --notch $NOTCH

echo "Job Complete."