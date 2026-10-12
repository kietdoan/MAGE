#!/bin/bash

# =============================================================================
# SHU-MI PREPROCESSING LAUNCHER
# =============================================================================
# INSTRUCTIONS:
# 1. Activate your python environment (e.g., conda activate mage).
# 2. Update 'ROOT_DIR' to the folder containing your .mat files.
# 3. Update 'OUT_DB' to your desired output location.
# =============================================================================

# ------------------ Configuration ----------------
# NOTE: Replace these paths with your actual directories
ROOT_DIR="/path/to/dataset/SHU-MI/mat"
OUT_DB="/path/to/output/SHUMI_lmdb"
TARGET_POINTS=800 
SCRIPT_PATH="preprocessing/downstream/prepare_shumi.py"

# ------------------ Optimization -----------------
# Optimizes numpy/scipy operations for speed
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32

# ------------------ Run Script -------------------
echo "Starting SHU-MI Preprocessing..."
echo "Script: $SCRIPT_PATH"
echo "Input : $ROOT_DIR"
echo "Output: $OUT_DB"

python "$SCRIPT_PATH" \
  --root_dir "$ROOT_DIR" \
  --out_db "$OUT_DB" \
  --target_points $TARGET_POINTS

echo "Job Complete."