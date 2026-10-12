#!/bin/bash

# =============================================================================
# KAGGLE ERN PREPROCESSING SCRIPT
# =============================================================================
# INSTRUCTIONS:
# 1. Activate your python environment.
# 2. Update 'ROOT_DIR' to point to the parent folder of 'KaggleERN'.
# 3. Update 'OUT_DB' to your desired output location.
# =============================================================================

# ------------------ Configuration ----------------
# REPLACE THESE PATHS WITH YOUR ACTUAL DIRECTORIES
ROOT_DIR="/path/to/dataset"   
OUT_DB="/path/to/output/KaggleERN_lmdb"

echo "=========================================="
echo "Starting Kaggle ERN Preprocessing"
echo "Input Root      : $ROOT_DIR"
echo "Output Database : $OUT_DB"
echo "Target Frequency: 200 Hz"
echo "=========================================="

# ------------------ Execution -------------------
python preprocessing/downstream/prepare_kaggle_ern.py \
  --root_dir "$ROOT_DIR" \
  --out_db "$OUT_DB" \
  --fs 200

echo "Job Complete."