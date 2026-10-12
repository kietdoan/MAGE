#!/bin/bash

# =============================================================================
# PHYSIONET MI PREPROCESSING LAUNCHER
# =============================================================================
# INSTRUCTIONS:
# 1. Activate your python environment (conda activate mage).
# 2. Update 'ROOT_DIR' to the 'files/eegmmidb/1.0.0' folder containing 
#    the Subject folders (S001, S002...).
# 3. Update 'OUT_DB' to your desired output location.
# =============================================================================

# ------------------ Configuration ----------------
# NOTE: Replace these paths with your actual directories
ROOT_DIR="/path/to/raw/data/files"
OUT_DB="/path/to/output/directory"

# Sampling frequency
FS=200

echo "=========================================="
echo "Starting PhysioNet MI Preprocessing"
echo "Environment     : mage"
echo "Input Root      : $ROOT_DIR"
echo "Output Database : $OUT_DB"
echo "Target Frequency: $FS Hz"
echo "=========================================="

# Increase number of open file handles (MNE/LMDB can use many)
ulimit -n 8192

# ------------------ Execution -------------------
python preprocessing/downstream/prepare_physiomi.py \
  --root_dir "$ROOT_DIR" \
  --out_db "$OUT_DB" \
  --fs $FS

echo "Job Complete."