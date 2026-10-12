#!/bin/bash

# =============================================================================
# FACED DATASET PREPROCESSING LAUNCHER
# =============================================================================
# INSTRUCTIONS:
# 1. Ensure your Python environment is active and has the required libraries
#    (numpy, scipy, sklearn, lmdb) installed.
# 2. Update the 'ROOT_DIR' variable below to point to the folder containing
#    your .pkl subject files.
# 3. Update the 'OUT_DB' variable to point to the folder where you want the
#    output LMDB created.
# =============================================================================

# ------------------ Configuration ----------------
# REPLACE THESE PATHS WITH YOUR ACTUAL DIRECTORIES
ROOT_DIR="/path/to/dataset/FACED_data/Processed_data"
OUT_DB="/path/to/output/FACED_lmdb"

echo "=========================================="
echo "Starting FACED Preprocessing Pipeline"
echo "Input Directory : $ROOT_DIR"
echo "Output Database : $OUT_DB"
echo "=========================================="

# ------------------ Execution -------------------
# Make sure the python script name matches your file
python preprocessing/downstream/prepare_faced.py \
  --root_dir "$ROOT_DIR" \
  --out_db "$OUT_DB"

echo "Job Complete."