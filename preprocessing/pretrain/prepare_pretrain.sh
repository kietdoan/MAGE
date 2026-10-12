#!/bin/bash

# =============================================================================
# TUH EEG PRETRAINING DATASET LAUNCHER
# =============================================================================
# INSTRUCTIONS:
# 1. Activate your python environment (e.g., conda activate my_env).
# 2. Update 'IN_ROOT' to the folder containing the TUH dataset subfolders.
# 3. Update 'OUT_DIR' to your desired output location for LMDB files.
# 4. Adjust resource settings (WORKERS, MEMORY) based on your machine.
# =============================================================================

# ------------------ Configuration ----------------
# NOTE: Replace these paths with your actual directories
IN_ROOT="/path/to/dataset/source_data"
OUT_DIR="/path/to/output_directory"

# Dataset Selection
DATASETS="tuh_eeg_artifact,tuh_eeg_epilepsy,tuh_eeg_seizure,tuh_eeg_slowing"
MONTAGES="01_tcp_ar"

# Split Configuration (90% Train / 10% Val)
FRAC_TRAIN=0.90
FRAC_VAL=0.10
SEED=12345

# Signal Processing
FS=200
L_FREQ=0.3
H_FREQ=75
MAINS=60
WIN_SEC=30

# Output Paths
TRAIN_DB="${OUT_DIR}/train.lmdb"
VAL_DB="${OUT_DIR}/val.lmdb"

# System Resources
WORKERS=8

echo "=========================================="
echo "Starting TUH Pretraining Preparation"
echo "Input Root      : $IN_ROOT"
echo "Output Train    : $TRAIN_DB"
echo "Output Val      : $VAL_DB"
echo "Montages        : $MONTAGES"
echo "Workers         : $WORKERS"
echo "=========================================="

# Increase open file limit (LMDB + Multiprocessing requirement)
ulimit -n 65536

# ------------------ Execution -------------------
python preprocessing/pretrain/prepare_pretrain.py \
  --in-root "$IN_ROOT" \
  --datasets "$DATASETS" \
  --allowed-montages "$MONTAGES" \
  --out-lmdb-train "$TRAIN_DB" \
  --out-lmdb-val   "$VAL_DB" \
  --fraction-train $FRAC_TRAIN \
  --fraction-val   $FRAC_VAL \
  --split-seed     $SEED \
  --fs $FS \
  --l-freq $L_FREQ \
  --h-freq $H_FREQ \
  --mains  $MAINS \
  --window-sec $WIN_SEC \
  --edge-trim-sec 60 \
  --min-len-sec 300 \
  --amp-max 100 \
  --dtype float32 \
  --workers $WORKERS

echo "Job Complete."