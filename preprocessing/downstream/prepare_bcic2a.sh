#!/bin/bash

# =============================================================================
# USAGE INSTRUCTIONS
# =============================================================================
# This script runs the preprocessing pipeline to convert BCIC IV 2a data to LMDB format.
#
# PREREQUISITES:
# 1. Ensure you have activated the appropriate Python/Conda environment.
# 2. Replace the placeholder paths below with your actual system paths:
#    - /path/to/dataset/BCIC2A_mat : Path to the folder containing raw .mat files.
#    - /path/to/output/BCIC2a_lmdb : Path where the output LMDB file should be saved.
#
# PARAMETER EXPLANATION:
# --band_low / --band_high : Bandpass filter range (0.3 - 50 Hz).
# --t_start 2              : Start sampling from the 2nd second of the trial.
# --window_sec 4           : Sampling window length is 4 seconds.
# --resample_hz 200        : Resample frequency to 200Hz.
# =============================================================================

echo "Starting data preprocessing..."

python preprocessing/downstream/prepare_bcic2a.py \
  --root_dir /path/to/dataset/BCIC2A_mat \
  --out_db /path/to/output/BCIC2a_lmdb \
  --band_low 0.3 --band_high 50 \
  --t_start 2 --window_sec 4 \
  --resample_hz 200

echo "Completed."