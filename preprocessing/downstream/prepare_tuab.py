#!/usr/bin/env python3
"""
TUAB (TUH Abnormal Corpus) -> .pkl Windows with Z-Score
=======================================================

Pipeline:
1. **File Discovery**: Scans train/eval folders for 'abnormal' (1) and 'normal' (0) EDFs.
2. **Channel Enforcement**: Selects standard 23 channels (01_tcp_ar).
3. **Preprocessing**:
   - Bandpass Filter (0.1–75 Hz).
   - Notch Filter (50Hz EU / 60Hz US).
   - Resampling (Target FS, e.g., 200 Hz).
4. **Windowing**: Splits data into 10-second non-overlapping windows.
5. **Normalization**: Per-Channel Per-Window Z-Score Normalization.
6. **Storage**: Saves individual windows as .pkl files preserving train/eval structure.

Usage:
    python prepare_tuab.py --root_dir /path/to/tuab --out_dir /path/to/output --notch 60
"""

import os
import argparse
import pickle
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import mne

# ================== CONFIG ==================
# Standard 23-channel order (01_tcp_ar with A1/A2 + T1/T2)
CH_ORDER_STANDARD = [
    'EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF',
    'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF',
    'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF',
    'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF',
    'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF'
]

# Drop non-EEG channels
DROP_CHANNELS = [
    'PHOTIC-REF', 'IBI', 'BURSTS', 'SUPPR', 'EEG ROC-REF', 'EEG LOC-REF',
    'EEG EKG1-REF', 'EMG-REF', 'EEG C3P-REF', 'EEG C4P-REF', 'EEG SP1-REF',
    'EEG SP2-REF', 'EEG LUC-REF', 'EEG RLC-REF', 'EEG RESP1-REF', 'EEG RESP2-REF',
    'EEG EKG-REF', 'RESP ABDOMEN-REF', 'ECG EKG-REF', 'PULSE RATE',
    'EEG PG2-REF', 'EEG PG1-REF'
]
# Also drop any extra EEG channels beyond 23
DROP_CHANNELS.extend([f'EEG {i}-REF' for i in range(20, 129)])

CHANNEL_STD = "01_tcp_ar"   # only this montage

# ================= ARGS =================
def parse_args():
    parser = argparse.ArgumentParser(description="TUAB Preprocessing")
    parser.add_argument("--root_dir", type=str, required=True, 
                        help="Root path to TUAB (containing 'edf' folder or split folders)")
    parser.add_argument("--out_dir", type=str, required=True, 
                        help="Output directory for processed data")
    parser.add_argument("--notch", type=float, default=60.0, 
                        help="Notch frequency (50.0 for EU, 60.0 for US). Default: 60.0")
    parser.add_argument("--fs", type=int, default=200, 
                        help="Target sampling frequency. Default: 200")
    parser.add_argument("--win_sec", type=int, default=10, 
                        help="Window length in seconds. Default: 10")
    return parser.parse_args()

# ================= HELPERS =================

def zscore_per_channel(x: np.ndarray) -> np.ndarray:
    """Z-score each channel independently within the window."""
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return (x - mean) / (std + 1e-8)


def process_one_edf(args_tuple):
    """Worker function to process a single EDF file."""
    edf_path, out_dir, label, notch_freq, target_sfreq, win_samples = args_tuple
    
    edf_path = Path(edf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")

        # Drop junk channels
        to_drop = [ch for ch in DROP_CHANNELS if ch in raw.ch_names]
        if to_drop:
            raw.drop_channels(to_drop)

        if len(raw.ch_names) != len(CH_ORDER_STANDARD):
            raw.pick_channels(CH_ORDER_STANDARD, ordered=True)
        else:
            raw.reorder_channels(CH_ORDER_STANDARD)

        if raw.ch_names != CH_ORDER_STANDARD:
             # Last check
             raw.reorder_channels(CH_ORDER_STANDARD)

        # Preprocessing
        raw.filter(l_freq=0.1, h_freq=75.0, verbose="ERROR")
        raw.notch_filter(notch_freq, verbose="ERROR")
        
        # Resample
        if raw.info['sfreq'] != target_sfreq:
            raw.resample(target_sfreq, n_jobs=1, verbose="ERROR")

        data = raw.get_data(units='uV').astype(np.float32)  # [C, T] in µV

    except Exception as e:
        print(f"Failed {edf_path.name}: {e}")
        return

    # Split into windows
    n_windows = data.shape[1] // win_samples
    stem = edf_path.stem

    for i in range(n_windows):
        start = i * win_samples
        end = start + win_samples
        seg = data[:, start:end]  # [23, samples]

        # Z-score normalization per channel per window
        seg = zscore_per_channel(seg)

        save_path = out_dir / f"{stem}_{i:04d}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump({"X": seg, "y": label}, f, protocol=pickle.HIGHEST_PROTOCOL)


def gather_tasks(root_path, target_path):
    tasks = []
    root = Path(root_path)
    target = Path(target_path)
    
    if (root / "edf").is_dir():
        search_root = root / "edf"
    else:
        search_root = root

    for split in ("train", "eval"):
        for cls_name, label in (("abnormal", 1), ("normal", 0)):
            in_dir = search_root / split / cls_name / CHANNEL_STD
            
            if not in_dir.is_dir():
                print(f"Skipping missing dir: {in_dir}")
                continue

            out_dir = target / split / str(label)
            
            files = list(in_dir.rglob("*.edf"))
            
            for edf_file in files:
                tasks.append((edf_file, out_dir, label))
                
    return tasks


if __name__ == "__main__":
    args = parse_args()
    
    WIN_SAMPLES = args.win_sec * args.fs

    print(f"Starting TUAB preprocessing -> {args.out_dir}")
    print(f"Settings: {args.fs} Hz, {args.win_sec}s windows ({WIN_SAMPLES} samples), notch={args.notch} Hz")

    file_tasks = gather_tasks(args.root_dir, args.out_dir)
    print(f"Found {len(file_tasks)} EDF files to process")

    if not file_tasks:
        raise RuntimeError("No tasks found! Check ROOT path and structure.")

    full_tasks = [
        (t[0], t[1], t[2], args.notch, args.fs, WIN_SAMPLES) 
        for t in file_tasks
    ]

    workers = min(32, os.cpu_count() or 1)
    print(f"Using {workers} workers")

    with Pool(processes=workers) as pool:
        list(pool.imap_unordered(process_one_edf, full_tasks))

    print(f"\nDone! All windows saved to:\n  {args.out_dir}")
    print("  Structure: train|eval / 0|1 / *.pkl")