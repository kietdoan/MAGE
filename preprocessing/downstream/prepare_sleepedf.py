#!/usr/bin/env python3
import os
import gc
from pathlib import Path
import numpy as np
import torch

from braindecode.datasets import SleepPhysionet
from braindecode.preprocessing import preprocess, Preprocessor
from braindecode.preprocessing.windowers import create_windows_from_events

# ---------------- Custom Functions ---------------- #

def zscore_per_window(data):
    """
    Perform Z-score normalization on a 3D array (n_windows, n_channels, n_times).
    Normalizes each channel in each window independently across the time axis.
    """
    # data shape: (n_windows, n_channels, n_times)
    mean = np.mean(data, axis=-1, keepdims=True)
    std = np.std(data, axis=-1, keepdims=True)
    return (data - mean) / (std + 1e-8)

# ---------------- Config ---------------- #
n_jobs = 4
target_sfreq = 200
window_size_samples = 30 * target_sfreq
high_cut_hz = 30.0
factor_v_to_uv = 1e6

mapping = {
    'Sleep stage W': 0, 'Sleep stage 1': 1, 'Sleep stage 2': 2,
    'Sleep stage 3': 3, 'Sleep stage 4': 3, 'Sleep stage R': 4,
}

# Output root - ANONYMIZED PATH
dataset_root = Path(os.environ.get("SLEEPEDF_PATH", "data/SLEEPEDF"))
dataset_root.mkdir(parents=True, exist_ok=True)
for lab in range(5):
    (dataset_root / str(lab)).mkdir(parents=True, exist_ok=True)

# Subject list (using a generic range for reproducibility)
all_subjects = [s for s in range(0, 83) if s not in {39, 68, 69, 78, 79}]

num_subs = len(all_subjects)
print(f"=== Starting Dataset Preprocessing ===")
print(f"Total subjects to process: {num_subs}")
print(f"Output directory: {dataset_root}")
print(f"========================================\n")

# ---------------- Preprocessors ---------------- #

preprocessors_record = [
    Preprocessor(lambda data: data * factor_v_to_uv), # V -> µV
    Preprocessor('filter', l_freq=None, h_freq=high_cut_hz, n_jobs=n_jobs),
    Preprocessor('resample', sfreq=target_sfreq, n_jobs=n_jobs),
]

preprocessors_window = [
    Preprocessor(zscore_per_window, apply_on_array=True)
]

# ---------------- Processing loop ---------------- #

for idx, sub in enumerate(all_subjects, 1):
    try:
        print(f"[{idx}/{num_subs}] Loading & processing Subject {sub}...")
        
        # Load subject
        dataset = SleepPhysionet(subject_ids=[sub], crop_wake_mins=30)
        
        # Standardize Channels
        for ds in dataset.datasets:
            try:
                ds.raw.pick_channels(['Fpz-Cz', 'Pz-Oz'])
            except ValueError:
                actual = ds.raw.ch_names
                pick_list = [ch for ch in actual if ch in ['EEG Fpz-Cz', 'EEG Pz-Oz', 'Fpz-Cz', 'Pz-Oz']]
                if len(pick_list) >= 2:
                    ds.raw.pick_channels(pick_list[:2])
                else:
                    print(f"!!! Warning Sub {sub}: Standard channels not found. Found: {actual}")
            
        preprocess(dataset, preprocessors_record)

        windows_dataset = create_windows_from_events(
            dataset,
            window_size_samples=window_size_samples,
            window_stride_samples=window_size_samples,
            preload=True,
            mapping=mapping,
            drop_bad_windows=True
        )

        preprocess(windows_dataset, preprocessors_window)

        # Save windows to disk
        count_win = 0
        for i, (X, y, meta) in enumerate(windows_dataset):
            lbl = int(y)
            save_path = dataset_root / str(lbl) / f"sub{sub}_win{i:05d}.pt"
            
            torch.save(torch.from_numpy(X).float(), save_path)
            count_win += 1

        print(f"   -> Finished Subject {sub}: Saved {count_win} windows.")

        del dataset, windows_dataset
        gc.collect()

    except Exception as e:
        print(f"!!! Error at Subject {sub}: {e}")

print(f"\nPreprocessing Complete. Data saved to {dataset_root}")