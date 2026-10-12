#!/usr/bin/env python3
"""
SHU-MI Dataset -> LMDB Builder
==============================

Pipeline:
1. **File Scanning**: Locates .mat files in the root directory.
2. **Resampling**: Resamples time-series to a fixed number of points (Default: 800).
3. **Normalization**: Applies Per-Channel Z-Score Normalization.
4. **Storage**: Saves processed trials to LMDB.
5. **Splitting**: Generates a 5-Fold Subject-Wise Split (with 10% validation).

Usage:
    python prepare_shumi.py --root_dir /path/to/mat_files --out_db /path/to/lmdb
"""

import os
import argparse
import scipy.io
from scipy import signal
import lmdb
import pickle
import numpy as np
from sklearn.model_selection import KFold
import tqdm

# ---------------- ARGUMENTS ---------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="SHU-MI -> LMDB with Z-Score & 5-Fold Split")
    
    parser.add_argument('--root_dir', type=str, required=True, 
                        help='Path to the folder containing .mat files')
    parser.add_argument('--out_db', type=str, required=True, 
                        help='Output path for the LMDB folder')
    
    # Default target points (e.g., 200Hz * 4s = 800 points)
    parser.add_argument('--target_points', type=int, default=800, 
                        help='Number of time points to resample to (default: 800)')
    
    parser.add_argument('--map_size', type=int, default=1099511627776, 
                        help='LMDB map size in bytes (default 1TB)')
    
    return parser.parse_args()

# ---------------- HELPERS ---------------- #
def zscore_per_channel(x):
    """
    Z-score normalization per channel (dim 0) along time (dim 1).
    Input: (Channels, Time) numpy array.
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    
    # Normalize
    return (x - mean) / (std + 1e-8)

# ---------------- MAIN ---------------- #
def main():
    args = parse_args()

    if not os.path.exists(args.out_db):
        os.makedirs(args.out_db)

    print(f"Scanning files in {args.root_dir}...")
    if not os.path.exists(args.root_dir):
         raise FileNotFoundError(f"Root dir not found: {args.root_dir}")

    files = sorted([f for f in os.listdir(args.root_dir) if f.endswith('.mat')])
    print(f"Found {len(files)} subjects.")

    file_to_keys = {f: [] for f in files}

    print(f"Opening LMDB at {args.out_db}...")
    db = lmdb.open(args.out_db, map_size=args.map_size)

    # ---------------- 1. PROCESS & STORE DATA ---------------- #
    print("Processing files and storing in LMDB...")
    
    for file in tqdm.tqdm(files):
        mat_path = os.path.join(args.root_dir, file)
        try:
            data = scipy.io.loadmat(mat_path)
            eeg = data['data']       # Shape: (Trials, Channels, Time)
            labels = data['labels'][0] # Shape: (Trials,)
        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue

        bz, ch_num, points = eeg.shape
        
        # Resample along time axis (axis 2)
        if points != args.target_points:
            eeg_resampled = signal.resample(eeg, args.target_points, axis=2) 
        else:
            eeg_resampled = eeg
        
        txn = db.begin(write=True)
        
        for i in range(bz):
            # Get single trial: Shape (Channels, Time)
            sample = eeg_resampled[i] 
            label = labels[i]

            # --- APPLY Z-SCORE ---
            sample = zscore_per_channel(sample)
            
            sample = sample.astype(np.float32)

            sample_key = f'{file[:-4]}-{i}'
            
            # Label adjustment: -1 to make it 0-indexed
            data_dict = {
                'sample': sample, 
                'label': int(label) - 1 
            }
            
            txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict))
            file_to_keys[file].append(sample_key)

        txn.commit()

    # ---------------- 2. GENERATE 5-FOLD SPLITS ---------------- #
    print("Generating 5-Fold Subject-Wise Splits...")
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    file_list = np.array(files)
    
    dataset_split = {}
    
    for fold_idx, (train_val_idx, test_idx) in enumerate(kf.split(file_list)):
        # Split Files (Subjects)
        test_files = file_list[test_idx]
        train_val_files_indices = train_val_idx
        
        # 10% Validation Split within the training set
        n_train_val = len(train_val_files_indices)
        n_val = int(n_train_val * 0.1) 
        
        val_idx_sub = train_val_files_indices[:n_val]
        train_idx_sub = train_val_files_indices[n_val:]
        
        val_files = file_list[val_idx_sub]
        train_files = file_list[train_idx_sub]

        # Retrieve Keys
        def get_keys_from_files(file_subset):
            keys = []
            for f in file_subset:
                keys.extend(file_to_keys[f])
            return keys

        train_keys = get_keys_from_files(train_files)
        val_keys = get_keys_from_files(val_files)
        test_keys = get_keys_from_files(test_files)

        fold_name = f'fold_{fold_idx}'
        dataset_split[fold_name] = {
            'train': train_keys,
            'val': val_keys,
            'test': test_keys
        }
        
        print(f"Fold {fold_idx}: Train: {len(train_keys)}, Val: {len(val_keys)}, Test: {len(test_keys)}")

    # ---------------- 3. SAVE KEYS ---------------- #
    print("Saving split configuration...")
    txn = db.begin(write=True)
    txn.put(key='__keys__'.encode(), value=pickle.dumps(dataset_split))
    txn.commit()
    db.close()
    
    print(f"Done! Processed LMDB saved to {args.out_db}")

if __name__ == "__main__":
    main()