#!/usr/bin/env python3
"""
Kaggle ERN Dataset -> LMDB Builder
==================================

This script processes the Kaggle ERN (Error-Related Negativity) dataset and 
stores it into an LMDB database.

Pipeline:
1. **Reads Raw CSVs**: Extracts EEG data for training and testing subjects.
2. **Channel Selection**: Selects 19 standard channels.
3. **Epoching**: Extracts time windows based on 'FeedBackEvent' markers.
4. **Resampling**: Interpolates to a fixed target length based on input frequency.
5. **Normalization**: Applies Per-Channel Z-Score Normalization.
6. **Splitting**: Generates a 5-Fold Subject-Independent Split.

Usage:
    python prepare_kaggle_ern.py --root_dir /path --out_db /path --fs 200
"""

import warnings
warnings.filterwarnings('ignore')

import argparse
import os
import csv
import random
import torch
import numpy as np
import tqdm
import pandas as pd
import lmdb
import pickle

# ---------------- CONFIGURATION ---------------- #
# Fixed Time Window parameters
TMIN = -0.7
TLEN = 2.0

# Channel definitions
CH_NAMES_FULL = "Fp1,Fp2,AF7,AF3,AF4,AF8,F7,F5,F3,F1,Fz,F2,F4,F6,F8,FT7,FC5,FC3,FC1,FCz,FC2,FC4,FC6,FT8,T7,C5,C3,C1,Cz,C2,C4,C6,T8,TP7,CP5,CP3,CP1,CPz,CP2,CP4,CP6,TP8,P7,P5,P3,P1,Pz,P2,P4,P6,P8,PO7,POz,PO8,O1,O2".split(',')
USE_CHANNELS = [
    'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
    'T7', 'C3', 'CZ', 'C4', 'T8', 'P7', 'P3', 'PZ', 'P4', 'P8', 'O1', 'O2'
]

# ---------------- ARGUMENT PARSER ---------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="Kaggle ERN -> LMDB with Z-Score & 5-Fold Split")
    
    parser.add_argument('--root_dir', type=str, required=True, 
                        help='Root path containing the KaggleERN folder')
    parser.add_argument('--out_db', type=str, required=True, 
                        help='Output path for the LMDB folder')
    parser.add_argument('--map_size', type=int, default=1099511627776 * 2, 
                        help='Maximum size of LMDB (default ~2TB)')
    
    # NEW: Sampling rate argument
    parser.add_argument('--fs', type=int, default=200, 
                        help='Target sampling rate in Hz (default: 200)')
    
    return parser.parse_args()

# ---------------- HELPERS ---------------- #

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def zscore_per_channel(x):
    """
    Z-score normalization per channel (dim 0) along time (dim 1).
    Input: (Channels, Time)
    """
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    return (x - mean) / (std + 1e-8)

def temporal_interpolation(x, desired_len):
    """Interpolate time dimension to match exact sample count"""
    # x shape: (Channels, Time) -> Unsqueeze for interpolate: (1, C, T)
    x = x.unsqueeze(0)
    out = torch.nn.functional.interpolate(x, size=desired_len, mode='nearest')
    return out.squeeze(0)

def get_channel_indices():
    """Map requested channel names to indices in the raw CSV"""
    full_lower = [x.lower().replace('.', '') for x in CH_NAMES_FULL]
    indices = []
    for ch in USE_CHANNELS:
        ch_clean = ch.lower().replace('.', '')
        if ch_clean in full_lower:
            indices.append(full_lower.index(ch_clean))
        else:
            raise ValueError(f"Channel {ch} not found in Kaggle header")
    return indices

# ---------------- CORE LOGIC ---------------- #

def read_csv_epochs(filename, channel_indices, fs):
    """
    Reads a single Kaggle CSV, extracts feedback epochs, resamples, z-scores.
    Now accepts 'fs' (sampling rate) as an argument.
    """
    if not os.path.exists(filename):
        print(f"Warning: File not found {filename}")
        return []

    df = pd.read_csv(filename)
    
    # Extract EEG Data (columns 1:-2 usually) and Feedback Events
    raw_data = torch.tensor(df.iloc[:, 1:-2].values).float() # (Time, All_Channels)
    feedback_events = df['FeedBackEvent'].values
    
    # Identify epoch start positions
    stim_indices = np.where(feedback_events > 0)[0]
    
    epochs = []
    # Calculate desired length based on dynamic input frequency
    desired_len = int(fs * TLEN)
    
    for idx in stim_indices:
        start_idx = int(idx + (fs * TMIN))
        end_idx = start_idx + desired_len
        
        # Skip incomplete edge trials
        if start_idx < 0 or end_idx > len(raw_data):
            continue 
            
        # Extract trial: (Time, All_Channels)
        trial_data = raw_data[start_idx:end_idx, :]
        
        # Transpose to (All_Channels, Time)
        trial_data = trial_data.t()
        
        # Select Channels
        trial_data = trial_data[channel_indices, :] 
        
        if trial_data.shape[-1] != desired_len:
             trial_data = temporal_interpolation(trial_data, desired_len)
             
        # Z-Score Normalization
        trial_data = zscore_per_channel(trial_data)
        
        epochs.append(trial_data)
        
    return epochs

def get_labels_map(path):
    """Load TrainLabels.csv into a dictionary mapping ID -> Label"""
    labels = {}
    label_path = os.path.join(path, 'KaggleERN', 'TrainLabels.csv')
    with open(label_path, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i > 0: # Skip header
                labels[row[0]] = int(row[1])
    return labels

def get_true_labels(path):
    """Load true_labels.csv for Test set"""
    true_labels = pd.read_csv(os.path.join(path, 'KaggleERN', 'true_labels.csv'))['label'].values
    return true_labels

# ---------------- MAIN PROCESSING ---------------- #

def main():
    args = parse_args()
    seed_torch(9)
    
    # Define Split Structure
    Folds = {
        1: ([11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 26], [2, 6, 7]),
        2: ([2, 6, 7, 14, 16, 17, 18, 20, 21, 22, 23, 24, 26], [11, 12, 13]),
        3: ([2, 6, 7, 11, 12, 13, 18, 20, 21, 22, 23, 24, 26], [14, 16, 17]),
        4: ([2, 6, 7, 11, 12, 13, 14, 16, 17, 18, 23, 24, 26], [20, 21, 22]),
        5: ([2, 6, 7, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22], [23, 24, 26]),
    }
    test_subjects = [1, 3, 4, 5, 8, 9, 10, 15, 19, 25]
    
    # Collect ALL subjects
    all_train_subjects = [2, 6, 7, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 26]
    all_subjects = sorted(list(set(all_train_subjects + test_subjects)))
    
    # Prepare Output
    if not os.path.exists(args.out_db):
        os.makedirs(args.out_db)
    
    print(f"Opening LMDB at {args.out_db}...")
    print(f"Target Frequency: {args.fs} Hz")
    
    db = lmdb.open(args.out_db, map_size=args.map_size)
    
    # Load Labels
    train_labels_map = get_labels_map(args.root_dir)
    test_labels_arr = get_true_labels(args.root_dir)
    
    # Prepare Channel Indices
    ch_indices = get_channel_indices()
    
    # Store keys per subject for splitting later
    subject_keys = {s: [] for s in all_subjects}
    test_label_idx = 0 
    
    print(f"Processing {len(all_subjects)} subjects...")
    
    for subj in tqdm.tqdm(all_subjects):
        is_test_subj = subj in test_subjects
        s_str = f"S0{subj}" if subj < 10 else f"S{subj}"
        
        for sess in range(1, 6):
            sess_str = f"Sess0{sess}"
            
            # File path construction
            if is_test_subj:
                filename = os.path.join(args.root_dir, "KaggleERN", "test", f"Data_{s_str}_{sess_str}.csv")
            else:
                filename = os.path.join(args.root_dir, "KaggleERN", "train", f"Data_{s_str}_{sess_str}.csv")
                
            # Skip known bad file
            if subj == 22 and sess == 5:
                continue
                
            if not os.path.exists(filename):
                continue
                
            # Process Epochs with dynamic Sampling Rate
            epochs = read_csv_epochs(filename, ch_indices, fs=args.fs)
            
            if len(epochs) == 0:
                continue
                
            txn = db.begin(write=True)
            
            for i, x in enumerate(epochs):
                fb_count = i + 1 
                
                # Determine Label
                if is_test_subj:
                    y = test_labels_arr[test_label_idx]
                    test_label_idx += 1
                else:
                    lbl_key = f"{s_str}_{sess_str}_FB{fb_count:03d}"
                    if lbl_key in train_labels_map:
                        y = train_labels_map[lbl_key]
                    else:
                        print(f"Label missing for {lbl_key}")
                        continue

                # Create Unique DB Key
                db_key = f"s{subj:02d}_sess{sess:02d}_{i:04d}"
                
                # Save
                payload = {
                    'sample': x.numpy().astype(np.float32), 
                    'label': int(y)
                }
                txn.put(key=db_key.encode(), value=pickle.dumps(payload))
                
                # Track key
                subject_keys[subj].append(db_key)
                
            txn.commit()

    # ---------------- GENERATE SPLITS ---------------- #
    print("Generating 5-Fold Splits...")
    dataset_split = {}
    
    # 1. Test Keys (Same for all folds)
    test_keys_all = []
    for s in test_subjects:
        test_keys_all.extend(subject_keys[s])
        
    # 2. Train/Val per Fold
    for fold_num, (train_subs, val_subs) in Folds.items():
        fold_train_keys = []
        for s in train_subs:
            fold_train_keys.extend(subject_keys[s])
            
        fold_val_keys = []
        for s in val_subs:
            fold_val_keys.extend(subject_keys[s])
            
        dataset_split[f'fold_{fold_num}'] = {
            'train': fold_train_keys,
            'val': fold_val_keys,
            'test': test_keys_all
        }
        
        print(f"Fold {fold_num}: Train={len(fold_train_keys)}, Val={len(fold_val_keys)}, Test={len(test_keys_all)}")

    # Save Splits
    txn = db.begin(write=True)
    txn.put(key='__keys__'.encode(), value=pickle.dumps(dataset_split))
    txn.commit()
    
    db.close()
    print("Done! LMDB created successfully.")

if __name__ == "__main__":
    main()