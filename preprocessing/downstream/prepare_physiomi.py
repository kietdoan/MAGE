#!/usr/bin/env python3
"""
PhysioNet Motor Imagery (EEGMMIDB) -> LMDB Builder
==================================================

Pipeline:
1. **Subject Discovery**: Scans for Subject folders (S001, S002, ...).
2. **Loading**: Reads specific EDF files for Motor Imagery tasks (04, 06, 08, 10, 12, 14).
3. **Preprocessing**:
   - Channel Selection (64 channels).
   - Bandpass Filter (0.3 Hz - None).
   - Notch Filter (60 Hz).
   - Resampling (Target FS, e.g., 200 Hz).
4. **Normalization**: Per-Channel Z-Score Normalization.
5. **Storage**: Saves epoched data to LMDB.

Usage:
    python prepare_physionet_mi.py --root_dir /path/to/eegmmidb --out_db /path/to/lmdb --fs 200
"""

import os
import argparse
import pickle
import lmdb
import numpy as np
import mne

# ---------------- CONFIG ---------------- #
TASKS = ['04', '06', '08', '10', '12', '14']   # Motor imagery runs

# 64-channel selection
SELECTED_CHANNELS = [
    'Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.', 'C5..', 'C3..', 'C1..', 'Cz..', 'C2..',
    'C4..', 'C6..', 'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.', 'Fp1.', 'Fpz.', 'Fp2.',
    'Af7.', 'Af3.', 'Afz.', 'Af4.', 'Af8.', 'F7..', 'F5..', 'F3..', 'F1..', 'Fz..', 'F2..', 'F4..',
    'F6..', 'F8..', 'Ft7.', 'Ft8.', 'T7..', 'T8..', 'T9..', 'T10.', 'Tp7.', 'Tp8.', 'P7..', 'P5..',
    'P3..', 'P1..', 'Pz..', 'P2..', 'P4..', 'P6..', 'P8..', 'Po7.', 'Po3.', 'Poz.', 'Po4.', 'Po8.',
    'O1..', 'Oz..', 'O2..', 'Iz..'
]

EPOCH_LEN_S  = 4.0      # Seconds
HP_CUT       = 0.3      # Hz
NOTCH_HZ     = 60       # Hz

# ---------------- ARGS ---------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="PhysioNet MI -> LMDB with Z-Score")
    
    parser.add_argument('--root_dir', type=str, required=True, 
                        help='Path to the "files" folder of eegmmidb (containing S001, S002...)')
    parser.add_argument('--out_db', type=str, required=True, 
                        help='Output path for the LMDB folder')
    parser.add_argument('--fs', type=float, default=200.0, 
                        help='Target sampling frequency (Hz). Default: 200')
    parser.add_argument('--map_size', type=int, default=5_000_000_000, 
                        help='LMDB map size (default 5GB)')
    
    return parser.parse_args()

# ---------------- HELPERS ---------------- #
def znorm_per_channel(x, eps=1e-6):
    """
    Standardize (Z-score) each channel over time.
    x shape: (n_channels, n_timepoints)
    """
    mu = np.mean(x, axis=1, keepdims=True)
    sigma = np.std(x, axis=1, keepdims=True)
    return (x - mu) / (sigma + eps)

# ---------------- MAIN ---------------- #
def main():
    args = parse_args()
    
    # MNE verbosity
    mne.set_log_level('ERROR')

    # 1. Discover Subjects
    if not os.path.exists(args.root_dir):
        raise FileNotFoundError(f"Root dir not found: {args.root_dir}")

    files = [
        f for f in os.listdir(args.root_dir)
        if os.path.isdir(os.path.join(args.root_dir, f)) and f.startswith('S') and f[1:].isdigit()
    ]
    files = sorted(files)

    files_dict = {
        'train': files[:70],
        'eval' : files[90:109], 
    }
    print(f"Subject split: Train={len(files_dict['train'])}, Eval={len(files_dict['eval'])}")

    dataset_keys = {'train': [], 'eval': []}

    # 2. Setup LMDB
    if not os.path.exists(args.out_db):
        os.makedirs(args.out_db)
        
    print(f"Opening LMDB at {args.out_db} (Target FS: {args.fs} Hz)...")
    db = lmdb.open(args.out_db, map_size=args.map_size, subdir=True, lock=True)
    txn = db.begin(write=True)
    written_since_commit = 0
    commit_every = 256

    try:
        for split_name, subject_list in files_dict.items():
            for subj in subject_list:
                for task in TASKS:
                    edf_path = os.path.join(args.root_dir, subj, f'{subj}R{task}.edf')
                    
                    if not os.path.isfile(edf_path):
                        print(f"[WARN] Missing file: {edf_path}")
                        continue

                    try:
                        # -- Load --
                        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
                        
                        # -- Channel Pick --
                        raw.pick_channels(SELECTED_CHANNELS, ordered=True)
                        
                        # -- Interpolate Bads --
                        if len(raw.info.get('bads', [])) > 0:
                            raw.interpolate_bads()

                        # -- Re-reference --
                        raw.set_eeg_reference(ref_channels='average')

                        # -- Filter --
                        raw.filter(l_freq=HP_CUT, h_freq=None, verbose=False)
                        raw.notch_filter(freqs=[NOTCH_HZ], verbose=False)

                        # -- Resample (Dynamic FS) --
                        if raw.info['sfreq'] != args.fs:
                            raw.resample(args.fs, npad="auto", verbose=False)

                        # -- Epoching --
                        events_from_annot, event_dict = mne.events_from_annotations(raw, verbose=False)
                        
                        epochs = mne.Epochs(
                            raw,
                            events_from_annot,
                            event_dict,
                            tmin=0.0,
                            tmax=EPOCH_LEN_S - 1.0 / raw.info['sfreq'],
                            baseline=None,
                            preload=True,
                            verbose=False,
                        )

                        data = epochs.get_data(units='uV')     # (N, C, T)
                        events = epochs.events[:, 2]           # (N,)

                        for i, (sample, ev) in enumerate(zip(data, events)):
                            if ev == 1:
                                continue
                            
                            if sample.ndim != 2:
                                continue

                            # -- Z-Score Normalization --
                            sample = znorm_per_channel(sample)

                            if task in ['04', '08', '12']:
                                label = int(ev - 2)
                            else:
                                label = int(ev)

                            # -- Write to DB --
                            key = f'{subj}R{task}-{i}'
                            data_dict = {'sample': sample.astype(np.float32), 'label': label}

                            txn.put(key=key.encode(), value=pickle.dumps(data_dict))
                            dataset_keys[split_name].append(key)
                            written_since_commit += 1

                            if written_since_commit >= commit_every:
                                txn.commit()
                                txn = db.begin(write=True)
                                written_since_commit = 0
                                
                    except Exception as e:
                        print(f"[ERROR] Failed {subj}R{task}: {e}")

        # Save keys
        txn.put(key='__keys__'.encode(), value=pickle.dumps(dataset_keys))
        txn.commit()

    finally:
        db.close()

    print("Done.")
    print({k: len(v) for k, v in dataset_keys.items()})

if __name__ == "__main__":
    main()