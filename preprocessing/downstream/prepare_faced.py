import argparse
import os
import lmdb
import pickle
import numpy as np
import scipy.signal as signal
from sklearn.model_selection import KFold, train_test_split

# ----------------- ARGS ----------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="FACED Dataset -> LMDB with Z-Score & 4-Fold Split")
    
    # Default paths (can be overridden by bash script)
    parser.add_argument('--root_dir', type=str, required=True, 
                        help='Path to the folder containing .pkl subject files')
    parser.add_argument('--out_db', type=str, required=True, 
                        help='Output path for the LMDB folder')
    parser.add_argument('--map_size', type=int, default=1099511627776 * 2, 
                        help='Maximum size of LMDB (default ~2TB)')
    
    return parser.parse_args()

# ----------------- HELPER ----------------- #
def zscore_2d(x):
    """Normalize (32, 2000) array channel-wise"""
    mean = np.mean(x, axis=1, keepdims=True)
    std = np.std(x, axis=1, keepdims=True)
    return (x - mean) / (std + 1e-8)

# ----------------- MAIN ----------------- #
def main():
    args = parse_args()
    
    # Labels (Hardcoded as they are specific to FACED dataset structure)
    labels = np.array([0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8])

    if not os.path.exists(args.root_dir):
        raise FileNotFoundError(f"Root dir not found: {args.root_dir}")

    # 1. GET SUBJECT FILES
    files = sorted([f for f in os.listdir(args.root_dir) if not f.startswith('.')])
    file_to_keys_map = {f: [] for f in files}

    # 2. SETUP LMDB
    if not os.path.exists(args.out_db):
        os.makedirs(args.out_db)
    
    print(f"Opening LMDB at {args.out_db}...")
    db = lmdb.open(args.out_db, map_size=args.map_size)
    
    print(f"Processing {len(files)} subjects...")

    # 3. PROCESSING LOOP
    for file_idx, file in enumerate(files):
        try:
            file_path = os.path.join(args.root_dir, file)
            with open(file_path, 'rb') as f:
                array = pickle.load(f)

            # Logic: Resample to 6000 -> Reshape to (Trials, Channels, Segments, Time)
            eeg = signal.resample(array, 6000, axis=2)
            eeg_ = eeg.reshape(28, 32, 30, 200)

            txn = db.begin(write=True)

            for i, (samples, label) in enumerate(zip(eeg_, labels)):
                for j in range(3):
                    # Extract 10s chunk (32, 10, 200) -> Flatten to (32, 2000)
                    chunk_3d = samples[:, 10*j:10*(j+1), :]
                    chunk_2d = chunk_3d.reshape(32, -1)
                    
                    # Z-Score
                    sample_norm = zscore_2d(chunk_2d)
                    
                    # Save
                    sample_key = f'{file}-{i}-{j}'
                    data_dict = {'sample': sample_norm.astype(np.float32), 'label': label}
                    
                    txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict))
                    file_to_keys_map[file].append(sample_key)

            txn.commit()
            if (file_idx + 1) % 10 == 0:
                print(f"Processed {file_idx + 1} / {len(files)} files")
                
        except Exception as e:
            print(f"Error processing {file}: {e}")

    # 4. CREATE SPLITS
    print("\nGenerating Subject-Wise Train/Val/Test Splits...")
    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    dataset_split = {}
    
    for fold_num, (train_val_idx, test_idx) in enumerate(kf.split(files)):
        test_files = [files[k] for k in test_idx]
        train_val_files = [files[k] for k in train_val_idx]
        
        # Split Train into Train (80%) and Val (20%)
        train_files, val_files = train_test_split(train_val_files, test_size=0.2, random_state=42)
        
        def get_keys(flist):
            k = []
            for f in flist: k.extend(file_to_keys_map[f])
            return k

        dataset_split[f'fold_{fold_num}'] = {
            'train': get_keys(train_files),
            'val':   get_keys(val_files),
            'test':  get_keys(test_files)
        }
        print(f"Fold {fold_num} | Train: {len(train_files)} subj | Val: {len(val_files)} subj | Test: {len(test_files)} subj")

    txn = db.begin(write=True)
    txn.put(key='__keys__'.encode(), value=pickle.dumps(dataset_split))
    txn.commit()
    db.close()
    print("Done.")

if __name__ == '__main__':
    main()