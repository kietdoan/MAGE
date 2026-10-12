import lmdb
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, db_path, keys):
        self.db_path = db_path
        self.keys = keys
        self.env = None

    def _init_env(self):
        self.env = lmdb.open(
            self.db_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        if self.env is None:
            self._init_env()
            
        key = self.keys[idx]
        with self.env.begin(write=False) as txn:
            byteflow = txn.get(key.encode())
        
        if byteflow is None:
            raise ValueError(f"Key {key} not found in LMDB")
            
        data = pickle.loads(byteflow)
        
        x = torch.from_numpy(data['sample'])
        
        y = torch.tensor(data['label']).long()
        
        return x, y

class LoadDataset:
    def __init__(self, params):
        self.params = params
        self.db_path = params.datasets_dir
        
        self.fold_num = params.fold
        
        print(f"[KaggleERN Dataset] Loading Fold {self.fold_num} from {self.db_path}...")
        
        env = lmdb.open(self.db_path, readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            keys_packed = txn.get(b'__keys__')
            if keys_packed is None:
                raise ValueError("Key '__keys__' not found. Did preprocessing finish?")
            all_folds = pickle.loads(keys_packed)
        env.close()
        
        fold_key = f"fold_{self.fold_num}"
        if fold_key not in all_folds:
            raise ValueError(f"Fold key {fold_key} not found. Available: {list(all_folds.keys())}")
            
        fold_data = all_folds[fold_key]
        
        self.train_keys = fold_data['train']
        self.val_keys   = fold_data['val']
        self.test_keys  = fold_data['test']
        
        print(f"  Train: {len(self.train_keys)}")
        print(f"  Val:   {len(self.val_keys)}")
        print(f"  Test:  {len(self.test_keys)}")

    def get_data_loader(self):
        train_ds = CustomDataset(self.db_path, self.train_keys)
        val_ds   = CustomDataset(self.db_path, self.val_keys)
        test_ds  = CustomDataset(self.db_path, self.test_keys)
        
        dataloaders = {}
        
        dataloaders['train'] = DataLoader(
            train_ds,
            batch_size=self.params.batch_size,
            shuffle=self.params.shuffle_train,
            num_workers=self.params.num_workers,
            pin_memory=self.params.pin_memory,
            drop_last=self.params.drop_last
        )
        
        dataloaders['val'] = DataLoader(
            val_ds,
            batch_size=self.params.batch_size,
            shuffle=False,
            num_workers=self.params.num_workers,
            pin_memory=self.params.pin_memory
        )
        
        dataloaders['test'] = DataLoader(
            test_ds,
            batch_size=self.params.batch_size,
            shuffle=False,
            num_workers=self.params.num_workers,
            pin_memory=self.params.pin_memory
        )
        
        return dataloaders