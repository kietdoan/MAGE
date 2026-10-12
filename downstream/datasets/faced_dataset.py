import lmdb
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, db_path, keys, transform=None):
        self.db_path = db_path
        self.keys = keys
        self.transform = transform
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
            raise ValueError(f"Key {key} not found in LMDB!")

        data = pickle.loads(byteflow)
        
        x = data['sample'] 
        y = data['label']

        x = torch.from_numpy(x).float()
        y = torch.tensor(y).long()

        if self.transform:
            x = self.transform(x)

        return x, y

class LoadDataset:
    """
    Wrapper class expected by the training runner.
    """
    def __init__(self, params):
        self.params = params
        self.db_path = params.datasets_dir
        
        self.fold_idx = getattr(params, 'fold_idx', 0) 
        
        print(f"[FACED Dataset] Loading Fold {self.fold_idx} from {self.db_path}...")

        env = lmdb.open(self.db_path, readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            keys_packed = txn.get(b'__keys__')
            if keys_packed is None:
                raise ValueError("Key '__keys__' not found in LMDB. Did preprocessing finish correctly?")
            self.all_folds = pickle.loads(keys_packed)
        env.close()

        fold_key = f'fold_{self.fold_idx}'
        if fold_key not in self.all_folds:
            available_folds = list(self.all_folds.keys())
            raise ValueError(f"Fold '{fold_key}' not found in DB keys. Available: {available_folds}")
            
        fold_data = self.all_folds[fold_key]
        
        self.train_keys = fold_data['train']
        self.val_keys = fold_data['val']
        self.test_keys = fold_data['test']
        
        print(f"[FACED Dataset] Fold {self.fold_idx} Loaded.")
        print(f"  Train samples: {len(self.train_keys)}")
        print(f"  Val samples:   {len(self.val_keys)}")
        print(f"  Test samples:  {len(self.test_keys)}")

    def get_data_loader(self):
        train_dataset = CustomDataset(self.db_path, self.train_keys)
        val_dataset = CustomDataset(self.db_path, self.val_keys)
        test_dataset = CustomDataset(self.db_path, self.test_keys)

        dataloaders = {}
        
        dataloaders['train'] = DataLoader(
            train_dataset,
            batch_size=self.params.batch_size,
            shuffle=self.params.shuffle_train,
            num_workers=self.params.num_workers,
            pin_memory=self.params.pin_memory,
            drop_last=self.params.drop_last
        )

        dataloaders['val'] = DataLoader(
            val_dataset,
            batch_size=self.params.batch_size,
            shuffle=False,
            num_workers=self.params.num_workers,
            pin_memory=self.params.pin_memory,
            drop_last=False
        )

        dataloaders['test'] = DataLoader(
            test_dataset,
            batch_size=self.params.batch_size,
            shuffle=False,
            num_workers=self.params.num_workers,
            pin_memory=self.params.pin_memory,
            drop_last=False
        )

        return dataloaders