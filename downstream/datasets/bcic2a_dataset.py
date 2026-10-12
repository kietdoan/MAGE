#!/usr/bin/env python3
from __future__ import annotations

import pickle, random, os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import lmdb
import torch
from torch.utils.data import Dataset, DataLoader

from downstream.utils import to_tensor


# ------------------------------ Dataset ------------------------------ #

class LMDBDataset(Dataset):
    def __init__(self, db_path: str, keys: Sequence[str], scale_div: float = 100.0):
        super().__init__()
        self.db_path = db_path
        self.keys = list(keys)
        self.scale_div = float(scale_div)
        self._env: Optional[lmdb.Environment] = None

    def _ensure_env(self):
        if self._env is None:
            self._env = lmdb.open(self.db_path, readonly=True, lock=False, readahead=True, meminit=False)

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int):
        self._ensure_env()
        k = self.keys[idx]
        with self._env.begin(write=False) as txn:
            pair = pickle.loads(txn.get(k.encode()))
        x = pair["sample"].astype(np.float32) / self.scale_div  # [C, T]
        y = int(pair["label"])
        return x, y

    @staticmethod
    def collate(batch):
        x = np.stack([b[0] for b in batch], axis=0).astype(np.float32)
        y = np.array([b[1] for b in batch], dtype=np.int64)
        return to_tensor(x), to_tensor(y).long()


# --------------------------- Small utilities -------------------------- #

def _load_flat_keys(db_path: str) -> List[str]:
    env = lmdb.open(db_path, readonly=True, lock=False, readahead=True, meminit=False)
    try:
        with env.begin(write=False) as txn:
            raw = pickle.loads(txn.get(b"__keys__"))
    finally:
        env.close()
    if isinstance(raw, dict):
        out, seen = [], set()
        for m in ("train", "val", "test"):
            for k in raw.get(m, []):
                if k not in seen:
                    out.append(k); seen.add(k)
        for m, lst in raw.items():
            if m in {"train", "val", "test"}: continue
            for k in lst:
                if k not in seen:
                    out.append(k); seen.add(k)
        return out
    return list(raw)

def _key_to_subject_token(key: str) -> str:
    return key.split("-", 1)[0][:3]

def _group_by_subject(keys: Sequence[str]) -> Tuple[List[str], Dict[str, List[str]]]:
    tokens: List[str] = []
    seen = set()
    by = {}
    for k in keys:
        s = _key_to_subject_token(k)
        if s not in seen:
            tokens.append(s); seen.add(s)
        by.setdefault(s, []).append(k)
    return tokens, by

def _normalize_ids(ids: Optional[Sequence[int]], n_subj: int) -> List[int]:
    """Accept 0- or 1-based indices. Return sorted unique 0-based."""
    if not ids:
        return []
    ids = list(dict.fromkeys(int(i) for i in ids))
    if all(1 <= i <= n_subj for i in ids) and not any(i == 0 for i in ids):
        ids = [i - 1 for i in ids]
    bad = [i for i in ids if i < 0 or i >= n_subj]
    if bad:
        raise ValueError(f"Subject ids out of range [0..{n_subj-1}]: {bad}")
    return sorted(ids)

def _read_labels_for_keys(db_path: str, keys: Sequence[str]) -> List[int]:
    labels: List[int] = []
    env = lmdb.open(db_path, readonly=True, lock=False, readahead=True, meminit=False)
    try:
        with env.begin(write=False) as txn:
            for k in keys:
                pair = pickle.loads(txn.get(k.encode()))
                labels.append(int(pair["label"]))
    finally:
        env.close()
    return labels

def _stratified_idx_split(labels: Sequence[int], val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    by_cls: Dict[int, List[int]] = {}
    for i, y in enumerate(labels):
        by_cls.setdefault(int(y), []).append(i)
    tr, va = [], []
    for idxs in by_cls.values():
        rng.shuffle(idxs)
        n = len(idxs)
        k = max(1, int(round(n * val_ratio))) if 0.0 < val_ratio < 1.0 else 0
        k = min(k, n - 1) if n >= 2 else k
        va.extend(idxs[:k]); tr.extend(idxs[k:])
    rng.shuffle(tr); rng.shuffle(va)
    return tr, va


# ------------------------------- Loader -------------------------------- #

class LoadDataset:
    """
    Split modes:

    - subject_explicit:
        • Provide any of: train_subject_ids, val_subject_ids, test_subject_ids (as ints).
        • IDs are indices into the discovered subject list (0- or 1-based accepted).
        • Any subject not listed goes to train by default.

    - mixed:
        • Provide test_subject_ids (int or list[int]) for subject-wise test.
        • Provide val_ratio in (0,1). We split sample-wise train/val within the remaining subjects.
        • Optional stratified_val (default True).

    Required params:
      - datasets_dir (LMDB path), batch_size

    Optional params (defaults):
      - split_mode: 'subject_explicit' | 'mixed'  (default 'subject_explicit')
      - train_subject_ids, val_subject_ids, test_subject_ids
      - val_ratio: 0.1 (mixed)
      - stratified_val: True (mixed)
      - seed: 7
      - num_workers: 0, pin_memory: True, drop_last: False, shuffle_train: True, scale_div: 100.0
    """

    def __init__(self, params):
        self.p = params
        self.db_path = params.datasets_dir
        self.batch_size = params.batch_size

        self.split_mode = getattr(params, "split_mode", "subject_explicit")
        self.seed = int(getattr(params, "seed", 7))

        # subject_explicit
        self.train_subject_ids = getattr(params, "train_subject_ids", None)
        self.val_subject_ids   = getattr(params, "val_subject_ids", None)
        self.test_subject_ids  = getattr(params, "test_subject_ids", None)

        # mixed
        self.val_ratio = float(getattr(params, "val_ratio", 0.1))
        self.stratified_val = bool(getattr(params, "stratified_val", True))

        # loader knobs
        self.num_workers = int(getattr(params, "num_workers", 0))
        self.pin_memory  = bool(getattr(params, "pin_memory", True))
        self.drop_last   = bool(getattr(params, "drop_last", False))
        self.shuffle_train = bool(getattr(params, "shuffle_train", True))
        self.scale_div = float(getattr(params, "scale_div", 100.0))

    # ---- API ----
    def get_data_loader(self) -> Dict[str, DataLoader]:
        flat = _load_flat_keys(self.db_path)
        subj_tokens, by_subj = _group_by_subject(flat)
        n_subj = len(subj_tokens)
        if n_subj == 0:
            raise RuntimeError("No subjects found in LMDB.")

        print("[BCIC2a] subject index mapping:")
        for i, tok in enumerate(subj_tokens):
            print(f"  {i}: {tok}")

        if self.split_mode == "subject_explicit":
            tr_ids = _normalize_ids(self.train_subject_ids, n_subj)
            va_ids = _normalize_ids(self.val_subject_ids,   n_subj)
            te_ids = _normalize_ids(self.test_subject_ids,  n_subj)

            if not tr_ids:
                listed = set(va_ids) | set(te_ids)
                tr_ids = [i for i in range(n_subj) if i not in listed]

            # Validate no overlaps
            if set(tr_ids) & set(va_ids) or set(tr_ids) & set(te_ids) or set(va_ids) & set(te_ids):
                raise ValueError("train/val/test subject id sets must be disjoint.")

            train_keys = self._gather(by_subj, subj_tokens, tr_ids)
            val_keys   = self._gather(by_subj, subj_tokens, va_ids)
            test_keys  = self._gather(by_subj, subj_tokens, te_ids)
            return self._make_loaders(train_keys, val_keys, test_keys)

        elif self.split_mode == "mixed":
            te_ids = _normalize_ids(self.test_subject_ids if isinstance(self.test_subject_ids, (list, tuple)) else [self.test_subject_ids], n_subj)
            if not te_ids:
                raise ValueError("mixed split requires test_subject_ids.")
            test_keys = self._gather(by_subj, subj_tokens, te_ids)

            pool_ids = [i for i in range(n_subj) if i not in te_ids]
            pool_keys = self._gather(by_subj, subj_tokens, pool_ids)

            if not (0.0 < self.val_ratio < 1.0):
                return self._make_loaders(pool_keys, [], test_keys)

            if self.stratified_val:
                labels = _read_labels_for_keys(self.db_path, pool_keys)
                tr_idx, va_idx = _stratified_idx_split(labels, self.val_ratio, self.seed)
            else:
                rng = random.Random(self.seed)
                idxs = list(range(len(pool_keys))); rng.shuffle(idxs)
                k = max(1, int(round(len(pool_keys) * self.val_ratio)))
                va_idx, tr_idx = idxs[:k], idxs[k:]

            train_keys = [pool_keys[i] for i in tr_idx]
            val_keys   = [pool_keys[i] for i in va_idx]
            return self._make_loaders(train_keys, val_keys, test_keys)

        else:
            raise ValueError(f"Unknown split_mode: {self.split_mode!r}")

    # ---- helpers ----
    @staticmethod
    def _gather(by_subj: Dict[str, List[str]], tokens: List[str], idxs: List[int]) -> List[str]:
        out: List[str] = []
        for i in idxs:
            out.extend(by_subj[tokens[i]])
        return out

    def _make_loader(self, ds: LMDBDataset, *, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            collate_fn=LMDBDataset.collate,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last if shuffle else False,
        )

    def _make_loaders(self, train_keys: List[str], val_keys: List[str], test_keys: List[str]) -> Dict[str, DataLoader]:
        print(f"[BCIC2a] train={len(train_keys)} | val={len(val_keys)} | test={len(test_keys)}")
        loaders: Dict[str, DataLoader] = {
            "train": self._make_loader(LMDBDataset(self.db_path, train_keys, self.scale_div), shuffle=self.shuffle_train),
            "test":  self._make_loader(LMDBDataset(self.db_path, test_keys,  self.scale_div), shuffle=False),
        }
        if val_keys:
            loaders["val"] = self._make_loader(LMDBDataset(self.db_path, val_keys, self.scale_div), shuffle=False)
        return loaders
