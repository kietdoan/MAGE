#!/usr/bin/env python3
"""
BCICIV2a -> LMDB (no split), 4s chunks, keep shape (C, T), no patchify.
Now with PER-TRIAL PER-CHANNEL Z-SCORE NORMALIZATION (default)
"""

from __future__ import annotations

import os
import pickle
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import lmdb
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, resample
from scipy.signal import lfilter

# --- EMS import (optional) ---
try:
    from braindecode.preprocessing import exponential_moving_standardize as _ems
except Exception:
    _ems = None


# ----------------------------- Utils --------------------------------- #
def setup_logger(verbosity: int = 1) -> None:
    level = logging.WARNING if verbosity <= 0 else (logging.INFO if verbosity == 1 else logging.DEBUG)
    logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=level)


def butter_bandpass(low_cut: float, high_cut: float, fs: float, order: int = 5):
    nyq = 0.5 * fs
    low = low_cut / nyq
    high = high_cut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def zscore_per_channel(x: np.ndarray) -> np.ndarray:
    """Z-score each channel independently (mean=0, std=1) within the trial."""
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return (x - mean) / (std + 1e-8)


# --------------------------- Core logic ------------------------------- #
def _unwrap_cell(x):
    while isinstance(x, np.ndarray) and x.dtype == object and x.size == 1:
        x = x.item()
    return x


def _extract_triplet_from_block(block, *, n_chans: int):
    b = _unwrap_cell(block)

    if hasattr(b, "__dict__") and not isinstance(b, (np.ndarray, list, tuple)):
        if all(hasattr(b, nm) for nm in ("X", "trial", "y")):
            raw, events, labels = b.X, b.trial, b.y
        else:
            raw_attr = next((getattr(b, nm, None) for nm in ("raw", "signal", "data", "X", "cnt", "eeg", "samples")), None)
            evt_attr = next((getattr(b, nm, None) for nm in ("trial", "events", "ev", "markers")), None)
            lab_attr = next((getattr(b, nm, None) for nm in ("y", "labels", "label", "class", "target")), None)
            if raw_attr is None or evt_attr is None or lab_attr is None:
                fields = sorted(k for k in b.__dict__.keys() if not k.startswith("_"))
                raise TypeError(f"Unrecognized MATLAB struct fields: {fields}")
            raw, events, labels = raw_attr, evt_attr, lab_attr

    elif isinstance(b, np.ndarray) and b.dtype == object and b.size >= 3:
        flat = b.ravel()
        raw, events, labels = flat[0], flat[1], flat[2]
    elif isinstance(b, (list, tuple)) and len(b) >= 3:
        raw, events, labels = b[0], b[1], b[2]
    else:
        if isinstance(block, np.ndarray) and block.dtype == object:
            try:
                return _extract_triplet_from_block(block[0, 0], n_chans=n_chans)
            except Exception:
                pass
            try:
                return _extract_triplet_from_block(block.item(), n_chans=n_chans)
            except Exception:
                pass
        raise TypeError(f"Unsupported block type: {type(b)} with shape={getattr(b, 'shape', None)}")

    raw = np.asarray(raw)
    events = np.asarray(events).ravel()
    labels = np.asarray(labels).ravel()

    if raw.ndim != 2:
        raise ValueError(f"Expected 2D raw, got shape {raw.shape}")
    S, C = raw.shape
    if C == n_chans:
        pass
    elif S == n_chans:
        raw = raw.T
    return raw, events, labels


def process_mat_file(
    mat_path: Path,
    *,
    fs: int,
    n_chans: int,
    band_low: float,
    band_high: float,
    butter_order: int,
    t_start: float,
    window_sec: float,
    resample_hz: int,
    apply_ems: bool = False,
    ems_factor_new: float = 0.001,
    ems_init_block_size: Optional[int] = None,
    ems_eps: float = 1e-4,
    zscore: bool = True,          # ← NOW DEFAULT: z-score per trial per channel
    key_prefix: Optional[str] = None,
) -> List[Tuple[str, bytes]]:

    logging.info(f"Reading {mat_path.name}")
    data = loadmat(mat_path, squeeze_me=False, struct_as_record=False)
    if "data" not in data:
        raise KeyError(f"'data' key not found in {mat_path}")

    blocks = data["data"]
    if not isinstance(blocks, np.ndarray):
        raise ValueError(f"Unexpected 'data' container type: {type(blocks)}")

    if apply_ems and _ems is None:
        raise RuntimeError("apply_ems=True but 'braindecode' not installed.")

    b, a = butter_bandpass(band_low, band_high, fs, order=butter_order)

    kvs: List[Tuple[str, bytes]] = []
    base_name = key_prefix if key_prefix is not None else mat_path.stem

    j_start = 3 if blocks.shape[-1] > 3 else 0
    for j in range(j_start, blocks.shape[-1]):
        cell = blocks.item(j) if blocks.dtype == object and blocks.ndim == 1 else blocks[(0, j)] if blocks.ndim == 2 else blocks[j]
        try:
            raw_full, events, labels = _extract_triplet_from_block(cell, n_chans=n_chans)
        except Exception as e_try:
            try:
                raw_full, events, labels = _extract_triplet_from_block(_unwrap_cell(cell)[0, 0], n_chans=n_chans)
            except Exception as e_fallback:
                raise TypeError(f"{mat_path.name} block {j}: failed to parse block. "
                                f"Primary error: {e_try}; Fallback error: {e_fallback}")

        raw_full = np.asarray(raw_full, dtype=np.float32)
        if raw_full.ndim != 2:
            logging.debug(f"Skip block {j}: raw ndim={raw_full.ndim}")
            continue
        if raw_full.shape[1] < n_chans and raw_full.shape[0] >= n_chans:
            raw_full = raw_full.T
        raw = raw_full[:, :n_chans]
        S, C = raw.shape
        if C != n_chans:
            logging.warning(f"{mat_path.name} block {j}: expected {n_chans} chans, got {C}")

        ev = events.astype(int).tolist()
        ev.append(S)
        segs = list(zip(ev[:-1], ev[1:]))

        s0 = int(round(t_start * fs))
        win_len = int(round(window_sec * fs))
        s1 = s0 + win_len

        for i, ((s, e), lab) in enumerate(zip(segs, labels)):
            x = raw[s:e].T  # -> [C, T_trial]
            x = x - x.mean(axis=0, keepdims=True)   # per-channel centering

            x = lfilter(b, a, x, axis=-1)           # bandpass (causal)

            if x.shape[-1] < s1:
                continue

            x = x[:, s0:s1]                         # fixed 4s window

            if resample_hz > 0 and resample_hz != fs:
                target_len = int(round(window_sec * resample_hz))
                x = resample(x, target_len, axis=-1)

            # === NORMALIZATION ===
            if zscore:
                x = zscore_per_channel(x)           # ← YOUR Z-SCORE (default)
            elif apply_ems:
                x = _ems(
                    x.astype(np.float32, copy=False),
                    factor_new=ems_factor_new,
                    init_block_size=ems_init_block_size,
                    eps=ems_eps,
                ).astype(np.float32, copy=False)

            y = int(lab) - 1
            key = f"{base_name}-{j}-{i}"
            kvs.append((key, pickle.dumps({"sample": x.astype(np.float32), "label": y}, protocol=pickle.HIGHEST_PROTOCOL)))

    logging.info(f"{mat_path.name}: produced {len(kvs)} samples")
    return kvs


def write_lmdb(kvs: List[Tuple[str, bytes]], out_path: Path, *, map_size: int, batch_size: int = 512) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(out_path), map_size=map_size, subdir=True, lock=True)
    try:
        count = 0
        txn = env.begin(write=True)
        try:
            for k, v in kvs:
                txn.put(k.encode("utf-8"), v)
                count += 1
                if count % batch_size == 0:
                    txn.commit()
                    txn = env.begin(write=True)
            txn.commit()
        finally:
            if txn is not None:
                try:
                    txn.abort()
                except lmdb.Error:
                    pass
    finally:
        env.close()


# ------------------------------ CLI ---------------------------------- #
def parse_args() -> argparse.Namespace:
    # FIXED (Fully Anonymous):
    default_root = Path(os.getenv("BCIC2A_MAT_DIR", "./data/BCIC2A_mat"))
    default_out  = Path(os.getenv("BCIC2A_LMDB", "./output/BCICIV2a.lmdb"))

    p = argparse.ArgumentParser(description="BCICIV2a -> LMDB with z-score normalization (default)")
    p.add_argument("--root_dir", type=Path, default=default_root)
    p.add_argument("--out_db", type=Path, default=default_out)
    p.add_argument("--map_size", type=int, default=1_500_000_000)

    p.add_argument("--fs", type=int, default=250)
    p.add_argument("--n_chans", type=int, default=22)
    p.add_argument("--band_low", type=float, default=0.3)
    p.add_argument("--band_high", type=float, default=50.0)
    p.add_argument("--butter_order", type=int, default=5)
    p.add_argument("--t_start", type=float, default=2.0)
    p.add_argument("--window_sec", type=float, default=4.0)
    p.add_argument("--resample_hz", type=int, default=200)

    # === NEW: z-score is default, EMS is optional ===
    p.add_argument("--no-zscore", action="store_true", help="Disable z-score normalization")
    p.add_argument("--apply_ems", action="store_true", help="Use EMS instead of z-score")

    p.add_argument("--ems_factor_new", type=float, default=0.001)
    p.add_argument("--ems_init_block_size", type=int, default=0)
    p.add_argument("--ems_eps", type=float, default=1e-4)

    p.add_argument("--verbosity", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logger(args.verbosity)

    if not args.root_dir.exists():
        raise FileNotFoundError(f"root_dir not found: {args.root_dir}")
    mat_files = sorted(args.root_dir.glob("*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files in {args.root_dir}")

    all_kvs = []
    all_keys = []

    for mat_path in mat_files:
        kvs = process_mat_file(
            mat_path,
            fs=args.fs,
            n_chans=args.n_chans,
            band_low=args.band_low,
            band_high=args.band_high,
            butter_order=args.butter_order,
            t_start=args.t_start,
            window_sec=args.window_sec,
            resample_hz=args.resample_hz,
            apply_ems=args.apply_ems,
            ems_factor_new=args.ems_factor_new,
            ems_init_block_size=None if args.ems_init_block_size == 0 else args.ems_init_block_size,
            ems_eps=args.ems_eps,
            zscore=not args.no_zscore,   # ← z-score by default
        )
        all_kvs.extend(kvs)
        all_keys.extend([k for k, _ in kvs])

    logging.info(f"Writing {len(all_kvs)} samples to {args.out_db}")
    write_lmdb(all_kvs, args.out_db, map_size=args.map_size)

    env = lmdb.open(str(args.out_db), map_size=args.map_size, subdir=True, lock=True)
    try:
        with env.begin(write=True) as txn:
            txn.put(b"__keys__", pickle.dumps(all_keys, protocol=pickle.HIGHEST_PROTOCOL))
    finally:
        env.close()

    logging.info("Done. Z-score normalized BCICIV2a LMDB created.")


if __name__ == "__main__":
    main()