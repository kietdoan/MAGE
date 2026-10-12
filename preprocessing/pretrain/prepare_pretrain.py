#!/usr/bin/env python3
"""
TUH EEG → LMDB windowed dataset builder — stable, resumable, parallel-safe
Per-channel z-normalization (mean=0, std=1) is ALWAYS applied per window.
This is the modern standard for deep learning on EEG.
"""

from __future__ import annotations
import os
# ---- Prevent oversubscription ----
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import hashlib
import json
import pickle
import random
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Set, Dict
import gc

import lmdb
import mne
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

# -------------------------- Channel templates --------------------------
SELECTED_CHANNELS = {
    "01_tcp_ar": [
        "EEG FP1-REF","EEG FP2-REF","EEG F3-REF","EEG F4-REF","EEG C3-REF","EEG C4-REF","EEG P3-REF",
        "EEG P4-REF","EEG O1-REF","EEG O2-REF","EEG F7-REF","EEG F8-REF","EEG T3-REF","EEG T4-REF",
        "EEG T5-REF","EEG T6-REF","EEG FZ-REF","EEG CZ-REF","EEG PZ-REF",
    ],
    "02_tcp_le": [
        "EEG FP1-LE","EEG FP2-LE","EEG F3-LE","EEG F4-LE","EEG C3-LE","EEG C4-LE","EEG P3-LE",
        "EEG P4-LE","EEG O1-LE","EEG O2-LE","EEG F7-LE","EEG F8-LE","EEG T3-LE","EEG T4-LE",
        "EEG T5-LE","EEG T6-LE","EEG FZ-LE","EEG CZ-LE","EEG PZ-LE",
    ],
    "03_tcp_ar_a": [
        "EEG FP1-REF","EEG FP2-REF","EEG F3-REF","EEG F4-REF","EEG C3-REF","EEG C4-REF","EEG P3-REF",
        "EEG P4-REF","EEG O1-REF","EEG O2-REF","EEG F7-REF","EEG F8-REF","EEG T3-REF","EEG T4-REF",
        "EEG T5-REF","EEG T6-REF","EEG FZ-REF","EEG CZ-REF","EEG PZ-REF",
    ],
}

# ------------------------------ Helpers --------------------------------
def setup_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)

def iter_edf_files_multi(parent_root: Path, dataset_names: List[str]):
    for dname in dataset_names:
        droot = parent_root / dname
        if not droot.exists():
            print(f"[warn] dataset folder missing: {droot}")
            continue
        for p in sorted(droot.rglob("*.edf")):
            if p.is_file():
                yield p

def detect_montage_key(file_path: Path) -> Optional[str]:
    s = str(file_path).lower()
    if "02_tcp_le" in s: return "02_tcp_le"
    if "01_tcp_ar" in s: return "01_tcp_ar"
    if "03_tcp_ar_a" in s: return "03_tcp_ar_a"
    return None

def pick_ordered_channels(raw: mne.io.BaseRaw, montage_key: str) -> bool:
    want = SELECTED_CHANNELS[montage_key]
    have = set(raw.info["ch_names"])
    if any(ch not in have for ch in want):
        return False
    raw.pick_channels(want, ordered=True)
    return True

def file_signature(path: Path) -> str:
    st = path.stat()
    sig = f"{path.resolve()}::{st.st_size}::{int(st.st_mtime)}"
    return hashlib.sha1(sig.encode("utf-8")).hexdigest()

def split_decision(path: Path, frac_train: float, frac_val: float, seed: int) -> str:
    h = hashlib.sha1((str(path.resolve()) + f"|{seed}").encode("utf-8")).digest()
    u = int.from_bytes(h[:8], "big") / float(2**64)
    if u < frac_train:
        return "train"
    if u < frac_train + frac_val:
        return "val"
    return "skip"

# ----------------------------- Core: Z-Normalization ALWAYS ON -----------------------------
def preprocess_and_segment(
    file_path: Path,
    *,
    fs: int,
    l_freq: float,
    h_freq: float,
    mains: Optional[int],
    window_sec: int,
    edge_trim_sec: int,
    min_len_sec: int,
    amp_max_uv: float,
    dtype: str,
    n_jobs: int,
    limit_windows: Optional[int] = None,
) -> Tuple[List[str], List[np.ndarray]]:
    raw = None
    try:
        raw = mne.io.read_raw_edf(str(file_path), preload=True, verbose="ERROR")
        mkey = detect_montage_key(file_path)
        if mkey is None or not pick_ordered_channels(raw, mkey):
            return ([], [])

        raw.resample(fs, npad="auto", n_jobs=n_jobs)
        raw.filter(l_freq=l_freq, h_freq=h_freq, n_jobs=n_jobs, verbose="ERROR")
        nyq = fs / 2.0
        if mains is not None and mains > 0 and mains <= nyq:
            freqs = [mains * k for k in range(1, int(nyq // mains) + 1)]
            raw.notch_filter(freqs=freqs, n_jobs=n_jobs, verbose="ERROR")

        try:
            data = raw.get_data(units="uV")
        except TypeError:
            data = raw.get_data() * 1e6

        if dtype == "float32":
            data = data.astype(np.float32, copy=False)
        elif dtype == "float64":
            data = data.astype(np.float64, copy=False)

        C, T = data.shape
        if T < min_len_sec * fs:
            return ([], [])

        wlen = window_sec * fs
        edge = edge_trim_sec * fs
        leftover = T % wlen
        start = edge
        end = T - (leftover + edge)
        if end <= start:
            return ([], [])

        data = data[:, start:end]
        n_win = data.shape[1] // wlen
        if n_win == 0:
            return ([], [])

        data = data[:, : n_win * wlen].reshape(C, n_win, window_sec, fs).transpose(1, 0, 2, 3)  # [N, C, S, Fs]

        # Amplitude filter
        mx = np.max(np.abs(data), axis=(1, 2, 3))
        keep_idx = np.nonzero(mx < amp_max_uv)[0]
        if keep_idx.size == 0:
            return ([], [])
        if limit_windows is not None and keep_idx.size > limit_windows:
            keep_idx = keep_idx[:limit_windows]

        # Z-NORMALIZE PER CHANNEL PER WINDOW (ALWAYS ON)
        def z_normalize_window(arr: np.ndarray) -> np.ndarray:
            # arr: [C, S, Fs]
            mean = np.mean(arr, axis=(1, 2), keepdims=True)
            std = np.std(arr, axis=(1, 2), keepdims=True)
            return (arr - mean) / (std + 1e-6)

        ids = [str(i) for i in keep_idx.tolist()]
        windows = data[keep_idx]  # [N, C, S, Fs]
        samples = [z_normalize_window(w) for w in windows]

        return (ids, samples)

    except Exception as e:
        print(f"[error] Failed {file_path.name}: {e}")
        return ([], [])
    finally:
        if raw is not None:
            try: raw.close()
            except: pass
        del raw
        gc.collect()

# --------------------------- Compaction --------------------------------
def compact_lmdb_inplace(db_dir: Path, map_size_tb: float) -> None:
    db_dir = db_dir.resolve()
    if not db_dir.exists():
        raise FileNotFoundError(db_dir)

    with tempfile.TemporaryDirectory(dir=db_dir.parent) as tmpd:
        tmp_path = Path(tmpd) / "lmdb_compact"
        tmp_path.mkdir(parents=True, exist_ok=True)

        env_src = lmdb.open(str(db_dir), readonly=True, lock=False, subdir=True, readahead=False)
        src_info = env_src.info()
        map_size_bytes = max(int(map_size_tb * (1024 ** 4)), src_info.get("map_size", 0))
        env_dst = lmdb.open(str(tmp_path), map_size=map_size_bytes, subdir=True, lock=True,
                            writemap=True, map_async=True, metasync=False, readahead=False, max_readers=2048)

        with env_src.begin() as s_txn, env_dst.begin(write=True) as d_txn:
            cur = s_txn.cursor()
            for k, v in cur:
                d_txn.put(k, v)

        env_src.close()
        env_dst.close()

        swap_old = db_dir.with_name(db_dir.name + "_to_delete")
        db_dir.rename(swap_old)
        shutil.move(str(tmp_path), str(db_dir))
        shutil.rmtree(swap_old)
        print(f"[compact] Completed compaction at {db_dir}")

# --------------------------- Parallel Producer --------------------------------
def _producer_worker(shard_files, cfg, q, key_scheme: str, limit_windows: Optional[int],
                     split_assign: Dict[Path, str], single_db_prefix: bool,
                     joblib_temp: Optional[Path]):
    mne.set_log_level("ERROR")
    if joblib_temp:
        os.environ["JOBLIB_TEMP_FOLDER"] = str(joblib_temp)

    for fp in shard_files:
        try:
            s = split_assign.get(fp, "skip")
            if s == "skip":
                q.put(("seen", file_signature(fp), None))
                continue

            fsig = file_signature(fp)
            ids, arrays = preprocess_and_segment(
                fp,
                fs=cfg["fs_hz"],
                l_freq=cfg["l_freq_hz"],
                h_freq=cfg["h_freq_hz"],
                mains=(None if cfg["mains_hz"] == 0 else cfg["mains_hz"]),
                window_sec=cfg["window_sec"],
                edge_trim_sec=cfg["edge_trim_sec"],
                min_len_sec=cfg["min_len_sec"],
                amp_max_uv=cfg["amp_max_uV"],
                dtype=cfg["dtype"],
                n_jobs=cfg["n_jobs"],
                limit_windows=limit_windows,
            )
            base = fp.stem
            fsig_short = fsig[:10]
            for wid, arr in zip(ids, arrays):
                base_key = f"{base}_{wid}" if key_scheme == "stem" else f"{base}_{fsig_short}_{wid}"
                if single_db_prefix:
                    pref = "trn_" if s == "train" else "val_"
                    base_key = pref + base_key
                payload = pickle.dumps(arr, protocol=pickle.HIGHEST_PROTOCOL)
                q.put(("data", s, base_key, payload), block=True)
            q.put(("seen", fsig, None), block=True)
        except Exception:
            q.put(("seen", file_signature(fp), None))
            continue
        finally:
            gc.collect()
    q.put(("done", None, None))

# --------------------------- Parallel Runner (unchanged logic) ----------------
def _run_parallel(files, env_train, env_val, using_config, *, present_keys_train: Set[str], present_keys_val: Set[str],
                  seen_files: Set[str], batch_writes: int, checkpoint_interval: int, seen_flush_every: int,
                  key_scheme: str, workers: int, writer_queue_cap: int, limit_windows: Optional[int], dry_run: bool,
                  split_assign: Dict[Path, str], single_db_prefix: bool, joblib_temp: Optional[Path]):
    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    q = manager.Queue(maxsize=writer_queue_cap)

    shards = [[] for _ in range(workers)]
    for i, fp in enumerate(files):
        shards[i % workers].append(fp)

    procs = []
    for shard in shards:
        p = ctx.Process(
            target=_producer_worker,
            args=(shard, using_config, q, key_scheme, limit_windows, split_assign, single_db_prefix, joblib_temp),
            daemon=True,
        )
        p.start()
        procs.append(p)

    pending_trn: List[Tuple[str, bytes]] = []
    pending_val: List[Tuple[str, bytes]] = []
    written_trn = written_val = 0
    skipped_trn = skipped_val = 0
    since_last_ckpt_trn = since_last_ckpt_val = 0
    seen_local = set(seen_files)
    live = workers

    pbar = tqdm(total=len(files), desc="Writing", unit="file")
    seen_files_counted = set()
    sample_printed = 0

    try:
        while live > 0:
            msg = q.get()
            # msg is a tuple like:
            # ("data", split, key_str, payload)  -> 4 items
            # ("seen", fsig, None)               -> 3 items
            # ("done", None, None)               -> 3 items
            if not isinstance(msg, (tuple, list)) or not msg:
                continue

            typ = msg[0]

            if typ == "data":
                # Unpack the 4-tuple produced by _producer_worker
                try:
                    _, split, key_str, payload = msg
                except Exception:
                    # Back-compat or malformed; skip safely
                    continue

                if dry_run:
                    if sample_printed < 5:
                        arr = pickle.loads(payload)
                        print(f"[dry-run] {split}:{key_str} -> shape={arr.shape}, dtype={arr.dtype}")
                        sample_printed += 1
                    continue

                if split == "train":
                    if key_str in present_keys_train:
                        skipped_trn += 1
                    else:
                        pending_trn.append((key_str, payload))
                        if len(pending_trn) >= batch_writes:
                            with env_train.begin(write=True) as txn:
                                for k, v in pending_trn:
                                    if txn.get(k.encode("utf-8")) is None:
                                        txn.put(k.encode("utf-8"), v)
                                        present_keys_train.add(k)
                                        written_trn += 1
                            pending_trn.clear()
                            since_last_ckpt_trn += batch_writes
                            if checkpoint_interval and since_last_ckpt_trn >= checkpoint_interval:
                                with env_train.begin(write=True) as txn:
                                    txn.put(b"__keys__", pickle.dumps(sorted(present_keys_train), protocol=pickle.HIGHEST_PROTOCOL))
                                since_last_ckpt_trn = 0
                else:  # split == "val"
                    if key_str in present_keys_val:
                        skipped_val += 1
                    else:
                        pending_val.append((key_str, payload))
                        if len(pending_val) >= batch_writes:
                            with env_val.begin(write=True) as txn:
                                for k, v in pending_val:
                                    if txn.get(k.encode("utf-8")) is None:
                                        txn.put(k.encode("utf-8"), v)
                                        present_keys_val.add(k)
                                        written_val += 1
                            pending_val.clear()
                            since_last_ckpt_val += batch_writes
                            if checkpoint_interval and since_last_ckpt_val >= checkpoint_interval:
                                with env_val.begin(write=True) as txn:
                                    txn.put(b"__keys__", pickle.dumps(sorted(present_keys_val), protocol=pickle.HIGHEST_PROTOCOL))
                                since_last_ckpt_val = 0

            elif typ == "seen":
                # ("seen", fsig, None)
                try:
                    _, fsig, _ = msg
                except Exception:
                    continue
                if fsig not in seen_local:
                    seen_local.add(fsig)
                    if fsig not in seen_files_counted:
                        pbar.update(1)
                        seen_files_counted.add(fsig)
                    if not dry_run and seen_flush_every and (len(seen_local) % seen_flush_every == 0):
                        for env in (env_train, env_val):
                            with env.begin(write=True) as txn:
                                txn.put(b"__seen_files__", pickle.dumps(sorted(seen_local), protocol=pickle.HIGHEST_PROTOCOL))

            elif typ == "done":
                live -= 1
            else:
                # Unknown message type; ignore safely
                continue


        # Flush leftovers
        if not dry_run:
            if pending_trn:
                with env_train.begin(write=True) as txn:
                    for k, v in pending_trn:
                        if txn.get(k.encode("utf-8")) is None:
                            txn.put(k.encode("utf-8"), v)
                            present_keys_train.add(k)
                            written_trn += 1
            if pending_val:
                with env_val.begin(write=True) as txn:
                    for k, v in pending_val:
                        if txn.get(k.encode("utf-8")) is None:
                            txn.put(k.encode("utf-8"), v)
                            present_keys_val.add(k)
                            written_val += 1
            for env, keys in ((env_train, present_keys_train), (env_val, present_keys_val)):
                with env.begin(write=True) as txn:
                    txn.put(b"__keys__", pickle.dumps(sorted(keys), protocol=pickle.HIGHEST_PROTOCOL))
                    txn.put(b"__seen_files__", pickle.dumps(sorted(seen_local), protocol=pickle.HIGHEST_PROTOCOL))

    finally:
        pbar.close()
        for p in procs:
            p.join(timeout=10)

    if dry_run:
        print(f"[parallel][dry-run] files_seen={len(seen_local)} (no writes)")
    else:
        print(f"[parallel] written(train)={written_trn} skipped(train)={skipped_trn} | "
              f"written(val)={written_val} skipped(val)={skipped_val} | files_seen={len(seen_local)}")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build LMDB from TUH EEG EDF files (windowed, global meta).")
    p.add_argument("--in-root", type=Path, required=True,
                   help="Parent folder containing TUH sub-datasets (e.g., tuh_eeg_*).")
    p.add_argument("--datasets", type=str,
                   default="tuh_eeg_artifact,tuh_eeg_epilepsy,tuh_eeg_seizure,tuh_eeg_slowing",
                   help="Comma-separated subfolders under --in-root to scan for EDFs.")
    p.add_argument("--allowed-montages", type=str, default="01_tcp_ar",
                   help="Comma-separated subset of {01_tcp_ar,02_tcp_le,03_tcp_ar_a} to include.")

    # Deterministic split (both produced in one run)
    p.add_argument("--fraction-train", type=float, default=0.10, help="Fraction of files assigned to train.")
    p.add_argument("--fraction-val", type=float, default=0.01, help="Fraction of files assigned to val.")
    p.add_argument("--split-seed", type=int, default=12345, help="Seed for deterministic file split.")

    # Output choices
    p.add_argument("--out-lmdb-train", type=Path, help="LMDB for TRAIN split.")
    p.add_argument("--out-lmdb-val", type=Path, help="LMDB for VAL split.")
    p.add_argument("--out-lmdb", type=Path,
                   help="Single LMDB output for BOTH splits (keys will be prefixed 'trn_'/'val_').")

    p.add_argument("--map-size-tb", type=float, default=1.6, help="LMDB map size in terabytes")
    p.add_argument("--seed", type=int, default=1)

    # Processing
    p.add_argument("--fs", type=int, default=200)
    p.add_argument("--l-freq", type=float, default=0.3)
    p.add_argument("--h-freq", type=float, default=75.0)
    p.add_argument("--mains", type=int, default=60, choices=[0, 50, 60])
    p.add_argument("--window-sec", type=int, default=30)
    p.add_argument("--edge-trim-sec", type=int, default=60)
    p.add_argument("--min-len-sec", type=int, default=300)
    p.add_argument("--amp-max", type=float, default=100)
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    p.add_argument("--n-jobs", type=int, default=1, help="Threads per MNE op (forced to 1 if --workers>1)")
    p.add_argument("--joblib-temp", type=Path, default=Path("/tmp/joblib"),
                   help="Folder for joblib temp files (helps on small /dev/shm).")

    # IO policy / resume
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--batch-writes", type=int, default=512)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--force-rewrite-config", action="store_true")
    p.add_argument("--checkpoint-interval", type=int, default=20000)
    p.add_argument("--seen-files-flush-every", type=int, default=50)
    p.add_argument("--key-scheme", type=str, default="stem_sig", choices=["stem", "stem_sig"])

    # Small run / smoke test
    p.add_argument("--limit-files", type=int, default=None, help="Process only first N EDF files")
    p.add_argument("--limit-windows", type=int, default=None, help="Keep at most N windows per file")
    p.add_argument("--dry-run", action="store_true", help="Run without writing LMDB; print sample shapes")

    # Parallel
    p.add_argument("--workers", type=int, default=1, help="Producer processes (set >1 for parallel)")
    p.add_argument("--writer-queue-cap", type=int, default=2048)

    # Modes
    p.add_argument("--compact", action="store_true", help="Compact an existing LMDB in place and exit")
    p.add_argument("--compact-db", type=Path, help="LMDB directory to compact (defaults to outputs)")
    return p.parse_args()

def main() -> None:
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    args = parse_args()
    mne.set_log_level("ERROR")

    if args.joblib_temp:
        os.environ["JOBLIB_TEMP_FOLDER"] = str(args.joblib_temp)
        Path(args.joblib_temp).mkdir(parents=True, exist_ok=True)

    if args.compact:
        target = args.compact_db or args.out_lmdb or args.out_lmdb_train or args.out_lmdb_val
        if not target:
            raise SystemExit("Provide output path with --compact.")
        compact_lmdb_inplace(target, args.map_size_tb)
        return

    # ... [rest of your original main() — only config changed] ...
    # Validate outputs
    if not (args.out_lmdb or (args.out_lmdb_train and args.out_lmdb_val)):
        raise SystemExit("Provide either --out-lmdb OR both --out-lmdb-train and --out-lmdb-val.")
    single_db_mode = args.out_lmdb is not None

    setup_seed(args.seed)

    in_root: Path = args.in_root
    dataset_names = [s.strip() for s in args.datasets.split(",") if s.strip()]
    files_all = list(iter_edf_files_multi(in_root, dataset_names))

    # Optional montage filter
    allowed_montages = set(s.strip() for s in args.allowed_montages.split(",") if s.strip())
    if allowed_montages:
        files_all = [fp for fp in files_all if (detect_montage_key(fp) in allowed_montages)]

    # Deterministic split assignment
    split_assign: Dict[Path, str] = {}
    train_files, val_files = [], []
    for fp in files_all:
        s = split_decision(fp, args.fraction_train, args.fraction_val, args.split_seed)
        split_assign[fp] = s
        if s == "train":
            train_files.append(fp)
        elif s == "val":
            val_files.append(fp)

    files = train_files + val_files  # both, in one pass
    if args.shuffle:
        random.shuffle(files)
    if args.limit_files is not None:
        files = files[: args.limit_files]

    print("========================================")
    print(f"[split] total files scanned : {len(files_all)}")
    print(f"[split] assigned -> train   : {len(train_files)} "
          f"({len(train_files)/max(1,len(files_all))*100:.2f}%)")
    print(f"[split] assigned -> val     : {len(val_files)} "
          f"({len(val_files)/max(1,len(files_all))*100:.2f}%)")
    print(f"[run]    files this pass    : {len(files)} (train+val)")
    print("========================================")

    # Open LMDBs
    map_size_bytes = int(args.map_size_tb * (1024 ** 4))
    if single_db_mode:
        args.out_lmdb.mkdir(parents=True, exist_ok=True)
        env_train = env_val = lmdb.open(
            str(args.out_lmdb),
            map_size=map_size_bytes, subdir=True, lock=True,
            writemap=True, map_async=True, metasync=False, readahead=False,
            max_readers=2048,
        )
        single_db_prefix = True
    else:
        args.out_lmdb_train.mkdir(parents=True, exist_ok=True)
        args.out_lmdb_val.mkdir(parents=True, exist_ok=True)
        env_train = lmdb.open(
            str(args.out_lmdb_train),
            map_size=map_size_bytes, subdir=True, lock=True,
            writemap=True, map_async=True, metasync=False, readahead=False,
            max_readers=2048,
        )
        env_val = lmdb.open(
            str(args.out_lmdb_val),
            map_size=map_size_bytes, subdir=True, lock=True,
            writemap=True, map_async=True, metasync=False, readahead=False,
            max_readers=2048,
        )
        single_db_prefix = False

    # Load resume meta
    def load_meta(env):
        present: Set[str] = set()
        existing_cfg = None
        seen: Set[str] = set()
        with env.begin(write=False) as txn:
            blob = txn.get(b"__keys__")
            if blob is not None:
                try:
                    present = set(pickle.loads(blob))
                except Exception:
                    cur = txn.cursor()
                    for k, _ in cur:
                        if k not in (b"__keys__", b"__config__", b"__seen_files__"):
                            present.add(k.decode("utf-8"))
            cfg_blob = txn.get(b"__config__")
            if cfg_blob is not None:
                try:
                    existing_cfg = json.loads(cfg_blob.decode("utf-8"))
                except Exception:
                    existing_cfg = None
            s_blob = txn.get(b"__seen_files__")
            if s_blob is not None:
                try:
                    seen = set(pickle.loads(s_blob))
                except Exception:
                    seen = set()
        return present, existing_cfg, seen

    present_keys_train, existing_config_tr, seen_files_tr = load_meta(env_train)
    present_keys_val,   existing_config_vl, seen_files_vl = load_meta(env_val)
    seen_files = seen_files_tr.union(seen_files_vl)

    desired_config = {
        "fs_hz": args.fs,
        "l_freq_hz": args.l_freq,
        "h_freq_hz": args.h_freq,
        "mains_hz": (0 if args.mains == 0 else args.mains),
        "window_sec": args.window_sec,
        "edge_trim_sec": args.edge_trim_sec,
        "min_len_sec": args.min_len_sec,
        "amp_max_uV": args.amp_max,
        "units": "uV",
        "dtype": args.dtype,
        "n_jobs": args.n_jobs,
        "normalization": "zscore_per_channel_per_window",
        "note": "Per-channel z-normalization applied per window (mean=0, std=1)",
        "fraction_train": args.fraction_train,
        "fraction_val": args.fraction_val,
        "split_seed": args.split_seed,
        "single_db_mode": args.out_lmdb is not None,
    }

    def ensure_config(env, existing_cfg):
        if existing_cfg is None or (existing_cfg != desired_config and args.force_rewrite_config):
            with env.begin(write=True) as txn:
                txn.put(b"__config__", json.dumps(desired_config, ensure_ascii=False).encode("utf-8"))
            return desired_config
        if existing_cfg != desired_config and not args.force_rewrite_config:
            raise RuntimeError("Existing __config__ differs. Use --force-rewrite-config to overwrite.")
        return existing_cfg

    using_config = ensure_config(env_train, existing_config_tr)
    ensure_config(env_val, existing_config_vl)

    print("========================================")
    print(f"[resume] keys(train) loaded : {len(present_keys_train)}")
    print(f"[resume] keys(val)   loaded : {len(present_keys_val)}")
    print(f"[resume] seen_files merged  : {len(seen_files)}")
    print("========================================")

    if args.resume and seen_files:
        files = [fp for fp in files if file_signature(fp) not in seen_files]

    if args.workers > 1 and using_config["n_jobs"] != 1:
        print(f"[info] workers={args.workers} > 1 → forcing MNE n_jobs=1 to avoid nested parallelism.")
        using_config["n_jobs"] = 1

    if args.workers <= 1:
        pending_trn: List[Tuple[str, bytes]] = []
        pending_val: List[Tuple[str, bytes]] = []
        written_trn = written_val = 0
        since_last_ckpt_trn = since_last_ckpt_val = 0
        files_processed_since_seen_flush = 0
        sample_printed = 0

        def flush_pending(force_checkpoint: bool = False) -> None:
            nonlocal pending_trn, pending_val, written_trn, written_val
            nonlocal since_last_ckpt_trn, since_last_ckpt_val
            if pending_trn:
                with env_train.begin(write=True) as txn:
                    for k, v in pending_trn:
                        if txn.get(k.encode("utf-8")) is None:
                            txn.put(k.encode("utf-8"), v)
                            present_keys_train.add(k); written_trn += 1; since_last_ckpt_trn += 1
                pending_trn.clear()
            if pending_val:
                with env_val.begin(write=True) as txn:
                    for k, v in pending_val:
                        if txn.get(k.encode("utf-8")) is None:
                            txn.put(k.encode("utf-8"), v)
                            present_keys_val.add(k); written_val += 1; since_last_ckpt_val += 1
                pending_val.clear()
            if args.checkpoint_interval > 0 and force_checkpoint:
                with env_train.begin(write=True) as txn:
                    txn.put(b"__keys__", pickle.dumps(sorted(present_keys_train), protocol=pickle.HIGHEST_PROTOCOL))
                with env_val.begin(write=True) as txn:
                    txn.put(b"__keys__", pickle.dumps(sorted(present_keys_val), protocol=pickle.HIGHEST_PROTOCOL))
                since_last_ckpt_trn = since_last_ckpt_val = 0

        for fp in tqdm(files, desc="Processing EDFs (train+val)", unit="file"):
            s = split_assign.get(fp, "skip")
            if s == "skip":
                continue
            fsig = file_signature(fp)
            try:
                win_ids, arrays = preprocess_and_segment(
                    file_path=fp,
                    fs=using_config["fs_hz"],
                    l_freq=using_config["l_freq_hz"],
                    h_freq=using_config["h_freq_hz"],
                    mains=(None if using_config["mains_hz"] == 0 else using_config["mains_hz"]),
                    window_sec=using_config["window_sec"],
                    edge_trim_sec=using_config["edge_trim_sec"],
                    min_len_sec=using_config["min_len_sec"],
                    amp_max_uv=using_config["amp_max_uV"],
                    dtype=using_config["dtype"],
                    n_jobs=using_config["n_jobs"],
                    limit_windows=args.limit_windows,
                )
            except Exception:
                gc.collect()
                continue

            base = fp.stem
            fsig_short = fsig[:10]
            for wid, arr in zip(win_ids, arrays):
                base_key = f"{base}_{wid}" if args.key_scheme == "stem" else f"{base}_{fsig_short}_{wid}"
                if single_db_mode:
                    pref = "trn_" if s == "train" else "val_"
                    base_key = pref + base_key
                payload = pickle.dumps(arr, protocol=pickle.HIGHEST_PROTOCOL)

                if args.dry_run:
                    if sample_printed < 5:
                        print(f"[dry-run] {s}:{base_key} -> shape={arr.shape}, dtype={arr.dtype}")
                        sample_printed += 1
                    continue

                if s == "train":
                    if not (args.resume and base_key in present_keys_train):
                        pending_trn.append((base_key, payload))
                else:
                    if not (args.resume and base_key in present_keys_val):
                        pending_val.append((base_key, payload))

                if len(pending_trn) + len(pending_val) >= args.batch_writes:
                    flush_pending(force_checkpoint=False)

            if not args.dry_run:
                seen_files.add(fsig)
                files_processed_since_seen_flush += 1
                if args.seen_files_flush_every and files_processed_since_seen_flush >= args.seen_files_flush_every:
                    for env in (env_train, env_val):
                        with env.begin(write=True) as txn:
                            txn.put(b"__seen_files__", pickle.dumps(sorted(seen_files), protocol=pickle.HIGHEST_PROTOCOL))
                    files_processed_since_seen_flush = 0

            gc.collect()

        flush_pending(force_checkpoint=True)
        # Final meta write
        with env_train.begin(write=True) as txn:
            txn.put(b"__keys__", pickle.dumps(sorted(present_keys_train), protocol=pickle.HIGHEST_PROTOCOL))
            txn.put(b"__seen_files__", pickle.dumps(sorted(seen_files), protocol=pickle.HIGHEST_PROTOCOL))
        with env_val.begin(write=True) as txn:
            txn.put(b"__keys__", pickle.dumps(sorted(present_keys_val), protocol=pickle.HIGHEST_PROTOCOL))
            txn.put(b"__seen_files__", pickle.dumps(sorted(seen_files), protocol=pickle.HIGHEST_PROTOCOL))

    else:
        _run_parallel(
            files=files,
            env_train=env_train,
            env_val=env_val,
            using_config=using_config,
            present_keys_train=present_keys_train,
            present_keys_val=present_keys_val,
            seen_files=seen_files,
            batch_writes=args.batch_writes,
            checkpoint_interval=args.checkpoint_interval,
            seen_flush_every=args.seen_files_flush_every,
            key_scheme=args.key_scheme,
            workers=args.workers,
            writer_queue_cap=args.writer_queue_cap,
            limit_windows=args.limit_windows,
            dry_run=args.dry_run,
            split_assign=split_assign,
            single_db_prefix=single_db_mode,  # prefix only if single DB
            joblib_temp=args.joblib_temp,
        )

    # Close
    env_train.sync(); env_train.close()
    if env_val is not env_train:
        env_val.sync(); env_val.close()

    # --- summary ---
    print("========================================")
    if not args.dry_run:
        if single_db_mode:
            print(f"LMDB (single): {args.out_lmdb}  [keys prefixed 'trn_'/'val_']")
        else:
            print(f"LMDB train  : {args.out_lmdb_train}")
            print(f"LMDB val    : {args.out_lmdb_val}")
        print(f"Seen files  : {len(seen_files)}")
    else:
        print("DRY-RUN complete (no LMDB writes).")
    print("========================================")


if __name__ == "__main__":
    main()