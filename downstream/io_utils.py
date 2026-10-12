# io_utils.py
# -*- coding: utf-8 -*-
"""
I/O helpers:
- Kaggle-ERN CSV readers
- Checkpoint discovery (Lightning-style)
"""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import List, Sequence, Tuple, Optional

import torch


USE_CHANNELS_19 = [
    "FP1","FP2","F7","F3","FZ","F4","F8","T7","C3","CZ","C4","T8","P7","P3","PZ","P4","P8","O1","O2"
]
CH_NAMES_KAGGLE_ERN = "Fp1,Fp2,AF7,AF3,AF4,AF8,F7,F5,F3,F1,Fz,F2,F4,F6,F8,FT7,FC5,FC3,FC1,FCz,FC2,FC4,FC6,FT8,T7,C5,C3,C1,Cz,C2,C4,C6,T8,TP7,CP5,CP3,CP1,CPz,CP2,CP4,CP6,TP8,P7,P5,P3,P1,Pz,P2,P4,P6,P8,PO7,POz,PO8,O1,O2".split(",")

def _map_channels_kaggle(names: Sequence[str]) -> List[int]:
    base = [x.lower().strip(".") for x in CH_NAMES_KAGGLE_ERN]
    idxs = []
    for ch in names:
        try:
            idxs.append(base.index(ch.lower()))
        except ValueError as e:
            raise KeyError(f"Channel '{ch}' not found in Kaggle-ERN header list.") from e
    return idxs


def read_csv_epochs(filename: str, tmin: float, tlen: float,
                    use_channels_names: Optional[Sequence[str]] = USE_CHANNELS_19,
                    data_max: Optional[float] = None,
                    data_min: Optional[float] = None) -> List[torch.Tensor]:
    """
    Read one Kaggle-ERN CSV file and extract per-feedback epochs.

    Returns a list of trials with shape [C, T] (possibly C+1 if scale row appended).
    """
    import pandas as pd
    import mne
    from utils import min_max_normalize
    mne.set_log_level("ERROR")

    sfreq = 200
    raw = pd.read_csv(filename)
    data = torch.tensor(raw.iloc[:, 1:-2].values, dtype=torch.float32)
    feed = torch.tensor(raw["FeedBackEvent"].values)
    stim_pos = torch.nonzero(feed > 0, as_tuple=False).flatten()

    use_idxs: Optional[List[int]] = None
    if use_channels_names is not None:
        use_idxs = _map_channels_kaggle(use_channels_names)
        if data_max is not None and data_min is not None:
            use_idxs += [-1]  # add scale row

    trials: List[torch.Tensor] = []
    for pos in stim_pos.tolist():
        start = max(int(pos + sfreq * tmin), 0)
        end = min(start + int(sfreq * tlen), data.shape[0])
        trial = data[start:end, :].T.numpy()  # [C, T]

        info = mne.create_info(ch_names=[str(i) for i in range(trial.shape[0])], ch_types="eeg", sfreq=sfreq)
        raw_mne = mne.io.RawArray(trial, info)

        trial_t = torch.tensor(raw_mne.get_data(), dtype=torch.float32)  # [C, T]
        trial_t = min_max_normalize(trial_t, data_max=data_max, data_min=data_min)
        if use_idxs is not None:
            trial_t = trial_t[use_idxs]
        trials.append(trial_t)
    return trials


def read_kaggle_ern_train(path: str,
                          subjects: Sequence[int],
                          sessions: Sequence[int] = (1, 2, 3, 4, 5),
                          tmin: float = -0.7, tlen: float = 2.0,
                          data_max: Optional[float] = None, data_min: Optional[float] = None,
                          use_channels_names: Optional[Sequence[str]] = USE_CHANNELS_19):
    """Return list of (trial_tensor, label) for Kaggle-ERN Train split."""
    from tqdm import tqdm

    labels = []
    with open(os.path.join(path, "KaggleERN", "TrainLabels.csv"), "r") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i > 0:
                labels.append(row)
    labels = dict(labels)

    data = []
    for i in tqdm(subjects, desc="KaggleERN-Train"):
        for j in sessions:
            if i > 9:
                if (i == 22 and j == 5):
                    print("Skipped known-bad file: Data_S22_Sess05.csv")
                    continue
                filename = os.path.join(path, "KaggleERN", "train", f"Data_S{i}_Sess0{j}.csv")
            else:
                filename = os.path.join(path, "KaggleERN", "train", f"Data_S0{i}_Sess0{j}.csv")

            trials = read_csv_epochs(filename, tmin=tmin, tlen=tlen,
                                     data_max=data_max, data_min=data_min,
                                     use_channels_names=use_channels_names)
            for fb, trial in enumerate(trials, 1):
                key = f"S{i:02d}_Sess{j:02d}_FB{fb:03d}"
                label = int(labels[key])
                data.append((trial, label))
    return data


def read_kaggle_ern_test(path: str,
                         subjects: Sequence[int],
                         sessions: Sequence[int] = (1, 2, 3, 4, 5),
                         tmin: float = -0.7, tlen: float = 2.0,
                         data_max: Optional[float] = None, data_min: Optional[float] = None,
                         use_channels_names: Optional[Sequence[str]] = USE_CHANNELS_19):
    """Return list of (trial_tensor, label) for Kaggle-ERN Test split (labels from true_labels.csv)."""
    from tqdm import tqdm
    import pandas as pd

    labels = pd.read_csv(os.path.join(path, "KaggleERN", "true_labels.csv"))["label"].tolist()
    label_it = iter(labels)

    data = []
    for i in tqdm(subjects, desc="KaggleERN-Test"):
        for j in sessions:
            filename = os.path.join(path, "KaggleERN", "test", f"Data_S{i:02d}_Sess{j:02d}.csv")
            trials = read_csv_epochs(filename, tmin=tmin, tlen=tlen,
                                     data_max=data_max, data_min=data_min,
                                     use_channels_names=use_channels_names)
            for trial in trials:
                data.append((trial, int(next(label_it))))
    return data


# ============================ Checkpoints ============================

_STEP_PAT  = re.compile(r"(?:^|[^0-9])step=([0-9]+)")
_EPOCH_PAT = re.compile(r"(?:^|[^0-9])epoch=([0-9]+)")

def _exists_ok(p: Path) -> bool:
    try:
        return p.is_file() and p.exists()
    except OSError:
        return False

def _extract_metric(name: str, metric: str):
    pats = [
        rf"{re.escape(metric)}=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        rf"val[_-]{re.escape(metric)}=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        rf"val={re.escape(metric)}=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    ]
    for p in pats:
        m = re.search(p, name)
        if m:
            return float(m.group(1))
    return None

def _extract_int(name: str, pat: re.Pattern) -> int:
    m = pat.search(name)
    return int(m.group(1)) if m else -1

def _sort_key(p: Path):
    step  = _extract_int(p.name, _STEP_PAT)
    epoch = _extract_int(p.name, _EPOCH_PAT)
    return (step, epoch, p.stat().st_mtime_ns)

def get_checkpoint_path(base_dir: str, which: str | int = "last",
                        metric: str = "val_loss", mode: str = "min") -> Tuple[str, int]:
    """
    Inspect a Lightning log dir and pick a checkpoint.

    Args:
      base_dir: Path to your Lightning version dir (parent of 'checkpoints').
      which: 'last' | 'best' | specific epoch number (int or 'epoch=XX').
      metric: metric key to parse from filename for 'best' fallback.
      mode: 'min' or 'max' when comparing metric values.

    Returns:
      (checkpoint_path, epoch_number)
    """
    ckpt_dir = (Path(base_dir).expanduser().resolve() / "checkpoints")
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_dir}")

    # ---- LAST ----
    if str(which) == "last":
        for pat in ("last*.ckpt", "*last*.ckpt"):
            cands = [p for p in ckpt_dir.glob(pat) if _exists_ok(p)]
            if cands:
                cands.sort(key=_sort_key, reverse=True)
                chosen = cands[0]
                return str(chosen), _extract_int(chosen.name, _EPOCH_PAT)
        cands = [p for p in ckpt_dir.glob("*.ckpt") if _exists_ok(p)]
        if not cands:
            raise FileNotFoundError(f"No .ckpt files in {ckpt_dir}")
        cands.sort(key=_sort_key, reverse=True)
        chosen = cands[0]
        return str(chosen), _extract_int(chosen.name, _EPOCH_PAT)

    # ---- BEST ----
    if str(which) == "best":
        best_named = []
        for pat in ("best*.ckpt", "*best*.ckpt"):
            best_named.extend([p for p in ckpt_dir.glob(pat) if _exists_ok(p)])
        if best_named:
            best_named.sort(key=_sort_key, reverse=True)
            chosen = best_named[0]
            return str(chosen), _extract_int(chosen.name, _EPOCH_PAT)

        # parse metric from filenames
        cands = [p for p in ckpt_dir.glob("*.ckpt") if _exists_ok(p)]
        scored = []
        for p in cands:
            val = _extract_metric(p.name, metric)
            if val is not None:
                scored.append((p, val))
        if scored:
            chosen = min(scored, key=lambda t: t[1])[0] if mode == "min" else max(scored, key=lambda t: t[1])[0]
            return str(chosen), _extract_int(chosen.name, _EPOCH_PAT)

        # fallback to newest
        if cands:
            cands.sort(key=_sort_key, reverse=True)
            chosen = cands[0]
            return str(chosen), _extract_int(chosen.name, _EPOCH_PAT)
        raise FileNotFoundError(f"No .ckpt files in {ckpt_dir}")

    # ---- SPECIFIC EPOCH ----
    ep = int(str(which).replace("epoch=", ""))
    patterns = (f"*epoch={ep}-*.ckpt", f"*epoch=epoch={ep}-*.ckpt")
    matches: List[Path] = []
    for pat in patterns:
        matches.extend([p for p in ckpt_dir.glob(pat) if _exists_ok(p)])
    if not matches:
        raise FileNotFoundError(f"No ckpts for epoch {ep} in {ckpt_dir}")
    matches.sort(key=_sort_key, reverse=True)
    chosen = matches[0]
    return str(chosen), _extract_int(chosen.name, _EPOCH_PAT)


__all__ = [
    # Kaggle
    "USE_CHANNELS_19", "CH_NAMES_KAGGLE_ERN",
    "read_csv_epochs", "read_kaggle_ern_train", "read_kaggle_ern_test",
    # ckpts
    "get_checkpoint_path",
]
