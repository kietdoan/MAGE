from __future__ import annotations

import os
import random
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# --------------------------- to_tensor helper --------------------------- #
def to_tensor(x, *, dtype: Optional[torch.dtype] = None,
              device: Optional[torch.device] = None, copy: bool = False) -> torch.Tensor:
    """
    Robust numpy -> torch converter.
    - Keeps numpy arrays contiguous (copies only if needed or copy=True).
    - Preserves dtype unless `dtype` is specified.
    - Moves to `device` if given.
    """
    if isinstance(x, torch.Tensor):
        if dtype is not None and x.dtype is not dtype:
            x = x.to(dtype)
        if device is not None:
            x = x.to(device)
        return x

    arr = np.asarray(x)
    if (copy or not arr.flags.c_contiguous):
        arr = np.ascontiguousarray(arr)

    t = torch.from_numpy(arr)
    if dtype is not None and t.dtype is not dtype:
        t = t.to(dtype)
    if device is not None:
        t = t.to(device)
    return t

# ============================ RNG & DEVICES ============================

def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU/CUDA) deterministically."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_devices(devices_arg) -> List[int]:
    """
    Accepts '0,1', '0', 0, [0,1], [], None -> returns list of GPU indices (or []).
    """
    if devices_arg is None:
        return [0] if torch.cuda.is_available() else []
    if isinstance(devices_arg, (list, tuple)):
        return [int(d) for d in devices_arg]
    if isinstance(devices_arg, int):
        return [devices_arg]
    if isinstance(devices_arg, str):
        s = devices_arg.strip()
        if s == "":
            return [0] if torch.cuda.is_available() else []
        return [int(x) for x in s.split(",") if x.strip()]
    return [0] if torch.cuda.is_available() else []


def default_num_workers(devices: List[int]) -> int:
    """Heuristic: min(8 * max(1, len(devices)), 32, max(0, cpu-1))."""
    try:
        cpu = os.cpu_count() or 8
    except Exception:
        cpu = 8
    return min(8 * max(1, len(devices)), 32, max(0, cpu - 1))


def select_devices(num_device: int, gpus: Optional[Sequence[int]] = None) -> List[torch.device]:
    """
    Pick `num_device` CUDA devices from the provided list (or all).
    Only selects cards with >= ~12.7GB total memory by default.
    """
    if gpus is None:
        gpus = list(range(torch.cuda.device_count()))
    res: List[torch.device] = []
    last = 0
    min_total_mem = 25447170048 // 2  # ≈12.7GB; tweak as needed
    for _ in range(num_device):
        if not gpus:
            break
        dev = gpus[last % len(gpus)]
        last += 1
        tries = 0
        while torch.cuda.get_device_properties(dev).total_memory < min_total_mem and tries < len(gpus):
            dev = gpus[last % len(gpus)]
            last += 1
            tries += 1
        res.append(torch.device(f"cuda:{dev}"))
    return res


def select_free_gpu() -> Optional[torch.device]:
    """
    Randomly sample a GPU and return the first with >= ~12.7GB total memory.
    (Total memory is a coarse proxy; adjust to your environment.)
    """
    count = torch.cuda.device_count()
    if count == 0:
        return None
    min_total_mem = 25447170048 // 2
    for _ in range(count):
        i = random.randint(0, count - 1)
        props = torch.cuda.get_device_properties(i)
        if props.total_memory >= min_total_mem:
            return torch.device(f"cuda:{i}")
    return torch.device("cuda:0") if count > 0 else None


# ============================= DATASET ==============================

class eeg_dataset(Dataset):
    """Minimal dataset wrapper for (x, y[, subject_id]) tensors."""
    def __init__(self, feature: torch.Tensor, label: torch.Tensor,
                 subject_id: Optional[torch.Tensor] = None):
        super().__init__()
        self.x = feature
        self.y = label
        self.s = subject_id

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]

    # Few-shot helpers (kept for backward-compat)
    def get_num_class(self, num_class: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        buckets = [[] for _ in num_class]
        remaining = list(num_class)
        idxs = list(range(len(self.y)))
        import random as _rnd
        while sum(remaining) > 0:
            i = _rnd.choice(idxs)
            lab = int(self.y[i])
            if remaining[lab] > 0:
                remaining[lab] -= 1
                buckets[lab].append((self.x[i], self.y[i]))
        flat = [p for b in buckets for p in b]
        xs = torch.stack([t[0] for t in flat], dim=0)
        ys = torch.stack([t[1] for t in flat], dim=0)
        return xs, ys

    def get_num_subject(self, num_subject: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.s is not None, "Subject ids not provided to dataset."
        buckets = [[] for _ in num_subject]
        remaining = list(num_subject)
        idxs = list(range(len(self.y)))
        import random as _rnd
        while sum(remaining) > 0:
            i = _rnd.choice(idxs)
            sid = int(self.s[i])
            if remaining[sid] > 0:
                remaining[sid] -= 1
                buckets[sid].append((self.x[i], self.y[i]))
        flat = [p for b in buckets for p in b]
        xs = torch.stack([t[0] for t in flat], dim=0)
        ys = torch.stack([t[1] for t in flat], dim=0)
        return xs, ys


# =================== Compatibility: get_checkpoint_path ===================

def get_checkpoint_path(base_dir: str, which: str | int = "last",
                        metric: str = "val_loss", mode: str = "min"):
    """
    Compatibility wrapper so existing code can still:
        from utils import get_checkpoint_path
    while the actual implementation lives in io_utils.
    """
    from io_utils import get_checkpoint_path as _gcp  # local import avoids circulars
    return _gcp(base_dir, which=which, metric=metric, mode=mode)




__all__ = [
    "to_tensor",
    "set_seed", "parse_devices", "default_num_workers", "select_devices", "select_free_gpu",
    "eeg_dataset",
    "get_checkpoint_path",
]
