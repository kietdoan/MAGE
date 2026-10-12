# peek_data.py
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any

import torch

from configs import (
    seed_torch,
    get_dataloaders,
    NUM_WORKERS,
)

def _extract_tensor(batch: Any) -> torch.Tensor:
    # Accept (x, ...), dicts, or a bare tensor
    if isinstance(batch, torch.Tensor):
        return batch
    if isinstance(batch, (tuple, list)) and len(batch) > 0:
        return batch[0]
    if isinstance(batch, dict):
        # pick the first tensor-like entry
        for v in batch.values():
            if isinstance(v, torch.Tensor):
                return v
    raise TypeError(f"Don't know how to extract tensor from batch type: {type(batch)}")
def describe_tensor(x: torch.Tensor, input_mode: str) -> None:
    # x: raw [B,C,T] or pretoken [B,C,N,D]
    x = x.detach().to(torch.float32)
    shape = tuple(x.shape)
    dtype = str(x.dtype)
    device = str(x.device)

    # overall stats (NaN-robust, works on old torch)
    finite = torch.isfinite(x)
    n_total = x.numel()
    n = finite.sum().item()
    nan_rate = 1.0 - (n / max(1, n_total))

    x_masked = torch.where(finite, x, torch.zeros_like(x))
    s = x_masked.sum()
    ss = (x_masked * x_masked).sum()
    mean = (s / max(1, n)) if n > 0 else torch.tensor(float("nan"))
    var = (ss / max(1, n)) - mean * mean
    std = torch.sqrt(torch.clamp(var, min=0.0))

    # min/max/quantiles (guard if all NaN)
    if n > 0:
        vmin = x[finite].min()
        vmax = x[finite].max()
        q = torch.quantile(
            x[finite],
            torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=x.device),
        )
    else:
        vmin = vmax = torch.tensor(float("nan"), device=x.device)
        q = torch.tensor([float("nan")] * 5, device=x.device)

    print(f"\n[peek] tensor shape={shape} dtype={dtype} device={device}")
    print(
        f"[peek] overall: mean={mean.item():.6g} std={std.item():.6g} "
        f"min={vmin.item():.6g} p25={q[1].item():.6g} med={q[2].item():.6g} "
        f"p75={q[3].item():.6g} max={vmax.item():.6g}  NaN_rate={nan_rate:.3%}"
    )

    # -------- per-channel stats (no torch.nanmean/nanstd needed) --------
    # collapse everything except channel dim=1
    reduce_dims = (0, 2) if x.ndim == 3 else (0, 2, 3)  # raw=[B,C,T], pretoken=[B,C,N,D]
    finite_ch = torch.isfinite(x)
    cnt_ch = finite_ch.sum(dim=reduce_dims)                          # [C]
    x_masked_ch = torch.where(finite_ch, x, torch.zeros_like(x))
    sum_ch = x_masked_ch.sum(dim=reduce_dims)                        # [C]
    mean_ch = sum_ch / cnt_ch.clamp(min=1)                           # [C]
    sumsq_ch = (x_masked_ch * x_masked_ch).sum(dim=reduce_dims)      # [C]
    var_ch = (sumsq_ch / cnt_ch.clamp(min=1)) - (mean_ch * mean_ch)  # [C]
    std_ch = torch.sqrt(torch.clamp(var_ch, min=0.0))                # [C]

    # Print a few channels
    C = mean_ch.numel()
    show = min(10, C)
    print(f"[peek] per-channel mean/std (first {show}/{C}):")
    for i in range(show):
        m_i = mean_ch[i].item() if cnt_ch[i].item() > 0 else float("nan")
        s_i = std_ch[i].item()  if cnt_ch[i].item() > 0 else float("nan")
        print(f"  C[{i:02d}]: mean={m_i:.6g} std={s_i:.6g}")


def main():
    ap = argparse.ArgumentParser("Quick LMDB data peek (train only)")
    ap.add_argument("--train_lmdb", type=str, required=True, help="Path to the training LMDB directory")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    ap.add_argument("--input_mode", choices=["raw", "pretoken"], default="raw")
    ap.add_argument("--pretoken_layout", choices=["CND", "NCD"], default="CND")
    ap.add_argument("--max_batches", type=int, default=1, help="how many train batches to load")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    lmdb_path = Path(args.train_lmdb)
    if not lmdb_path.exists():
        raise FileNotFoundError(f"LMDB path not found: {lmdb_path}")

    seed_torch(args.seed)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("medium")

    # Build ONLY the training loader from LMDB
    train_loader, _ = get_dataloaders(
        train_root=None,
        valid_root=None,
        extensions=[],                 # not needed for LMDB
        global_bs=args.batch_size,
        num_workers=args.num_workers,
        devs=[],                       # CPU is fine for peeking
        input_mode=args.input_mode,
        pretoken_layout=args.pretoken_layout,
        data_backend="lmdb",           # force LMDB
        train_lmdb=lmdb_path,
        valid_lmdb=None,
        val_fraction=0.0,
        seed=args.seed,
    )

    if train_loader is None:
        raise RuntimeError("No train loader returned. Check --train_lmdb path and configs.get_dataloaders.")

    loaded = 0
    total_samples = 0
    for batch in train_loader:
        x = _extract_tensor(batch)
        describe_tensor(x, args.input_mode)
        total_samples += x.shape[0]
        loaded += 1
        if loaded >= args.max_batches:
            break

    print(f"\n[done] inspected {loaded} batch(es), ~{total_samples} sample(s).")

if __name__ == "__main__":
    main()

