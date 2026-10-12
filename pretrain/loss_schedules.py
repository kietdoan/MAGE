# ---- loss_schedules.py ----
from typing import Dict

# Static per-size anchors
_BASES = {
    "nano":   {"lambda_recon": 10.0, "lambda_align_vis_start": 1.00, "lambda_align_vis_end": 0.20, "lambda_align_mask": 1.00, "vis_warmup_frac": 0.0},
    "tiny":   {"lambda_recon": 10.0, "lambda_align_vis_start": 0.80, "lambda_align_vis_end": 0.10, "lambda_align_mask": 1.20, "vis_warmup_frac": 0.0},
    "little": {"lambda_recon":  8.0, "lambda_align_vis_start": 0.50, "lambda_align_vis_end": 0.05, "lambda_align_mask": 1.50, "vis_warmup_frac": 0.0},
    "base":   {"lambda_recon":  8.0, "lambda_align_vis_start": 0.25, "lambda_align_vis_end": 0.00, "lambda_align_mask": 2.00, "vis_warmup_frac": 0.0},
    "large":  {"lambda_recon":  6.0, "lambda_align_vis_start": 0.10, "lambda_align_vis_end": 0.00, "lambda_align_mask": 2.50, "vis_warmup_frac": 0.05},
}

def _cosine_decay(x: float) -> float:
    # x in [0,1] -> cosine from 1 to 0
    import math
    x = max(0.0, min(1.0, x))
    return 0.5 * (1.0 + math.cos(math.pi * x))

def get_loss_weights(tag: str, global_step: int, total_steps: int) -> Dict[str, float]:
    """
    Returns scheduled loss weights for the given model size and training step.
    - Visible alignment: cosine decay from start -> end (optional short warmup for 'large').
    - Masked alignment: slightly anti-correlated with visible (small boost when vis decays).
    - Reconstruction: constant per size (can combine with your LR schedule).
    """
    if tag not in _BASES:
        raise KeyError(f"Unknown tag '{tag}'. Available: {list(_BASES.keys())}")

    b = _BASES[tag]
    T = max(1, total_steps)
    s = max(0, min(global_step, T))

    warm_frac = b["vis_warmup_frac"]
    warm_steps = int(T * warm_frac)

    if s <= warm_steps:
        vis_w = b["lambda_align_vis_start"]
        decay_progress = 0.0
    else:
        decay_progress = (s - warm_steps) / max(1, T - warm_steps)
        cos = _cosine_decay(decay_progress)  # 1 -> 0 over training
        vis_w = b["lambda_align_vis_end"] + (b["lambda_align_vis_start"] - b["lambda_align_vis_end"]) * cos

    mask_base = b["lambda_align_mask"]
    mask_boost = 0.20 * mask_base * (1.0 - _cosine_decay(decay_progress))
    mask_w = mask_base + mask_boost

    return {
        "lambda_recon": b["lambda_recon"],
        "lambda_align_vis": float(vis_w),
        "lambda_align_mask": float(mask_w),
    }
