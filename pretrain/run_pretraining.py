from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import pytorch_lightning as pl
from pytorch_lightning import callbacks as pl_cb, loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

try:
    from argparse import BooleanOptionalAction
except Exception:
    BooleanOptionalAction = None

# Env hygiene
os.environ.setdefault("OMP_NUM_THREADS", "1")
torch.multiprocessing.set_sharing_strategy("file_system")

# Project imports
from configs import (
    seed_torch,
    tag as CFG_TAG,
    pretrain_max_lr as CFG_LR,
    global_batch_size as CFG_BS,
    devices as CFG_DEVICES,
    RESUME as CFG_RESUME,
    version as CFG_VERSION,
    PIN_LOG_VERSION as CFG_PIN,
    variant, preprocess, encoder,
    get_dataloaders, TRAIN_ROOT, VALID_ROOT, FILE_EXTENSIONS, NUM_WORKERS,
    max_epochs as CFG_MAX_EPOCHS, steps_per_epoch_from_loader,
    mask_ratio as CFG_MASK_RATIO,
    USE_LOSS_R as CFG_USE_LOSS_R,
    USE_LOSS_AM as CFG_USE_LOSS_AM,
    USE_LOSS_AV as CFG_USE_LOSS_AV,
    USE_LN as CFG_USE_LN,
    LAMBDA_RECON, LAMBDA_ALIGN_VIS, LAMBDA_ALIGN_MASK,
)

from engine_pretraining import LitMAGE
from utils import get_checkpoint_path


# ---------------- CLI ---------------- #

def _add_bool_flag(p: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    if BooleanOptionalAction is not None:
        p.add_argument(f"--{name}", action=BooleanOptionalAction, default=default, help=help_text)
    else:
        p.add_argument(f"--{name}", dest=name, action="store_true", default=default, help=help_text)
        p.add_argument(f"--no_{name}", dest=name, action="store_false", help=f"Disable: {help_text}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("MAGE pretraining")

    # Training/model
    p.add_argument("--tag", default=CFG_TAG, help="Model size tag (nano/tiny/little/base/large)")
    p.add_argument("--pretrain_max_lr", type=float, default=CFG_LR)
    p.add_argument("--batch_size", type=int, default=CFG_BS)
    p.add_argument("--max_epochs", type=int, default=CFG_MAX_EPOCHS)
    p.add_argument("--devices", type=str, default=",".join(map(str, CFG_DEVICES)),
                   help="Comma-separated GPU ids, e.g. '0' or '0,1'. Empty -> CPU.")
    _add_bool_flag(p, "resume", CFG_RESUME, "Resume from last checkpoint if available.")
    p.add_argument("--ckpt_path", type=str, default=None, help="Explicit path to a checkpoint file to resume from.")
    p.add_argument("--version", type=int, default=CFG_VERSION)
    p.add_argument("--pin_log_version", type=int, default=CFG_PIN,
                   help="Pinned version (0,1,2...). Use -1 to auto-create with timestamp.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--log_root", type=str, default="./logs")

    # Data roots
    p.add_argument("--train_root", type=str, default=str(TRAIN_ROOT))
    p.add_argument("--valid_root", type=str, default=str(VALID_ROOT))
    p.add_argument("--exts", type=str, default=",".join(FILE_EXTENSIONS))
    p.add_argument("--num_workers", type=int, default=NUM_WORKERS)

    # Multi-scale configuration
    p.add_argument("--time_length", type=int, default=6000, help="Total time length of input signal")
    p.add_argument("--num_channels", type=int, default=19, help="Number of EEG channels")
    p.add_argument("--patch_sizes", type=str, default="50,100,200", 
                   help="Comma-separated patch sizes for multi-scale (e.g., '50,100,200')")
    p.add_argument("--montage", choices=["", "58", "19"], default="19", help="Channel montage")
    
    # Model architecture
    p.add_argument("--d_model", type=int, default=256, help="Model hidden dimension")
    p.add_argument("--encoder_depth", type=int, default=6, help="Number of encoder layers")
    p.add_argument("--predictor_depth", type=int, default=3, help="Number of predictor layers")
    p.add_argument("--encoder_heads", type=int, default=8, help="Number of attention heads in encoder")
    p.add_argument("--predictor_heads", type=int, default=4, help="Number of attention heads in predictor")
    
    # Patch embedding config
    p.add_argument("--k_temporal", type=int, default=3, help="Temporal kernel size for patch embedding")
    p.add_argument("--temporal_extra_convs", type=int, default=2, help="Extra conv layers in patch embedding")
    _add_bool_flag(p, "use_spectral", True, "Use spectral branch in patch embedding")
    
    # Aggregator config
    p.add_argument("--aggregator_hidden_dim", type=int, default=128, help="Hidden dim for aggregator")
    p.add_argument("--aggregator_freq_dim", type=int, default=64, help="Frequency feature dim for aggregator")
    _add_bool_flag(p, "use_frequency", True, "Use frequency domain features in aggregator")

    # [NEW] Aggregator Regularization / Noise
    p.add_argument("--agg_noise_std", type=float, default=0.0, help="Std dev of noise added to aggregator logits")
    p.add_argument("--scale_dropout", type=float, default=0.0, help="Probability of dropping a scale")
    p.add_argument("--entropy_lambda", type=float, default=0.0, help="Coefficient for entropy regularization")
    
    # Decoder config
    p.add_argument("--decoder_hidden_ratio", type=float, default=2.0, help="Hidden ratio for decoder")
    
    # Attention config
    p.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    _add_bool_flag(p, "output_attention", False, "Output attention weights")
    _add_bool_flag(p, "no_inter_attn_enc", False, "Disable inter-scale attention in encoder")
    _add_bool_flag(p, "no_inter_attn_pred", True, "Disable inter-scale attention in predictor")
    _add_bool_flag(p, "use_channel_attn", True, "Use channel attention")
    _add_bool_flag(p, "use_rope_time", True, "Use RoPE for time dimension")
    
    # Masking
    p.add_argument("--mask_ratio", type=float, default=CFG_MASK_RATIO, 
                   help="Hierarchical mask ratio (applied at coarsest scale)")

    # Channels override
    p.add_argument("--channels", type=str, default="", help="Comma-separated channel list.")

    # Feature switches (loss components)
    _add_bool_flag(p, "use_loss_r", CFG_USE_LOSS_R, "Enable reconstruction loss.")
    _add_bool_flag(p, "use_align_vis", CFG_USE_LOSS_AV, "Enable visible-token alignment.")
    _add_bool_flag(p, "use_align_mask", CFG_USE_LOSS_AM, "Enable masked-token alignment.")
    _add_bool_flag(p, "use_ln", CFG_USE_LN, "LayerNorm before alignment.")

    # Backend / LMDB
    p.add_argument("--data_backend", choices=["files", "lmdb"], default="files")
    p.add_argument("--train_lmdb", type=str, default=None)
    p.add_argument("--valid_lmdb", type=str, default=None)

    # Validation
    p.add_argument("--no_valid", action="store_true", help="Disable validation.")
    p.add_argument("--val_fraction", type=float, default=0.0,
                   help="If >0 and no valid set given, split this fraction from train.")

    # Loss weights
    p.add_argument("--lambda_recon", type=float, default=LAMBDA_RECON)
    p.add_argument("--lambda_align_vis", type=float, default=LAMBDA_ALIGN_VIS)
    p.add_argument("--lambda_align_mask", type=float, default=LAMBDA_ALIGN_MASK)

    # Loss weighting scheme
    p.add_argument(
        "--loss_schedule",
        choices=["fixed", "scale_cosine"],
        default="fixed",
        help="fixed = use provided lambda_* as-is; "
             "scale_cosine = size-aware schedule w/ cosine decay of visible alignment."
    )

    return p


# ---------------- Main ---------------- #

def main() -> None:
    args = build_parser().parse_args()
    print(f"[args] {args}")

    seed_torch(args.seed)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("medium")

    # Devices / accelerator
    devices: List[int] = [int(x) for x in args.devices.split(",") if x.strip() != ""] if args.devices else []
    accelerator = "gpu" if (torch.cuda.is_available() and len(devices) > 0) else "cpu"

    # Parse patch sizes
    patch_sizes = [int(p.strip()) for p in args.patch_sizes.split(",") if p.strip()]
    if not patch_sizes:
        raise ValueError("patch_sizes must not be empty")
    
    # Validate patch sizes divide time_length
    for p in patch_sizes:
        if args.time_length % p != 0:
            raise ValueError(f"time_length={args.time_length} must be divisible by patch_size={p}")
    
    print(f"[info] Multi-scale patch sizes: {patch_sizes}")

    # Dataloaders - use raw input mode for multi-scale
    exts = [e if e.startswith(".") else f".{e}" for e in args.exts.split(",") if e.strip()]
    train_loader, valid_loader = get_dataloaders(
        train_root=Path(args.train_root),
        valid_root=Path(args.valid_root),
        extensions=exts,
        global_bs=args.batch_size,
        num_workers=args.num_workers,
        devs=devices,
        input_mode="raw",  # Multi-scale always uses raw input
        pretoken_layout="CND",  # Not used for raw mode
        data_backend=args.data_backend,
        train_lmdb=Path(args.train_lmdb) if args.train_lmdb else None,
        valid_lmdb=None if args.no_valid else (Path(args.valid_lmdb) if args.valid_lmdb else None),
        val_fraction=(0.0 if args.no_valid else args.val_fraction),
        seed=args.seed,
        norm_fixed_scale=1.0,
    )

    print(f"Train loader: {len(train_loader)}, Val loader: {len(valid_loader)}")

    steps_per_epoch = steps_per_epoch_from_loader(train_loader)
    if steps_per_epoch <= 0:
        raise RuntimeError("Computed steps_per_epoch <= 0. Check dataset and batch size.")
    print(f"[info] steps_per_epoch = {steps_per_epoch}")

    # Verify data shape
    batch = next(iter(train_loader))
    x = batch[0] if isinstance(batch, (list, tuple)) else batch
    
    if x.ndim != 3:
        raise ValueError(f"Expected raw input [B, C, T], got shape {x.shape}")
    
    _, C_auto, T_auto = x.shape
    if C_auto != args.num_channels:
        print(f"[warning] Data has {C_auto} channels but expected {args.num_channels}")
    if T_auto != args.time_length:
        print(f"[warning] Data has {T_auto} time points but expected {args.time_length}")

    # --- Run name coding ---
    patch_str = "-".join(map(str, patch_sizes))
    mon_code = f"M{args.montage or 'XX'}"
    a_code = f"V{int(bool(args.use_align_vis))}M{int(bool(args.use_align_mask))}"
    
    flags_code = "".join([
        "R" if args.use_loss_r else "r",
        "-", a_code, "-",
        "N" if args.use_ln else "n",
        f"-MS-{mon_code}",
    ])

    var_str = variant if isinstance(variant, str) else str(variant)
    pre_str = preprocess if isinstance(preprocess, str) else str(preprocess)
    enc_str = encoder if isinstance(encoder, str) else str(encoder)
    
    # Append loss schedule and frequency feature flags
    loss_code = "FIX" if args.loss_schedule == "fixed" else "SCH"
    freq_code = "FREQ" if args.use_frequency else "TEMP"
    
    model_type = (f"MAGE_{args.tag}_{var_str}_{pre_str}_{enc_str}_"
                  f"P{patch_str}_{flags_code}_L{loss_code}_{freq_code}")

    # Loggers (pinned or timestamped)
    pin = None if args.pin_log_version in (-1, None) else args.pin_log_version
    csv_root = Path(args.log_root) / f"{args.pretrain_max_lr}" / f"{model_type}_csv"
    tb_root = Path(args.log_root) / f"{args.pretrain_max_lr}" / f"{model_type}_tb"

    if pin is None:
        csv_logger = pl_loggers.CSVLogger(save_dir=str(csv_root), name="", version=None)
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=str(tb_root), name="", version=None)
    else:
        csv_logger = pl_loggers.CSVLogger(save_dir=str(csv_root), name="", version=pin)
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=str(tb_root), name="", version=pin)

    # Checkpoints under TB run dir
    CKPT_DIR = Path(tb_logger.log_dir) / "checkpoints"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    monitor_metric = "valid_loss" if valid_loader is not None else "train_loss"
    best_name = "val" if valid_loader is not None else "train"

    ckpt_best = ModelCheckpoint(
        dirpath=str(CKPT_DIR),
        filename=f"best-{best_name}-" + "{epoch}-{step}-" + f"{monitor_metric}" + "={" + monitor_metric + ":.6f}",
        monitor=monitor_metric, mode="min", save_top_k=1, save_last=False,
    )
    ckpt_last = ModelCheckpoint(
        dirpath=str(CKPT_DIR),
        filename="last-{epoch}-{step}",
        save_top_k=1, every_n_epochs=1, save_on_train_epoch_end=True,
    )

    # Channels override
    channels_override: Optional[List[str]] = None
    if args.channels.strip():
        channels_override = [c.strip() for c in args.channels.split(",") if c.strip()]

    print(f"[model] {model_type}")
    print(f"[model] time_length={args.time_length}, num_channels={args.num_channels}")
    print(f"[model] patch_sizes={patch_sizes}, d_model={args.d_model}")
    print(f"[model] encoder_depth={args.encoder_depth}, predictor_depth={args.predictor_depth}")
    print(f"[model] use_frequency={args.use_frequency}, use_spectral={args.use_spectral}")

    # Create model
    model = LitMAGE(
        # Data config
        time_length=args.time_length,
        num_channels=args.num_channels,
        patch_sizes=patch_sizes,
        channels=channels_override,
        montage=args.montage if not channels_override else None,
        mask_ratio=args.mask_ratio,
        # Model config
        d_model=args.d_model,
        encoder_depth=args.encoder_depth,
        predictor_depth=args.predictor_depth,
        encoder_heads=args.encoder_heads,
        predictor_heads=args.predictor_heads,
        k_temporal=args.k_temporal,
        temporal_extra_convs=args.temporal_extra_convs,
        use_spectral=args.use_spectral,
        dropout=args.dropout,
        output_attention=args.output_attention,
        no_inter_attn_enc=args.no_inter_attn_enc,
        no_inter_attn_pred=args.no_inter_attn_pred,
        use_channel_attn=args.use_channel_attn,
        use_rope_time=args.use_rope_time,
        # Decoder config
        decoder_hidden_ratio=args.decoder_hidden_ratio,
        # Training schedules
        steps_per_epoch=steps_per_epoch,
        max_epochs=args.max_epochs,
        pretrain_max_lr=args.pretrain_max_lr,
        ref_wd=5e-2,
        final_wd=5e-2,
        ema_start_end=(0.996, 0.9995),
        # Loss config
        USE_LOSS_R=args.use_loss_r,
        USE_LOSS_A=bool(args.use_align_vis or args.use_align_mask),
        USE_ALIGN_VIS=args.use_align_vis,
        USE_ALIGN_MASK=args.use_align_mask,
        USE_LN=args.use_ln,
        lambda_recon=args.lambda_recon,
        lambda_align_vis=args.lambda_align_vis,
        lambda_align_mask=args.lambda_align_mask,
        tag_name=args.tag,
        loss_schedule=args.loss_schedule,
        # [NEW] Aggregator Regularization
        agg_noise_std=args.agg_noise_std,
        scale_dropout=args.scale_dropout,
        entropy_lambda=args.entropy_lambda,
    )

    # Update aggregator config (can't be done in __init__ easily, so do it here)
    model.aggregator = type(model.aggregator)(
        time_length=args.time_length,
        num_channels=args.num_channels,
        patch_sizes=patch_sizes,
        hidden_dim=args.aggregator_hidden_dim,
        use_frequency=args.use_frequency,
        freq_dim=args.aggregator_freq_dim,
        act="gelu",
    ).to(model.device)
    ckpt_path_for_fit: Optional[str] = None
    
    # 1. Prioritize explicit checkpoint path if provided
    if args.ckpt_path:
        ckpt_path_for_fit = args.ckpt_path
        print(f"[resume] Explicitly resuming from {ckpt_path_for_fit}")
        
    # 2. Fall back to auto-resume logic if --resume flag is passed
    elif args.resume:
        try:
            ckpt_path, ep = get_checkpoint_path(tb_logger.log_dir, which="last")
        except Exception as e:
            print(f"[resume] no checkpoint found in {CKPT_DIR}: {e}")
        else:
            ckpt_path_for_fit = ckpt_path
            print(f"[resume] from {ckpt_path} at epoch {ep}")

    # Strategy
    strategy = "ddp_find_unused_parameters_true" if (accelerator == "gpu" and len(devices) > 1) else "auto"

    # Trainer
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=(devices if accelerator == "gpu" else None),
        strategy=strategy,
        max_epochs=args.max_epochs,
        callbacks=[pl_cb.LearningRateMonitor(logging_interval="epoch"), ckpt_best, ckpt_last],
        logger=[tb_logger, csv_logger],
        num_sanity_val_steps=0,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
    )

    # Train
    t0 = time.time()
    if valid_loader is None:
        trainer.fit(model, train_loader, ckpt_path=ckpt_path_for_fit)
    else:
        trainer.fit(model, train_loader, valid_loader, ckpt_path=ckpt_path_for_fit)
    elapsed_hours = (time.time() - t0) / 3600.0

    # Log hparams snapshot
    hparams = dict(vars(args))
    hparams.update({
        "time_length": args.time_length,
        "num_channels": args.num_channels,
        "patch_sizes": patch_sizes,
        "steps_per_epoch": steps_per_epoch,
        "devices": devices,
        "accelerator": accelerator,
        "strategy": strategy,
        "channels_override": channels_override,
        "model_type": model_type,
        "ckpt_dir": str(CKPT_DIR),
    })
    tb_logger.log_hyperparams(hparams)

    # Wall time
    time_file = Path(csv_logger.log_dir) / "pretraining_time.txt"
    with open(time_file, "w") as f:
        f.write(f"{elapsed_hours:.4f} hours\n")

    print(f"[logs] TB:  {tb_logger.log_dir}")
    print(f"[logs] CSV: {csv_logger.log_dir}")
    print(f"[ckpts]    {CKPT_DIR}")
    print(f"[time]     {elapsed_hours:.4f} h -> {time_file}")


if __name__ == "__main__":
    main()