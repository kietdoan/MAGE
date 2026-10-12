from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
import math
from sklearn.model_selection import KFold
import importlib

# -----------------------------------------------------------------------------
# Config + local imports
# -----------------------------------------------------------------------------
try:
    CFG = importlib.import_module("finetune_configs")
except ImportError:
    # Fallback generic config if file missing
    CFG = type("CFG", (), {
        "SEED": 42, "BATCH_SIZE": 32, "MAX_EPOCHS": 50,
        "MAX_LR_HEAD": 1e-4, "MAX_LR_ENC": 1e-4, "WD_HEAD": 0.0, "WD_ENC": 0.0,
        "SCHEDULER": "cosine", "WARMUP_PCT": 0.1, "devices": [0],
        "USE_DYNAMIC_PE": False, "ROPE_TIME_DEFAULT": False, "ROPE_CHAN_DEFAULT": False,
        "ROPE_DIM_FRAC_DEFAULT": 0.5, "ROPE_INTERP_TIME_DEFAULT": 1.0, "ROPE_INTERP_CHAN_DEFAULT": 1.0,
        "IN_CHANS": 22, "TARGET_CHANS": 19, "SEQ_LEN": 1024, "PATCH": 64,
        "DROPOUT": 0.5, "HEAD_TYPE": "avg", "POOL": "mean",
    })

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import default_num_workers, get_checkpoint_path, parse_devices  
from channels import CHANNELS_BY_DATASET, DEFAULT_EEGPT58, MONTAGE_BY_DATASET  

from downstream.modules.mage_ft import LitMAGEDownstream

try:
    from argparse import BooleanOptionalAction
except Exception:
    BooleanOptionalAction = None


# =============================================================================
# Utilities
# =============================================================================

def _add_bool_flag(p: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    if BooleanOptionalAction is not None:
        p.add_argument(f"--{name}", action=BooleanOptionalAction, default=default, help=help_text)
    else:
        p.add_argument(f"--{name}", dest=name, action="store_true", default=default, help=help_text)
        p.add_argument(f"--no-{name}", dest=name, action="store_false", help=f"Disable: {help_text}")


def write_csv(path: str, rows: List[dict]) -> None:
    if not rows: return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    file_exists = out.exists()
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with out.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists: w.writeheader()
        w.writerows(rows)
    print("[SUMMARY] wrote:", str(out))


def _flatten(prefix: str, d: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict): out.update(_flatten(f"{key}.", v))
        else: out[key] = v
    return out


def _save_encoder_config_json(base_dir: Path, payload: Dict[str, Any]) -> None:
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / "encoder_config.json"
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print("[CONFIG] wrote:", str(path))


def _log_encoder_hparams_to_tb(loggers, hparams: Dict[str, Any]) -> None:
    for lg in (loggers or []):
        try:
            if isinstance(lg, pl_loggers.TensorBoardLogger):
                lg.log_hyperparams(hparams)
                break
        except Exception: pass

def _family_from_scheme(s: str) -> str:
    return (s or "mage").upper()

def resolve_pretrain_checkpoint(args) -> Tuple[Optional[str], str]:
    from pathlib import Path
    import re
    family = _family_from_scheme(getattr(args, "scheme", ""))

    if not getattr(args, "load_pretrained", True):
        run_tag = f"{family}_{args.tag}_{args.variant}_{args.preprocess}_{args.encoder}_{args.transformer}_randinit"
        return None, run_tag

    if getattr(args, "pretrain_ckpt", None):
        ck = Path(str(args.pretrain_ckpt))
        if not ck.is_file(): raise FileNotFoundError(f"--pretrain_ckpt not found: {ck}")
        try: run_tag = ck.parent.parent.parent.name.replace("_tb", "")
        except Exception: run_tag = f"{family}_custom"
        return str(ck), run_tag

    if getattr(args, "pretrain_dir", None):
        ver_dir = Path(str(args.pretrain_dir))
        if not ver_dir.is_dir(): raise FileNotFoundError(f"--pretrain_dir is not a directory: {ver_dir}")
        run_tag = ver_dir.parent.name.replace("_tb", "")
        ckpt_path, _ = get_checkpoint_path(str(ver_dir), which="best" if args.epoch_select == "best" else "last")
        m = re.search(r"(?:^|/)version_(\d+)(?:/|$)", str(ver_dir))
        if m: args._pretrain_version = int(m.group(1))
        return ckpt_path, run_tag

    raise ValueError("Checkpoint discovery is disabled. Pass --pretrain_ckpt or --pretrain_dir, or run with --no-load_pretrained.")

def downstream_log_base(dataset: str, args, model_type: str, extra: str = "") -> Path:
    ver = getattr(args, "_pretrain_version", args.version)
    head_name = args.head_type
    use_val = float(getattr(args, "val_ratio", 0.0) or 0.0) > 0.0
    family = _family_from_scheme(args.scheme)

    # Append active head info to folder path if a specific head was selected.
    head_tag = ""
    ahi = int(getattr(args, "active_head_index", -1))
    if ahi != -1:
        head_tag = f"Head{ahi}"

    if not getattr(args, "load_pretrained", True):
        run_tag = f"{family}_{args.tag}_{args.variant}_{args.preprocess}_{args.encoder}_{args.transformer}_randinit"
        parts = [run_tag, dataset, "randinit", "val" if use_val else "noval", extra or None, head_tag or None]
        parts = [p for p in parts if p]
        return Path(args.log_root).joinpath(*parts)

    topk_tag = None
    if not getattr(args, "freeze_encoder", False):
        k = int(getattr(args, "ft_top_k", 0) or 0)
        if k > 0: topk_tag = f"topk{k}"

    parts = [
        f"{args.epoch_select}_v{ver}_{args.pretrain_max_lr}_{model_type}_{head_name}",
        dataset, "pretrained", "frozen" if getattr(args, "freeze_encoder", False) else None,
        topk_tag, "val" if use_val else "noval", extra or None, head_tag or None,
    ]
    parts = [p for p in parts if p]
    return Path(args.log_root).joinpath(*parts)


def make_loggers(base_dir: Path, version: str, *, enable_csv: bool = True):
    base_dir = Path(base_dir)
    tb = pl_loggers.TensorBoardLogger(save_dir=str(base_dir), name="tb", version=version, default_hp_metric=False)
    if enable_csv:
        csv = pl_loggers.CSVLogger(save_dir=str(base_dir), name="csv", version=version)
        return [tb, csv]
    return [tb]


def build_trainer_with_callbacks(
    devices: List[int], max_epochs: int, loggers, precision: str = "32",
    monitor_metric: Optional[str] = "valid_cohen_kappa", ckpt_dir: Optional[Path] = None,
) -> Tuple[pl.Trainer, ModelCheckpoint]:

    if monitor_metric:
        ckpt_kwargs: Dict[str, Any] = dict(save_top_k=1, monitor=monitor_metric, mode="max", filename=f"best-{{epoch:02d}}-{{{monitor_metric}:.4f}}", save_last=True)
    else:
        ckpt_kwargs = dict(save_top_k=0, save_last=True, filename="last")

    if ckpt_dir is not None: ckpt_kwargs["dirpath"] = str(ckpt_dir)
    ckpt_cb = ModelCheckpoint(**ckpt_kwargs)
    
    has_devices = devices is not None and len(devices) > 0 if isinstance(devices, list) else bool(devices)
    cuda_available = torch.cuda.is_available()
    
    if has_devices and cuda_available:
        accelerator = "gpu"
        devices_arg = devices if isinstance(devices, list) else [devices]
        strategy = "auto" if len(devices_arg) == 1 else "ddp_find_unused_parameters_true"
    else:
        accelerator = "cpu"
        devices_arg = 1
        strategy = "auto"

    trainer = pl.Trainer(
        accelerator=accelerator, devices=devices_arg, precision=precision, max_epochs=max_epochs,
        callbacks=[ckpt_cb], enable_checkpointing=True, logger=loggers, strategy=strategy, enable_progress_bar=True,
    )
    return trainer, ckpt_cb

# ----------------- Encoder-config capture helpers -----------------

def _collect_encoder_build_flags(args, scheme: str) -> Dict[str, Any]:
    flags: Dict[str, Any] = dict(
        scheme=scheme, transformer=args.transformer,
        rope_time=getattr(args, "rope_time", None),
        rope_chan=getattr(args, "rope_chan", None),
        rope_dim_frac=getattr(args, "rope_dim_frac", None),
        rope_interp_time=getattr(args, "rope_interp_time", None),
        rope_interp_chan=getattr(args, "rope_interp_chan", None),
        use_dynamic_pe=getattr(args, "use_dynamic_pe", None),
        load_pretrained=bool(getattr(args, "load_pretrained", True)),
        init_scheme=str(getattr(args, "init_scheme", "")) if not getattr(args, "load_pretrained", True) else "",
        freeze_encoder=bool(getattr(args, "freeze_encoder", False)),
        ft_top_k=int(getattr(args, "ft_top_k", 0) or 0),
        active_head_index=getattr(args, "active_head_index", -1),
    )
    if scheme == "mage":
        flags.update({
            "mage_patch_sizes": getattr(args, "mage_patch_sizes", None),
            "mage_active_scale": getattr(args, "mage_active_scale", None),
            "scale_dropout": getattr(args, "scale_dropout", 0.0),
            "agg_noise_std": getattr(args, "agg_noise_std", 0.0),
        })
    return {k: v for k, v in flags.items() if v is not None}


def _stash_encoder_logging_payload(args, models_cfg: Dict[str, Any], scheme: str) -> None:
    enc_cfg = models_cfg.get("encoder", {}) if isinstance(models_cfg, dict) else {}
    flags = _collect_encoder_build_flags(args, scheme)
    setattr(args, "_encoder_cfg_for_log", enc_cfg)
    setattr(args, "_encoder_flags_for_log", flags)


def _encoder_csv_fields(args) -> Dict[str, Any]:
    enc = getattr(args, "_encoder_cfg_for_log", {}) or {}
    flg = getattr(args, "_encoder_flags_for_log", {}) or {}
    csv_bits: Dict[str, Any] = {}
    for key in ("embed_dim", "embed_num", "depth", "num_heads", "use_criss_cross"):
        if key in enc: csv_bits[f"enc_{key}"] = enc[key]
    for key in ("rope_time", "rope_chan", "rope_dim_frac", "rope_interp_time", "rope_interp_chan", "use_dynamic_pe"):
        if key in flg: csv_bits[f"enc_{key}"] = flg[key]
    for key in ("load_pretrained", "init_scheme", "freeze_encoder", "ft_top_k", "active_head_index"):
        if key in flg: csv_bits[key] = flg[key]
    return csv_bits


def _log_encoder_everywhere(base_dir: Path, loggers, args) -> None:
    enc = getattr(args, "_encoder_cfg_for_log", {}) or {}
    flg = getattr(args, "_encoder_flags_for_log", {}) or {}
    payload = dict(
        encoder_config=enc, encoder_flags=flg, tag=getattr(args, "tag", None),
        variant=getattr(args, "variant", None), preprocess=getattr(args, "preprocess", None),
        encoder_name=getattr(args, "encoder", None), transformer_name=getattr(args, "transformer", None),
        pretrain_max_lr=getattr(args, "pretrain_max_lr", None),
        pretrain_version=getattr(args, "_pretrain_version", getattr(args, "version", None)),
    )
    _save_encoder_config_json(base_dir, payload)
    tb_hparams = {}
    tb_hparams.update(_flatten("enc.", enc))
    tb_hparams.update(_flatten("flag.", flg))
    _log_encoder_hparams_to_tb(loggers, tb_hparams)


def make_result_row(
    *, dataset: str, seed: int, args, test_metrics: dict, ckpt_cb: ModelCheckpoint, extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    use_names = getattr(args, "use_channel_names", None)
    model_target_chans = (len(use_names) if use_names else int(getattr(args, "target_chans", 0) or 0) or int(getattr(args, "in_chans", 0) or 0))
    montage_preview = ",".join(use_names[:10]) + ("..." if len(use_names) > 10 else "") if use_names else ""
    row: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "dataset": dataset, "seed": seed,
        "dataset_in_chans": args.in_chans, "model_target_chans": model_target_chans, "model_montage": montage_preview,
        "best_valid_cohen_kappa": (float(ckpt_cb.best_model_score.item()) if ckpt_cb.best_model_score else float("nan")),
        "test_balanced_accuracy": float(test_metrics.get("test_balanced_accuracy", float("nan"))),
        "test_accuracy": float(test_metrics.get("test_accuracy", test_metrics.get("test_acc", float("nan")))),
        "test_f1_macro": float(test_metrics.get("test_f1_macro", float("nan"))),
        "test_f1_weighted": float(test_metrics.get("test_f1_weighted", float("nan"))),
        "test_cohen_kappa": float(test_metrics.get("test_cohen_kappa", float("nan"))),
        "scheme": args.scheme, "tag": args.tag, "version_arg": args.version,
        "pretrain_version": getattr(args, "_pretrain_version", args.version),
        "frozen": int(bool(getattr(args, "freeze_encoder", False))),
        "ft_top_k": int(getattr(args, "ft_top_k", 0) or 0),
        "load_pretrained": int(bool(getattr(args, "load_pretrained", True))),
        "init_scheme": getattr(args, "init_scheme", ""),
        "active_head_index": getattr(args, "active_head_index", -1),
    }
    row.update(_encoder_csv_fields(args))
    # Surface any per-head metrics produced by --test_separate_heads.
    for k, v in test_metrics.items():
        if k.startswith("head") or k not in row:
            try: row[k] = float(v)
            except Exception: row[k] = str(v)
    if extra: row.update(extra)
    return row


# =============================================================================
# Split helpers
# =============================================================================

def _parse_id_list(s: Optional[str]) -> Optional[List[int]]:
    if not s: return None
    out: List[int] = []
    for tok in s.split(","):
        t = tok.strip()
        if not t: continue
        if "-" in t:
            a, b = t.split("=", 1)[0].split("-", 1) if "=" in t else t.split("-", 1)
            a, b = int(a), int(b)
            if a > b: a, b = b, a
            out.extend(range(a, b + 1))
        else: out.append(int(t))
    seen, final = set(), []
    for v in out:
        if v not in seen: final.append(v); seen.add(v)
    return final


def _parse_split_spec(spec: Optional[str]) -> Dict[str, Optional[List[int]]]:
    spec = (spec or os.getenv("EEGPT_SPLIT_SPEC", "")).strip()
    if not spec: return {"train": None, "val": None, "test": None}
    parts = [p.strip() for p in spec.split(";") if p.strip()]
    out: Dict[str, Optional[List[int]]] = {"train": None, "val": None, "test": None}
    for p in parts:
        if "=" not in p: continue
        k, v = p.split("=", 1)
        key = k.strip().lower()
        if key in out: out[key] = _parse_id_list(v.strip())
    return out


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser("Cleaned Downstream Evaluation Runner")

    # Task / scheme
    p.add_argument("--task", type=str, required=True,
                   choices=["BCIC2a", "PhysioMI", "TUAB", "SLEEPEDF", "KaggleERN", "FACED", "SHUMI"])
    p.add_argument("--scheme", type=str, default="mage", choices=["mage"],
                   help="MAGE downstream model")

    # Split controller
    p.add_argument("--split", type=str, default="loso",
                   choices=["loso", "fixed", "4fold", "5fold"],
                   help="For 'fixed', pass --train_subject_ids/--val_subject_ids/--test_subject_ids or --split_spec.")
    p.add_argument("--split_spec", type=str, default=None)
    p.add_argument("--train_subject_ids", type=str, default=None)
    p.add_argument("--val_subject_ids", type=str, default=None)
    p.add_argument("--test_subject_ids", type=str, default=None)

    # Generic overrides
    p.add_argument("--batch_size", type=int, default=CFG.BATCH_SIZE)
    p.add_argument("--epochs", type=int, default=CFG.MAX_EPOCHS)
    p.add_argument("--seed", type=int, default=CFG.SEED)
    p.add_argument("--fold", type=int, default=-1)

    # Optim & schedule
    p.add_argument("--max_lr_head", type=float, default=CFG.MAX_LR_HEAD)
    p.add_argument("--max_lr_enc", type=float, default=CFG.MAX_LR_ENC)
    p.add_argument("--wd_head", type=float, default=CFG.WD_HEAD)
    p.add_argument("--wd_enc", type=float, default=CFG.WD_ENC)
    p.add_argument("--scheduler", type=str, default=CFG.SCHEDULER, choices=["onecycle", "cosine", "none"])
    p.add_argument("--warmup_pct", type=float, default=CFG.WARMUP_PCT)

    # train policy
    p.add_argument("--freeze_encoder", action="store_true", default=False)
    p.add_argument("--ft_top_k", type=int, default=0)

    # Validation & dataset toggles
    p.add_argument("--val_ratio", type=float, default=0.0)
    _add_bool_flag(p, "stratified_val", True, "Use stratified split when building a val set from train.")
    p.add_argument("--scale_div", type=float, default=1.0)

    # Pretrain ckpt selection
    p.add_argument("--epoch_select", type=str, default=getattr(CFG, "epoch_select", "last"), choices=["best", "last"])
    p.add_argument("--version", type=int, default=getattr(CFG, "version", 0))
    p.add_argument("--pretrain_max_lr", type=float, default=getattr(CFG, "pretrain_max_lr", 5e-4))
    p.add_argument("--pretrain_dir", type=str, default=None)
    p.add_argument("--pretrain_ckpt", type=str, default=None)
    _add_bool_flag(p, "load_pretrained", True, "Load encoder weights")
    p.add_argument("--init_scheme", type=str, default="xavier_uniform")

    # Model fingerprint
    p.add_argument("--tag", type=str, default=CFG.tag)
    p.add_argument("--variant", type=str, default=CFG.variant)
    p.add_argument("--preprocess", type=str, default=CFG.preprocess)
    p.add_argument("--encoder", type=str, default=CFG.encoder)
    p.add_argument("--transformer", type=str, default=CFG.transformer)
    p.add_argument("--loss_switch", type=str, default=getattr(CFG, "loss_switch", "default"))

    # RoPE toggles (ADDED BACK for MAGE)
    _add_bool_flag(p, "use_dynamic_pe", CFG.USE_DYNAMIC_PE, "Enable dynamic positional encoding.")
    _add_bool_flag(p, "rope_time", CFG.ROPE_TIME_DEFAULT, "Enable RoPE along time axis.")
    _add_bool_flag(p, "rope_chan", CFG.ROPE_CHAN_DEFAULT, "Enable RoPE along channel axis.")
    p.add_argument("--rope_dim_frac", type=float, default=CFG.ROPE_DIM_FRAC_DEFAULT)
    p.add_argument("--rope_interp_time", type=float, default=CFG.ROPE_INTERP_TIME_DEFAULT)
    p.add_argument("--rope_interp_chan", type=float, default=CFG.ROPE_INTERP_CHAN_DEFAULT)

    # Frontend / head options
    p.add_argument("--dataset_channel_names", type=str, default=None)
    p.add_argument("--dataset_channel_set", type=str, default=None, choices=sorted(CHANNELS_BY_DATASET.keys()))
    p.add_argument("--model_channel_set", type=str, default=None, choices=sorted(MONTAGE_BY_DATASET.keys()))
    p.add_argument('--patch_list', type=int, nargs='+', default=[50, 100, 200])

    p.add_argument("--in_chans", type=int, default=getattr(CFG, "IN_CHANS", 22))
    p.add_argument("--target_chans", type=int, default=getattr(CFG, "TARGET_CHANS", 19))
    p.add_argument("--seq_len", type=int, default=getattr(CFG, "SEQ_LEN", 1024))
    p.add_argument("--patch", type=int, default=getattr(CFG, "PATCH", 64))
    p.add_argument("--dropout", type=float, default=getattr(CFG, "DROPOUT", 0.5))
    p.add_argument("--head_type", type=str, default=getattr(CFG, "HEAD_TYPE", "avg"),
                   choices=["avg", "per_time_attn", "flatten", "eegpt_probe", "all_patch_reps", "attention"])
    p.add_argument("--pool", type=str, default=getattr(CFG, "POOL", "mean"))
    p.add_argument("--attn_temperature", type=float, default=None)
    p.add_argument("--attn_project_kv", dest="attn_project_kv", action="store_true")
    p.add_argument("--no-attn_project_kv", dest="attn_project_kv", action="store_false")
    p.set_defaults(attn_project_kv=True)
    p.add_argument("--flatten_n_time", type=int, default=None)
    p.add_argument("--flatten_embed_slots", type=int, default=None)
    p.add_argument("--val_n_subjects", type=int, default=1)
    p.add_argument("--val_seed", type=int, default=CFG.SEED)

    # Data roots (Sanitized default paths)
    p.add_argument("--bcic2a_path", type=str, default="./data/BCIC_2a")
    p.add_argument("--physiomi_path", type=str, default="./data/PhysioMI/")
    p.add_argument("--kaggle_ern_path", type=str, default="./data/KaggleERN")
    p.add_argument("--sleepedf_path", type=str, default="./data/sleep_edf/")
    p.add_argument("--tuab_path", type=str, default="./data/TUAB")
    p.add_argument("--faced_path", type=str, default="./data/Faced/")
    p.add_argument("--shumi_path", type=str, default="./data/SHUMI/")
    p.add_argument("--log_root", type=str, default="./logs")


    _add_bool_flag(p, "use_chan_conv", True, "1x1 channel-mixing Conv1d before encoder.")
    _add_bool_flag(p, "use_time_chan_conv", False, "EEGNet-style temporal depthwise Conv2d before encoder.")
    p.add_argument("--chan_max_norm", type=float, default=1.0)
    p.add_argument("--conv_kernel_time", type=int, default=55)
    p.add_argument("--conv_dropout", type=float, default=0.8)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--class_weights", type=float, nargs='+', default=None)
    p.add_argument("--freeze_patch_embed", action="store_true", default=True)
    p.add_argument("--mage_patch_sizes", type=int, nargs='+', default=[200, 300, 600])
    p.add_argument("--mage_active_scale", type=int, default=None,
                   help="Train using one MAGE patch scale by its zero-based index in --mage_patch_sizes.")
    # MAGE scale selection (mirrors run_downstream_multi_scale_single_scale.py)
    p.add_argument("--active_head_index", type=int, default=-1,
                   help="Select a specific head/scale index (0, 1, 2...). Default -1 uses Ensemble/Aggregator.")
    p.add_argument("--test_separate_heads", action="store_true", default=False,
                   help="If True, runs evaluation per scale head (0, 1, 2) and logs separate metrics.")
    p.add_argument("--mage_k_temporal", type=int, default=3)
    p.add_argument("--mage_temporal_extra_convs", type=int, default=2)
    p.add_argument("--mage_use_spectral", action="store_true", default=True)
    p.add_argument("--mage_no_inter_attn_enc", action="store_true", default=False)
    p.add_argument("--mage_use_channel_attn", action="store_true", default=True)
    p.add_argument("--mage_aggregator_hidden_dim", type=int, default=128)
    p.add_argument("--mage_use_frequency", action="store_true", default=True)
    p.add_argument("--mage_aggregator_freq_dim", type=int, default=64)
    p.add_argument("--scale_dropout", type=float, default=0.0)
    p.add_argument("--agg_noise_std", type=float, default=0.0)
    p.add_argument("--entropy_lambda", type=float, default=0.0)

    # Devices & Logging
    p.add_argument("--devices", default=",".join(map(str, getattr(CFG, "devices", []))))
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--precision", type=str, default="32")
    _add_bool_flag(p, "resume", False, "Resume training from last.ckpt")
    _add_bool_flag(p, "skip_completed_folds", True, "Skip folds that already have a .done sentinel")
    _add_bool_flag(p, "log_csv", True, "Also write CSV metrics alongside TensorBoard")

    args = p.parse_args()

    # Post-processing
    args.devices = parse_devices(args.devices)
    if args.num_workers is None:
        args.num_workers = default_num_workers(args.devices)

    cfg_path = getattr(CFG, "PHYSIOMI_PATH", None)
    if (not args.physiomi_path) or args.physiomi_path.startswith("./data"):
        if cfg_path: args.physiomi_path = cfg_path
    args.test_fold_index = getattr(CFG, "PHYSIOMI_TEST_FOLD_INDEX", 4)
    args.cv_fold_index = getattr(CFG, "PHYSIOMI_CV_FOLD_INDEX", None)
    args.physiomi_folds_json = getattr(CFG, "PHYSIOMI_FOLDS_JSON", None)

    # Parse channel names
    if isinstance(args.dataset_channel_names, str) and args.dataset_channel_names:
        s = args.dataset_channel_names.strip()
        try:
            if s.startswith("["): lst = json.loads(s)
            else: lst = [t.strip() for t in s.split(",") if t.strip()]
            if lst:
                args.dataset_channel_names = lst
                args.in_chans = len(lst)
        except Exception: pass

    # Montage logic
    if getattr(args, "dataset_channel_set", None):
        args.dataset_channel_names = CHANNELS_BY_DATASET.get(args.dataset_channel_set)
        if args.dataset_channel_names: args.in_chans = len(args.dataset_channel_names)
    elif not args.dataset_channel_names and args.task in CHANNELS_BY_DATASET:
        args.dataset_channel_names = CHANNELS_BY_DATASET.get(args.task)
    
    if getattr(args, "model_channel_set", None):
        args.use_channel_names = MONTAGE_BY_DATASET.get(args.model_channel_set) or list(DEFAULT_EEGPT58)
    else:
        args.use_channel_names = MONTAGE_BY_DATASET.get(args.task, list(DEFAULT_EEGPT58))
    args.target_chans = len(args.use_channel_names)

    # Model Config Load
    if args.scheme == "mage":
        try: _models_cfg = CFG.get_full_config(args.tag)
        except: _models_cfg = CFG.get_config(args.tag)
        if "encoder" in _models_cfg: args._enc_embed_dim = int(_models_cfg["encoder"]["embed_dim"])
        else: args._enc_embed_dim = int(_models_cfg.get("embed_dim", 128))

    if args.flatten_n_time is None:
        try: args.flatten_n_time = max(1, int(args.seq_len) // int(args.patch))
        except Exception: args.flatten_n_time = None

    args._split_spec = _parse_split_spec(args.split_spec) if args.split_spec else {
        "train": _parse_id_list(args.train_subject_ids), "val": _parse_id_list(args.val_subject_ids), "test": _parse_id_list(args.test_subject_ids)
    }
    
    return args


# =============================================================================
# Engines / runners
# =============================================================================

def build_engine(args, *, ckpt_path: Optional[str], steps_per_epoch: int, num_classes: int):
    if args.scheme.lower() == "mage":
        if LitMAGEDownstream is None: raise ImportError("LitMAGEDownstream not found.")
        full_cfg = CFG.get_full_config(args.tag)
        encoder_cfg = full_cfg["encoder"]
        ms_cfg = full_cfg["multiscale"]
        
        # Override with CLI
        patch_sizes = getattr(args, "mage_patch_sizes", None) or ms_cfg.get("patch_sizes", [200, 300, 600])
        active_scale = getattr(args, "mage_active_scale", None)
        active_head_index = int(getattr(args, "active_head_index", -1))
        # -1 = ensemble; >=0 = single scale. Map onto active_scale when not explicitly set.
        if active_scale is None and active_head_index >= 0:
            active_scale = active_head_index
        if active_scale is not None:
            if active_scale < 0 or active_scale >= len(patch_sizes):
                raise ValueError(f"--mage_active_scale/--active_head_index must index --mage_patch_sizes (0..{len(patch_sizes) - 1})")
        k_temporal = getattr(args, "mage_k_temporal", None) or ms_cfg.get("k_temporal", 3)
        temporal_extra_convs = getattr(args, "mage_temporal_extra_convs", None) or ms_cfg.get("temporal_extra_convs", 2)
        use_spectral = getattr(args, "mage_use_spectral", None)
        if use_spectral is None: use_spectral = ms_cfg.get("use_spectral", True)
        aggregator_hidden_dim = getattr(args, "mage_aggregator_hidden_dim", None) or ms_cfg.get("aggregator_hidden_dim", 128)
        use_frequency = getattr(args, "mage_use_frequency", None)
        if use_frequency is None: use_frequency = ms_cfg.get("use_frequency", True)
        aggregator_freq_dim = getattr(args, "mage_aggregator_freq_dim", None) or ms_cfg.get("aggregator_freq_dim", 64)
        
        combined_cfg = {
            "encoder": encoder_cfg,
            "multiscale": {
                "patch_sizes": patch_sizes, "k_temporal": k_temporal, "temporal_extra_convs": temporal_extra_convs,
                "use_spectral": use_spectral, "aggregator_hidden_dim": aggregator_hidden_dim,
                "use_frequency": use_frequency, "aggregator_freq_dim": aggregator_freq_dim,
                "scale_dropout": getattr(args, "scale_dropout", 0.0),
                "agg_noise_std": getattr(args, "agg_noise_std", 0.0),
                "entropy_lambda": getattr(args, "entropy_lambda", 0.0),
            }
        }
        _stash_encoder_logging_payload(args, combined_cfg, "mage")
        
        return LitMAGEDownstream(
            load_path=str(ckpt_path) if ckpt_path else None, in_chans=int(args.in_chans),
            channel_names=args.dataset_channel_names, target_chans=int(args.target_chans),
            use_channel_names=args.use_channel_names, time_length=int(args.seq_len),
            patch_sizes=patch_sizes, d_model=int(encoder_cfg["embed_dim"]),
            encoder_depth=int(encoder_cfg.get("depth", 6)), encoder_heads=int(encoder_cfg.get("num_heads", 8)),
            dropout=float(args.dropout), no_inter_attn_enc=bool(getattr(args, "mage_no_inter_attn_enc", False)),
            use_channel_attn=bool(getattr(args, "mage_use_channel_attn", True)),
            use_rope_time=bool(getattr(args, "rope_time", True)), k_temporal=int(k_temporal),
            temporal_extra_convs=int(temporal_extra_convs), use_spectral=bool(use_spectral),
            aggregator_hidden_dim=int(aggregator_hidden_dim), use_frequency=bool(use_frequency),
            aggregator_freq_dim=int(aggregator_freq_dim), head_type=str(args.head_type), num_classes=int(num_classes),
            active_scale=active_scale, active_head_index=active_head_index,
            use_chan_conv=bool(getattr(args, "use_chan_conv", True)), chan_max_norm=float(getattr(args, "chan_max_norm", 1.0)),
            freeze_encoder=bool(args.freeze_encoder), freeze_patch_embed=bool(getattr(args, "freeze_patch_embed", True)),
            freeze_aggregator=False, ft_top_k=int(getattr(args, "ft_top_k", 0)),
            lr_head=float(args.max_lr_head), lr_enc=float(args.max_lr_enc), wd_head=float(args.wd_head), wd_enc=float(args.wd_enc),
            max_epochs=int(args.epochs), steps_per_epoch=int(steps_per_epoch), scheduler=str(args.scheduler),
            warmup_pct=float(args.warmup_pct), label_smoothing=float(getattr(args, "label_smoothing", 0.0)),
            class_weights=getattr(args, "class_weights", None), scale_dropout=float(getattr(args, "scale_dropout", 0.0)),
            agg_noise_std=float(getattr(args, "agg_noise_std", 0.0)), entropy_lambda=float(getattr(args, "entropy_lambda", 0.0)),
            load_pretrained=bool(getattr(args, "load_pretrained", True)), init_scheme=str(getattr(args, "init_scheme", "xavier_uniform")),
        )

    raise NotImplementedError(f"Scheme {args.scheme} not implemented in build_engine.")

# =============================================================================
# Runners
# =============================================================================

def _bcic2a_discover_subjects(db_path: str) -> Tuple[List[str], Dict[str, List[str]]]:
    import lmdb
    env = lmdb.open(db_path, readonly=True, lock=False, readahead=True, meminit=False)
    try:
        with env.begin(write=False) as txn:
            raw = pickle.loads(txn.get(b"__keys__"))
    finally:
        env.close()

    if isinstance(raw, dict):
        keys, seen = [], set()
        for m in ("train", "val", "test"):
            for k in raw.get(m, []):
                if k not in seen:
                    keys.append(k); seen.add(k)
        for m, lst in raw.items():
            if m in {"train", "val", "test"}: continue
            for k in lst:
                if k not in seen:
                    keys.append(k); seen.add(k)
    else:
        keys = list(raw)

    def _tok(key: str) -> str:
        return key.split("-", 1)[0][:3]

    toks: List[str] = []
    by: Dict[str, List[str]] = {}
    seen = set()
    for k in keys:
        s = _tok(k)
        if s not in seen:
            toks.append(s); seen.add(s)
        by.setdefault(s, []).append(k)
    return toks, by

def run_bcic2a_loso(args):
    try: from bcic2a_dataset import LoadDataset
    except Exception: from downstream.datasets.bcic2a_dataset import LoadDataset
    ckpt_path, model_type = resolve_pretrain_checkpoint(args)
    base_dir = downstream_log_base("BCIC2a", args, args.scheme, extra="LOSO")
    subj_tokens, _ = _bcic2a_discover_subjects(args.bcic2a_path)
    n_subj = len(subj_tokens)
    rows = []
    metric_keys = None
    metric_accum = {}

    for sid, tok in enumerate(subj_tokens):
        print(f"\n[BCIC2a] === LOSO fold {sid}/{n_subj-1} — test subject: {tok} ===")
        class _Params:
            datasets_dir = args.bcic2a_path
            batch_size = args.batch_size
            split_mode = "mixed"
            test_subject_ids = [sid]
            val_ratio = args.val_ratio
            stratified_val = args.stratified_val
            seed = args.seed
            num_workers = args.num_workers
            pin_memory = True
            drop_last = False
            shuffle_train = True
            scale_div = args.scale_div
        
        loaders = LoadDataset(_Params()).get_data_loader()
        train_loader = loaders["train"]
        val_loader = loaders.get("val", None)
        test_loader = loaders["test"]
        steps = len(train_loader)
        model = build_engine(args, ckpt_path=ckpt_path, steps_per_epoch=steps, num_classes=4)
        fold_dir = base_dir / f"s{sid:02d}_{tok}"
        (fold_dir / "ckpts").mkdir(parents=True, exist_ok=True)
        loggers = make_loggers(fold_dir, version="run", enable_csv=getattr(args, "log_csv", True))
        _log_encoder_everywhere(fold_dir, loggers, args)
        monitor = "valid_cohen_kappa" if (val_loader is not None) else None
        trainer, ckpt_cb = build_trainer_with_callbacks(args.devices, args.epochs, loggers, args.precision, monitor_metric=monitor, ckpt_dir=fold_dir / "ckpts")
        trainer.fit(model, train_loader, val_loader)
        best_path = ckpt_cb.best_model_path or (ckpt_cb.last_model_path if hasattr(ckpt_cb, "last_model_path") else None)
        test_metrics = trainer.test(model if best_path is None else None, ckpt_path=best_path, dataloaders=test_loader, verbose=False)[0]
        rows.append(make_result_row(dataset="BCIC2a", seed=args.seed, args=args, test_metrics=test_metrics, ckpt_cb=ckpt_cb, extra={"strategy": "loso", "fold_test_subject_id": sid, "fold_test_subject_token": tok, "best_ckpt": ckpt_cb.best_model_path or ""}))
        if metric_keys is None:
            metric_keys = list(test_metrics.keys())
            metric_accum = {k: 0.0 for k in metric_keys}
        for k in metric_keys: metric_accum[k] += float(test_metrics[k])
    
    if metric_keys:
        avg_row = {"dataset": "BCIC2a", "strategy": "loso", "fold_test_subject_id": "MEAN", "fold_test_subject_token": "MEAN"}
        for k in metric_keys: avg_row[k] = metric_accum[k] / n_subj
        rows.append(avg_row)
    out_csv = str(base_dir / "csv" / "summary.csv")
    write_csv(out_csv, rows)
    print(f"\n[BCIC2a] LOSO finished. Summary at: {out_csv}")

def run_bcic2a_fixed(args):
    try: from bcic2a_dataset import LoadDataset
    except Exception: from downstream.datasets.bcic2a_dataset import LoadDataset
    ckpt_path, model_type = resolve_pretrain_checkpoint(args)
    base_dir = downstream_log_base("BCIC2a", args, model_type, extra="Fixed")
    _ = _bcic2a_discover_subjects(args.bcic2a_path)
    spec = args._split_spec
    train_ids, val_ids, test_ids = spec.get("train"), spec.get("val"), spec.get("test")
    if not (train_ids and test_ids): raise ValueError("For --split=fixed specify train/test IDs.")
    
    class _Params:
        datasets_dir = args.bcic2a_path
        batch_size = args.batch_size
        split_mode = "subject_explicit"
        train_subject_ids = train_ids
        val_subject_ids = val_ids if val_ids is not None else []
        test_subject_ids = test_ids
        num_workers = args.num_workers
        pin_memory = True
        drop_last = False
        shuffle_train = True
        scale_div = args.scale_div
        seed = args.seed
    
    loaders = LoadDataset(_Params()).get_data_loader()
    train_loader = loaders["train"]
    val_loader = loaders.get("val", None)
    test_loader = loaders["test"]
    steps = len(train_loader)
    model = build_engine(args, ckpt_path=ckpt_path, steps_per_epoch=steps, num_classes=4)
    (base_dir / "ckpts").mkdir(parents=True, exist_ok=True)
    loggers = make_loggers(base_dir, version="run", enable_csv=getattr(args, "log_csv", True))
    _log_encoder_everywhere(base_dir, loggers, args)
    monitor = "valid_cohen_kappa" if (val_loader is not None) else None
    trainer, ckpt_cb = build_trainer_with_callbacks(args.devices, args.epochs, loggers, args.precision, monitor_metric=monitor, ckpt_dir=base_dir / "ckpts")
    trainer.fit(model, train_loader, val_loader)
    best_path = ckpt_cb.best_model_path or (ckpt_cb.last_model_path if hasattr(ckpt_cb, "last_model_path") else None)
    test_metrics = trainer.test(model if best_path is None else None, ckpt_path=best_path, dataloaders=test_loader, verbose=False)[0]
    row = make_result_row(dataset="BCIC2a", seed=args.seed, args=args, test_metrics=test_metrics, ckpt_cb=ckpt_cb, extra={"strategy": "fixed_subject", "best_ckpt": ckpt_cb.best_model_path or ""})
    write_csv(str(base_dir / "csv" / "summary.csv"), [row])
    print(f"[BCIC2a] Fixed split finished.")

def run_physiomi_kfold(args):
    try: from physiomi_dataset import LoadDataset
    except Exception: from downstream.datasets.physiomi_dataset import LoadDataset
    dataset_key = "PhysioMI"
    ckpt_path, model_type = resolve_pretrain_checkpoint(args)
    non_test_ids = [i for i in range(5) if i != int(args.test_fold_index)]
    inner_round_ids = non_test_ids if args.cv_fold_index is None else [non_test_ids[int(args.cv_fold_index)]]
    base_dir = downstream_log_base(dataset_key, args, model_type, extra=f"K5fold_SUBJ_fixT{int(args.test_fold_index)}")
    rows, metric_keys, metric_accum = [], None, {}

    for val_fold_id in inner_round_ids:
        class _Params:
            datasets_dir = args.physiomi_path
            batch_size = args.batch_size
            num_workers = args.num_workers
            pin_memory = True; drop_last = False
            seed = args.seed
            test_fold_index = int(args.test_fold_index)
            cv_fold_index = [i for i in range(5) if i != int(args.test_fold_index)].index(val_fold_id)
            folds_json = getattr(args, "physiomi_folds_json", None)
            scale_div = getattr(args, "scale_div", 1.0)
        
        loaders = LoadDataset(_Params()).get_data_loader()
        train_loader, val_loader, test_loader = loaders["train"], loaders["val"], loaders["test"]
        steps = len(train_loader)
        model = build_engine(args, ckpt_path=ckpt_path, steps_per_epoch=steps, num_classes=getattr(args, "num_classes", 4))
        fold_dir = base_dir / f"valFold{val_fold_id}"
        (fold_dir / "ckpts").mkdir(parents=True, exist_ok=True)
        loggers = make_loggers(fold_dir, version="run", enable_csv=getattr(args, "log_csv", True))
        _log_encoder_everywhere(fold_dir, loggers, args)
        trainer, ckpt_cb = build_trainer_with_callbacks(args.devices, args.epochs, loggers, args.precision, monitor_metric="valid_cohen_kappa", ckpt_dir=fold_dir / "ckpts")
        
        done_flag = fold_dir / ".done"
        if bool(getattr(args, "skip_completed_folds", True)) and done_flag.exists():
            print(f"[PhysioMI][VAL={val_fold_id}] .done exists.")
            best_path = None # Logic to find best_path usually here
        else:
            trainer.fit(model, train_loader, val_loader)
            try: done_flag.touch()
            except: pass
            best_path = ckpt_cb.best_model_path or getattr(ckpt_cb, "last_model_path", None)
        
        test_metrics = trainer.test(model=model, ckpt_path=best_path, dataloaders=test_loader, verbose=False)[0]
        rows.append(make_result_row(dataset=dataset_key, seed=args.seed, args=args, test_metrics=test_metrics, ckpt_cb=ckpt_cb, extra={"val_fold_id": val_fold_id}))
        if metric_keys is None:
            metric_keys = list(test_metrics.keys())
            metric_accum = {k: 0.0 for k in metric_keys}
        for k in metric_keys: metric_accum[k] += float(test_metrics[k])
    
    if args.cv_fold_index is None and metric_keys:
        mean_row = {"dataset": dataset_key, "val_fold_id": "MEAN"}
        for k in metric_keys: mean_row[k] = metric_accum[k] / float(len(inner_round_ids))
        rows.append(mean_row)
    write_csv(str(base_dir / "csv" / "summary.csv"), rows)
    print(f"[{dataset_key}] Finished.")

def run_tuab_4fold(args):
    try: from tuab_dataset import LoadTrainValDataset, LoadTest, _scan_tuab
    except Exception: from downstream.datasets.tuab_dataset import LoadTrainValDataset, LoadTest, _scan_tuab
    ckpt_path, model_type = resolve_pretrain_checkpoint(args)
    base_dir = downstream_log_base("TUAB", args, model_type, extra="4fold")
    _, _, subj_ids = _scan_tuab(args.tuab_path,"train")
    subj_tokens = sorted(set(subj_ids))
    kf = KFold(n_splits=4)
    fold_subjects = [[] for _ in range(4)]
    for fold_idx, (train_val_idx, test_idx) in enumerate(kf.split(subj_tokens)):
        for idx in test_idx: fold_subjects[fold_idx].append(subj_tokens[idx])
    
    class _TestParams:
        datasets_dir = args.tuab_path
        batch_size = args.batch_size
        num_workers = args.num_workers
        pin_memory = True; drop_last = False; shuffle_Test = False; scale_div = args.scale_div
        transform = None
    test_loader = LoadTest(_TestParams()).get_data_loader()
    
    folds = [args.fold] if args.fold != -1 else range(4)
    rows, metric_keys, metric_accum = [], None, {}
    
    for fold_idx in folds:
        val_subjects = fold_subjects[fold_idx]
        print(f"\n[TUAB] === 4-fold {fold_idx+1}/4 ===")
        class _Params:
            datasets_dir = args.tuab_path
            batch_size = args.batch_size
            split_mode = "subject_random"
            val_subject_ids = val_subjects
            seed = args.val_seed
            num_workers = args.num_workers
            pin_memory = True; drop_last = False; shuffle_train = True; scale_div = args.scale_div
            transform = None
        loaders = LoadTrainValDataset(_Params()).get_data_loader()
        train_loader, val_loader = loaders["train"], loaders.get("val", None)
        steps = math.ceil(len(train_loader))
        model = build_engine(args, ckpt_path=ckpt_path, steps_per_epoch=steps, num_classes=2)
        fold_dir = base_dir / f"fold{fold_idx}"
        (fold_dir / "ckpts").mkdir(parents=True, exist_ok=True)
        loggers = make_loggers(fold_dir, version="run")
        trainer, ckpt_cb = build_trainer_with_callbacks(args.devices, args.epochs, loggers, args.precision, monitor_metric="valid_balanced_accuracy", ckpt_dir=fold_dir/"ckpts")
        trainer.fit(model, train_loader, val_loader)
        best_path = ckpt_cb.best_model_path or None
        test_metrics = trainer.test(model if best_path is None else None, ckpt_path=best_path, dataloaders=test_loader, verbose=False)[0]
        rows.append(make_result_row(dataset="TUAB", seed=args.seed, args=args, test_metrics=test_metrics, ckpt_cb=ckpt_cb, extra={"fold_number": fold_idx}))
        if metric_keys is None:
            metric_keys = list(test_metrics.keys())
            metric_accum = {k: 0.0 for k in metric_keys}
        for k in metric_keys: metric_accum[k] += float(test_metrics[k])
    
    if len(folds) == 4 and metric_keys:
        avg_row = {"dataset": "TUAB", "fold_number": "MEAN"}
        for k in metric_keys: avg_row[f"test/{k}"] = metric_accum[k] / 4
        rows.append(avg_row)
    write_csv(str(base_dir / "csv" / "summary.csv"), rows)

def run_evaluation_with_heads(trainer, model, ckpt_path, dataloaders, args):
    """
    Runs trainer.test() possibly multiple times:
    1. Once with the configured active_head_index (ensemble if -1).
    2. If --test_separate_heads is True, loops through indices 0, 1, 2,
       sets model.active_head_index, and collects per-head metrics.
    """
    target_idx = int(getattr(args, "active_head_index", -1))

    if hasattr(model, "active_head_index"):
        model.active_head_index = target_idx
        status = "Ensemble" if target_idx == -1 else f"Head {target_idx}"
        print(f"[Eval] Running Test with: {status} ...")
    else:
        print("[Eval] Running Standard Test (Model does not support active_head_index)...")

    main_metrics = trainer.test(model, ckpt_path=ckpt_path, dataloaders=dataloaders, verbose=False)[0]

    if getattr(args, "test_separate_heads", False):
        print("[Eval] --test_separate_heads is ON. Testing scales [0, 1, 2] separately.")
        if not hasattr(model, "active_head_index"):
            print("[WARNING] Model does not have 'active_head_index'. Skipping separate head tests.")
            return main_metrics

        for h_idx in range(3):
            if h_idx == target_idx:
                print(f"--- [Eval] Head {h_idx} already tested. Copying metrics. ---")
                for k, v in main_metrics.items():
                    if not k.startswith("head"):
                        main_metrics[f"head{h_idx}_{k}"] = v
                continue
            print(f"--- [Eval] Testing Head/Scale {h_idx} ---")
            model.active_head_index = h_idx
            head_metrics = trainer.test(model, ckpt_path=ckpt_path, dataloaders=dataloaders, verbose=False)[0]
            for k, v in head_metrics.items():
                main_metrics[f"head{h_idx}_{k}"] = v

        model.active_head_index = target_idx

    return main_metrics


def run_sleepedf_5fold(args):
    try: from sleepedf_dataset import LoadDataset, _scan_sleepedf_auto
    except Exception: from downstream.datasets.sleepedf_dataset import LoadDataset, _scan_sleepedf_auto
    ckpt_path, model_type = resolve_pretrain_checkpoint(args)
    base_dir = downstream_log_base("SLEEPEDF", args, model_type, extra="5fold")
    _, _, subj_ids = _scan_sleepedf_auto(args.sleepedf_path)
    subj_tokens = sorted(set(subj_ids))
    kf = KFold(n_splits=5)
    fold_subjects = [[] for _ in range(5)]
    for fold_idx, (train_val_idx, test_idx) in enumerate(kf.split(subj_tokens)):
        for idx in test_idx: fold_subjects[fold_idx].append(subj_tokens[idx])
    
    folds = [args.fold] if args.fold != -1 else range(5)
    rows, metric_keys, metric_accum = [], None, {}
    
    for fold_idx in folds:
        test_subjects = fold_subjects[fold_idx]
        print(f"\n[SLEEPEDF] === 5-fold {fold_idx+1}/5 ===")
        class _Params:
            datasets_dir = args.sleepedf_path
            batch_size = args.batch_size
            split_mode = "subject_random"
            test_subject_ids = test_subjects
            val_ratio = args.val_ratio
            seed = args.val_seed
            num_workers = args.num_workers
            pin_memory = True; drop_last = False; shuffle_train = True; scale_div = args.scale_div
            transform = None
        loaders = LoadDataset(_Params()).get_data_loader()
        train_loader, val_loader, test_loader = loaders["train"], loaders.get("val", None), loaders["test"]
        steps = math.ceil(len(train_loader))
        model = build_engine(args, ckpt_path=ckpt_path, steps_per_epoch=steps, num_classes=5)
        fold_dir = base_dir / f"fold{fold_idx}"
        (fold_dir / "ckpts").mkdir(parents=True, exist_ok=True)
        loggers = make_loggers(fold_dir, version="run")
        trainer, ckpt_cb = build_trainer_with_callbacks(args.devices, args.epochs, loggers, args.precision, monitor_metric="valid_balanced_accuracy" if val_loader is not None else None, ckpt_dir=fold_dir/"ckpts")
        trainer.fit(model, train_loader, val_loader)
        best_path = ckpt_cb.best_model_path or None
        test_metrics = run_evaluation_with_heads(trainer, model, best_path, test_loader, args)
        rows.append(make_result_row(dataset="SLEEPEDF", seed=args.seed, args=args, test_metrics=test_metrics, ckpt_cb=ckpt_cb, extra={"fold_number": fold_idx}))
        if metric_keys is None:
            metric_keys = list(test_metrics.keys())
            metric_accum = {k: 0.0 for k in metric_keys}
        for k in metric_keys: metric_accum[k] += float(test_metrics[k])
    
    if len(folds) == 5 and metric_keys:
        avg_row = {"dataset": "SLEEPEDF", "fold_number": "MEAN"}
        for k in metric_keys: avg_row[f"test/{k}"] = metric_accum[k] / 5
        rows.append(avg_row)
    write_csv(str(base_dir / "csv" / "summary.csv"), rows)

def run_kaggle_ern(args):
    try: from kaggle_ern_dataset import LoadDataset
    except Exception: from downstream.datasets.kaggle_ern_dataset import LoadDataset
    ckpt_path, model_type = resolve_pretrain_checkpoint(args)
    base_dir = downstream_log_base("KaggleERN", args, model_type, extra="5fold")
    folds = [args.fold] if args.fold != -1 else range(1, 6)
    rows, metric_keys, metric_accum = [], None, {}
    
    for fold_idx in folds:
        print(f"\n[KaggleERN] === Fold {fold_idx}/5 ===")
        class _Params:
            datasets_dir = args.kaggle_ern_path
            batch_size = args.batch_size
            num_workers = args.num_workers
            pin_memory = True; drop_last = False; shuffle_train = True; scale_div = args.scale_div
            fold = fold_idx
        loaders = LoadDataset(_Params()).get_data_loader()
        train_loader, val_loader, test_loader = loaders["train"], loaders.get("val", None), loaders["test"]
        steps = math.ceil(len(train_loader))
        model = build_engine(args, ckpt_path=ckpt_path, steps_per_epoch=steps, num_classes=2)
        fold_dir = base_dir / f"fold{fold_idx}"
        (fold_dir / "ckpts").mkdir(parents=True, exist_ok=True)
        loggers = make_loggers(fold_dir, version="run")
        _log_encoder_everywhere(fold_dir, loggers, args)
        trainer, ckpt_cb = build_trainer_with_callbacks(args.devices, args.epochs, loggers, args.precision, monitor_metric="valid_balanced_accuracy", ckpt_dir=fold_dir/"ckpts")
        trainer.fit(model, train_loader, val_loader)
        best_path = ckpt_cb.best_model_path or None
        test_metrics = trainer.test(model if best_path is None else None, ckpt_path=best_path, dataloaders=test_loader, verbose=False)[0]
        rows.append(make_result_row(dataset="KaggleERN", seed=args.seed, args=args, test_metrics=test_metrics, ckpt_cb=ckpt_cb, extra={"fold_number": fold_idx}))
        if metric_keys is None:
            metric_keys = list(test_metrics.keys())
            metric_accum = {k: 0.0 for k in metric_keys}
        for k in metric_keys: metric_accum[k] += float(test_metrics[k])
    
    if len(folds) == 5 and metric_keys:
        avg_row = {"dataset": "KaggleERN", "fold_number": "MEAN"}
        for k in metric_keys: avg_row[f"test/{k}"] = metric_accum[k] / 5
        rows.append(avg_row)
    write_csv(str(base_dir / "csv" / "summary.csv"), rows)

def run_faced_4fold(args):
    """
    Run 4-Fold Cross Validation on FACED dataset.
    """
    try:
        from downstream.datasets.faced_dataset import LoadDataset
    except ImportError:
        try:
            from faced_dataset import LoadDataset
        except ImportError:
            raise ImportError("Could not import LoadDataset for FACED. Ensure downstream/datasets/faced_dataset.py exists.")

    ckpt_path, model_type = resolve_pretrain_checkpoint(args)
    base_dir = downstream_log_base("FACED", args, model_type, extra="4fold")

    folds = range(4)
    if args.fold != -1:
        folds = [args.fold]
        print(f"[FACED] Running specific fold: {args.fold}")
    else:
        print(f"[FACED] Running all 4 folds sequentially.")

    rows: List[Dict[str, Any]] = []
    metric_keys: Optional[List[str]] = None
    metric_accum: Dict[str, float] = {}

    for f_idx in folds:
        print(f"\n[FACED] === 4-fold {f_idx+1}/4 ===")

        class _Params:
            datasets_dir = args.faced_path
            batch_size = args.batch_size
            num_workers = args.num_workers
            pin_memory = True
            drop_last = False
            shuffle_train = True
            scale_div = args.scale_div
            fold_idx = f_idx  
        loaders = LoadDataset(_Params()).get_data_loader()
        train_loader = loaders["train"]
        val_loader = loaders.get("val", None)
        test_loader = loaders["test"]

        steps = math.ceil(len(train_loader))
        model = build_engine(args, ckpt_path=ckpt_path, steps_per_epoch=steps, num_classes=9)

        fold_dir = base_dir / f"fold{f_idx}"
        (fold_dir / "ckpts").mkdir(parents=True, exist_ok=True)

        done_flag = fold_dir / ".done"
        if bool(getattr(args, "skip_completed_folds", True)) and done_flag.exists():
            print(f"[FACED][Fold {f_idx}] .done exists — skipping.")
            continue

        loggers = make_loggers(fold_dir, version="run")
        _log_encoder_everywhere(fold_dir, loggers, args)

        trainer, ckpt_cb = build_trainer_with_callbacks(
            args.devices, args.epochs, loggers, args.precision,
            monitor_metric="valid_accuracy" if val_loader is not None else None,
            ckpt_dir=fold_dir/"ckpts",
        )

        resume_ckpt = None
        if bool(getattr(args, "resume", True)):
            ckpt_dir = fold_dir / "ckpts"
            last_ckpt = ckpt_dir / "last.ckpt"
            if last_ckpt.is_file(): resume_ckpt = str(last_ckpt)
            else:
                ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
                if ckpts: resume_ckpt = str(ckpts[0])

        trainer.fit(model, train_loader, val_loader, ckpt_path=resume_ckpt)

        best_path = ckpt_cb.best_model_path or None
        test_metrics = trainer.test(model if best_path is None else None,
                                    ckpt_path=best_path, dataloaders=test_loader, verbose=False)[0]

        row = make_result_row(dataset="FACED", seed=args.seed, args=args, test_metrics=test_metrics, ckpt_cb=ckpt_cb,
                              extra={
                                  "strategy": "4fold",
                                  "fold_number": f_idx,
                                  "best_ckpt": ckpt_cb.best_model_path or "",
                              })
        rows.append(row)

        if metric_keys is None:
            metric_keys = list(test_metrics.keys())
            metric_accum = {k: 0.0 for k in metric_keys}
        for k in metric_keys:
            metric_accum[k] += float(test_metrics[k])
        
        try: done_flag.touch()
        except Exception: pass

    if metric_keys and len(folds) > 1:
        avg_row = {"dataset": "FACED", "strategy": "4fold", "fold_number": "MEAN"}
        for k in metric_keys:
            avg_row[f"test/{k}"] = metric_accum[k] / len(folds)
        rows.append(avg_row)

    out_csv = str((base_dir / "csv" / "summary.csv"))
    write_csv(out_csv, rows)
    print(f"\n[FACED] 4-fold finished. Summary at: {out_csv}")
    
def run_shumi_5fold(args):
    """
    Run 5-Fold Cross Validation on SHUMI (MODMA) dataset.
    """
    try:
        from downstream.datasets.shumi_dataset import LoadDataset
    except ImportError:
        try:
            from shumi_dataset import LoadDataset
        except ImportError:
            raise ImportError("Could not import LoadDataset for SHUMI. Ensure downstream/datasets/shumi_dataset.py exists.")

    ckpt_path, model_type = resolve_pretrain_checkpoint(args)
    base_dir = downstream_log_base("SHUMI", args, model_type, extra="5fold")

    folds = range(5)
    if args.fold != -1:
        folds = [args.fold]
        print(f"[SHUMI] Running specific fold: {args.fold}")
    else:
        print(f"[SHUMI] Running all 5 folds sequentially.")

    rows: List[Dict[str, Any]] = []
    metric_keys: Optional[List[str]] = None
    metric_accum: Dict[str, float] = {}

    for f_idx in folds:
        print(f"\n[SHUMI] === Fold {f_idx+1}/5 ===")

        class _Params:
            datasets_dir = args.shumi_path
            batch_size = args.batch_size
            num_workers = args.num_workers
            pin_memory = True
            drop_last = False
            shuffle_train = True
            scale_div = args.scale_div
            fold_idx = f_idx

        loaders = LoadDataset(_Params()).get_data_loader()
        train_loader = loaders["train"]
        val_loader = loaders.get("val", None)
        test_loader = loaders["test"]

        num_classes = 2 
        
        steps = math.ceil(len(train_loader))
        model = build_engine(args, ckpt_path=ckpt_path, steps_per_epoch=steps, num_classes=num_classes)

        fold_dir = base_dir / f"fold{f_idx}"
        (fold_dir / "ckpts").mkdir(parents=True, exist_ok=True)

        done_flag = fold_dir / ".done"
        if bool(getattr(args, "skip_completed_folds", True)) and done_flag.exists():
            print(f"[SHUMI][Fold {f_idx}] .done exists — skipping.")
            continue

        loggers = make_loggers(fold_dir, version="run")
        _log_encoder_everywhere(fold_dir, loggers, args)

        monitor_metric = "valid_balanced_accuracy" 
        
        trainer, ckpt_cb = build_trainer_with_callbacks(
            args.devices, args.epochs, loggers, args.precision,
            monitor_metric=monitor_metric,
            ckpt_dir=fold_dir/"ckpts",
        )

        resume_ckpt = None
        if bool(getattr(args, "resume", True)):
            ckpt_dir = fold_dir / "ckpts"
            last_ckpt = ckpt_dir / "last.ckpt"
            if last_ckpt.is_file(): resume_ckpt = str(last_ckpt)
            else:
                ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
                if ckpts: resume_ckpt = str(ckpts[0])

        trainer.fit(model, train_loader, val_loader, ckpt_path=resume_ckpt)

        best_path = ckpt_cb.best_model_path or None
        test_metrics = trainer.test(model if best_path is None else None,
                                    ckpt_path=best_path, dataloaders=test_loader, verbose=False)[0]

        row = make_result_row(dataset="SHUMI", seed=args.seed, args=args, test_metrics=test_metrics, ckpt_cb=ckpt_cb,
                              extra={
                                  "strategy": "5fold",
                                  "fold_number": f_idx,
                                  "best_ckpt": ckpt_cb.best_model_path or "",
                              })
        rows.append(row)

        if metric_keys is None:
            metric_keys = list(test_metrics.keys())
            metric_accum = {k: 0.0 for k in metric_keys}
        for k in metric_keys:
            metric_accum[k] += float(test_metrics[k])
        
        try: done_flag.touch()
        except Exception: pass

    if metric_keys and len(folds) > 1:
        avg_row = {"dataset": "SHUMI", "strategy": "5fold", "fold_number": "MEAN"}
        for k in metric_keys:
            avg_row[f"test/{k}"] = metric_accum[k] / len(folds)
        rows.append(avg_row)

    out_csv = str((base_dir / "csv" / "summary.csv"))
    write_csv(out_csv, rows)
    print(f"\n[SHUMI] 5-fold finished. Summary at: {out_csv}")


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)
    
    if args.task == "BCIC2a":
        if args.split == "loso": run_bcic2a_loso(args)
        elif args.split == "fixed": run_bcic2a_fixed(args)
        else: raise ValueError(f"Unsupported split for BCIC2a: {args.split}")
    elif args.task == "FACED":
        if args.split == "4fold": run_faced_4fold(args)
        else: raise ValueError(f"Unsupported split for FACED: {args.split}. Only '4fold' is supported.")
    elif args.task == "SHUMI":
        if args.split == "5fold": run_shumi_5fold(args)
        else: raise ValueError(f"Unsupported split for SHUMI: {args.split}. Only '5fold' is supported.")
    elif args.task == "PhysioMI":
        run_physiomi_kfold(args)
    elif args.task == "TUAB":
        if args.split == "4fold": run_tuab_4fold(args)
        else: raise ValueError(f"Unsupported split for TUAB: {args.split}. Only '4fold' is supported.")
    elif args.task == "SLEEPEDF":
        if args.split == "5fold": run_sleepedf_5fold(args)
        else: raise ValueError(f"Unsupported split for SLEEPEDF: {args.split}. Only '5fold' is supported.")
    elif args.task == "KaggleERN":
        if args.split == "5fold": run_kaggle_ern(args)
        else: raise ValueError(f"Unsupported split for KaggleERN: {args.split}. Only '5fold' is supported.")
    else:
        raise ValueError(f"Unknown task: {args.task}")

if __name__ == "__main__":
    main()
