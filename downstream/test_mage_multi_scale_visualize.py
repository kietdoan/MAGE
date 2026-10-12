#!/usr/bin/env python3
"""
Test-only aggregator-output exporter for MAGE (scheme=mage) checkpoints.

This rebuilds the downstream fold loader, loads a downstream Lightning checkpoint,
runs inference on the fold test set only, saves the full aggregator output, and
visualizes each scale weight across test samples.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.model_selection import KFold


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import run_downstream_full as _base  # noqa: E402

try:
    from utils_eval import get_metrics  # noqa: E402
except Exception:
    from downstream.utils_eval import get_metrics  # type: ignore  # noqa: E402


def _parse_extra_args() -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ckpt", type=str, required=True, help="Downstream .ckpt to evaluate.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory. Defaults to <checkpoint_dir>/aggregator_test_outputs.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Fold id. If omitted, inferred from .../foldX/... in --ckpt.",
    )
    parser.add_argument(
        "--plot_prefix",
        type=str,
        default="faced_test",
        help="Prefix for plot/data filenames.",
    )
    parser.add_argument(
        "--max_plot_samples",
        type=int,
        default=0,
        help="If >0, only plot the first N samples in the line plot. Saved arrays still contain all samples.",
    )
    return parser.parse_known_args()


def _infer_fold_from_path(path: Path) -> int:
    for part in path.parts:
        match = re.fullmatch(r"(?:fold|valFold)(\d+)", part)
        if match:
            return int(match.group(1))
    raise ValueError("Could not infer fold from checkpoint path. Pass --fold explicitly.")


def _make_faced_loaders(args: argparse.Namespace, fold_id: int):
    try:
        from downstream.datasets.faced_dataset import LoadDataset
    except Exception:
        from datasets.faced_dataset import LoadDataset  # type: ignore

    attrs = {
        "datasets_dir": args.faced_path,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "drop_last": False,
        "shuffle_train": False,
        "scale_div": args.scale_div,
        "fold_idx": fold_id,
    }
    return LoadDataset(type("_Params", (), attrs)()).get_data_loader()


def _make_sleepedf_loaders(args: argparse.Namespace, fold_id: int):
    try:
        from downstream.datasets.sleepedf_dataset import LoadDataset, _scan_sleepedf_auto
    except Exception:
        from datasets.sleepedf_dataset import LoadDataset, _scan_sleepedf_auto  # type: ignore

    _, _, subj_ids = _scan_sleepedf_auto(args.sleepedf_path)
    subj_tokens = sorted(set(subj_ids))
    fold_subjects = [[] for _ in range(5)]
    for idx, (_, test_idx) in enumerate(KFold(n_splits=5).split(subj_tokens)):
        fold_subjects[idx] = [subj_tokens[j] for j in test_idx]

    attrs = {
        "datasets_dir": args.sleepedf_path,
        "batch_size": args.batch_size,
        "split_mode": "subject_random",
        "test_subject_ids": fold_subjects[fold_id],
        "val_ratio": args.val_ratio,
        "seed": args.val_seed,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "drop_last": False,
        "shuffle_train": False,
        "scale_div": args.scale_div,
        "transform": None,
    }
    return LoadDataset(type("_Params", (), attrs)()).get_data_loader()


def _make_shumi_loaders(args: argparse.Namespace, fold_id: int):
    try:
        from downstream.datasets.shumi_dataset import LoadDataset
    except Exception:
        from datasets.shumi_dataset import LoadDataset  # type: ignore

    attrs = {
        "datasets_dir": args.shumi_path,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "drop_last": False,
        "shuffle_train": False,
        "scale_div": args.scale_div,
        "fold_idx": fold_id,
    }
    return LoadDataset(type("_Params", (), attrs)()).get_data_loader()


def _make_physiomi_loaders(args: argparse.Namespace, val_fold_id: int):
    try:
        from downstream.datasets.physiomi_dataset import LoadDataset
    except Exception:
        from datasets.physiomi_dataset import LoadDataset  # type: ignore

    test_fold_index = int(getattr(args, "test_fold_index", 4))
    non_test_ids = [i for i in range(5) if i != test_fold_index]
    if val_fold_id not in non_test_ids:
        raise ValueError(
            f"PhysioMI valFold{val_fold_id} is invalid for fixed test fold {test_fold_index}; "
            f"expected one of {non_test_ids}."
        )

    attrs = {
        "datasets_dir": args.physiomi_path,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "drop_last": False,
        "seed": args.seed,
        "test_fold_index": test_fold_index,
        "cv_fold_index": non_test_ids.index(val_fold_id),
        "folds_json": getattr(args, "physiomi_folds_json", None),
        "scale_div": args.scale_div,
    }
    return LoadDataset(type("_Params", (), attrs)()).get_data_loader()


def _make_kaggleern_loaders(args: argparse.Namespace, fold_id: int):
    try:
        from downstream.datasets.kaggle_ern_dataset import LoadDataset
    except Exception:
        from datasets.kaggle_ern_dataset import LoadDataset  # type: ignore

    attrs = {
        "datasets_dir": args.kaggle_ern_path,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "drop_last": False,
        "shuffle_train": False,
        "scale_div": args.scale_div,
        "fold": fold_id,
    }
    return LoadDataset(type("_Params", (), attrs)()).get_data_loader()


def _make_tuab_loaders(args: argparse.Namespace, fold_id: int):
    del fold_id  # TUAB uses one fixed held-out test set for every training fold.
    try:
        from downstream.datasets.tuab_dataset import LoadTest
    except Exception:
        from datasets.tuab_dataset import LoadTest  # type: ignore

    attrs = {
        "datasets_dir": args.tuab_path,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "drop_last": False,
        "shuffle_Test": False,
        "scale_div": args.scale_div,
        "transform": None,
    }
    test_loader = LoadTest(type("_Params", (), attrs)()).get_data_loader()
    # The train loader is only used to derive scheduler steps before restoring
    # the complete checkpoint; using the test loader avoids loading unused data.
    return {"train": test_loader, "test": test_loader}


def _make_bcic2a_loaders(args: argparse.Namespace, fold_id: int):
    try:
        from downstream.datasets.bcic2a_dataset import LoadDataset
    except Exception:
        from datasets.bcic2a_dataset import LoadDataset  # type: ignore

    attrs = {
        "datasets_dir": args.bcic2a_path,
        "batch_size": args.batch_size,
        "split_mode": "mixed",
        "test_subject_ids": [fold_id],
        "val_ratio": args.val_ratio,
        "stratified_val": args.stratified_val,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "drop_last": False,
        "shuffle_train": False,
        "scale_div": args.scale_div,
    }
    return LoadDataset(type("_Params", (), attrs)()).get_data_loader()


def _make_loaders(args: argparse.Namespace, fold_id: int):
    if args.task == "FACED":
        return _make_faced_loaders(args, fold_id)
    if args.task == "SLEEPEDF":
        return _make_sleepedf_loaders(args, fold_id)
    if args.task == "SHUMI":
        return _make_shumi_loaders(args, fold_id)
    if args.task == "PhysioMI":
        return _make_physiomi_loaders(args, fold_id)
    if args.task == "KaggleERN":
        return _make_kaggleern_loaders(args, fold_id)
    if args.task == "TUAB":
        return _make_tuab_loaders(args, fold_id)
    if args.task == "BCIC2a":
        return _make_bcic2a_loaders(args, fold_id)
    raise ValueError(f"Unsupported task: {args.task}")


def _num_classes(task: str) -> int:
    if task == "FACED":
        return 9
    if task == "SLEEPEDF":
        return 5
    if task == "SHUMI":
        return 2
    if task == "PhysioMI":
        return 4
    if task in {"KaggleERN", "TUAB"}:
        return 2
    if task == "BCIC2a":
        return 4
    raise ValueError(f"Unsupported task: {task}")


def _load_checkpoint_state(model: torch.nn.Module, ckpt_path: Path) -> None:
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[CKPT] loaded: {ckpt_path}")
    if missing:
        print(f"[CKPT][WARN] missing keys: {len(missing)}")
    if unexpected:
        print(f"[CKPT][WARN] unexpected keys: {len(unexpected)}")


@torch.no_grad()
def _forward_with_aggregator_capture(model, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    xa = x
    if model.use_chan_conv and model.chan_conv is not None:
        xa = model.chan_conv(xa)

    x_list = model.patch_embed(xa)
    z_list, _ = model.encoder(x_list, mask_list=None)

    scale_logits = []
    for z, head in zip(z_list, model.scale_heads):
        z = z.permute(0, 2, 1, 3)
        scale_logits.append(head(z))
    scale_logits_t = torch.stack(scale_logits, dim=0)

    scale_weights, agg_info = model.aggregator(xa)
    final_logits = (scale_logits_t * scale_weights.t().unsqueeze(-1)).sum(dim=0)

    out = {
        "scale_weights": scale_weights.detach(),
        "scale_logits": scale_logits_t.detach(),
        "final_logits": final_logits.detach(),
    }
    if isinstance(agg_info, dict) and "scale_logits" in agg_info:
        out["aggregator_logits"] = agg_info["scale_logits"].detach()
    return out


@torch.no_grad()
def _run_test(model, loader, device: torch.device) -> Dict[str, Any]:
    model.to(device)
    model.eval()
    model.encoder.eval()
    model.patch_embed.eval()
    model.aggregator.eval()
    model.scale_heads.eval()

    labels: List[torch.Tensor] = []
    final_logits: List[torch.Tensor] = []
    scale_weights: List[torch.Tensor] = []
    scale_logits: List[torch.Tensor] = []
    aggregator_logits: List[torch.Tensor] = []

    loss_sum = 0.0
    total = 0
    correct = 0

    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device).long()
        captured = _forward_with_aggregator_capture(model, x)
        logits = captured["final_logits"]
        loss = model._loss_fn(logits, y)
        pred = logits.argmax(dim=-1)

        labels.append(y.detach().cpu())
        final_logits.append(logits.detach().cpu())
        scale_weights.append(captured["scale_weights"].detach().cpu())
        scale_logits.append(captured["scale_logits"].detach().cpu())
        if "aggregator_logits" in captured:
            aggregator_logits.append(captured["aggregator_logits"].detach().cpu())

        loss_sum += float(loss.detach().item()) * int(y.numel())
        correct += int((pred == y).sum().item())
        total += int(y.numel())
        completed = batch_idx + 1
        if completed == 1 or completed % 50 == 0 or completed == len(loader):
            print(f"[TEST] batch {completed}/{len(loader)} done")

    y_true = torch.cat(labels, dim=0).numpy()
    logits_np = torch.cat(final_logits, dim=0).numpy()
    weights_np = torch.cat(scale_weights, dim=0).numpy()
    scale_logits_np = torch.cat(scale_logits, dim=1).numpy()
    agg_logits_np = (
        torch.cat(aggregator_logits, dim=0).numpy()
        if aggregator_logits
        else np.empty((weights_np.shape[0], weights_np.shape[1]), dtype=np.float32)
    )

    pred_np = logits_np.argmax(axis=-1)
    metric_values = get_metrics(
        logits_np,
        y_true,
        ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted", "f1_macro", "f1_micro"],
        is_binary=False,
    )
    metrics = {
        "test_loss": loss_sum / max(1, total),
        "test_accuracy_simple": correct / max(1, total),
    }
    metrics.update({f"test_{k}": float(np.nan_to_num(v, nan=0.0)) for k, v in metric_values.items()})

    return {
        "labels": y_true,
        "preds": pred_np,
        "final_logits": logits_np,
        "scale_weights": weights_np,
        "scale_logits": scale_logits_np,
        "aggregator_logits": agg_logits_np,
        "metrics": metrics,
    }


def _write_sample_csv(path: Path, data: Dict[str, Any], patch_sizes: List[int]) -> None:
    labels = data["labels"]
    preds = data["preds"]
    final_logits = data["final_logits"]
    weights = data["scale_weights"]
    scale_logits = data["scale_logits"]
    agg_logits = data["aggregator_logits"]
    num_classes = final_logits.shape[1]

    fieldnames = ["sample_index", "label", "pred"]
    fieldnames += [f"final_logit_class_{i}" for i in range(num_classes)]
    fieldnames += [f"scale_weight_p{p}" for p in patch_sizes]
    fieldnames += [f"aggregator_logit_p{p}" for p in patch_sizes]
    for scale_idx, patch in enumerate(patch_sizes):
        fieldnames += [f"scale_logit_p{patch}_class_{c}" for c in range(num_classes)]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(labels.shape[0]):
            row: Dict[str, Any] = {
                "sample_index": i,
                "label": int(labels[i]),
                "pred": int(preds[i]),
            }
            for c in range(num_classes):
                row[f"final_logit_class_{c}"] = float(final_logits[i, c])
            for s, patch in enumerate(patch_sizes):
                row[f"scale_weight_p{patch}"] = float(weights[i, s])
                row[f"aggregator_logit_p{patch}"] = float(agg_logits[i, s])
                for c in range(num_classes):
                    row[f"scale_logit_p{patch}_class_{c}"] = float(scale_logits[s, i, c])
            writer.writerow(row)


def _write_metrics_csv(path: Path, metrics: Dict[str, float], ckpt_path: Path, fold_id: int) -> None:
    row: Dict[str, Any] = {"checkpoint": str(ckpt_path), "fold": fold_id}
    row.update(metrics)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def _write_scale_summary_csv(path: Path, weights: np.ndarray, patch_sizes: List[int]) -> None:
    rows = []
    for scale_idx, patch in enumerate(patch_sizes):
        values = weights[:, scale_idx]
        rows.append(
            {
                "scale_index": scale_idx,
                "patch_size": int(patch),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_outputs(out_dir: Path, task: str, prefix: str, weights: np.ndarray, patch_sizes: List[int], max_samples: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_plot = weights.shape[0] if max_samples <= 0 else min(weights.shape[0], max_samples)
    xs = np.arange(n_plot)

    plt.figure(figsize=(12, 5))
    for scale_idx, patch in enumerate(patch_sizes):
        plt.plot(xs, weights[:n_plot, scale_idx], linewidth=1.1, label=f"patch {patch}")
    plt.xlabel("Test sample index")
    plt.ylabel("Aggregator softmax weight")
    plt.title(f"{task} test aggregator weights by scale")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    line_path = out_dir / f"{prefix}_scale_weights_line.png"
    plt.savefig(line_path, dpi=180)
    plt.close()

    means = weights.mean(axis=0)
    stds = weights.std(axis=0)
    plt.figure(figsize=(7, 5))
    plt.bar([str(p) for p in patch_sizes], means, yerr=stds, capsize=5)
    plt.xlabel("Patch size / scale")
    plt.ylabel("Mean aggregator weight on test set")
    plt.title(f"{task} test mean aggregator weight")
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    bar_path = out_dir / f"{prefix}_scale_weights_mean.png"
    plt.savefig(bar_path, dpi=180)
    plt.close()

    fig, axes = plt.subplots(len(patch_sizes), 1, figsize=(9, max(4, 2.8 * len(patch_sizes))), sharex=True)
    if len(patch_sizes) == 1:
        axes = [axes]
    for scale_idx, patch in enumerate(patch_sizes):
        values = weights[:, scale_idx]
        mean = values.mean()
        std = values.std()
        axes[scale_idx].hist(values, bins=40, range=(0.0, 1.0), density=False, alpha=0.78)
        axes[scale_idx].axvline(mean, color="black", linestyle="--", linewidth=1.2, label=f"mean={mean:.3f}")
        axes[scale_idx].set_title(f"Scale {scale_idx}: patch size {patch} | weight mean={mean:.3f}, std={std:.3f}", loc="left")
        axes[scale_idx].set_ylabel("Count")
        axes[scale_idx].grid(True, alpha=0.2)
        axes[scale_idx].legend(loc="upper right")
    axes[-1].set_xlabel("Aggregator softmax weight after softmax; 0 = ignored, 1 = dominates")
    fig.suptitle("Distribution of aggregator weights on test set")
    fig.tight_layout()
    hist_path = out_dir / f"{prefix}_scale_weights_distribution_labeled.png"
    fig.savefig(hist_path, dpi=180)
    plt.close(fig)

    plt.figure(figsize=(9, 5))
    for scale_idx, patch in enumerate(patch_sizes):
        values = weights[:, scale_idx]
        plt.hist(values, bins=40, range=(0.0, 1.0), density=True, histtype="step", linewidth=1.8, label=f"scale {scale_idx}, patch {patch} (mean={values.mean():.3f})")
    plt.xlabel("Aggregator softmax weight after softmax")
    plt.ylabel("Density")
    plt.title("Overlay distribution of scale weights")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    overlay_path = out_dir / f"{prefix}_scale_weights_distribution_overlay.png"
    plt.savefig(overlay_path, dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.boxplot([weights[:, i] for i in range(len(patch_sizes))], tick_labels=[str(p) for p in patch_sizes], showmeans=True)
    plt.xlabel("Patch size / scale")
    plt.ylabel("Aggregator softmax weight")
    plt.title(f"{task} test scale-weight distribution summary")
    plt.ylim(0.0, 1.0)
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    box_path = out_dir / f"{prefix}_scale_weights_boxplot.png"
    plt.savefig(box_path, dpi=180)
    plt.close()

    print(f"[PLOT] wrote: {line_path}")
    print(f"[PLOT] wrote: {bar_path}")
    print(f"[PLOT] wrote: {hist_path}")
    print(f"[PLOT] wrote: {overlay_path}")
    print(f"[PLOT] wrote: {box_path}")


def main() -> None:
    extra, downstream_argv = _parse_extra_args()

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]] + downstream_argv
        args = _base.parse_args()
    finally:
        sys.argv = old_argv

    supported_tasks = {"FACED", "SLEEPEDF", "SHUMI", "PhysioMI", "KaggleERN", "TUAB", "BCIC2a"}
    if args.task not in supported_tasks:
        raise ValueError(f"Unsupported --task {args.task}; expected one of {sorted(supported_tasks)}.")
    if args.scheme.lower() != "mage":
        raise ValueError("This script expects --scheme mage.")
    expected_splits = {
        "FACED": {"4fold"},
        "SLEEPEDF": {"5fold"},
        "SHUMI": {"5fold"},
        "PhysioMI": {"loso"},
        "KaggleERN": {"5fold"},
        "TUAB": {"4fold"},
        "BCIC2a": {"loso"},
    }
    if args.split not in expected_splits[args.task]:
        raise ValueError(f"{args.task} test-only visualization expects --split in {sorted(expected_splits[args.task])}.")

    ckpt_path = Path(extra.ckpt).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    fold_id = int(extra.fold) if extra.fold is not None else _infer_fold_from_path(ckpt_path)
    out_dir = Path(extra.output_dir).resolve() if extra.output_dir else ckpt_path.parent / "aggregator_test_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    pl.seed_everything(int(args.seed), workers=True)
    loaders = _make_loaders(args, fold_id)
    train_loader = loaders["train"]
    test_loader = loaders["test"]
    steps = max(1, len(train_loader))

    # Build architecture only. The full downstream checkpoint is loaded below.
    args.load_pretrained = False
    model = _base.build_engine(args, ckpt_path=None, steps_per_epoch=steps, num_classes=_num_classes(args.task))
    _load_checkpoint_state(model, ckpt_path)

    device = torch.device(f"cuda:{args.devices[0]}" if args.devices and torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")
    print(f"[{args.task}] fold={fold_id} test_samples={len(test_loader.dataset)}")
    data = _run_test(model, test_loader, device)

    patch_sizes = [int(p) for p in args.mage_patch_sizes]
    sample_csv = out_dir / f"{extra.plot_prefix}_aggregator_outputs_fold{fold_id}.csv"
    _write_sample_csv(sample_csv, data, patch_sizes)
    print(f"[DATA] wrote: {sample_csv}")

    scale_summary_csv = out_dir / f"{extra.plot_prefix}_scale_weight_summary_fold{fold_id}.csv"
    _write_scale_summary_csv(scale_summary_csv, data["scale_weights"], patch_sizes)
    print(f"[DATA] wrote: {scale_summary_csv}")

    metrics_csv = out_dir / f"{extra.plot_prefix}_metrics_fold{fold_id}.csv"
    _write_metrics_csv(metrics_csv, data["metrics"], ckpt_path, fold_id)
    print(f"[DATA] wrote: {metrics_csv}")

    try:
        _plot_outputs(out_dir, args.task, f"{extra.plot_prefix}_fold{fold_id}", data["scale_weights"], patch_sizes, extra.max_plot_samples)
    except Exception as exc:
        print(f"[PLOT][WARN] failed to write plots: {exc}")
    print("[METRICS]", data["metrics"])


if __name__ == "__main__":
    main()
