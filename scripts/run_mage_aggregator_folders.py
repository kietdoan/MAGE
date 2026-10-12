#!/usr/bin/env python3
"""Export and summarize aggregator weights for standard MAGE folders."""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

from run_shared_encoder_scale_weights_folders import (
    AGGREGATOR,
    combine_dataset_sample_outputs,
    combine_fold_csvs,
    seed_in_path,
    summarize_by_seed,
    summarize_matching_folds,
    write_seed_fold_weights,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORTER = REPO_ROOT / "downstream/test_mage_multi_scale_visualize.py"
FOLD_RE = re.compile(r"(?:(?:fold|valFold)(\d+)|s(\d+)_.*)$")

DATASETS = {
    "FACED": {
        "split_dir": "4fold", "split": "4fold",
        "seq_len": 2000, "in_chans": 32, "workers": 10,
        "path_arg": "--faced_path",
        "data_path": os.environ.get("FACED_PATH", str(REPO_ROOT / "data/FACED")),
        "extra": [],
    },
    "SLEEPEDF": {
        "split_dir": "5fold", "split": "5fold",
        "seq_len": 6000, "in_chans": 2, "workers": 10,
        "path_arg": "--sleepedf_path",
        "data_path": os.environ.get("SLEEPEDF_PATH", str(REPO_ROOT / "data/SLEEPEDF")),
        "extra": ["--target_chans", "19", "--val_ratio", "0.2"],
    },
    "PhysioMI": {
        "split_dir": "K5fold_SUBJ_fixT4", "split": "loso",
        "seq_len": 800, "in_chans": 64, "workers": 10,
        "path_arg": "--physiomi_path",
        "data_path": os.environ.get("PHYSIOMI_PATH", str(REPO_ROOT / "data/PhysioMI")),
        "extra": [],
    },
    "SHUMI": {
        "split_dir": "5fold", "split": "5fold",
        "seq_len": 800, "in_chans": 32, "workers": 6,
        "path_arg": "--shumi_path",
        "data_path": os.environ.get("SHUMI_PATH", str(REPO_ROOT / "data/SHUMI")),
        "extra": [],
    },
    "KaggleERN": {
        "split_dir": "5fold", "split": "5fold",
        "seq_len": 400, "in_chans": 19, "workers": 6,
        "path_arg": "--kaggle_ern_path",
        "data_path": os.environ.get("KAGGLE_ERN_PATH", str(REPO_ROOT / "data/KaggleERN")),
        "extra": [],
    },
    "TUAB": {
        "split_dir": "4fold", "split": "4fold",
        "seq_len": 2000, "in_chans": 23, "workers": 6,
        "path_arg": "--tuab_path",
        "data_path": os.environ.get("TUAB_PATH", str(REPO_ROOT / "data/TUAB")),
        "extra": [],
    },
    "BCIC2a": {
        "split_dir": "LOSO", "split": "loso",
        "seq_len": 800, "in_chans": 22, "workers": 6,
        "path_arg": "--bcic2a_path",
        "data_path": os.environ.get("BCIC2A_PATH", str(REPO_ROOT / "data/BCIC2a")),
        "extra": [],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT.parent / "MAGE_private_artifacts/aggregator_eval_selected")
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def dataset_from_path(path: Path) -> str | None:
    lookup = {name.upper(): name for name in DATASETS}
    for part in reversed(path.parts):
        if part.upper() in lookup:
            return lookup[part.upper()]
    return None


def discover(folders: list[Path]) -> list[tuple[int, Path, str, Path, list[tuple[int, Path]]]]:
    groups = []
    seen: set[Path] = set()
    for source_index, raw_root in enumerate(folders, start=1):
        root = raw_root.expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"Folder does not exist: {root}")
        split_dirs: dict[Path, dict[int, Path]] = {}
        for ckpt in sorted(root.rglob("*.ckpt")):
            if not (ckpt.name.startswith("best-") or ckpt.name == "last.ckpt"):
                continue
            if ckpt.parent.name != "ckpts":
                continue
            match = FOLD_RE.fullmatch(ckpt.parent.parent.name)
            if not match:
                continue
            split_dir = ckpt.parent.parent.parent.resolve()
            dataset = dataset_from_path(split_dir)
            if dataset is None or split_dir.name != DATASETS[dataset]["split_dir"]:
                continue
            fold = int(match.group(1) or match.group(2))
            previous = split_dirs.setdefault(split_dir, {}).get(fold)
            if previous is not None and previous.name.startswith("best-"):
                if ckpt.name.startswith("best-"):
                    raise ValueError(f"Multiple best checkpoints for {split_dir}/{ckpt.parent.parent.name}")
                continue
            split_dirs[split_dir][fold] = ckpt.resolve()
        for split_dir, fold_map in sorted(split_dirs.items()):
            if split_dir in seen:
                continue
            seen.add(split_dir)
            dataset = dataset_from_path(split_dir)
            assert dataset is not None
            groups.append((source_index, root, dataset, split_dir, sorted(fold_map.items())))
    if not groups:
        raise ValueError("No supported MAGE checkpoints found")
    return groups


def python_for(dataset: str) -> str:
    del dataset  # Run the exporter with the same interpreter as this orchestrator.
    return sys.executable


def run_fold(dataset: str, seed: int, fold: int, ckpt: Path, raw_dir: Path, args: argparse.Namespace) -> None:
    cfg = DATASETS[dataset]
    workers = args.num_workers if args.num_workers is not None else cfg["workers"]
    command = [
        python_for(dataset), str(EXPORTER),
        "--ckpt", str(ckpt), "--output_dir", str(raw_dir), "--fold", str(fold),
        "--plot_prefix", dataset.lower(), "--task", dataset,
        "--scheme", "mage", "--split", str(cfg["split"]),
        "--tag", "custom_64d_6L", "--batch_size", str(args.batch_size),
        "--epochs", "1", "--max_lr_head", "1e-3", "--max_lr_enc", "1e-4",
        "--num_workers", str(workers), "--seq_len", str(cfg["seq_len"]),
        "--devices", str(args.device), "--freeze_encoder",
        "--mage_patch_sizes", "50", "100", "200",
        "--mage_k_temporal", "3", "--mage_temporal_extra_convs", "2",
        "--mage_use_spectral", "--mage_use_channel_attn",
        "--mage_aggregator_hidden_dim", "32", "--mage_use_frequency",
        "--mage_aggregator_freq_dim", "16", "--mage_no_inter_attn_enc",
        "--rope_time", "--scale_div", "1.0", "--use_chan_conv", "--chan_max_norm", "1.0",
        "--head_type", "eegpt_probe", "--dropout", "0.5", "--in_chans", str(cfg["in_chans"]),
        "--seed", str(seed), "--val_seed", str(seed), str(cfg["path_arg"]), str(cfg["data_path"]),
        "--scale_dropout", "0.0", "--agg_noise_std", "0.0", "--entropy_lambda", "0.0",
        *cfg["extra"],
    ]
    raw_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def convert_fold(dataset: str, fold: int, raw_dir: Path, sample_dir: Path) -> tuple[dict[str, object], int]:
    prefix = dataset.lower()
    raw_samples = raw_dir / f"{prefix}_aggregator_outputs_fold{fold}.csv"
    raw_summary = raw_dir / f"{prefix}_scale_weight_summary_fold{fold}.csv"
    raw_metrics = raw_dir / f"{prefix}_metrics_fold{fold}.csv"

    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_out = sample_dir / f"{prefix}_fold{fold}_sample_weights.csv"
    sample_count = 0
    with raw_samples.open(newline="") as src, sample_out.open("w", newline="") as dst:
        reader = csv.DictReader(src)
        fields = ["sample_index", "label", "pred", "weight_patch_50", "weight_patch_100", "weight_patch_200"]
        writer = csv.DictWriter(dst, fieldnames=fields)
        writer.writeheader()
        for row in reader:
            writer.writerow({
                "sample_index": row["sample_index"], "label": row["label"], "pred": row["pred"],
                "weight_patch_50": row["scale_weight_p50"],
                "weight_patch_100": row["scale_weight_p100"],
                "weight_patch_200": row["scale_weight_p200"],
            })
            sample_count += 1

    with raw_summary.open(newline="") as handle:
        summaries = {int(row["scale_index"]): row for row in csv.DictReader(handle)}
    with raw_metrics.open(newline="") as handle:
        metrics = next(csv.DictReader(handle))
    result: dict[str, object] = {
        "dataset": dataset, "strategy": DATASETS[dataset]["split"], "fold_number": fold,
        "checkpoint": metrics.pop("checkpoint"), "num_test_samples": sample_count,
    }
    metrics.pop("fold", None)
    result.update(metrics)
    for index in range(3):
        result[f"test_agg_weight_scale_{index}"] = summaries[index]["mean"]
    result["test_agg_weight_sum"] = sum(float(summaries[index]["mean"]) for index in range(3))
    return result, sample_count


def write_group_csv(path: Path, rows: list[dict[str, object]]) -> None:
    mean_row: dict[str, object] = {
        "dataset": rows[0]["dataset"], "strategy": rows[0]["strategy"],
        "fold_number": "MEAN", "checkpoint": "",
    }
    numeric = [key for key in rows[0] if key not in {"dataset", "strategy", "fold_number", "checkpoint"}]
    for key in numeric:
        try:
            mean_row[key] = sum(float(row[key]) for row in rows) / len(rows)
        except (KeyError, TypeError, ValueError):
            pass
    all_rows = [*rows, mean_row]
    fields = sorted({key for row in all_rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id and (output_dir / f".supersede_job_{job_id}").is_file():
        print(f"[SUPERSEDED] Slurm job {job_id} was replaced; exiting before inference.", flush=True)
        return
    try:
        groups = discover(args.folders)
    except ValueError as error:
        raise SystemExit(f"error: {error}") from error

    csv_paths: list[Path] = []
    source_info: dict[Path, tuple[int, Path]] = {}
    print(f"[DISCOVERED] {len(groups)} dataset folder(s)", flush=True)
    for group_number, (source_index, source_root, dataset, split_dir, folds) in enumerate(groups, start=1):
        group_stem = f"{dataset.lower()}_source{source_index:02d}_group{group_number:02d}"
        group_csv = output_dir / "per_folder" / f"{group_stem}.csv"
        sample_dir = output_dir / "per_folder" / f"{group_stem}_sample_outputs"
        print(f"[GROUP] dataset={dataset} seed={seed_in_path(source_root)} split={split_dir}", flush=True)
        for fold, ckpt in folds:
            print(f"  [CKPT] fold={fold} {ckpt}", flush=True)
        if args.dry_run:
            continue
        rows = []
        for fold, ckpt in folds:
            raw_dir = sample_dir / f"raw_fold{fold}"
            expected = raw_dir / f"{dataset.lower()}_aggregator_outputs_fold{fold}.csv"
            if args.skip_existing and expected.is_file():
                print(f"[SKIP] reuse {expected}", flush=True)
            else:
                run_fold(dataset, seed_in_path(source_root), fold, ckpt, raw_dir, args)
            row, _ = convert_fold(dataset, fold, raw_dir, sample_dir)
            rows.append(row)
        write_group_csv(group_csv, rows)
        csv_paths.append(group_csv)
        source_info[group_csv] = (source_index, source_root)

    if args.dry_run:
        return
    combine_fold_csvs(csv_paths, source_info, output_dir / "all_fold_scale_weights.csv")
    write_seed_fold_weights(csv_paths, source_info, output_dir / "scale_weight_by_seed_and_fold.csv")
    summarize_by_seed(csv_paths, source_info, output_dir / "scale_weight_mean_std_by_seed.csv")
    summarize_matching_folds(csv_paths, output_dir / "scale_weight_mean_std_by_fold_across_seeds.csv")
    combine_dataset_sample_outputs(csv_paths, source_info, output_dir / "all_sample_distributions")
    subprocess.run([
        sys.executable, str(AGGREGATOR), "--inputs", *map(str, csv_paths),
        "--output_csv", str(output_dir / "scale_weight_mean_std_by_dataset.csv"),
    ], cwd=REPO_ROOT, check=True)
    print(f"[DONE] {output_dir}", flush=True)


if __name__ == "__main__":
    main()
