"""CSV aggregation and plotting helpers for MAGE evaluation."""

from __future__ import annotations

import argparse

import csv

import re

import statistics

import subprocess

import sys

from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

AGGREGATOR = REPO_ROOT / "scripts/aggregate_mage_scale_weights.py"

SEED_RE = re.compile(r"results_seed(\d+)$", re.IGNORECASE)

WEIGHT_RE = re.compile(r"test_agg_weight_scale_(\d+)$")

PATCH_SIZES = [50, 100, 200]

TIMESCALE_LABELS = {50: "Timescale 250 ms", 100: "Timescale 500 ms", 200: "Timescale 1 s"}

DISPLAY_DATASET_NAMES = {
    "FACED": "FACED",
    "PHYSIOMI": "PhysioNet-MI",
    "SHUMI": "SHU-MI",
    "SLEEPEDF": "Sleep-EDFx",
    "KAGGLEERN": "KaggleERN",
    "TUAB": "TUAB",
    "BCIC2A": "BCIC2a",
}

def seed_in_path(path: Path) -> int:
    for parent in (path, *path.parents):
        match = SEED_RE.fullmatch(parent.name)
        if match:
            return int(match.group(1))
    return 100

def combine_fold_csvs(csv_paths: list[Path], source_info: dict[Path, tuple[int, Path]], output: Path) -> None:
    rows: list[dict[str, str | int]] = []
    for csv_path in csv_paths:
        source_index, source_folder = source_info[csv_path]
        with csv_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("fold_number", "")).upper() == "MEAN":
                    continue
                rows.append({
                    "source_index": source_index,
                    "seed": seed_in_path(source_folder),
                    "source_folder": str(source_folder),
                    **row,
                })
    if not rows:
        raise ValueError("No fold rows were produced")
    fields = ["source_index", "seed", "source_folder"] + sorted(
        {key for row in rows for key in row if key not in {"source_index", "seed", "source_folder"}}
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[FOLDS] {output} ({len(rows)} fold rows)", flush=True)

def write_seed_fold_weights(
    csv_paths: list[Path], source_info: dict[Path, tuple[int, Path]], output: Path
) -> None:
    """Write one explicit row per seed and fold, with scales in columns."""
    rows: list[dict[str, object]] = []
    for csv_path in csv_paths:
        _, source_folder = source_info[csv_path]
        seed = seed_in_path(source_folder)
        with csv_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                fold = str(row.get("fold_number", ""))
                if not fold or fold.upper() == "MEAN":
                    continue
                rows.append({
                    "dataset": row.get("dataset", ""),
                    "seed": seed,
                    "fold_number": fold,
                    "num_test_samples": row.get("num_test_samples", ""),
                    "mean_weight_patch_50": row.get("test_agg_weight_scale_0", ""),
                    "mean_weight_patch_100": row.get("test_agg_weight_scale_1", ""),
                    "mean_weight_patch_200": row.get("test_agg_weight_scale_2", ""),
                    "weight_sum": row.get("test_agg_weight_sum", ""),
                    "checkpoint": row.get("checkpoint", ""),
                    "source_folder": str(source_folder),
                })
    rows.sort(key=lambda row: (str(row["dataset"]), int(row["seed"]), int(row["fold_number"])))
    fields = [
        "dataset", "seed", "fold_number", "num_test_samples",
        "mean_weight_patch_50", "mean_weight_patch_100", "mean_weight_patch_200",
        "weight_sum", "checkpoint", "source_folder",
    ]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[SEED/FOLD] {output} ({len(rows)} rows)", flush=True)

def summarize_by_seed(
    csv_paths: list[Path], source_info: dict[Path, tuple[int, Path]], output: Path
) -> None:
    """Compute mean/std across folds separately inside each seed."""
    grouped: dict[tuple[str, int, int], list[float]] = {}
    for csv_path in csv_paths:
        _, source_folder = source_info[csv_path]
        seed = seed_in_path(source_folder)
        with csv_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("fold_number", "")).upper() == "MEAN":
                    continue
                dataset = str(row.get("dataset", ""))
                for key, raw_value in row.items():
                    match = WEIGHT_RE.fullmatch(key)
                    if match and raw_value not in {None, ""}:
                        grouped.setdefault((dataset, seed, int(match.group(1))), []).append(float(raw_value))

    rows = []
    for (dataset, seed, scale_index), values in sorted(grouped.items()):
        mean = statistics.fmean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        rows.append({
            "dataset": dataset,
            "seed": seed,
            "scale_index": scale_index,
            "patch_size": PATCH_SIZES[scale_index] if scale_index < len(PATCH_SIZES) else "",
            "n_folds": len(values),
            "mean": mean,
            "std": std,
            "mean_pm_std": f"{mean:.6f}+-{std:.6f}",
        })
    fields = ["dataset", "seed", "scale_index", "patch_size", "n_folds", "mean", "std", "mean_pm_std"]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[SEED SUMMARY] {output} ({len(rows)} rows)", flush=True)

def summarize_matching_folds(csv_paths: list[Path], output: Path) -> None:
    """Average the same fold ID across input folders/seeds."""
    grouped: dict[tuple[str, str, int], list[float]] = {}
    for csv_path in csv_paths:
        with csv_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                fold = str(row.get("fold_number", ""))
                if not fold or fold.upper() == "MEAN":
                    continue
                dataset = str(row.get("dataset", ""))
                for key, raw_value in row.items():
                    match = WEIGHT_RE.fullmatch(key)
                    if match and raw_value not in {None, ""}:
                        grouped.setdefault((dataset, fold, int(match.group(1))), []).append(float(raw_value))

    rows = []
    for (dataset, fold, scale_index), values in sorted(
        grouped.items(), key=lambda item: (item[0][0], int(item[0][1]), item[0][2])
    ):
        mean = statistics.fmean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        rows.append({
            "dataset": dataset,
            "fold_number": fold,
            "scale_index": scale_index,
            "patch_size": PATCH_SIZES[scale_index] if scale_index < len(PATCH_SIZES) else "",
            "n_folders": len(values),
            "mean": mean,
            "std": std,
            "mean_pm_std": f"{mean:.6f}+-{std:.6f}",
        })
    fields = ["dataset", "fold_number", "scale_index", "patch_size", "n_folders", "mean", "std", "mean_pm_std"]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[MATCHING FOLDS] {output} ({len(rows)} rows)", flush=True)

def combine_dataset_sample_outputs(
    csv_paths: list[Path],
    source_info: dict[Path, tuple[int, Path]],
    output_dir: Path,
) -> None:
    """Pool sample weights from every supplied seed/fold and plot per dataset."""
    by_dataset: dict[str, list[dict[str, object]]] = {}
    sample_file_re = re.compile(r"(.+)_fold(\d+)_sample_weights\.csv$")
    for result_csv in csv_paths:
        _, source_folder = source_info[result_csv]
        seed = seed_in_path(source_folder)
        sample_dir = result_csv.parent / f"{result_csv.stem}_sample_outputs"
        for sample_csv in sorted(sample_dir.glob("*_fold*_sample_weights.csv")):
            match = sample_file_re.fullmatch(sample_csv.name)
            if not match:
                continue
            dataset = match.group(1).upper()
            fold = int(match.group(2))
            with sample_csv.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    by_dataset.setdefault(dataset, []).append({
                        "dataset": dataset,
                        "seed": seed,
                        "fold_number": fold,
                        "sample_index_in_fold": int(row["sample_index"]),
                        "label": int(row["label"]),
                        "pred": int(row["pred"]),
                        "weight_patch_50": float(row["weight_patch_50"]),
                        "weight_patch_100": float(row["weight_patch_100"]),
                        "weight_patch_200": float(row["weight_patch_200"]),
                    })

    if not by_dataset:
        print("[SAMPLES][WARN] no per-sample CSVs found; dataset-level plots were not created", flush=True)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset", "seed", "fold_number", "sample_index_in_fold", "label", "pred",
        "weight_patch_50", "weight_patch_100", "weight_patch_200",
    ]
    for dataset, rows in sorted(by_dataset.items()):
        rows.sort(key=lambda row: (int(row["seed"]), int(row["fold_number"]), int(row["sample_index_in_fold"])))
        combined_csv = output_dir / f"{dataset.lower()}_all_samples_scale_weights.csv"
        with combined_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

        summary_rows = []
        arrays = []
        for patch in PATCH_SIZES:
            values = np.asarray([float(row[f"weight_patch_{patch}"]) for row in rows], dtype=float)
            arrays.append(values)
            summary_rows.append({
                "dataset": dataset,
                "patch_size": patch,
                "n_samples": values.size,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "min": float(values.min()),
                "p05": float(np.quantile(values, 0.05)),
                "median": float(np.median(values)),
                "p95": float(np.quantile(values, 0.95)),
                "max": float(values.max()),
            })
        summary_csv = output_dir / f"{dataset.lower()}_all_samples_summary.csv"
        with summary_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
            writer.writeheader()
            writer.writerows(summary_rows)

        plot_path = write_sample_histogram(dataset, arrays, output_dir)
        print(f"[DATASET SAMPLES] wrote: {combined_csv}", flush=True)
        print(f"[DATASET SAMPLES] wrote: {summary_csv}", flush=True)
        print(f"[DATASET PLOT] wrote: {plot_path}", flush=True)

def write_sample_histogram(dataset: str, arrays: list[np.ndarray], output_dir: Path) -> Path:
    """Write a publication-readable three-timescale histogram for one dataset."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(PATCH_SIZES), 1, figsize=(13, 9), sharex=True)
    for axis, patch, values in zip(axes, PATCH_SIZES, arrays):
        axis.hist(values, bins=50, range=(0.0, 1.0), alpha=0.82)
        axis.axvline(
            values.mean(),
            color="black",
            linestyle="--",
            linewidth=2.0,
            label=f"mean={values.mean():.3f}, median={np.median(values):.3f}",
        )
        axis.set_title(
            TIMESCALE_LABELS.get(patch, f"Patch {patch}"),
            loc="left",
            fontsize=22,
            fontweight="semibold",
            pad=10,
        )
        axis.set_ylabel("Samples", fontsize=18)
        axis.tick_params(axis="both", labelsize=16)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=18, framealpha=0.9)
    axes[-1].set_xlabel("Aggregator softmax weight", fontsize=18, labelpad=9)
    display_name = DISPLAY_DATASET_NAMES.get(dataset, dataset)
    fig.suptitle(
        f'Histogram of weight distribution of dataset "{display_name}"',
        fontsize=22,
        fontweight="bold",
        y=0.985,
    )
    # Keep identical panel geometry for datasets whose y tick labels vary from
    # hundreds to hundreds of thousands; automatic layout can otherwise crush
    # the vertical gaps and hide the per-timescale titles.
    for axis, bottom in zip(axes, (0.68, 0.39, 0.10)):
        axis.set_position((0.12, bottom, 0.855, 0.20))
    plot_path = output_dir / f"{dataset.lower()}_all_samples_histograms.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    return plot_path
