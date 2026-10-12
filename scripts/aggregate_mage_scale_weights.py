#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


WEIGHT_RE = re.compile(r"test_agg_weight_scale_(\d+)$")
PATCH_SIZES = [50, 100, 200]


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    # Pool fold observations by dataset and scale across every input CSV.  This
    # also preserves the old behavior when there is one CSV per dataset.
    grouped: Dict[Tuple[str, int], Dict[str, object]] = {}
    for input_name in args.inputs:
        path = Path(input_name)
        rows = [
            row for row in _read_rows(path)
            if str(row.get("fold_number", "")).upper() != "MEAN"
            and str(row.get("fold_test_subject_id", "")).upper() != "MEAN"
            and str(row.get("val_fold_id", "")).upper() != "MEAN"
        ]
        if not rows:
            continue

        dataset = rows[0].get("dataset") or path.stem
        weight_keys = sorted(
            [key for key in rows[0] if WEIGHT_RE.match(key)],
            key=lambda key: int(WEIGHT_RE.match(key).group(1)),
        )
        for key in weight_keys:
            scale_idx = int(WEIGHT_RE.match(key).group(1))
            values = np.array([float(row[key]) for row in rows if row.get(key) not in {"", None}], dtype=float)
            if values.size == 0:
                continue
            group = grouped.setdefault((dataset, scale_idx), {
                "dataset": dataset,
                "scale_index": scale_idx,
                "patch_size": PATCH_SIZES[scale_idx] if scale_idx < len(PATCH_SIZES) else "",
                "values": [],
                "sources": [],
            })
            group["values"].extend(values.tolist())
            group["sources"].append(str(path))

    summary_rows: List[Dict[str, object]] = []
    for (_, _), group in sorted(grouped.items()):
        values = np.asarray(group.pop("values"), dtype=float)
        sources = group.pop("sources")
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if values.size > 1 else 0.0
        summary_rows.append({
            **group,
            "n_folds": int(values.size),
            "mean": mean,
            "std": std,
            "mean_pm_std": f"{mean:.6f}+-{std:.6f}",
            "source_csv": ";".join(sources),
        })

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "scale_index", "patch_size", "n_folds", "mean", "std", "mean_pm_std", "source_csv"]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[DONE] wrote {len(summary_rows)} row(s): {out}")


if __name__ == "__main__":
    main()
