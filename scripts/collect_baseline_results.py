#!/usr/bin/env python3
"""Collect best validation and frozen test metrics into a paper-ready CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "runs/baseline_nano"
TEST_ROOT = ROOT / "runs/baseline_nano_test"
OUTPUT = ROOT / "docs/baseline-nano-results.csv"
RUN_NAMES = [
    f"{family}n_{initialization}_seed42"
    for family in ("yolov5", "yolo26")
    for initialization in ("pretrained", "scratch")
]
MODEL_METADATA = {
    "yolov5": {"parameters": 1_769_329, "gflops": 4.2},
    "yolo26": {"parameters": 2_505_360, "gflops": 5.8},
}


def value(row: dict, *names: str) -> float:
    for name in names:
        if name in row:
            return float(row[name])
    raise KeyError(f"None of {names} found in {tuple(row)}")


def main() -> None:
    records = []
    for name in RUN_NAMES:
        training_csv = TRAIN_ROOT / name / "results.csv"
        test_json = TEST_ROOT / name / "metrics.json"
        if not training_csv.exists():
            continue
        with training_csv.open(newline="") as stream:
            rows = list(csv.DictReader(stream, skipinitialspace=True))
        best = max(
            rows,
            key=lambda row: value(row, "metrics/mAP_0.5:0.95", "metrics/mAP50-95(B)"),
        )
        test = json.loads(test_json.read_text()) if test_json.exists() else {}
        weights = TRAIN_ROOT / name / "weights/best.pt"
        family = name.split("n_")[0]
        records.append(
            {
                "run": name,
                "family": family,
                "initialization": "pretrained" if "pretrained" in name else "scratch",
                "seed": 42,
                **MODEL_METADATA[family],
                "epochs_completed": len(rows),
                "best_validation_epoch": int(float(best["epoch"]))
                + (1 if name.startswith("yolov5") else 0),
                "validation_mAP50": value(best, "metrics/mAP_0.5", "metrics/mAP50(B)"),
                "validation_mAP50-95": value(best, "metrics/mAP_0.5:0.95", "metrics/mAP50-95(B)"),
                "test_mAP50": test.get("metrics/mAP50", test.get("metrics/mAP50(B)")),
                "test_mAP50-95": test.get("metrics/mAP50-95", test.get("metrics/mAP50-95(B)")),
                "test_precision": test.get("metrics/precision", test.get("metrics/precision(B)")),
                "test_recall": test.get("metrics/recall", test.get("metrics/recall(B)")),
                "weights_mb": weights.stat().st_size / 1_000_000 if weights.exists() else None,
            }
        )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {len(records)} records to {OUTPUT}")


if __name__ == "__main__":
    main()
