#!/usr/bin/env python3
"""Collect validation-only tuning results and select each condition's winner."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = ROOT / "runs/baseline_nano"
TUNING_ROOT = ROOT / "runs/tuning_nano"
TABLE = ROOT / "docs/tuning-nano-validation-results.csv"
SELECTION = ROOT / "configs/tuning_nano_selection.json"


def value(row: dict, *names: str) -> float:
    for name in names:
        if name in row and row[name] != "":
            return float(row[name])
    raise KeyError(f"None of {names} found in {tuple(row)}")


def record(
    run_dir: Path, family: str, initialization: str, trial: str, complete: bool
) -> dict | None:
    path = run_dir / "results.csv"
    if not path.exists():
        return None
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream, skipinitialspace=True))
    if not rows:
        return None
    best = max(rows, key=lambda row: value(row, "metrics/mAP_0.5:0.95", "metrics/mAP50-95(B)"))
    return {
        "run": run_dir.name,
        "family": family,
        "initialization": initialization,
        "trial": trial,
        "complete": complete,
        "epochs_completed": len(rows),
        "best_validation_epoch": int(float(best["epoch"])) + (1 if family == "yolov5" else 0),
        "validation_mAP50": value(best, "metrics/mAP_0.5", "metrics/mAP50(B)"),
        "validation_mAP50-95": value(best, "metrics/mAP_0.5:0.95", "metrics/mAP50-95(B)"),
        "checkpoint": str(run_dir / "weights/best.pt"),
    }


def main() -> None:
    records = []
    for family in ("yolov5", "yolo26"):
        for initialization in ("pretrained", "scratch"):
            baseline = BASELINE_ROOT / f"{family}n_{initialization}_seed42"
            item = record(baseline, family, initialization, "native_default", True)
            if item:
                records.append(item)
            pattern = f"{family}n_{initialization}_*_seed42"
            for run_dir in sorted(TUNING_ROOT.glob(pattern)):
                trial = run_dir.name.removeprefix(f"{family}n_{initialization}_").removesuffix("_seed42")
                item = record(
                    run_dir,
                    family,
                    initialization,
                    trial,
                    (run_dir / "tuning-complete.json").exists(),
                )
                if item:
                    records.append(item)
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    if records:
        with TABLE.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=records[0].keys(), lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
    winners = {}
    for family in ("yolov5", "yolo26"):
        for initialization in ("pretrained", "scratch"):
            candidates = [
                row for row in records
                if row["family"] == family and row["initialization"] == initialization and row["complete"]
            ]
            if candidates:
                winners[f"{family}_{initialization}"] = max(
                    candidates, key=lambda row: row["validation_mAP50-95"]
                )
    SELECTION.write_text(json.dumps({"criterion": "validation_mAP50-95", "winners": winners}, indent=2) + "\n")
    print(f"Wrote {len(records)} candidates to {TABLE} and {len(winners)} selections to {SELECTION}")


if __name__ == "__main__":
    main()
