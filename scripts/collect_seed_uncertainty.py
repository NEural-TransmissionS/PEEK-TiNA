#!/usr/bin/env python3
"""Collect overall and per-class uncertainty across frozen final seeds."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from topoprun.uncertainty import summarize_seeds

SEED_PATTERN = re.compile(r"_seed(?P<seed>\d+)$")
METRIC_KEYS = {
    "precision": ("metrics/precision", "metrics/precision(B)"),
    "recall": ("metrics/recall", "metrics/recall(B)"),
    "mAP50": ("metrics/mAP50", "metrics/mAP50(B)"),
    "mAP50-95": ("metrics/mAP50-95", "metrics/mAP50-95(B)"),
}


def first(payload: dict, names: tuple[str, ...]) -> float:
    for name in names:
        if name in payload:
            return float(payload[name])
    raise KeyError(f"Missing all metric keys {names}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/confirmation_seeds.yaml")
    parser.add_argument("--runs", type=Path, default=ROOT / "runs/final_nano_test")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/final-seed-uncertainty.csv")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    grouped = defaultdict(list)
    observed_seeds = defaultdict(set)
    for path in sorted(args.runs.glob("*/metrics.json")):
        match = SEED_PATTERN.search(path.parent.name)
        if not match:
            continue
        seed = int(match.group("seed"))
        condition = SEED_PATTERN.sub("", path.parent.name)
        payload = json.loads(path.read_text())
        observed_seeds[condition].add(seed)
        for metric, keys in METRIC_KEYS.items():
            grouped[(condition, "all", metric)].append(first(payload, keys))
        for item in payload.get("per_class", []):
            for metric in METRIC_KEYS:
                grouped[(condition, item["class"], metric)].append(float(item[metric]))
    records = []
    for (condition, class_name, metric), values in sorted(grouped.items()):
        summary = summarize_seeds(
            values,
            confidence=config["confidence_level"],
            minimum_reportable=config["minimum_reportable_seeds"],
        )
        records.append(
            {
                "condition": condition, "class": class_name, "metric": metric,
                "seeds": ";".join(map(str, sorted(observed_seeds[condition]))),
                **summary.as_dict(),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "condition", "class", "metric", "seeds", "count", "mean", "standard_deviation",
        "confidence_level", "confidence_low", "confidence_high", "reportable",
    ]
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {len(records)} uncertainty rows to {args.output}")


if __name__ == "__main__":
    main()
