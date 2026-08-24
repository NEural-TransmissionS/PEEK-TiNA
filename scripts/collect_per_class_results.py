#!/usr/bin/env python3
"""Collect per-class metrics from every completed frozen test evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/per-class-test-results.csv"


def main() -> None:
    records = []
    for path in sorted((ROOT / "runs").glob("*_test/*/metrics.json")):
        metrics = json.loads(path.read_text())
        for class_metrics in metrics.get("per_class", []):
            records.append({"run": path.parent.name, **class_metrics})
    if not records:
        raise SystemExit("No structured per-class test metrics found")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {len(records)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
