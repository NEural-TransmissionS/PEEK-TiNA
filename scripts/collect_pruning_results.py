#!/usr/bin/env python3
"""Validate pruning records, apply frozen gates, and write reproducible tables."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pruning_yolo26.yaml")
    parser.add_argument("--records", type=Path, nargs="+", required=True)
    parser.add_argument("--reference-map", type=float, required=True)
    parser.add_argument("--table", type=Path, default=ROOT / "artifacts/pruning-results.csv")
    parser.add_argument("--selection", type=Path, default=ROOT / "artifacts/pruning-selection.json")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    from topoprun.pruning import CandidateMeasurement, PruningProtocol, evaluate_candidate

    config = yaml.safe_load(args.config.read_text())
    protocol = PruningProtocol.from_dict(config["pruning"])
    rows = []
    for path in args.records:
        payload = json.loads(path.read_text())
        candidate = CandidateMeasurement.from_dict(payload)
        evaluation = evaluate_candidate(candidate, protocol, args.reference_map)
        rows.append(
            {
                **evaluation.as_dict(),
                "method": candidate.method,
                "target_flops_reduction": candidate.target_flops_reduction,
                "actual_flops_reduction": candidate.actual_flops_reduction,
                "actual_parameter_reduction": candidate.actual_parameter_reduction,
                "validation_map50_95": candidate.validation_map50_95,
                "peek_variance_retention": candidate.peek_variance_retention,
                "checkpoint": candidate.checkpoint,
                "record": str(path),
            }
        )
    if len({row["candidate_id"] for row in rows}) != len(rows):
        raise ValueError("candidate_id values must be unique")
    rows.sort(key=lambda row: (row["target_flops_reduction"], row["method"]))
    args.table.parent.mkdir(parents=True, exist_ok=True)
    with args.table.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    winners = {}
    for target in protocol.target_flops_reductions:
        eligible = [
            row
            for row in rows
            if row["accepted"] and abs(row["target_flops_reduction"] - target) < 1e-9
        ]
        if eligible:
            winners[f"{target:.2f}"] = min(eligible, key=lambda row: row["score"])
    args.selection.parent.mkdir(parents=True, exist_ok=True)
    args.selection.write_text(
        json.dumps(
            {
                "selection_split": "val",
                "test_split_used": False,
                "reference_validation_map50_95": args.reference_map,
                "criterion": "lowest composite score among candidates passing all frozen gates",
                "winners": winners,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {len(rows)} records and {len(winners)} budget winners")


if __name__ == "__main__":
    main()
