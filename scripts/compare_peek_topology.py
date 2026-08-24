#!/usr/bin/env python3
"""Compare matched reference/candidate PEEK-map trees and aggregate topology drift."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def map_path(root: Path, image: str, module: int) -> Path:
    return root / Path(image).stem / f"module_{module}.npy"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "docs/wsd-v71-peek-calibration.json"
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pruning_yolo26.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    from topoprun.topology import (
        TopologyDistanceRecord,
        aggregate_topology_distances,
        compare_diagrams,
        persistence_diagrams,
    )

    manifest = json.loads(args.manifest.read_text())
    images = tuple(item["image"] for item in manifest["items"])
    protocol = yaml.safe_load(args.config.read_text())["pruning"]
    modules = tuple(protocol["modules"])
    records = []
    for image in images:
        for module in modules:
            reference_path = map_path(args.reference_root, image, module)
            candidate_path = map_path(args.candidate_root, image, module)
            if not reference_path.is_file() or not candidate_path.is_file():
                raise FileNotFoundError(
                    f"missing matched map: reference={reference_path}, candidate={candidate_path}"
                )
            reference = persistence_diagrams(np.load(reference_path), filtration="superlevel")
            candidate = persistence_diagrams(np.load(candidate_path), filtration="superlevel")
            distances = compare_diagrams(reference, candidate, wasserstein_order=2.0)
            for dimension in (0, 1):
                records.append(
                    TopologyDistanceRecord(
                        image=image,
                        module=module,
                        dimension=dimension,
                        bottleneck=distances[dimension]["bottleneck"],
                        wasserstein_2=distances[dimension]["wasserstein_2"],
                    )
                )
    aggregate = aggregate_topology_distances(
        records,
        images,
        modules,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    result = {
        "reference_root": str(args.reference_root),
        "candidate_root": str(args.candidate_root),
        "manifest": str(args.manifest),
        "filtration": "superlevel",
        "records": [record.as_dict() for record in records],
        "aggregate": aggregate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "records": len(records),
                "images": len(images),
                "modules": list(modules),
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
