#!/usr/bin/env python3
"""Select a deterministic, metadata-stratified WSD train calibration set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict, deque
from pathlib import Path

CLASSES = ["antenna", "body", "solar", "thruster"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=Path("docs/wsd-v71-robustness-slices.csv"))
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("docs/wsd-v71-peek-calibration.json"))
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    metadata = {
        row["image"]: row for row in csv.DictReader(args.metadata.open()) if row["split"] == "train"
    }
    class_frequency = Counter()
    items = []
    for image, row in sorted(metadata.items()):
        label = source / "train/labels" / f"{Path(image).stem}.txt"
        classes = sorted({int(line.split()[0]) for line in label.read_text().splitlines() if line.strip()})
        class_frequency.update(classes)
        items.append((image, row, classes))
    groups = defaultdict(list)
    for image, row, classes in items:
        primary = min(classes, key=lambda class_id: (class_frequency[class_id], class_id)) if classes else -1
        key = (row["slice"], "background" if primary == -1 else CLASSES[primary])
        rank = hashlib.sha256(f"{args.seed}:{image}".encode()).hexdigest()
        groups[key].append((rank, image, row, classes))
    queues = {key: deque(sorted(values)) for key, values in groups.items()}
    selected = []
    keys = sorted(queues)
    while len(selected) < args.count and any(queues.values()):
        for key in keys:
            if queues[key] and len(selected) < args.count:
                _, image, row, classes = queues[key].popleft()
                selected.append(
                    {
                        "image": image, "sha256": row["sha256"], "slice": row["slice"],
                        "classes": [CLASSES[class_id] for class_id in classes],
                    }
                )
    payload = {
        "dataset": "WSD v71 train split", "seed": args.seed, "count": len(selected),
        "selection": (
            "round-robin over robustness-slice and rarest-present-class strata; "
            "SHA-256(seed:image) order within strata"
        ),
        "correct_detections_only": False,
        "empty_scene_policy": "background stratum retained when selected by round-robin",
        "items": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {len(selected)} images to {args.output}")


if __name__ == "__main__":
    main()
