"""Audit WSD and generate a portable Ultralytics dataset manifest."""

from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import yaml

CLASSES = ["antenna", "body", "solar", "thruster"]
SPLITS = {"train": "train", "val": "valid", "test": "test"}


def audit_wsd(source: Path) -> dict:
    source = source.expanduser().resolve()
    counts = {}
    problems = []
    content_hashes = defaultdict(list)
    for key, folder in SPLITS.items():
        images_dir, labels_dir = source / folder / "images", source / folder / "labels"
        images = [p for p in images_dir.glob("*") if p.is_file()]
        labels = [p for p in labels_dir.glob("*.txt") if p.is_file()]
        image_stems, label_stems = {p.stem for p in images}, {p.stem for p in labels}
        missing_labels = sorted(image_stems - label_stems)
        orphan_labels = sorted(label_stems - image_stems)
        counts[key] = {"images": len(images), "labels": len(labels)}
        for image in images:
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            content_hashes[digest].append(f"{key}/{image.name}")
        if missing_labels:
            problems.append(f"{key}: {len(missing_labels)} images lack labels")
        if orphan_labels:
            problems.append(f"{key}: {len(orphan_labels)} labels lack images")
    if problems:
        raise ValueError("; ".join(problems))
    exact_duplicates = [paths for paths in content_hashes.values() if len(paths) > 1]
    return {
        "path": str(source),
        **{k: f"{v}/images" for k, v in SPLITS.items()},
        "names": CLASSES,
        "nc": 4,
        "counts": counts,
        "exact_duplicate_groups": exact_duplicates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_wsd(args.source)
    counts = result.pop("counts")
    duplicate_groups = result.pop("exact_duplicate_groups")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(result, sort_keys=False))
    print(yaml.safe_dump(counts, sort_keys=False).strip())
    print(f"Exact SHA-256 duplicate groups: {len(duplicate_groups)}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
