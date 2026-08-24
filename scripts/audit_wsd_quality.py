#!/usr/bin/env python3
"""Audit WSD image/label quality and assign deterministic robustness metalabels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageStat, UnidentifiedImageError

CLASSES = ["antenna", "body", "solar", "thruster"]
SPLITS = {"train": "train", "val": "valid", "test": "test"}
FEATURES = [
    "brightness", "contrast", "saturation", "box_count", "box_area_total",
    "box_area_median", "box_area_min", "object_extent",
]


def parse_label(path: Path) -> tuple[list[tuple[int, float, float, float, float]], list[str]]:
    boxes, errors = [], []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        fields = line.split()
        if len(fields) != 5:
            errors.append(f"{path}:{line_number}: expected 5 fields, got {len(fields)}")
            continue
        try:
            raw_class, *raw_box = map(float, fields)
        except ValueError:
            errors.append(f"{path}:{line_number}: non-numeric field")
            continue
        if not all(np.isfinite([raw_class, *raw_box])):
            errors.append(f"{path}:{line_number}: non-finite field")
            continue
        class_id = int(raw_class)
        x, y, width, height = raw_box
        if raw_class != class_id or not 0 <= class_id < len(CLASSES):
            errors.append(f"{path}:{line_number}: invalid class {raw_class}")
        if width <= 0 or height <= 0:
            errors.append(f"{path}:{line_number}: non-positive box size")
        if min(x - width / 2, y - height / 2) < -1e-6 or max(
            x + width / 2, y + height / 2
        ) > 1 + 1e-6:
            errors.append(f"{path}:{line_number}: box exceeds normalized image bounds")
        boxes.append((class_id, x, y, width, height))
    return boxes, errors


def image_features(path: Path, boxes: list[tuple[int, float, float, float, float]]) -> list[float]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        gray = rgb.convert("L")
        brightness = ImageStat.Stat(gray).mean[0] / 255
        contrast = ImageStat.Stat(gray).stddev[0] / 255
        hsv = np.asarray(rgb.convert("HSV"), dtype=np.float32)
        saturation = float(hsv[..., 1].mean() / 255)
    areas = np.array([box[3] * box[4] for box in boxes], dtype=float)
    if boxes:
        left = min(box[1] - box[3] / 2 for box in boxes)
        right = max(box[1] + box[3] / 2 for box in boxes)
        top = min(box[2] - box[4] / 2 for box in boxes)
        bottom = max(box[2] + box[4] / 2 for box in boxes)
        extent = (right - left) * (bottom - top)
    else:
        extent = 0.0
    return [
        brightness, contrast, saturation, float(len(boxes)), float(areas.sum()),
        float(np.median(areas)) if len(areas) else 0.0,
        float(areas.min()) if len(areas) else 0.0, extent,
    ]


def kmeans(features: np.ndarray, clusters: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    mean, scale = features.mean(0), features.std(0)
    standardized = (features - mean) / np.where(scale == 0, 1, scale)
    rng = np.random.default_rng(seed)
    centers = [standardized[rng.integers(len(standardized))]]
    for _ in range(1, clusters):
        distance = np.min(
            np.stack([np.sum((standardized - center) ** 2, axis=1) for center in centers]), axis=0
        )
        centers.append(standardized[rng.choice(len(standardized), p=distance / distance.sum())])
    centers = np.stack(centers)
    labels = np.zeros(len(features), dtype=int)
    for _ in range(100):
        distances = np.stack([np.sum((standardized - center) ** 2, axis=1) for center in centers])
        new_labels = distances.argmin(0)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        centers = np.stack([standardized[labels == i].mean(0) for i in range(clusters)])
    return labels, centers * np.where(scale == 0, 1, scale) + mean


def cluster_name(center: np.ndarray, medians: np.ndarray) -> str:
    light = "bright" if center[0] >= medians[0] else "dark"
    size = "large_objects" if center[5] >= medians[5] else "small_objects"
    density = "many_parts" if center[3] >= medians[3] else "few_parts"
    return f"{light}_{size}_{density}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/wsd-v71-quality-audit.json"))
    parser.add_argument("--metadata", type=Path, default=Path("docs/wsd-v71-robustness-slices.csv"))
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    rows, errors, class_counts = [], [], Counter()
    split_class_counts = {split: Counter() for split in SPLITS}
    split_counts = {}
    for split, folder in SPLITS.items():
        images = sorted((source / folder / "images").glob("*"))
        split_counts[split] = {"images": len(images), "background_images": 0}
        for image_path in images:
            label_path = source / folder / "labels" / f"{image_path.stem}.txt"
            boxes, label_errors = parse_label(label_path) if label_path.exists() else ([], [f"missing {label_path}"])
            errors.extend(label_errors)
            if not boxes:
                split_counts[split]["background_images"] += 1
            class_counts.update(box[0] for box in boxes if 0 <= box[0] < len(CLASSES))
            split_class_counts[split].update(
                box[0] for box in boxes if 0 <= box[0] < len(CLASSES)
            )
            try:
                features = image_features(image_path, boxes)
            except (OSError, SyntaxError, UnidentifiedImageError, ValueError) as exc:
                errors.append(f"{image_path}: corrupt/unreadable: {exc}")
                continue
            rows.append({
                "split": split, "image": image_path.name,
                "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                **dict(zip(FEATURES, features, strict=True)),
            })
    matrix = np.array([[row[key] for key in FEATURES] for row in rows], dtype=float)
    # Log-transform the count for clustering so two unusually dense annotations
    # do not consume an otherwise tiny outlier cluster. Report centroids below in
    # the original, directly interpretable units.
    cluster_matrix = matrix.copy()
    cluster_matrix[:, FEATURES.index("box_count")] = np.log1p(
        cluster_matrix[:, FEATURES.index("box_count")]
    )
    labels, _ = kmeans(cluster_matrix, args.clusters, args.seed)
    centers = np.stack([matrix[labels == i].mean(0) for i in range(args.clusters)])
    medians = np.median(matrix, axis=0)
    names, used = [], Counter()
    for center in centers:
        base = cluster_name(center, medians)
        used[base] += 1
        names.append(base if used[base] == 1 else f"{base}_{used[base]}")
    cluster_counts = Counter()
    for row, label in zip(rows, labels, strict=True):
        row["slice_id"] = int(label)
        row["slice"] = names[label]
        cluster_counts[(row["split"], names[label])] += 1
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    with args.metadata.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    audit = {
        "dataset": "WSD v71", "source": str(source), "seed": args.seed,
        "robustness_method": (
            "deterministic k-means on standardized image and annotation metadata; "
            "log1p transform applied to box_count for clustering"
        ),
        "feature_names": FEATURES, "split_counts": split_counts,
        "class_instance_counts": {CLASSES[i]: class_counts[i] for i in range(len(CLASSES))},
        "class_instance_counts_by_split": {
            split: {CLASSES[i]: split_class_counts[split][i] for i in range(len(CLASSES))}
            for split in SPLITS
        },
        "class_max_to_min_ratio": max(class_counts.values()) / min(class_counts.values()),
        "errors": errors, "valid": not errors,
        "slices": [
            {
                "id": i, "name": names[i],
                "centroid": dict(zip(FEATURES, map(float, center), strict=True)),
                "counts": {split: cluster_counts[(split, names[i])] for split in SPLITS},
            }
            for i, center in enumerate(centers)
        ],
        "metadata_csv": str(args.metadata),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
