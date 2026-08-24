#!/usr/bin/env python3
"""Perform the frozen two-stage exact and perceptual WSD duplicate review."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

SPLITS = {"train": "train", "val": "valid", "test": "test"}


def difference_hash(gray: Image.Image) -> int:
    sample = gray.resize((9, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(sample, dtype=np.int16)
    bits = pixels[:, :-1] > pixels[:, 1:]
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return value


def normalized_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = first - first.mean()
    second = second - second.mean()
    denominator = np.sqrt(np.square(first).sum() * np.square(second).sum())
    return float((first * second).sum() / denominator) if denominator else 1.0


def review(source: Path, hash_distance: int = 2, correlation_threshold: float = 0.98) -> dict:
    records = []
    exact = defaultdict(list)
    split_fingerprints = {}
    for split, folder in SPLITS.items():
        fingerprint = hashlib.sha256()
        image_paths = sorted((source / folder / "images").iterdir())
        for path in image_paths:
            if not path.is_file():
                continue
            image_bytes = path.read_bytes()
            label_path = source / folder / "labels" / f"{path.stem}.txt"
            fingerprint.update(path.name.encode())
            fingerprint.update(image_bytes)
            fingerprint.update(label_path.name.encode())
            fingerprint.update(label_path.read_bytes())
            exact[hashlib.sha256(image_bytes).hexdigest()].append(f"{split}/{path.name}")
            with Image.open(path) as image:
                gray = ImageOps.exif_transpose(image).convert("L")
                comparison = np.asarray(
                    gray.resize((128, 128), Image.Resampling.BILINEAR), dtype=np.float32
                )
            records.append((split, path.name, difference_hash(gray), comparison))
        split_fingerprints[split] = fingerprint.hexdigest()

    candidates = []
    confirmed = []
    for index, first in enumerate(records):
        for second in records[index + 1 :]:
            if first[0] == second[0]:
                continue
            distance = (first[2] ^ second[2]).bit_count()
            if distance > hash_distance:
                continue
            correlation = normalized_correlation(first[3], second[3])
            item = {
                "first": f"{first[0]}/{first[1]}",
                "second": f"{second[0]}/{second[1]}",
                "dhash_distance": distance,
                "normalized_correlation": correlation,
            }
            candidates.append(item)
            if correlation >= correlation_threshold:
                confirmed.append(item)

    return {
        "method": {
            "exact": "SHA-256 of encoded image bytes",
            "perceptual_screen": "64-bit grayscale difference hash at 9x8",
            "screen_max_hamming_distance": hash_distance,
            "confirmation": "normalized grayscale correlation after 128x128 resize",
            "confirmation_min_correlation": correlation_threshold,
            "scope": "all cross-split image pairs",
        },
        "images": len(records),
        "split_fingerprints_sha256": split_fingerprints,
        "exact_duplicate_groups": [group for group in exact.values() if len(group) > 1],
        "perceptual_candidates": candidates,
        "confirmed_near_duplicate_pairs": confirmed,
        "conclusion": (
            "No exact or confirmed near-duplicate cross-split pairs."
            if not confirmed and all(len(group) == 1 for group in exact.values())
            else "Review confirmed duplicate pairs before using these splits."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = review(args.source.expanduser().resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "perceptual_candidates"},
            indent=2,
        )
    )
    print(f"Perceptual candidates reviewed: {len(report['perceptual_candidates'])}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
