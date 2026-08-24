#!/usr/bin/env python3
"""Capture raw post-activation latents for a WSD split (or all splits).

One pickle per image: {module_index: torch.Tensor}. Thin CLI wrapper around
PEEK's own peek.extractors.ultralytics.extract_ultralytics_latents, the same
entry point used in third_party/PEEK/notebooks/Ultralytics_Demo.ipynb.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SPLIT_KEYS = {"train": "train", "valid": "val", "test": "test"}
MODULE_INVENTORY = ROOT / "docs/yolo26n-peek-module-inventory.json"


def peek_compatible_modules() -> list[int]:
    """Every module documented as producing a genuine spatial BCHW map.

    Excludes e.g. the Detect head, whose (1, num_boxes, fields) output is
    not a spatial map even though it happens to be 3D after squeezing.
    """
    inventory = json.loads(MODULE_INVENTORY.read_text())
    return [m["index"] for m in inventory["modules"] if m.get("peek_compatible")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data/wsd.yaml")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=list(SPLIT_KEYS), default=list(SPLIT_KEYS))
    parser.add_argument(
        "--modules", type=int, nargs="+", default=None,
        help="Default: every module in docs/yolo26n-peek-module-inventory.json "
             "marked peek_compatible. An explicit list is used as-is, including "
             "any non-spatial module -- save_peek_maps.py will then raise on it.",
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    modules = args.modules if args.modules is not None else peek_compatible_modules()

    sys.path.insert(0, str(ROOT / "third_party/ultralytics"))
    sys.path.insert(0, str(ROOT / "third_party/PEEK"))
    from ultralytics import YOLO  # noqa: F401  (imported first: pins this sys.path's copy)
    from peek.extractors.ultralytics import extract_ultralytics_latents

    data = yaml.safe_load(args.data.read_text())
    dataset_root = Path(data["path"])
    weights = str(args.weights.resolve())

    for split in args.splits:
        image_dir = (dataset_root / data[SPLIT_KEYS[split]]).resolve(strict=True)
        out_dir = args.output_root.resolve() / split
        extract_ultralytics_latents(
            weights=weights,
            image_dir=str(image_dir),
            out_dir=str(out_dir),
            device=args.device,
            imgsz=args.imgsz,
            modules=modules,
            to_cpu=True,
            verbose=True,
        )


if __name__ == "__main__":
    main()
