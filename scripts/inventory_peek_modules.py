#!/usr/bin/env python3
"""Inventory official-PEEK-compatible YOLO26 hook outputs and shapes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/yolo26n-peek-module-inventory.json")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "third_party/ultralytics"))
    sys.path.insert(0, str(ROOT / "third_party/PEEK"))
    from peek.extractors.hooks import LatentExtractor
    from ultralytics import YOLO

    model = YOLO(args.weights)
    target = model.model.model
    extractor = LatentExtractor(target, to_cpu=True, fp16=False)
    extractor.start()
    model.predict(source=str(args.image), imgsz=args.imgsz, device="cpu", verbose=False)
    records = []
    for index, module in enumerate(target):
        activation = extractor.cache.get(index)
        shape = list(activation.shape) if activation is not None else None
        records.append(
            {
                "index": index,
                "module": module.__class__.__name__,
                "parameters": sum(parameter.numel() for parameter in module.parameters()),
                "output_shape": shape,
                "peek_compatible": bool(shape and len(shape) == 4 and min(shape[-2:]) > 1),
            }
        )
    extractor.stop()
    try:
        weights_record = str(args.weights.resolve().relative_to(ROOT))
    except ValueError:
        weights_record = args.weights.name
    payload = {
        "weights": weights_record,
        "image": args.image.name,
        "image_sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
        "image_size": args.imgsz,
        "hook_semantics": "output after top-level model.model[index]",
        "modules": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {len(records)} modules to {args.output}")


if __name__ == "__main__":
    main()
