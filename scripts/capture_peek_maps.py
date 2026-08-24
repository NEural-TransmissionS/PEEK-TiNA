#!/usr/bin/env python3
"""Capture canonical per-image/module PEEK maps for the frozen train calibration set."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "docs/wsd-v71-peek-calibration.json"
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pruning_yolo26.yaml")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "third_party/ultralytics"))
    sys.path.insert(0, str(ROOT / "third_party/PEEK"))
    sys.path.insert(0, str(ROOT / "src"))
    from peek.extractors.hooks import LatentExtractor
    from ultralytics import YOLO

    from topoprun.peek_adapter import to_peek_map

    data = yaml.safe_load(args.data.read_text())
    dataset_root = Path(data["path"])
    train_root = (dataset_root / data["train"]).resolve(strict=True)
    manifest = json.loads(args.manifest.read_text())
    items = manifest["items"]
    modules = tuple(yaml.safe_load(args.config.read_text())["pruning"]["modules"])
    model = YOLO(args.weights)
    target = model.model.model
    extractor = LatentExtractor(target, modules=modules, to_cpu=True, fp16=False)
    extractor.start()
    captured = []
    try:
        for item in items:
            image_path = train_root / item["image"]
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
            if digest != item["sha256"]:
                raise ValueError(f"calibration image hash mismatch: {image_path.name}")
            output_dir = args.output_root / image_path.stem
            paths = [output_dir / f"module_{module}.npy" for module in modules]
            if any(path.exists() for path in paths):
                if args.skip_existing and all(path.is_file() for path in paths):
                    captured.append({"image": image_path.name, "sha256": digest, "skipped": True})
                    continue
                raise FileExistsError(f"refusing to overwrite PEEK maps in {output_dir}")
            extractor.clear()
            model.predict(
                source=str(image_path),
                imgsz=args.imgsz,
                device=args.device,
                batch=1,
                verbose=False,
            )
            missing = set(modules) - set(extractor.cache)
            if missing:
                raise RuntimeError(f"missing hook activations for modules {sorted(missing)}")
            output_dir.mkdir(parents=True, exist_ok=False)
            shapes = {}
            for module, path in zip(modules, paths, strict=True):
                peek_map = to_peek_map(extractor.cache[module])
                np.save(path, peek_map.astype(np.float32))
                shapes[str(module)] = list(peek_map.shape)
            captured.append(
                {"image": image_path.name, "sha256": digest, "shapes": shapes, "skipped": False}
            )
    finally:
        extractor.stop()
    payload = {
        "weights": str(args.weights),
        "data": str(args.data),
        "selection_split": "train",
        "test_split_used": False,
        "image_size": args.imgsz,
        "device": args.device,
        "capture_dtype": "float32",
        "modules": list(modules),
        "images": len(captured),
        "items": captured,
    }
    output_manifest = args.output_root / "capture-manifest.json"
    output_manifest.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {len(captured)} image/module PEEK-map groups to {args.output_root}")


if __name__ == "__main__":
    main()
