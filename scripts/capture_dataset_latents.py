#!/usr/bin/env python3
"""Capture post-activation latents and PEEK maps for every image in one or more splits.

Unlike capture_peek_maps.py (which is scoped to the frozen 128-image train
calibration set for pruning), this captures the full requested dataset scope.

Latent extraction is delegated to the vendored PEEK package's own Ultralytics
tooling (peek.extractors.ultralytics.extract_ultralytics_latents), the same
entry point used in third_party/PEEK/notebooks/Ultralytics_Demo.ipynb, rather
than reimplementing hook management here. PEEK-map conversion reuses this
project's to_peek_map (topoprun.peek_adapter), which every module's raw
latent is run through in a second pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]

SPLIT_KEYS = {"train": "train", "valid": "val", "test": "test"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data/wsd.yaml")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pruning_yolo26.yaml")
    parser.add_argument(
        "--modules", type=int, nargs="+", default=None,
        help="Override the frozen pruning-config module list, e.g. for full-coverage capture",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--splits", nargs="+", choices=list(SPLIT_KEYS), default=list(SPLIT_KEYS)
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip a split entirely if its latents directory is already populated",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "third_party/ultralytics"))
    sys.path.insert(0, str(ROOT / "third_party/PEEK"))
    sys.path.insert(0, str(ROOT / "src"))
    from ultralytics import YOLO  # noqa: F401  (imported first: pins this sys.path's copy)

    from peek.extractors.ultralytics import extract_ultralytics_latents
    from topoprun.peek_adapter import to_peek_map

    data = yaml.safe_load(args.data.read_text())
    dataset_root = Path(data["path"])
    if args.modules is not None:
        modules = tuple(args.modules)
    else:
        modules = tuple(yaml.safe_load(args.config.read_text())["pruning"]["modules"])

    weights = args.weights.resolve()
    output_root = args.output_root.resolve()
    manifest_items = []

    for split in args.splits:
        image_dir = (dataset_root / data[SPLIT_KEYS[split]]).resolve(strict=True)
        latents_dir = output_root / split / "latents"
        peek_dir = output_root / split / "peek_maps"
        images = sorted(p for p in image_dir.glob("*") if p.is_file())

        already_done = latents_dir.exists() and any(latents_dir.iterdir())
        if already_done and not args.skip_existing:
            raise FileExistsError(f"refusing to overwrite {latents_dir}")
        if not already_done:
            # One pickle per image: {module_index: torch.Tensor}, whole split in one pass.
            extract_ultralytics_latents(
                weights=str(weights),
                image_dir=str(image_dir),
                out_dir=str(latents_dir),
                device=args.device,
                imgsz=args.imgsz,
                modules=list(modules),
                to_cpu=True,
                verbose=True,
            )

        # Second pass: raw latents -> PEEK maps (module-by-module), plus provenance.
        for image_path in images:
            pkl_path = latents_dir / f"{image_path.stem}.pkl"
            out_dir = peek_dir / image_path.stem
            peek_paths = [out_dir / f"module_{m}.npy" for m in modules]
            if args.skip_existing and all(p.is_file() for p in peek_paths):
                manifest_items.append(
                    {"split": split, "image": image_path.name, "skipped": True}
                )
                continue

            with pkl_path.open("rb") as stream:
                cache = pickle.load(stream)
            missing = set(modules) - set(cache)
            if missing:
                raise RuntimeError(f"{pkl_path} missing modules {sorted(missing)}")

            out_dir.mkdir(parents=True, exist_ok=True)
            shapes = {}
            for module, peek_path in zip(modules, peek_paths, strict=True):
                peek_map = to_peek_map(cache[module])
                np.save(peek_path, peek_map.astype(np.float32))
                shapes[str(module)] = list(peek_map.shape)

            manifest_items.append(
                {
                    "split": split,
                    "image": image_path.name,
                    "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                    "shapes": shapes,
                    "skipped": False,
                }
            )

    payload = {
        "weights": str(weights),
        "data": str(args.data),
        "splits": args.splits,
        "image_size": args.imgsz,
        "device": args.device,
        "modules": list(modules),
        "images": len(manifest_items),
        "items": manifest_items,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "capture-manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {len(manifest_items)} image/module PEEK-map records to {output_root}")


if __name__ == "__main__":
    main()
