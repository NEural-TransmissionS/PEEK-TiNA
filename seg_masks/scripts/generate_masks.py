#!/usr/bin/env python3
"""Generate pseudo segmentation masks for a WSD split from its YOLO boxes.

Writes one ``<stem>.png`` label mask per image (0 = background, class_id+1 = fg)
under ``--output-root``, plus ``mask-manifest.json`` recording, per image, the
method used and the GrabCut-vs-boxfill Betti difference -- the feasibility signal
for ``docs/feasibility.md``: if GrabCut never changes the topology, option B is
not worth the machinery.

Needs the WSD export (parent ``README.md`` "WSD dataset mapping"); no GPU, no
detector. SAM (``--method sam``) additionally needs the detector stack / weights.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "seg_masks" / "src"))

from seg_masks.box_to_mask import boxfill_mask, grabcut_mask, sam_mask, to_binary
from seg_masks.topology_check import betti_numbers
from seg_masks.wsd_labels import WSD_CLASSES, parse_label_file


def _iter_split(images_dir: Path):
    for image_path in sorted(images_dir.glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        label_path = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
        yield image_path, label_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True,
                        help="e.g. ../datasets/satellite_components-71/train/images")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--method", choices=["grabcut", "boxfill", "sam"], default="grabcut")
    parser.add_argument("--sam-weights", default=None)
    parser.add_argument("--limit", type=int, default=0, help="0 = all images")
    args = parser.parse_args()

    from PIL import Image

    args.output_root.mkdir(parents=True, exist_ok=True)
    records = []
    pairs = list(_iter_split(args.images_dir))
    if args.limit:
        pairs = pairs[: args.limit]

    for image_path, label_path in pairs:
        image = np.array(Image.open(image_path).convert("RGB"))
        h, w = image.shape[:2]
        boxes = parse_label_file(label_path, w, h)

        if args.method == "boxfill":
            result = boxfill_mask((h, w), boxes)
        elif args.method == "sam":
            result = sam_mask(image, boxes, weights=args.sam_weights)
        else:
            result = grabcut_mask(image, boxes)
        fill = boxfill_mask((h, w), boxes)

        drift = {"d_beta0": 0, "d_beta1": 0}
        for class_id in range(len(WSD_CLASSES)):
            a = betti_numbers(to_binary(result.label_map, class_id))
            b = betti_numbers(to_binary(fill.label_map, class_id))
            drift["d_beta0"] += abs(a["beta0"] - b["beta0"])
            drift["d_beta1"] += abs(a["beta1"] - b["beta1"])

        Image.fromarray(result.label_map).save(args.output_root / f"{image_path.stem}.png")
        records.append({
            "image": image_path.name,
            "boxes": len(boxes),
            "method": result.method,
            "coverage": result.coverage,
            "betti_drift_vs_boxfill": drift,
        })

    manifest = {
        "images_dir": str(args.images_dir),
        "method": args.method,
        "classes": list(WSD_CLASSES),
        "label_convention": "0 = background, class_id + 1 = foreground",
        "images": len(records),
        "images_where_grabcut_changed_topology": sum(
            1 for r in records
            if r["betti_drift_vs_boxfill"]["d_beta0"] or r["betti_drift_vs_boxfill"]["d_beta1"]
        ),
        "records": records,
    }
    (args.output_root / "mask-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(records)} masks + manifest to {args.output_root}")
    print(f"GrabCut changed the topology vs box-fill on "
          f"{manifest['images_where_grabcut_changed_topology']}/{len(records)} images")


if __name__ == "__main__":
    main()
