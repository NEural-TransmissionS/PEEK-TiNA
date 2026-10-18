#!/usr/bin/env python3
"""Generate the four synthetic shape datasets (plain filled shapes on white).

For each of ``circles`` / ``triangles`` / ``squares`` (1 class, 1 object/image)
and ``mixed`` (3 classes, 2-4 objects/image) it writes, under ``--out/<name>/``:

  images/{train,val}/NNNNNN.png   labels/{train,val}/NNNNNN.txt   (YOLO cls cx cy w h)
  data.yaml                       Ultralytics training config
  manifest.json                   val split in the comparison-protocol harness format

Deterministic given ``--seed``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SS = 4  # supersample factor for anti-aliasing
# Per dataset, classes are remapped to a contiguous 0.. range (below); this is
# only the fixed drawing order.
KIND_ORDER = ["circle", "triangle", "square"]
PALETTE = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
    (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207),
    (0, 0, 0),
]
DATASETS = {
    "circles": ["circle"],
    "triangles": ["triangle"],
    "squares": ["square"],
    "mixed": ["circle", "triangle", "square"],
}


def _poly(kind: str, cx: float, cy: float, r: float, rot: float) -> list[tuple[float, float]]:
    if kind == "triangle":
        pts = [(cx + r * math.cos(rot + k * 2 * math.pi / 3),
                cy + r * math.sin(rot + k * 2 * math.pi / 3)) for k in range(3)]
    else:  # square
        d = r * math.sqrt(2)
        pts = [(cx + d * math.cos(rot + math.pi / 4 + k * math.pi / 2),
                cy + d * math.sin(rot + math.pi / 4 + k * math.pi / 2)) for k in range(4)]
    return pts


def _place(kind: str, rng: np.random.Generator, size: int) -> tuple[dict, tuple]:
    """Sample a shape's geometry + colour (no drawing). Returns (spec, xyxy px)."""
    r = float(rng.uniform(28, 100)) * SS
    cx = float(rng.uniform(r + 6 * SS, size * SS - r - 6 * SS))
    cy = float(rng.uniform(r + 6 * SS, size * SS - r - 6 * SS))
    fill = tuple(PALETTE[int(rng.integers(len(PALETTE)))])
    if kind == "circle":
        spec = {"kind": kind, "box": [cx - r, cy - r, cx + r, cy + r], "fill": fill}
        x0, y0, x1, y1 = spec["box"]
    else:
        pts = _poly(kind, cx, cy, r, float(rng.uniform(0, 2 * math.pi)))
        spec = {"kind": kind, "pts": pts, "fill": fill}
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    return spec, (x0 / SS, y0 / SS, x1 / SS, y1 / SS)


def _render(draw: ImageDraw.ImageDraw, spec: dict) -> None:
    if spec["kind"] == "circle":
        draw.ellipse(spec["box"], fill=spec["fill"], outline=(0, 0, 0), width=2 * SS)
    else:
        draw.polygon(spec["pts"], fill=spec["fill"], outline=(0, 0, 0))


def _iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _size_bin(w: float, h: float) -> str:
    area = w * h
    return "small" if area < 32 * 32 else ("large" if area > 96 * 96 else "medium")


def _make_split(name: str, kinds: list[str], n: int, split: str, out: Path, size: int,
                rng: np.random.Generator, manifest_rows: list, ann_id: list):
    local = {kind: idx for idx, kind in enumerate(kinds)}  # contiguous class ids
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    n_obj = 1 if name != "mixed" else None
    for i in range(n):
        canvas = Image.new("RGB", (size * SS, size * SS), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        k = n_obj or int(rng.integers(2, 5))
        boxes: list[tuple] = []
        labels: list[str] = []
        for _ in range(k):
            kind = kinds[0] if len(kinds) == 1 else kinds[int(rng.integers(len(kinds)))]
            for _try in range(20):
                spec, xyxy = _place(kind, rng, size)
                if all(_iou(xyxy, b[1]) < 0.15 for b in boxes):
                    _render(draw, spec)
                    boxes.append((kind, xyxy))
                    break
        img = canvas.resize((size, size), Image.LANCZOS)
        stem = f"{i:06d}"
        img.save(out / "images" / split / f"{stem}.png")
        for kind, (x0, y0, x1, y1) in boxes:
            w, h = x1 - x0, y1 - y0
            cx, cy = x0 + w / 2, y0 + h / 2
            labels.append(
                f"{local[kind]} {cx / size:.6f} {cy / size:.6f} {w / size:.6f} {h / size:.6f}"
            )
        (out / "labels" / split / f"{stem}.txt").write_text("\n".join(labels) + "\n")

        if split == "val":
            per_cat: dict[int, list] = {}
            names_by_cls: dict[int, str] = {}
            for kind, (x0, y0, x1, y1) in boxes:
                per_cat.setdefault(local[kind], []).append([x0, y0, x1 - x0, y1 - y0])
                names_by_cls[local[kind]] = kind
            for cls, cat_boxes in list(per_cat.items())[:2]:  # <=2 target objects / image
                bx = cat_boxes[0]
                ann_id[0] += 1
                manifest_rows.append({
                    "annotation_id": ann_id[0], "image_id": i, "file": f"{stem}.png",
                    "target_cls": cls, "category": cls,
                    "target_category_name": names_by_cls[cls],
                    "size_bin": _size_bin(bx[2], bx[3]),
                    "primary_box_xywh": bx, "target_boxes_xywh": cat_boxes,
                })


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "data")
    ap.add_argument("--train", type=int, default=800)
    ap.add_argument("--val", type=int, default=200)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    for di, (name, kinds) in enumerate(DATASETS.items()):
        out = (args.out / name).resolve()
        rng = np.random.default_rng(args.seed * 100 + di)
        rows: list = []
        ann_id = [0]
        _make_split(name, kinds, args.train, "train", out, args.imgsz, rng, rows, ann_id)
        _make_split(name, kinds, args.val, "val", out, args.imgsz, rng, rows, ann_id)
        (out / "data.yaml").write_text(
            f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\n"
            f"names:\n" + "".join(f"  {i}: {k}\n" for i, k in enumerate(kinds))
        )
        (out / "manifest.json").write_text(json.dumps(
            {"coco_root": out.as_posix(), "seed": args.seed, "count": len(rows), "objects": rows},
            indent=1))
        print(f"{name}: {args.train} train + {args.val} val images, {len(rows)} manifest objects")


if __name__ == "__main__":
    main()
