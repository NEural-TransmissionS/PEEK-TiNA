#!/usr/bin/env python3
"""Isolated YOLOv5 evaluator used by evaluate_baselines.py."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/yolov5"))

import val


def capture_class_metrics() -> dict:
    """Wrap YOLOv5's metric reducer without changing the pinned submodule."""
    captured = {}
    original = val.ap_per_class

    def wrapper(*args, **kwargs):
        result = original(*args, **kwargs)
        _, _, precision, recall, _, ap, class_ids = result
        captured["precision"] = precision.tolist()
        captured["recall"] = recall.tolist()
        captured["ap50"] = ap[:, 0].tolist()
        captured["ap50_95"] = ap.mean(1).tolist()
        captured["class_ids"] = class_ids.astype(int).tolist()
        return result

    val.ap_per_class = wrapper
    return captured


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--conf", type=float, required=True)
    parser.add_argument("--iou", type=float, required=True)
    parser.add_argument("--max-det", type=int, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    captured = capture_class_metrics()
    results, maps, timing = val.run(
        data=args.data,
        weights=args.weights,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        conf_thres=args.conf,
        iou_thres=args.iou,
        max_det=args.max_det,
        task="test",
        device=args.device,
        workers=args.workers,
        project=args.output.parent,
        name=args.output.parent.name,
        exist_ok=True,
        plots=True,
    )
    precision, recall, map50, map50_95, *losses = results
    payload = {
        "metrics/precision": precision,
        "metrics/recall": recall,
        "metrics/mAP50": map50,
        "metrics/mAP50-95": map50_95,
        "validation_losses": losses,
        "per_class_AP50-95": maps.tolist(),
        "per_class": [
            {
                "class_id": class_id,
                "class": val.check_dataset(args.data)["names"][class_id],
                "precision": captured["precision"][i],
                "recall": captured["recall"][i],
                "mAP50": captured["ap50"][i],
                "mAP50-95": captured["ap50_95"][i],
            }
            for i, class_id in enumerate(captured["class_ids"])
        ],
        "timing_ms": timing,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
