#!/usr/bin/env python3
"""Evaluate completed nano baselines on the frozen WSD test protocol."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs/baseline_nano"


def evaluate_yolov5(run_name: str, config: dict, device: str) -> None:
    output = ROOT / config["output"] / run_name
    output.mkdir(parents=True, exist_ok=True)
    helper = ROOT / "scripts/evaluate_yolov5.py"
    subprocess.run(
        [
            sys.executable,
            str(helper),
            "--weights",
            str(RUNS / run_name / "weights/best.pt"),
            "--data",
            str(ROOT / config["data"]),
            "--output",
            str(output / "metrics.json"),
            "--imgsz",
            str(config["image_size"]),
            "--batch-size",
            str(config["batch_size"]),
            "--workers",
            str(config["workers"]),
            "--conf",
            str(config["confidence"]),
            "--iou",
            str(config["nms_iou"]),
            "--max-det",
            str(config["max_detections"]),
            "--device",
            device,
        ],
        cwd=ROOT / "third_party/yolov5",
        check=True,
    )


def evaluate_yolo26(run_name: str, config: dict, device: str) -> None:
    sys.path.insert(0, str(ROOT / "third_party/ultralytics"))
    from ultralytics import YOLO

    output = ROOT / config["output"] / run_name
    model = YOLO(RUNS / run_name / "weights/best.pt")
    metrics = model.val(
        data=str(ROOT / config["data"]),
        split=config["split"],
        imgsz=config["image_size"],
        batch=config["batch_size"],
        workers=config["workers"],
        conf=config["confidence"],
        iou=config["nms_iou"],
        max_det=config["max_detections"],
        device=device,
        project=str(ROOT / config["output"]),
        name=run_name,
        exist_ok=True,
        plots=True,
    )
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        **metrics.results_dict,
        "per_class": [
            {
                "class_id": int(class_id),
                "class": metrics.names[int(class_id)],
                "precision": metrics.box.class_result(i)[0],
                "recall": metrics.box.class_result(i)[1],
                "mAP50": metrics.box.class_result(i)[2],
                "mAP50-95": metrics.box.class_result(i)[3],
            }
            for i, class_id in enumerate(metrics.box.ap_class_index)
        ],
    }
    (output / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/evaluation.yaml")
    parser.add_argument("--family", choices=["all", "yolov5", "yolo26"], default="all")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    for family in ("yolov5", "yolo26"):
        if args.family not in {"all", family}:
            continue
        for initialization in ("pretrained", "scratch"):
            name = f"{family}n_{initialization}_seed42"
            if family == "yolov5":
                evaluate_yolov5(name, config, args.device)
            else:
                evaluate_yolo26(name, config, args.device)


if __name__ == "__main__":
    main()
