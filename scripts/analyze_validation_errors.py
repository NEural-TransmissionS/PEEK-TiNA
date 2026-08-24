#!/usr/bin/env python3
"""Run validation inference and stratify detector FP/FN errors without using test data."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["antenna", "body", "solar", "thruster"]


def boxes(path: Path, prediction: bool = False) -> list[dict]:
    output = []
    if not path.exists():
        return output
    for line in path.read_text().splitlines():
        fields = list(map(float, line.split()))
        if len(fields) < 5:
            continue
        class_id, x, y, width, height = fields[:5]
        output.append(
            {
                "class_id": int(class_id), "x": x, "y": y, "w": width, "h": height,
                "confidence": fields[5] if prediction and len(fields) > 5 else 1.0,
            }
        )
    return output


def corners(box: dict) -> np.ndarray:
    return np.array(
        [box["x"] - box["w"] / 2, box["y"] - box["h"] / 2,
         box["x"] + box["w"] / 2, box["y"] + box["h"] / 2]
    )


def iou(a: dict, b: dict) -> float:
    aa, bb = corners(a), corners(b)
    intersection = np.maximum(0, np.minimum(aa[2:], bb[2:]) - np.maximum(aa[:2], bb[:2])).prod()
    union = a["w"] * a["h"] + b["w"] * b["h"] - intersection
    return float(intersection / union) if union > 0 else 0.0


def match(gt: list[dict], predictions: list[dict], threshold: float) -> tuple[set[int], set[int]]:
    candidates = sorted(
        (
            (iou(truth, prediction), gi, pi)
            for gi, truth in enumerate(gt)
            for pi, prediction in enumerate(predictions)
            if truth["class_id"] == prediction["class_id"]
        ),
        reverse=True,
    )
    matched_gt, matched_predictions = set(), set()
    for overlap, gi, pi in candidates:
        if overlap < threshold:
            break
        if gi not in matched_gt and pi not in matched_predictions:
            matched_gt.add(gi)
            matched_predictions.add(pi)
    return matched_gt, matched_predictions


def size_bin(box: dict) -> str:
    area = box["w"] * box["h"]
    return "small_lt1pct" if area < 0.01 else "medium_1to10pct" if area < 0.10 else "large_ge10pct"


def run_inference(family: str, weights: Path, data: dict, output: Path, config: dict) -> Path:
    labels = output / "predictions" / "labels"
    if (output / "inference-complete.json").exists():
        return labels
    image_dir = Path(data["path"]) / data["val"]
    if family == "yolov5":
        command = [
            sys.executable, str(ROOT / "third_party/yolov5/detect.py"), "--weights", str(weights),
            "--source", str(image_dir), "--imgsz", str(config["image_size"]),
            "--conf-thres", str(config["confidence"]), "--iou-thres", str(config["nms_iou"]),
            "--max-det", str(config["max_detections"]), "--device", config["device"],
            "--save-txt", "--save-conf", "--nosave", "--project", str(output),
            "--name", "predictions", "--exist-ok",
        ]
        subprocess.run(command, cwd=ROOT / "third_party/yolov5", check=True)
    else:
        sys.path.insert(0, str(ROOT / "third_party/ultralytics"))
        from ultralytics import YOLO

        model = YOLO(weights)
        model.predict(
            source=str(image_dir), imgsz=config["image_size"], conf=config["confidence"],
            iou=config["nms_iou"], max_det=config["max_detections"], device=config["device"],
            save=False, save_txt=True, save_conf=True, project=str(output), name="predictions",
            exist_ok=True, verbose=False,
        )
    (output / "inference-complete.json").write_text(
        json.dumps({"split": "val", "confidence": config["confidence"]}) + "\n"
    )
    return labels


def analyze(family: str, weights: Path, prediction_dir: Path, data: dict, output: Path, config: dict) -> None:
    metadata = {
        row["image"]: row
        for row in csv.DictReader((ROOT / "docs/wsd-v71-robustness-slices.csv").open())
        if row["split"] == "val"
    }
    dataset_root = Path(data["path"])
    image_dir = dataset_root / data["val"]
    label_dir = Path(str(image_dir).replace("/images", "/labels"))
    totals = Counter()
    by_class, by_slice, by_size, confusions = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter), Counter()
    image_rows = []
    sweep_thresholds = [value for value in (0.001, 0.01, 0.025, 0.05, 0.10, 0.20, 0.25, 0.40) if value >= config["confidence"]]
    sweep = {threshold: {"all": Counter(), **{name: Counter() for name in CLASSES}} for threshold in sweep_thresholds}
    for image in sorted(path for path in image_dir.iterdir() if path.is_file()):
        gt = boxes(label_dir / f"{image.stem}.txt")
        predictions = boxes(prediction_dir / f"{image.stem}.txt", prediction=True)
        for threshold in sweep_thresholds:
            kept = [prediction for prediction in predictions if prediction["confidence"] >= threshold]
            sweep_gt, sweep_predictions = match(gt, kept, config["match_iou"])
            sweep[threshold]["all"].update(
                {"tp": len(sweep_gt), "fp": len(kept) - len(sweep_predictions), "fn": len(gt) - len(sweep_gt)}
            )
            for class_id, class_name in enumerate(CLASSES):
                class_gt = [truth for truth in gt if truth["class_id"] == class_id]
                class_predictions = [prediction for prediction in kept if prediction["class_id"] == class_id]
                class_matched_gt, class_matched_predictions = match(
                    class_gt, class_predictions, config["match_iou"]
                )
                sweep[threshold][class_name].update(
                    {"tp": len(class_matched_gt), "fp": len(class_predictions) - len(class_matched_predictions),
                     "fn": len(class_gt) - len(class_matched_gt)}
                )
        matched_gt, matched_predictions = match(gt, predictions, config["match_iou"])
        tp, fp, fn = len(matched_gt), len(predictions) - len(matched_predictions), len(gt) - len(matched_gt)
        totals.update({"tp": tp, "fp": fp, "fn": fn})
        slice_name = metadata[image.name]["slice"]
        by_slice[slice_name].update({"images": 1, "tp": tp, "fp": fp, "fn": fn})
        for index, truth in enumerate(gt):
            outcome = "tp" if index in matched_gt else "fn"
            by_class[CLASSES[truth["class_id"]]][outcome] += 1
            by_size[size_bin(truth)][outcome] += 1
        for index, prediction in enumerate(predictions):
            if index not in matched_predictions:
                by_class[CLASSES[prediction["class_id"]]]["fp"] += 1
        for gi, truth in enumerate(gt):
            if gi in matched_gt:
                continue
            alternatives = [
                (iou(truth, prediction), prediction) for pi, prediction in enumerate(predictions)
                if pi not in matched_predictions and prediction["class_id"] != truth["class_id"]
            ]
            if alternatives:
                overlap, prediction = max(alternatives, key=lambda item: item[0])
                if overlap >= config["match_iou"]:
                    confusions[f"{CLASSES[truth['class_id']]}->{CLASSES[prediction['class_id']]}"] += 1
        image_rows.append(
            {"image": image.name, "slice": slice_name, "gt": len(gt), "predictions": len(predictions),
             "tp": tp, "fp": fp, "fn": fn, "error_total": fp + fn}
        )
    def rates(groups: dict) -> dict:
        result = {}
        for name, counts in sorted(groups.items()):
            denominator = counts["tp"] + counts["fn"]
            result[name] = {
                **dict(counts),
                "recall": counts["tp"] / denominator if denominator else None,
                "false_negatives_per_gt": counts["fn"] / denominator if denominator else None,
            }
        return result
    def sweep_rates(counts: Counter) -> dict:
        precision = counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else 0.0
        recall = counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else 0.0
        return {
            **dict(counts), "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        }
    payload = {
        "family": family, "weights": weights.name, "split": "val",
        "confidence": config["confidence"], "nms_iou": config["nms_iou"],
        "match_iou": config["match_iou"], "totals": dict(totals),
        "by_class": rates(by_class), "by_slice": rates(by_slice), "by_gt_size": rates(by_size),
        "class_confusions": dict(confusions.most_common()),
        "confidence_sweep": {
            str(threshold): {name: sweep_rates(counts) for name, counts in groups.items()}
            for threshold, groups in sweep.items()
        },
        "worst_images": sorted(image_rows, key=lambda row: (-row["error_total"], row["image"]))[:25],
    }
    (output / "error-analysis.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (output / "per-image-errors.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=image_rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(image_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["yolov5", "yolo26"], required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data/wsd.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.60)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--max-detections", type=int, default=300)
    args = parser.parse_args()
    args.weights = args.weights.resolve()
    args.output = args.output.resolve()
    args.data = args.data.resolve()
    data = yaml.safe_load(args.data.read_text())
    config = vars(args)
    args.output.mkdir(parents=True, exist_ok=True)
    prediction_dir = run_inference(args.family, args.weights, data, args.output, config)
    analyze(args.family, args.weights, prediction_dir, data, args.output, config)
    print(f"Wrote validation-only error analysis to {args.output}")


if __name__ == "__main__":
    main()
