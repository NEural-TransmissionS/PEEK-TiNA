#!/usr/bin/env python3
"""Train the frozen YOLOv5n and YOLO26n WSD baseline matrix."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from topoprun.reproducibility import seed_everything


def revision(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def write_manifest(config: dict, output: Path) -> None:
    import torch

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "revisions": {
            "repository": revision(ROOT),
            "yolov5": revision(ROOT / "third_party/yolov5"),
            "ultralytics": revision(ROOT / "third_party/ultralytics"),
            "peek": revision(ROOT / "third_party/PEEK"),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def run_name(model: dict, initialization: str, seed: int) -> str:
    return f"{model['family']}n_{initialization}_seed{seed}"


def train_yolov5(config: dict, model: dict, initialization: str, output: Path) -> None:
    name = run_name(model, initialization, config["seed"])
    command = [
        sys.executable,
        str(ROOT / "third_party/yolov5/train.py"),
        "--weights",
        model["weights"] if initialization == "pretrained" else "",
        "--cfg",
        str(ROOT / model["config"]),
        "--data",
        str(ROOT / config["data"]),
        "--imgsz",
        str(config["image_size"]),
        "--epochs",
        str(config["epochs"]),
        "--batch-size",
        str(config["batch_size"]),
        "--workers",
        str(config["workers"]),
        "--patience",
        str(config["patience"]),
        "--seed",
        str(config["seed"]),
        "--device",
        str(config["runtime_device"]),
        "--project",
        str(output),
        "--name",
        name,
        "--exist-ok",
    ]
    subprocess.run(command, cwd=ROOT / "third_party/yolov5", check=True)


def train_yolo26(config: dict, model: dict, initialization: str, output: Path) -> None:
    sys.path.insert(0, str(ROOT / "third_party/ultralytics"))
    from ultralytics import YOLO

    detector = YOLO(model["weights"] if initialization == "pretrained" else model["config"])
    name = run_name(model, initialization, config["seed"])
    detector.train(
        data=str(ROOT / config["data"]),
        imgsz=config["image_size"],
        epochs=config["epochs"],
        batch=config["batch_size"],
        workers=config["workers"],
        patience=config["patience"],
        seed=config["seed"],
        deterministic=config["deterministic"],
        device=config["runtime_device"],
        project=str(output),
        name=name,
        exist_ok=True,
        verbose=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baseline_nano.yaml")
    parser.add_argument("--family", choices=["all", "yolov5", "yolo26"], default="all")
    parser.add_argument(
        "--initialization",
        choices=["all", "pretrained", "scratch"],
        default="all",
    )
    parser.add_argument("--epochs", type=int, help="Override for smoke tests; recorded in manifest")
    parser.add_argument("--device", help="Override GPU/device for scheduling; recorded in manifest")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.epochs is not None:
        config["epochs"] = args.epochs
        config["patience"] = args.epochs
    if args.device is not None:
        config["device"] = args.device
    requested_device = str(config["device"])
    if requested_device.isdigit():
        # Isolate the physical GPU before importing/initializing torch. Both detector
        # frameworks then consistently address their sole visible accelerator as cuda:0.
        os.environ["CUDA_VISIBLE_DEVICES"] = requested_device
        config["runtime_device"] = "0"
    else:
        config["runtime_device"] = requested_device
    output = ROOT / config["output"]
    os.environ["WANDB_MODE"] = "disabled"
    seed_everything(config["seed"], config["deterministic"])
    for model in config["models"]:
        if args.family not in {"all", model["family"]}:
            continue
        for initialization in config["initializations"]:
            if args.initialization not in {"all", initialization}:
                continue
            name = run_name(model, initialization, config["seed"])
            run_config = {
                **config,
                "current_run": {"model": model, "initialization": initialization},
            }
            write_manifest(run_config, output / name)
            if model["family"] == "yolov5":
                train_yolov5(config, model, initialization, output)
            else:
                train_yolo26(config, model, initialization, output)


if __name__ == "__main__":
    main()
