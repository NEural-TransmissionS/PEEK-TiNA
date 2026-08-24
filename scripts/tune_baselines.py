#!/usr/bin/env python3
"""Run the frozen validation-only hyperparameter search for WSD nano models."""

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


def revision(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def name(model: dict, initialization: str, trial: str, seed: int) -> str:
    model_name = model.get("name", f"{model['family']}n")
    return f"{model_name}_{initialization}_{trial}_seed{seed}"


def write_manifest(config: dict, run_dir: Path, model: dict, initialization: str, trial: dict) -> None:
    import torch

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_split": "val",
        "test_split_used": False,
        "model": model,
        "initialization": initialization,
        "trial": trial,
        "protocol": config,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "revisions": {
            "repository": revision(ROOT),
            "yolov5": revision(ROOT / "third_party/yolov5"),
            "ultralytics": revision(ROOT / "third_party/ultralytics"),
            "peek": revision(ROOT / "third_party/PEEK"),
        },
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "tuning-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def train_yolov5(config: dict, model: dict, initialization: str, trial: dict, run: str) -> None:
    hyp = yaml.safe_load((ROOT / model["base_hyp"]).read_text())
    optimizer = trial["yolov5"].get("optimizer", "SGD")
    trainer_flags = {"cos_lr"}
    hyp.update(
        {k: v for k, v in trial["yolov5"].items() if k not in {"optimizer", *trainer_flags}}
    )
    hyp_path = ROOT / config["output"] / run / "tuning-hyp.yaml"
    hyp_path.write_text(yaml.safe_dump(hyp, sort_keys=False))
    command = [
        sys.executable, str(ROOT / "third_party/yolov5/train.py"),
        "--weights", model["weights"] if initialization == "pretrained" else "",
        "--cfg", str(ROOT / model["config"]), "--data", str(ROOT / config["data"]),
        "--hyp", str(hyp_path), "--optimizer", optimizer,
        "--imgsz", str(config["image_size"]), "--epochs", str(config["epochs"]),
        "--batch-size", str(config["batch_size"]), "--workers", str(config["workers"]),
        "--patience", str(config["patience"]), "--seed", str(config["seed"]),
        "--device", str(config["runtime_device"]), "--project", str(ROOT / config["output"]),
        "--name", run, "--exist-ok",
    ]
    if trial["yolov5"].get("cos_lr", False):
        command.append("--cos-lr")
    subprocess.run(command, cwd=ROOT / "third_party/yolov5", check=True)


def train_yolo26(config: dict, model: dict, initialization: str, trial: dict, run: str) -> None:
    sys.path.insert(0, str(ROOT / "third_party/ultralytics"))
    from ultralytics import YOLO

    if initialization == "pretrained" and model.get("initialize_architecture", False):
        detector = YOLO(model["config"]).load(model["weights"])
    else:
        detector = YOLO(model["weights"] if initialization == "pretrained" else model["config"])
    detector.train(
        data=str(ROOT / config["data"]), imgsz=config["image_size"], epochs=config["epochs"],
        batch=config["batch_size"], workers=config["workers"], patience=config["patience"],
        seed=config["seed"], deterministic=config["deterministic"],
        device=config["runtime_device"], project=str(ROOT / config["output"]), name=run,
        exist_ok=True, verbose=True, **trial["yolo26"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/tuning_nano.yaml")
    parser.add_argument("--family", choices=["all", "yolov5", "yolo26"], default="all")
    parser.add_argument("--initialization", choices=["all", "pretrained", "scratch"], default="all")
    parser.add_argument("--trial", action="append", help="Run only named trial(s); repeat as needed")
    parser.add_argument("--device", help="Physical GPU index or runtime device")
    parser.add_argument("--data", type=Path, help="Machine-local dataset YAML override")
    parser.add_argument("--seed", type=int, help="Frozen seed override for confirmation runs")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.data is not None:
        config["data"] = str(args.data.resolve())
    if args.seed is not None:
        config["seed"] = args.seed
    if args.device is not None:
        config["device"] = args.device
    requested = str(config["device"])
    if requested.isdigit():
        os.environ["CUDA_VISIBLE_DEVICES"] = requested
        config["runtime_device"] = "0"
    else:
        config["runtime_device"] = requested
    os.environ["WANDB_MODE"] = "disabled"
    sys.path.insert(0, str(ROOT / "src"))
    from topoprun.reproducibility import seed_everything

    seed_everything(config["seed"], config["deterministic"])
    for model in config["models"]:
        family = model["family"]
        if args.family not in {"all", family}:
            continue
        for initialization in config["initializations"]:
            if args.initialization not in {"all", initialization}:
                continue
            for trial in config["trials"]:
                if args.trial and trial["name"] not in args.trial:
                    continue
                run = name(model, initialization, trial["name"], config["seed"])
                run_dir = ROOT / config["output"] / run
                if (run_dir / "tuning-complete.json").exists():
                    print(f"Skipping completed run: {run}")
                    continue
                write_manifest(config, run_dir, model, initialization, trial)
                if family == "yolov5":
                    train_yolov5(config, model, initialization, trial, run)
                else:
                    train_yolo26(config, model, initialization, trial, run)
                (run_dir / "tuning-complete.json").write_text(
                    json.dumps({"completed_utc": datetime.now(timezone.utc).isoformat()}) + "\n"
                )


if __name__ == "__main__":
    main()
