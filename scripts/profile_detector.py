#!/usr/bin/env python3
"""Record reproducible raw-forward resource proxies for one detector checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["yolov5", "yolo26"], required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import resource

    import torch
    from thop import profile

    if args.family == "yolov5":
        sys.path.insert(0, str(ROOT / "third_party/yolov5"))
        from models.experimental import attempt_load

        model = attempt_load(args.weights, device=torch.device(args.device), fuse=True)
    else:
        sys.path.insert(0, str(ROOT / "third_party/ultralytics"))
        from ultralytics import YOLO

        model = YOLO(args.weights).model.to(args.device).fuse()
    model.eval()
    dummy = torch.zeros(1, 3, args.imgsz, args.imgsz, device=args.device)
    macs, _ = profile(model, inputs=(dummy,), verbose=False)
    before_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(dummy)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(args.trials):
            start = time.perf_counter_ns()
            model(dummy)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            times.append((time.perf_counter_ns() - start) / 1e6)
    times.sort()
    after_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "family": args.family, "weights": args.weights.name,
        "serialized_mb": args.weights.stat().st_size / 1_000_000,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "macs_g": macs / 1e9, "flops_g": 2 * macs / 1e9,
        "device": args.device, "precision": "float32", "batch": 1, "image_size": args.imgsz,
        "warmup": args.warmup, "trials": args.trials,
        "latency_ms": {
            "median": statistics.median(times),
            "p90": times[math.ceil(0.90 * len(times)) - 1],
            "p95": times[math.ceil(0.95 * len(times)) - 1],
            "p99": times[math.ceil(0.99 * len(times)) - 1],
        },
        "peak_cuda_mb": (
            torch.cuda.max_memory_allocated() / 1_000_000 if args.device.startswith("cuda") else None
        ),
        "process_peak_rss_mb": after_rss / 1024,
        "profile_incremental_peak_rss_mb": max(0, after_rss - before_rss) / 1024,
        "host": platform.platform(), "torch": torch.__version__,
        "scope": "desktop raw-forward proxy; excludes decode, preprocessing, NMS, and transfer",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
