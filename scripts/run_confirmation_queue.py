#!/usr/bin/env python3
"""Run assigned YOLO26 confirmation seeds sequentially on one physical GPU."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", choices=["native", "cls075"], required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    for seed in args.seeds:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/tune_baselines.py"),
                "--config",
                str(ROOT / "configs/confirmation_yolo26_cls075.yaml"),
                "--data",
                str(args.data),
                "--family",
                "yolo26",
                "--initialization",
                "pretrained",
                "--trial",
                args.trial,
                "--seed",
                str(seed),
                "--device",
                args.device,
            ],
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
