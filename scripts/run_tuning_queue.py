#!/usr/bin/env python3
"""Run an ordered list of unique tuning trials sequentially on one GPU."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trials", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", required=True)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    for trial in args.trials:
        subprocess.run(
            [
                sys.executable, str(ROOT / "scripts/tune_baselines.py"),
                "--config", str(args.config.resolve()), "--data", str(args.data),
                "--family", "yolo26", "--initialization", "pretrained",
                "--trial", trial, "--seed", str(args.seed), "--device", args.device,
            ],
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
