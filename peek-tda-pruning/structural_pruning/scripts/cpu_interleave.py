"""End-to-end CPU latency of the COCO s block cuts, measured round-robin.

bench_inference.py times its 59 models one after another in one process; profile_cpu_layers.py found
no layer that explains the cuts that came out slower than the uncut model. This times the same six
models interleaved (each round runs every model once, in a shuffled order), so drift or process state
affects all of them alike, with PyTorch's oneDNN CPU kernels on and off.

    python scripts/cpu_interleave.py --threads 8 --rounds 40 --out outputs/bench/cpu_interleave.txt
"""

import argparse
import random
import statistics as st
import sys
import time
from pathlib import Path

import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from profile_cpu_layers import MODELS, RUNS  # noqa: E402  (same six models)


def run(nets, x, rounds, per_round):
    times = {k: [] for k in nets}
    rng = random.Random(0)
    with torch.inference_mode():
        for k in nets:  # warm-up
            for _ in range(5):
                nets[k](x)
        for _ in range(rounds):
            keys = list(nets)
            rng.shuffle(keys)
            for k in keys:
                for _ in range(per_round):
                    t = time.perf_counter()
                    nets[k](x)
                    times[k].append(1000 * (time.perf_counter() - t))
    return {k: st.median(v) for k, v in times.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--per-round", type=int, default=5)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    x = torch.randn(1, 3, 640, 640)
    nets = {lab: YOLO(str(RUNS / run / "weights/best.pt")).model.fuse(verbose=False).eval().float()
            for lab, run in MODELS.items()}
    lines = [f"torch {torch.__version__}, threads {a.threads}, {a.rounds} rounds x {a.per_round} runs, median ms"]
    for mk in (True, False):
        torch.backends.mkldnn.enabled = mk
        med = run(nets, x, a.rounds, a.per_round)
        base = med["uncut (k0)"]
        lines.append(f"\noneDNN {'on' if mk else 'off'}:")
        lines += [f"  {k:34s} {v:7.2f} ms  {100 * (v / base - 1):+5.1f}%" for k, v in med.items()]
    a.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
