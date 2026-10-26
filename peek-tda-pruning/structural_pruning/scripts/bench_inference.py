"""Inference timing of the pruned YOLO26 models (channel pruning and block replacement, WSD + COCO).

Times the network forward pass only (no image decoding, letterboxing or post-processing) on random
640x640 input, after Ultralytics' conv+BN fusion:
  gpu: FP16, batch 1 latency (CUDA events) and batch 32 throughput;
  cpu: FP32, batch 1 latency with a fixed thread count.
All models are loaded first and timed round-robin: each of --rounds rounds runs every model
--per-round times, in a freshly shuffled order, so drift on the node (other jobs, memory, clocks) hits
every model alike. Timing them one after another instead made some CPU results wrong by 10-30%
(scripts/cpu_interleave.py). Medians over all timed runs; p90 for spread.

    python scripts/bench_inference.py --device gpu --out bench_gpu.csv
    python scripts/bench_inference.py --device cpu --threads 8 --out bench_cpu.csv
"""

import argparse
import csv
import platform
import random
import statistics as st
import sys
import time
from pathlib import Path

import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # pruned checkpoints pickle peek_rvc_pruning classes
T = Path("/shared/projects/tina/PEEK-TiNA")
CHANNEL = T / "PEEK-rvc-pruning/outputs/yolo26_tda"
BLOCKS = T / "topo-prune-shapes/runs"
RATIOS = ("0.10", "0.20", "0.30", "0.40", "0.50")


def models() -> list[dict]:
    """Every architecture once: channel pruning has the same per-layer widths for every criterion at a
    ratio (field_h1 stands for all), block replacement one model per ranked cut set."""
    out = []
    for ds in ("wsd", "coco"):
        for m in "ns":
            runs = CHANNEL / f"{ds}_{m}_protect6_peek_ft20/runs"
            out.append({"dataset": ds, "model": m, "method": "unpruned", "setting": "",
                        "weights": runs / "none_r0.00_s42/finetune/weights/best.pt"})
            for r in RATIOS:
                out.append({"dataset": ds, "model": m, "method": "channel", "setting": r,
                            "weights": runs / f"field_h1_r{r}_s42/finetune/weights/best.pt"})
            for k in (1, 2, 3):
                for lab, name in (("head", f"{ds}_{m}_k{k}_peek_ft"), ("grid", f"{ds}_{m}_k{k}_peek_all_grid_ft"),
                                  ("null", f"{ds}_{m}_k{k}_peek_all_null_ft")):
                    w = BLOCKS / name / "weights/best.pt"
                    if w.exists():
                        out.append({"dataset": ds, "model": m, "method": f"block_{lab}", "setting": f"k{k}", "weights": w})
    return out


def once(fn, cuda: bool) -> float:
    """One timed call, in ms."""
    if cuda:
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        return a.elapsed_time(b)
    t = time.perf_counter(); fn()
    return 1000 * (time.perf_counter() - t)


def round_robin(nets: list, x, rounds: int, per_round: int, warm: int, cuda: bool) -> list[list[float]]:
    with torch.inference_mode():
        for net in nets:
            for _ in range(warm):
                net(x)
        times = [[] for _ in nets]
        rng = random.Random(0)
        for _ in range(rounds):
            idx = list(range(len(nets)))
            rng.shuffle(idx)
            for i in idx:
                times[i] += [once(lambda: nets[i](x), cuda) for _ in range(per_round)]
    return times


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", choices=["gpu", "cpu"], required=True)
    ap.add_argument("--threads", type=int, default=8, help="CPU threads (cpu run)")
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--per-round", type=int, default=5, help="timed runs per model per round (batch 1)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    cuda = a.device == "gpu"
    torch.backends.cudnn.benchmark = True
    if not cuda:
        torch.set_num_threads(a.threads)
    hw = torch.cuda.get_device_name(0) if cuda else f"{platform.processor() or platform.machine()}, {a.threads} threads"
    specs = models()
    nets = []
    for s in specs:
        net = YOLO(str(s["weights"])).model.fuse(verbose=False).eval()
        nets.append(net.cuda().half() if cuda else net.float())
    dtype, dev = (torch.float16, "cuda") if cuda else (torch.float32, "cpu")
    lat = round_robin(nets, torch.randn(1, 3, 640, 640, dtype=dtype, device=dev), a.rounds, a.per_round, 10, cuda)
    thr = round_robin(nets, torch.randn(32, 3, 640, 640, dtype=dtype, device=dev), a.rounds // 4, 2, 3, cuda) if cuda else None
    rows = []
    for i, s in enumerate(specs):
        r = {k: v for k, v in s.items() if k != "weights"} | {
            "hardware": hw, "run": s["weights"].parts[-4 if "finetune" in s["weights"].parts else -3],
            "params": sum(p.numel() for p in nets[i].parameters()),
            "b1_ms_median": round(st.median(lat[i]), 3), "b1_ms_p90": round(sorted(lat[i])[int(0.9 * len(lat[i]))], 3)}
        if thr:
            r["b32_img_per_s"] = round(32 * 1000 / st.median(thr[i]), 1)
        rows.append(r)
        print(r, flush=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
