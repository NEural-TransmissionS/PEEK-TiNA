"""Per-layer CPU timing of block-cut YOLO26 models, to explain why some cuts run slower than the
uncut model on CPU (bench_inference.py found COCO s grid k2 +13%, null k2 +21% while every cut is
faster on GPU).

Same setting as bench_inference.py's CPU run: fused conv+BN, FP32, batch 1, 640x640, fixed threads.
For each model: median wall time of every top-level layer over --reps forward passes (forward
pre/post hooks), the layer's output shape and memory format, then torch.profiler's top CPU ops for
the layers whose time changed most against the uncut model.

    python scripts/profile_cpu_layers.py --threads 8 --out outputs/bench/cpu_layers.txt
"""

import argparse
import statistics as st
import sys
import time
from pathlib import Path

import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
RUNS = Path("/shared/projects/tina/PEEK-TiNA/topo-prune-shapes/runs")
MODELS = {  # label -> run (uncut first: the reference)
    "uncut (k0)": "coco_s_k0_peek_ft",
    "grid k1 [16] (normal, -8%)": "coco_s_k1_peek_all_grid_ft",
    "grid k2 [2 16] (+13%)": "coco_s_k2_peek_all_grid_ft",
    "null k2 [8 16] (+21%)": "coco_s_k2_peek_all_null_ft",
    "head k3 [13 19 22] (+4%)": "coco_s_k3_peek_ft",
    "null k3 [8 13 16] (normal, -17%)": "coco_s_k3_peek_all_null_ft",
}


def layer_times(net, x, reps):
    t, start = {}, {}
    hooks = []
    for i, m in enumerate(net.model):
        hooks.append(m.register_forward_pre_hook(lambda mod, inp, i=i: start.__setitem__(i, time.perf_counter())))
        hooks.append(m.register_forward_hook(lambda mod, inp, out, i=i: t.setdefault(i, []).append(
            1000 * (time.perf_counter() - start[i]))))
    shapes = {}
    with torch.inference_mode():
        for _ in range(10):
            net(x)
        t.clear()
        for _ in range(reps):
            net(x)
        probe = [m.register_forward_hook(lambda mod, inp, out, i=i: shapes.__setitem__(i, out))
                 for i, m in enumerate(net.model)]
        net(x)
    for h in hooks + probe:
        h.remove()
    info = {}
    for i, m in enumerate(net.model):
        o = shapes.get(i)
        o = o[0] if isinstance(o, (list, tuple)) else o
        fmt = ("NHWC" if o.is_contiguous(memory_format=torch.channels_last) and not o.is_contiguous() else "NCHW") \
            if torch.is_tensor(o) and o.dim() == 4 else "-"
        info[i] = (type(m).__name__, tuple(o.shape) if torch.is_tensor(o) else "-", fmt, st.median(t[i]))
    return info


def top_ops(net, x, layers, n=6):
    """torch.profiler table of the slowest CPU ops inside the given top-level layers."""
    from torch.profiler import ProfilerActivity, profile, record_function
    hooks = []
    ctx = {}
    for i in layers:
        m = net.model[i]
        hooks.append(m.register_forward_pre_hook(lambda mod, inp, i=i: ctx.__setitem__(i, record_function(f"layer{i}").__enter__())))
        hooks.append(m.register_forward_hook(lambda mod, inp, out, i=i: ctx[i].__exit__(None, None, None)))
    with torch.inference_mode(), profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        for _ in range(20):
            net(x)
    for h in hooks:
        h.remove()
    return prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_time_total", row_limit=n)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    x = torch.randn(1, 3, 640, 640)
    lines = [f"torch {torch.__version__}, mkldnn {torch.backends.mkldnn.is_available()}, threads {a.threads}"]
    res = {}
    for lab, run in MODELS.items():
        net = YOLO(str(RUNS / run / "weights/best.pt")).model.fuse(verbose=False).eval().float()
        res[lab] = (net, layer_times(net, x, a.reps))
        tot = sum(v[3] for v in res[lab][1].values())
        lines.append(f"{lab}: sum of layer medians {tot:.1f} ms")
    ref = res["uncut (k0)"][1]
    for lab, (net, info) in res.items():
        lines.append(f"\n=== {lab}  (layer: type, output shape, memory format, ms | uncut ms, delta)")
        diffs = []
        for i, (typ, shp, fmt, ms) in info.items():
            r = ref[i][3]
            diffs.append((ms - r, i))
            lines.append(f"  {i:2d} {typ:10s} {str(shp):22s} {fmt}  {ms:7.2f} | {r:7.2f}  {ms - r:+6.2f}")
        if lab != "uncut (k0)":
            worst = [i for d, i in sorted(diffs, reverse=True)[:3]]
            lines.append(f"  slowest-vs-uncut layers {worst}; their top CPU ops:")
            lines.append(top_ops(net, x, worst))
    a.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
