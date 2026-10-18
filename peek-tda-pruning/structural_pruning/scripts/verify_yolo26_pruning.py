"""
Sanity checks for the YOLO26 structural pruning in peek_rvc_pruning/yolo_tda.py.

1. equivalence: structurally removing channels must give the same output as
   keeping them and forcing them to zero (BN weight = bias = 0, SiLU(0) = 0).
   Checked for every prunable group on its own, then for all groups at once.
   A bookkeeping bug (wrong indices, missed coupled layer) breaks equality.

2. single-layer mAP: prune ONE group at --ratio and evaluate; repeat for every
   group. Pruning one layer should cost little, never collapse the model.

    python scripts/verify_yolo26_pruning.py --data coco.yaml --device 0
"""

from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_yolo26_tda_pruning import calibration_images, evaluate, load_prunable  # noqa: E402

from peek_rvc_pruning import yolo_tda  # noqa: E402

tp = yolo_tda.tp


def groups(net, make_importance, ratio, imgsz, protect=0):
    """Interactive MetaPruner: yields each group with the channels the criterion picked.
    Importance is built on `net` itself: persistence scores are keyed by module identity."""
    net.train()
    pruner = tp.pruner.MetaPruner(
        net, torch.zeros(1, 3, imgsz, imgsz), importance=make_importance(net), pruning_ratio=ratio,
        global_pruning=False, ignored_layers=yolo_tda.ignored_layers(net, protect),
    )
    return pruner.step(interactive=True)


def zero_group(masked, group, names):
    """Force the group's channels to zero in `masked` (an unpruned copy)."""
    modules = dict(masked.named_modules())
    for dep, idxs in group:
        if dep.pruning_fn == tp.pruner.function.prune_batchnorm_out_channels:
            bn = modules[names[id(dep.layer)]]
            bn.weight.data[idxs] = 0
            bn.bias.data[idxs] = 0


@torch.no_grad()
def max_diff(a, b, x):
    a.eval(), b.eval()
    ya, yb = a(x), b(x)
    ya, yb = (ya[0], yb[0]) if isinstance(ya, (list, tuple)) else (ya, yb)
    return (ya - yb).abs().max().item(), ya.abs().max().item()


def check_equivalence(base, importance, args, x):  # importance: net -> Importance
    print(f"\n== equivalence: pruned vs zeroed, ratio {args.ratio} ==")
    n = sum(1 for _ in groups(copy.deepcopy(base), importance, args.ratio, args.imgsz))
    worst = 0.0
    for gi in range(n):
        pruned, masked = copy.deepcopy(base), copy.deepcopy(base)
        names = {id(m): k for k, m in pruned.named_modules()}
        for j, g in enumerate(groups(pruned, importance, args.ratio, args.imgsz)):
            if j == gi:
                zero_group(masked, g, names)
                root, k = names[id(g[0].dep.layer)], len(g[0].idxs)
                g.prune()
                break
        d, scale = max_diff(pruned, masked, x)
        worst = max(worst, d / scale)
        print(f"  group {gi:2d} {root:28s} -{k:3d} ch  max|diff| {d:.2e} (output scale {scale:.0f})")

    pruned, masked = copy.deepcopy(base), copy.deepcopy(base)
    names = {id(m): k for k, m in pruned.named_modules()}
    for g in groups(pruned, importance, args.ratio, args.imgsz):
        zero_group(masked, g, names)
        g.prune()
    d, scale = max_diff(pruned, masked, x)
    worst = max(worst, d / scale)
    print(f"  ALL {n} groups together: max|diff| {d:.2e} (output scale {scale:.0f})")
    ok = n > 0 and worst < 1e-4
    print(f"  worst relative diff {worst:.1e} over {n} groups -> {'PASS' if ok else 'FAIL'}")
    return ok


def single_layer_map(base, args, scores):
    print(f"\n== single-layer pruning mAP50-95, ratio {args.ratio}, data {args.data} ==")
    run_dir = args.output_dir / "val"
    dense = evaluate(args.weights, base, args, run_dir, "dense")["map50_95"]
    print(f"  unpruned: {dense:.4f}")
    rows = []
    n = sum(1 for _ in groups(copy.deepcopy(base), lambda net: yolo_tda.build_importance("l2", net), args.ratio, args.imgsz))
    for gi in range(n):
        row = {"group": gi}
        for crit in args.criteria:
            net = copy.deepcopy(base)
            torch.manual_seed(0)
            for j, g in enumerate(groups(net, lambda net: yolo_tda.build_importance(crit, net, scores), args.ratio, args.imgsz)):
                if j == gi:
                    row["layer"] = next(k for k, m in net.named_modules() if m is g[0].dep.layer)
                    g.prune()
                    break
            row[crit] = evaluate(args.weights, net, args, run_dir, "single")["map50_95"]
        rows.append(row)
        print(f"  group {gi:2d} {row['layer']:28s} " + "  ".join(f"{c} {row[c]:.4f}" for c in args.criteria), flush=True)

    with (args.output_dir / "single_layer_map.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["group", "layer", *args.criteria])
        w.writeheader()
        w.writerows(rows)
    print(f"\n  unpruned {dense:.4f}")
    for c in args.criteria:
        vals = sorted(r[c] for r in rows)
        print(f"  {c:7s} min {vals[0]:.4f}  median {vals[len(vals) // 2]:.4f}  max {vals[-1]:.4f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default="yolo26n.pt")
    p.add_argument("--data", default="coco.yaml")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0")
    p.add_argument("--ratio", type=float, default=0.3)
    p.add_argument("--criteria", nargs="+", default=["h0", "l2", "random"])
    p.add_argument("--calib-images", type=int, default=32)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/yolo26_tda/verify"))
    p.add_argument("--skip-map", action="store_true")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base = load_prunable(args.weights)
    images = calibration_images(args.data, args.calib_images, args.imgsz, seed=0)
    scores = yolo_tda.compute_persistence_scores(base, images, max_side=64, workers=4)
    ok = check_equivalence(base, lambda net: yolo_tda.build_importance("h0", net, scores), args, images[:4])
    if not args.skip_map:
        single_layer_map(base, args, scores)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
