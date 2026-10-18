#!/usr/bin/env python3
"""Copy the cluster outputs into results/ (run on the cluster).

    python scripts/collect_results.py --structural ../PEEK-rvc-pruning/outputs/yolo26_tda \
        --block ../topo-prune-shapes --wsd-baselines ../runs/wsd71_scratch

Every run's Ultralytics args.yaml is copied too (machine-local paths reduced to file
names), and results/base_models/ records how the starting checkpoints were trained.

Both TDA bases were run on one setup (same models, seed, calibration images, 20-epoch
fine-tune); they differ only in the map the topology is read from.

Structural: ``<tag>_protect6_peek_ft20`` (full grid) -> results/structural_pruning/peek/<tag>/
and ``<tag>_protect6_activation_ft20`` (the 6 TDA criteria; l2 / random / none are
basis-independent and live in the PEEK grid) -> results/structural_pruning/activation/<tag>/,
each with curves.csv (per-epoch val mAP of every fine-tune), plus the random-mask spread
for every model (one-shot seeds 0-9, 20-epoch seeds 43/44).

Block: each basis's rankings, and one row per (dataset, size, criterion, k) giving the k
lowest-ranked blocks. A cut model depends only on which blocks are cut, so each row
points at the trained run with exactly that cut set (``source_run``); runs/ holds them all.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import yaml

RES = Path(__file__).resolve().parent.parent / "results"
CRITERIA = {"combined": "combined", "h0": "h0_secondary_sum", "h1": "h1_sum"}
BASES = {"peek": "ranking_peek", "activation": "ranking"}
FILES = ("results.csv", "map_vs_ratio.png", "layer_persistence.csv", "layer_field_drift.csv")


def _scrub(value):
    """Machine-local paths (cluster and laptop) -> file name only."""
    if isinstance(value, str) and ("/" in value or "\\" in value):
        return value.replace("\\", "/").rstrip("/").split("/")[-1]
    return value


def write_args(args: dict, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(yaml.safe_dump({k: _scrub(v) for k, v in args.items()}, sort_keys=False))


def run_args(out: Path, dst: Path) -> None:
    """Every fine-tune's Ultralytics args.yaml under ``out`` -> ``dst/<run>.yaml``."""
    for a in sorted(out.glob("runs/*/finetune/args.yaml")):
        write_args(yaml.safe_load(a.read_text()), dst / f"{a.parent.parent.name}.yaml")


def base_models(shapes_weights: Path, wsd_runs: Path) -> None:
    """Training args of the checkpoints every pruning run starts from."""
    import torch

    dst = RES / "base_models"
    shutil.rmtree(dst, ignore_errors=True)
    for w in sorted(shapes_weights.glob("*.pt")):
        if not w.stem.startswith("wsd"):
            ck = torch.load(w, map_location="cpu", weights_only=False)
            write_args(ck["train_args"], dst / f"shapes_{w.stem}.yaml")
    for m in "ns":
        args = yaml.safe_load((wsd_runs / f"yolo26{m}_scratch_seed42" / "args.yaml").read_text())
        write_args(args, dst / f"wsd_{m}.yaml")


def curves(out: Path, dst: Path) -> None:
    """Per-epoch val mAP of every fine-tune under ``out`` -> one long CSV."""
    rows = []
    for log in sorted(out.glob("runs/*/finetune/results.csv")):
        r = json.loads((log.parent.parent / "result.json").read_text())
        for e in csv.DictReader(log.open()):
            e = {k.strip(): v for k, v in e.items()}
            rows.append({"criterion": r["criterion"], "ratio": r["ratio"], "seed": r["seed"],
                         "epoch": int(e["epoch"]), "map50_95": float(e["metrics/mAP50-95(B)"]),
                         "map50": float(e["metrics/mAP50(B)"]), "time_s": float(e["time"])})
    with dst.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def structural(src: Path) -> None:
    dst_root = RES / "structural_pruning"
    shutil.rmtree(dst_root, ignore_errors=True)
    tags = set()
    for basis in BASES:
        for out in sorted(src.glob(f"*_protect6_{basis}_ft20")):
            tag = out.name.replace(f"_protect6_{basis}_ft20", "")
            tags.add(tag)
            dst = dst_root / basis / tag
            (dst / "runs").mkdir(parents=True)
            for f in FILES:
                if (out / f).exists():
                    shutil.copy2(out / f, dst / f)
            for r in sorted(out.glob("runs/*/result.json")):
                shutil.copy2(r, dst / "runs" / f"{r.parent.name}.json")
            curves(out, dst / "curves.csv")
            run_args(out, dst / "args")
    # random pruning on more seeds: 10 one-shot masks (seeds 0-9), 20-epoch seeds 43/44
    spread = dst_root / "random_spread"
    spread.mkdir()
    for tag in sorted(tags):
        shutil.copy2(src / f"{tag}_protect6_oneshot" / "results.csv", spread / f"{tag}_oneshot.csv")
        shutil.copy2(src / f"{tag}_protect6_ft20" / "results.csv", spread / f"{tag}_ft20_random.csv")
        curves(src / f"{tag}_protect6_ft20", spread / f"{tag}_ft20_curves.csv")
        run_args(src / f"{tag}_protect6_ft20", spread / f"{tag}_ft20_args")


def block(src: Path) -> None:
    dst_root = RES / "block_pruning"
    shutil.rmtree(dst_root, ignore_errors=True)
    (dst_root / "runs").mkdir(parents=True)
    runs = {p.stem: json.loads(p.read_text()) for p in (src / "results").glob("*.json")}
    used = set()
    for basis, rank_dir in BASES.items():
        dst = dst_root / basis
        shutil.copytree(src / rank_dir, dst / "ranking")
        rows = []
        for rank_path in sorted((dst / "ranking").glob("*.json")):
            ranking = json.loads(rank_path.read_text())
            ds, size = rank_path.stem.split("_")
            same = {n: r for n, r in runs.items() if (r["dataset"], r["size"]) == (ds, size)}
            for crit, key in CRITERIA.items():
                order = sorted(ranking["candidates"], key=lambda i: ranking[key][str(i)])
                for k in range(4):
                    cut = order[:k]
                    tag = "" if crit == "combined" or k == 0 else f"_{crit}"
                    prefer = [f"{ds}_{size}{tag}_k{k}_{basis}", f"{ds}_{size}{tag}_k{k}"]
                    match = [n for n in prefer if n in same and set(same[n]["layers_cut"]) == set(cut)]
                    match += sorted(n for n, r in same.items() if r["k"] == k and set(r["layers_cut"]) == set(cut))
                    if not match:
                        raise SystemExit(f"{basis}: no trained run for {ds}_{size} {crit} k={k} cut={cut}")
                    run = same[match[0]]
                    used.add(match[0])
                    rows.append({
                        "dataset": ds, "size": size, "criterion": crit, "k": k, "layers_cut": cut,
                        "cut_persistence": [ranking[key][str(i)] for i in cut],
                        **{c: run[c] for c in ("mAP50-95", "mAP50", "gflops", "params", "epochs")},
                        "source_run": match[0],
                    })
        base = {(r["dataset"], r["size"]): r for r in rows if r["k"] == 0}
        for r in rows:
            b = base[(r["dataset"], r["size"])]
            r["gflops_vs_full_%"] = round(100 * (r["gflops"] - b["gflops"]) / b["gflops"], 2)
            r["mAP_delta_vs_full"] = round(r["mAP50-95"] - b["mAP50-95"], 4)
        with (dst / "results.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"block/{basis}: {len(rows)} rows")
    (dst_root / "curves").mkdir()
    for name in sorted(used):
        shutil.copy2(src / "results" / f"{name}.json", dst_root / "runs" / f"{name}.json")
        write_args(yaml.safe_load((src / "runs" / name / "args.yaml").read_text()), dst_root / "args" / f"{name}.yaml")
        shutil.copy2(src / "runs" / name / "results.csv", dst_root / "curves" / f"{name}.csv")  # per-epoch log
    print(f"block: {len(used)} trained runs")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--structural", type=Path, required=True)
    ap.add_argument("--block", type=Path, required=True)
    ap.add_argument("--wsd-baselines", type=Path, required=True,
                    help="dir holding yolo26{n,s}_scratch_seed42/args.yaml")
    args = ap.parse_args()
    structural(args.structural)
    block(args.block)
    base_models(Path(__file__).resolve().parent.parent / "block_pruning" / "weights", args.wsd_baselines)
