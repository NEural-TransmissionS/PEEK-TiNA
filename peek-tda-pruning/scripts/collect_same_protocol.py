"""Collect the WSD + COCO + viewsphere same-protocol runs into results/same_protocol/*.csv.

Same protocol = 20-epoch fine-tune (AdamW, momentum 0.9, warmup 1, batch 16) after structural
channel pruning or block cut, train-loader seeds fixed. WSD uses lr 1.25e-3 (the lr its models were
trained with); COCO uses lr 1.25e-4 and a fixed 10k train2017 subset; viewsphere (v1_wsd4, base =
WSD -> viewsphere fine-tunes) uses lr 1.25e-3 with the matched warmup bias lr throughout (see README).
Writes channel_runs / block_runs (20-epoch results), curves (their per-epoch val mAP), and for the
recovery study probe_curves (every 50 batches), budget_runs (separate fine-tunes of B batches, incl.
COCO on full train2017) and inference_{gpu,cpu} (scripts/bench_inference.py timings).
Tables and figures: scripts/same_protocol_tables.py and scripts/recovery_tables.py; the older
WSD/shapes results are collected by scripts/collect_results.py.

    python scripts/collect_same_protocol.py --channel-root .../PEEK-rvc-pruning/outputs/yolo26_tda \
        --block-results .../topo-prune-shapes/results
"""

import argparse
import csv
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "results" / "same_protocol"
# output dir -> (dataset, model, set, basis)
CHANNEL_DIRS = {}
for m in "ns":
    CHANNEL_DIRS |= {
        f"wsd_{m}_protect6_peek_ft20": ("wsd", m, "grid", "peek"),
        f"wsd_{m}_protect6_seeds_ft20": ("wsd", m, "seeds", "peek"),
        f"coco_{m}_protect6_peek_ft20": ("coco", m, "grid", "peek"),
        f"coco_{m}_protect6_activation_ft20": ("coco", m, "grid", "activation"),
        f"coco_{m}_protect6_seeds_ft20": ("coco", m, "seeds", "peek"),
        f"coco_{m}_protect6_oneshot": ("coco", m, "oneshot", "peek"),
        f"viewsphere_{m}_protect6_peek_ft20": ("viewsphere", m, "grid", "peek"),
        f"viewsphere_{m}_protect6_activation_ft20": ("viewsphere", m, "grid", "activation"),
        f"viewsphere_{m}_protect6_seeds_ft20": ("viewsphere", m, "seeds", "peek"),
        f"viewsphere_{m}_protect6_oneshot": ("viewsphere", m, "oneshot", "peek"),
    }
DATASETS = ("wsd", "coco", "viewsphere")
CHANNEL_DIRS |= {
    "coco_n_protect6_activation_oneshot": ("coco", "n", "oneshot", "activation"),
    "coco_n_protect6_lrpilot_lr0.000125": ("coco", "n", "lrpilot", "peek"),
    "coco_n_protect6_lrpilot_lr0.00025": ("coco", "n", "lrpilot", "peek"),
    "coco_n_protect6_peek_ft20_lr1.25e-3": ("coco", "n", "lrpilot", "peek"),
}
CHANNEL_COLS = ["dataset", "model", "set", "basis", "criterion", "ratio", "seed", "epochs", "optimizer",
                "lr0", "finetune_data", "params", "gflops", "map_before", "map_bn", "map_ft"]
BLOCK_COLS = ["dataset", "model", "name", "criterion", "k", "scope", "norm", "random_order", "layers_cut", "lr0",
              "epochs_run", "params", "gflops", "map50_95", "map50"]


def channel(root: Path) -> list[dict]:
    rows = []
    for d, (ds, m, kind, basis) in CHANNEL_DIRS.items():
        for f in sorted((root / d / "runs").glob("*/result.json")):
            r = json.loads(f.read_text())
            if kind == "lrpilot" and d.endswith("1.25e-3"):
                r.setdefault("lr0", 0.00125)
            ft = r.get("after_finetune") or {}
            rows.append({
                "dataset": ds, "model": m, "set": kind, "basis": basis, "criterion": r["criterion"],
                "ratio": r["ratio"], "seed": r["seed"], "epochs": r["epochs"], "optimizer": r.get("optimizer", ""),
                "lr0": r.get("lr0", ""), "finetune_data": r.get("finetune_data", ""),
                "params": r["pruned"]["params"], "gflops": round(r["pruned"]["gflops"], 4),
                "map_before": round(r["before_finetune"]["map50_95"], 4),
                "map_bn": round(r["after_bn_recal"]["map50_95"], 4),
                "map_ft": round(ft["map50_95"], 4) if ft else "",
            })
    return rows


def blocks(results: Path) -> list[dict]:
    rows = []
    for ds in DATASETS:
        for f in sorted(results.glob(f"{ds}_*_ft.json")):
            r = json.loads(f.read_text())
            rows.append({
                "dataset": ds, "model": r["size"], "name": f.stem, "criterion": r.get("criterion", "combined"), "k": r["k"], "scope": r.get("scope", "head"),
                "norm": r.get("norm", "raw"), "random_order": "" if r.get("random_order") is None else r["random_order"],
                "layers_cut": " ".join(map(str, sorted(r["layers_cut"]))), "lr0": r.get("lr0", ""),
                "epochs_run": r.get("epochs_run", ""), "params": r["params"], "gflops": r["gflops"],
                "map50_95": r["mAP50-95"], "map50": r["mAP50"],
            })
    return rows


def scratch_blocks(results: Path) -> list[dict]:
    """Viewsphere block cuts retrained from scratch (1000 epochs, like WSD's), kept apart from the
    cut + fine-tune runs above: results/<name>.json without the _ft suffix."""
    rows = []
    for f in sorted(results.glob("viewsphere_*.json")):
        if "_ft" in f.stem:
            continue
        r = json.loads(f.read_text())
        rows.append({
            "dataset": "viewsphere", "model": r["size"], "name": f.stem, "criterion": r.get("criterion", "combined"),
            "k": r["k"], "scope": r.get("scope", "head"), "norm": r.get("norm", "raw"),
            "random_order": "" if r.get("random_order") is None else r["random_order"],
            "layers_cut": " ".join(map(str, sorted(r["layers_cut"]))), "lr0": r.get("lr0", ""),
            "epochs_run": r.get("epochs_run", ""), "params": r["params"], "gflops": r["gflops"],
            "map50_95": r["mAP50-95"], "map50": r["mAP50"],
        })
    return rows


CURVE_COLS = ["dataset", "model", "method", "basis", "criterion", "ratio", "seed", "layers_cut", "scope", "norm",
              "random_order", "variant", "budget", "batch", "epoch", "map50_95"]
BUDGETS = (250, 500, 1000, 2500, 5000, 25000, 50000)  # 25k/50k: COCO only


def _block_id(r: dict) -> dict:
    return {"method": "block", "basis": "peek", "criterion": r.get("criterion", "combined"), "ratio": "",
            "seed": r.get("seed", 42), "layers_cut": " ".join(map(str, sorted(r["layers_cut"]))),
            "scope": r.get("scope", "head"), "norm": r.get("norm", "raw"),
            "random_order": "" if r.get("random_order") is None else r["random_order"]}


def _channel_id(r: dict, basis: str) -> dict:
    return {"method": "channel", "basis": basis, "criterion": r["criterion"], "ratio": r["ratio"], "seed": r["seed"],
            "layers_cut": "", "scope": "", "norm": "", "random_order": ""}


def curves(root: Path, results: Path) -> list[dict]:
    """Per-epoch val mAP50-95 of every 20-epoch fine-tune (Ultralytics results.csv); epoch 0 = the
    starting model (channel: after BN re-estimation; block: precut_eval.json when present)."""
    rows = []
    for d, (ds, m, kind, basis) in CHANNEL_DIRS.items():
        if kind not in ("grid", "seeds"):
            continue
        for f in sorted((root / d / "runs").glob("*/finetune/results.csv")):
            r = json.loads((f.parents[1] / "result.json").read_text())
            key = {"dataset": ds, "model": m, **_channel_id(r, basis), "variant": "", "budget": ""}
            rows.append({**key, "batch": 0, "epoch": 0, "map50_95": r["after_bn_recal"]["map50_95"]})
            with open(f) as fh:
                for e in csv.DictReader(fh):
                    rows.append({**key, "batch": "", "epoch": int(e["epoch"]),
                                 "map50_95": float(e["metrics/mAP50-95(B)"])})
    pre = results / "precut_eval.json"
    pre = {(p["dataset"], p["model"], p["layers_cut"]): p["map50_95_precut"]
           for p in (json.loads(pre.read_text()) if pre.exists() else [])}
    for ds in DATASETS:
        for f in sorted(results.glob(f"{ds}_*_ft.json")):
            r = json.loads(f.read_text())
            key = {"dataset": ds, "model": r["size"], **_block_id(r), "variant": "", "budget": ""}
            if (ds, r["size"], key["layers_cut"]) in pre:
                rows.append({**key, "batch": 0, "epoch": 0, "map50_95": pre[(ds, r["size"], key["layers_cut"])]})
            with open(results.parent / "runs" / f.stem / "results.csv") as fh:
                for e in csv.DictReader(fh):
                    rows.append({**key, "batch": "", "epoch": int(e["epoch"]),
                                 "map50_95": float(e["metrics/mAP50-95(B)"])})
    return rows


def probes(root: Path, results: Path) -> list[dict]:
    """Every-50-batch curves of the recovery probes (iter_probe.py): channel runs in
    <ds>_<m>_protect6_probe[_wblr]/, block runs named *_ft_probe[_wblr]. variant "wblr" = warmup bias lr
    started at lr0 instead of Ultralytics' 0.1."""
    rows = []
    for v in ("", "_wblr"):
      for ds in DATASETS:
        for m in "ns":
            for f in sorted((root / f"{ds}_{m}_protect6_probe{v}" / "runs").glob("*/probe.csv")):
                r = json.loads((f.parent / "result.json").read_text())
                key = {"dataset": ds, "model": m, **_channel_id(r, "peek"), "variant": v[1:], "budget": ""}
                with open(f) as fh:
                    rows += [{**key, "batch": int(e["batch"]), "epoch": int(e["epoch"]), "map50_95": float(e["map50_95"])}
                             for e in csv.DictReader(fh)]
        for f in sorted(results.glob(f"{ds}_*_ft_probe{v}.json")):
            r = json.loads(f.read_text())
            key = {"dataset": ds, "model": r["size"], **_block_id(r), "variant": v[1:], "budget": ""}
            with open(results.parent / "runs" / f.stem / "probe.csv") as fh:
                rows += [{**key, "batch": int(e["batch"]), "epoch": int(e["epoch"]), "map50_95": float(e["map50_95"])}
                         for e in csv.DictReader(fh)]
    return rows


def budgets(root: Path, results: Path) -> list[dict]:
    """Final full-val mAP of the training-budget sweep (separate fine-tunes of B batches); variant "wblr" =
    warmup bias lr started at lr0 (<ds>_<m>_protect6_budget<B>_wblr/, blocks *_ft_b<B>_wblr)."""
    rows = []
    for v, b in [(v, b) for v in ("", "_wblr") for b in BUDGETS]:
        for ds in DATASETS:
            for m in "ns":
                for f in sorted((root / f"{ds}_{m}_protect6_budget{b}{v}" / "runs").glob("*/result.json")):
                    r = json.loads(f.read_text())
                    if "after_finetune" in r:
                        rows.append({"dataset": ds, "model": m, **_channel_id(r, "peek"), "variant": v[1:], "budget": b, "batch": "",
                                     "epoch": r["epochs"], "map50_95": r["after_finetune"]["map50_95"]})
            for f in sorted(results.glob(f"{ds}_*_ft_b{b}{v}.json")):
                r = json.loads(f.read_text())
                if r.get("mAP50-95") is not None:
                    rows.append({"dataset": ds, "model": r["size"], **_block_id(r), "variant": v[1:], "budget": b, "batch": "",
                                 "epoch": r["epochs"], "map50_95": r["mAP50-95"]})
    # COCO fine-tuned on all of train2017 (7,393 batches per epoch), matched warmup: variant "wblr_full",
    # budget = epochs x 7,393 (<ds>_<m>_protect6_full_e<E>_wblr/, blocks *_ft_full_e<E>_wblr)
    for e in (3, 7):
        b = e * 7393
        for m in "ns":
            for f in sorted((root / f"coco_{m}_protect6_full_e{e}_wblr" / "runs").glob("*/result.json")):
                r = json.loads(f.read_text())
                if "after_finetune" in r:
                    rows.append({"dataset": "coco", "model": m, **_channel_id(r, "peek"), "variant": "wblr_full", "budget": b,
                                 "batch": "", "epoch": e, "map50_95": r["after_finetune"]["map50_95"]})
        for f in sorted(results.glob(f"coco_*_ft_full_e{e}_wblr.json")):
            r = json.loads(f.read_text())
            if r.get("mAP50-95") is not None:
                rows.append({"dataset": "coco", "model": r["size"], **_block_id(r), "variant": "wblr_full", "budget": b,
                             "batch": "", "epoch": e, "map50_95": r["mAP50-95"]})
    return rows


def write(name: str, cols: list[str], rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"{name}: {len(rows)} rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channel-root", type=Path, required=True, help="PEEK-rvc-pruning/outputs/yolo26_tda")
    ap.add_argument("--block-results", type=Path, required=True, help="topo-prune-shapes/results")
    a = ap.parse_args()
    write("channel_runs.csv", CHANNEL_COLS, channel(a.channel_root))
    write("block_runs.csv", BLOCK_COLS, blocks(a.block_results))
    write("block_scratch_runs.csv", BLOCK_COLS, scratch_blocks(a.block_results))
    write("curves.csv", CURVE_COLS, curves(a.channel_root, a.block_results))
    write("probe_curves.csv", CURVE_COLS, probes(a.channel_root, a.block_results))
    write("budget_runs.csv", CURVE_COLS, budgets(a.channel_root, a.block_results))
    bench = a.channel_root.parent / "bench"
    for dev in ("gpu", "cpu"):  # inference timing (scripts/bench_inference.py), copied as is;
        # bench_<dev>_vs.csv = all three datasets timed together on one node (preferred when present)
        src = next((f for f in (bench / f"bench_{dev}_vs.csv", bench / f"bench_{dev}.csv") if f.exists()), None)
        if src:
            (OUT / f"inference_{dev}.csv").write_text(src.read_text())
            print(f"inference_{dev}.csv copied from {src.name}")
