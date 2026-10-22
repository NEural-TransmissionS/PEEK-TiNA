"""Collect the WSD + COCO same-protocol runs into results/same_protocol/*.csv.

Same protocol = 20-epoch fine-tune (AdamW, momentum 0.9, warmup 1, batch 16) after structural
channel pruning or block cut, train-loader seeds fixed. WSD uses lr 1.25e-3 (the lr its models were
trained with); COCO uses lr 1.25e-4 and a fixed 10k train2017 subset (see README).
Tables and figures are made from these CSVs by scripts/same_protocol_tables.py; the older WSD/shapes
results are collected by scripts/collect_results.py.

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
    }
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
    for ds in ("wsd", "coco"):
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
