"""Plan the YOLOv5u block cuts of one dataset for submit_yolov5.sh (run from topo-prune-shapes).

A cut model depends only on which blocks are cut, so each cut set is trained once, under the first
ranking (or random order) that picks it. Writes into <out dir>:
  yolov5_<ds>_blocks_ft.tsv    ds  m  label  cut  args  seeds   cut + 20-epoch fine-tune: k0 and ranked sets
                                                                seeds 42-44, random-order-only sets seed 42
  yolov5_<ds>_blocks_rec.tsv   ds  m  args                      k0 + ranked sets, for the recovery tests
  yolov5_<ds>_blocks_full.tsv  ds  m  args                      COCO full train2017: null on v5n, grid on v5s,
                                                                k = 1 and 3, all three scores (as YOLO26)
    python plan_yolov5.py <ds> <out dir> [tag (default yolov5)] [models (default v5n v5s)]
With tag yolo26 and models n s it plans YOLO26 (added 2026-11-03 for viewsphere2); files are <tag>_<ds>_*.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from topo_prune import cut_order, ranking_path, scopes

ds, out = sys.argv[1], Path(sys.argv[2])
tag = sys.argv[3] if len(sys.argv) > 3 else "yolov5"
models = sys.argv[4:] or ["v5n", "v5s"]
ft, rec, full = [], [], []
for m in models:
    seen = {frozenset(): ("k0", "--k 0", True)}  # cut -> (label, args, ranked)
    full_cuts = {}
    head = json.loads(ranking_path(ds, m, "peek", "head", "").read_text())
    allr = json.loads(ranking_path(ds, m, "peek", "all", "norm").read_text())
    for crit in ("combined", "h0", "h1"):
        for lab, r, scope in (("head", head, "--scope head"), ("grid", {**allr, **allr["grid"]}, "--scope all --norm grid"),
                              ("null", {**allr, **allr["null"]}, "--scope all --norm null")):
            order = cut_order(r, crit)[0]
            for k in (1, 2, 3):
                key, args = frozenset(order[:k]), f"--criterion {crit} --k {k} {scope}"
                seen.setdefault(key, (f"{crit}-{lab}-k{k}", args, True))
                if ds == "coco" and lab == ("null" if m.endswith("n") else "grid") and k in (1, 3):
                    full_cuts.setdefault(key, args)
    for s in range(23):  # random block orders, as _train_one.py --random-order
        order = [int(i) for i in np.random.default_rng(s).permutation(scopes(m)["all"])]
        for k in (1, 2, 3):
            seen.setdefault(frozenset(order[:k]), (f"rand{s}-k{k}", f"--k {k} --scope all --random-order {s}", False))
    for cut, (lab, args, ranked) in seen.items():
        ft.append(f"{ds}\t{m}\t{lab}\t{sorted(cut)}\t{args}\t{'42,43,44' if ranked else '42'}")
        if ranked:
            rec.append(f"{ds}\t{m}\t{args}")
    full += [f"{ds}\t{m}\t{args}" for args in full_cuts.values()]
for name, lines in (("ft", ft), ("rec", rec), ("full", full)):
    (out / f"{tag}_{ds}_blocks_{name}.tsv").write_text("".join(l + "\n" for l in lines))
    print(f"{tag}_{ds}_blocks_{name}.tsv: {len(lines)} cut sets")
print("block fine-tune jobs:", sum(len(l.split("\t")[5].split(",")) for l in ft))
