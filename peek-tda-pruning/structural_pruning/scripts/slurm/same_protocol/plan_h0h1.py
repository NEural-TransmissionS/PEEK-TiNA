"""Plan the H0-only / H1-only block cuts for submit_h0h1.sh (run from topo-prune-shapes).

A cut model depends only on which blocks are cut, so a cut set an earlier run already trained (any ranking,
or a random order at seed 42) is not trained again. Writes three TSVs into <out dir>:
  h0h1_blocks_ft.tsv    ds  m  label  cut  args  seeds   cut + 20-epoch fine-tune, seeds still missing
  h0h1_blocks_rec.tsv   ds  m  args                      cut sets with no recovery runs (probe / budget)
  h0h1_blocks_full.tsv  ds  m  args                      COCO full-train2017 cuts (n: null k1/k3, s: grid k1/k3)
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, ".")
from topo_prune import cut_order, ranking_path

out = Path(sys.argv[1])
done = {}  # (ds, m, cut) -> kinds trained: ft<seed>, b<B>_s<seed>, full<E>, probe, probe_wblr
for f in Path("results").glob("*.json"):
    mm = re.match(r"(wsd|coco|viewsphere)_([ns])_.*_ft(_s(\d+))?(_b(\d+)_wblr|_full_e(\d)_wblr|_probe(_wblr)?)?$", f.stem)
    if not mm:
        continue
    r = json.loads(f.read_text())
    seed = int(mm.group(4) or 42)
    kind = (f"b{mm.group(6)}_s{seed}" if mm.group(6) else f"full{mm.group(7)}" if mm.group(7)
            else "probe" + (mm.group(8) or "") if mm.group(5) else f"ft{seed}")
    done.setdefault((mm.group(1), mm.group(2), frozenset(r["layers_cut"])), set()).add(kind)

cuts = {}  # (ds, m, cut) -> (label, args) of the first ranking that picks it
full_cuts = {}  # COCO full-train2017 subset, as before: null on n, grid on s, k = 1 and 3
for ds in ("wsd", "coco", "viewsphere"):
    for m in "ns":
        head = json.loads(ranking_path(ds, m, "peek", "head", "").read_text())
        allr = json.loads(ranking_path(ds, m, "peek", "all", "norm").read_text())
        for crit in ("h0", "h1"):
            for lab, r, scope in (("head", head, "--scope head"), ("grid", {**allr, **allr["grid"]}, "--scope all --norm grid"),
                                  ("null", {**allr, **allr["null"]}, "--scope all --norm null")):
                order = cut_order(r, crit)[0]
                for k in (1, 2, 3):
                    key, args = (ds, m, frozenset(order[:k])), f"--criterion {crit} --k {k} {scope}"
                    cuts.setdefault(key, (f"{crit}-{lab}-k{k}", args))
                    if ds == "coco" and lab == ("null" if m == "n" else "grid") and k in (1, 3):
                        full_cuts.setdefault(key, args)

ft, rec, full = [], [], []
for (ds, m, cut), (lab, args) in cuts.items():
    have = done.get((ds, m, cut), set())
    seeds = [s for s in (42, 43, 44) if f"ft{s}" not in have]
    if seeds:
        ft.append(f"{ds}\t{m}\t{lab}\t{sorted(cut)}\t{args}\t{','.join(map(str, seeds))}")
    if not any(k.startswith("b") for k in have):
        rec.append(f"{ds}\t{m}\t{args}")
full = [f"{ds}\t{m}\t{args}" for (ds, m, cut), args in full_cuts.items() if "full7" not in done.get((ds, m, cut), set())]
for name, lines in (("ft", ft), ("rec", rec), ("full", full)):
    (out / f"h0h1_blocks_{name}.tsv").write_text("".join(l + "\n" for l in lines))
    print(f"h0h1_blocks_{name}.tsv: {len(lines)} cut sets")
print("block fine-tune jobs:", sum(len(l.split("\t")[5].split(",")) for l in ft))
