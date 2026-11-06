import csv, json, glob, os, statistics as st
from collections import defaultdict
root = "/shared/projects/tina/PEEK-TiNA/PEEK-rvc-pruning/outputs/yolo26_tda"
rows = []
for m in ("n", "s"):
    d = f"{root}/wsd_{m}_protect6"
    base = json.load(open(f"{d}/baseline.json"))["map50_95"] / 100
    for f in sorted(glob.glob(f"{d}/runs/*/finetune/results.csv")):
        name = f.split("/runs/")[1].split("/")[0]
        crit, r = name.rsplit("_r", 1)[0], float(name.rsplit("_r", 1)[1].split("_")[0])
        ys = [float({k.strip(): v for k, v in x.items()}["metrics/mAP50-95(B)"]) for x in csv.DictReader(open(f))]
        hit = lambda p: next((i + 1 for i, y in enumerate(ys) if y >= p * base), None)
        rows.append(dict(model=m, crit=crit, ratio=r, base=base, ep1=ys[0], e90=hit(.90), e95=hit(.95),
                         auc10=st.mean(ys[:10]), auc25=st.mean(ys[:25]), final=ys[-1], best=max(ys)))
with open(f"{root}/wsd_recovery.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, rows[0].keys()); w.writeheader(); w.writerows(rows)
fmt = lambda v: "  -" if v is None else f"{v:3d}"
order = ["l2", "field_total", "field_h0", "field_h1", "total", "h0", "h1", "random"]
for m in ("n", "s"):
    R = [x for x in rows if x["model"] == m]
    print(f"\n=== {m}  baseline mAP50-95 {R[0]['base']:.4f} ===")
    nn = [x for x in R if x["crit"] == "none"][0]
    print(f"none(unpruned FT): ep1 {nn['ep1']:.3f} e90 {fmt(nn['e90'])} e95 {fmt(nn['e95'])} auc10 {nn['auc10']:.3f}")
    print("crit         | e95 by ratio .1 .2 .3 .4 .5 | e90 by ratio .1 .2 .3 .4 .5 | mean ep1 auc10 auc25 final | never95")
    for c in order:
        C = sorted([x for x in R if x["crit"] == c], key=lambda x: x["ratio"])
        print(f"{c:12s} | {' '.join(fmt(x['e95']) for x in C)} | {' '.join(fmt(x['e90']) for x in C)} | "
              f"{st.mean(x['ep1'] for x in C):.3f} {st.mean(x['auc10'] for x in C):.3f} {st.mean(x['auc25'] for x in C):.3f} "
              f"{st.mean(x['final'] for x in C):.3f} | {sum(x['e95'] is None for x in C)}")
    print("auc10 rank per ratio:")
    for r in (.1, .2, .3, .4, .5):
        C = sorted([x for x in R if x["ratio"] == r and x["crit"] != "none"], key=lambda x: -x["auc10"])
        print(f"  r{r}: " + " > ".join(f"{x['crit']}({x['auc10']:.3f})" for x in C))
