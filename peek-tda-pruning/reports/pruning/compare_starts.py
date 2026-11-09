"""Builds tables/start_compare.tex: channel pruning from the from-scratch vs the COCO-pretrained starting models.

Per dataset and starting model: how many of the 20 settings (YOLO26n/s, YOLOv5n/s x 10-50% pruning) a
persistence score (field H1 / H0 / H0+H1) wins, and how far those three scores are from L2 on average,
one-shot and after the 20-epoch fine-tune. Reads results/same_protocol (COCO start) and
results/same_protocol_scratch (from scratch, the results before the COCO-base re-run); the per-dataset
tables come from make_tables.py.

    python compare_starts.py
"""

import csv
import statistics as st

import make_tables as mt

STARTS = {"same_protocol_scratch": "From scratch", "same_protocol": "COCO-pretrained"}
DATA = {"wsd": "WSD", "viewsphere": "Viewsphere", "viewsphere2": "Viewsphere v2", "coco": "COCO"}
MODELS = ("n", "s", "v5n", "v5s")
RATIOS = ("0.1", "0.2", "0.3", "0.4", "0.5")
PERS = ("field_h1", "field_h0", "field_total")
GAPS = ("0.3", "0.5")  # ratios whose gap to L2 is shown


def summary(t, stage):
    """(settings a persistence score wins, mean gap of the persistence scores to L2 at each GAPS ratio)."""
    wins = 0
    for m in MODELS:
        for x in RATIOS:
            means = {c: st.mean(t[(stage, m, c, x)]) for c in mt.CRIT if t.get((stage, m, c, x))}
            wins += max(means[c] for c in PERS) == max(means.values())
    gap = {x: st.mean(st.mean(t[(stage, m, c, x)]) - st.mean(t[(stage, m, "l2", x)]) for m in MODELS for c in PERS)
           for x in GAPS}
    return wins, gap


def cells(t, stage):
    wins, gap = summary(t, stage)
    return [f"{wins}/20"] + [f"{gap[x]:+.3f}".replace("-", "$-$") for x in GAPS]


if __name__ == "__main__":
    lines = [r"\begin{tabular}{ll" + "r" * 6 + "}", r"\toprule",
             r" & & \multicolumn{3}{c}{One-shot} & \multicolumn{3}{c}{After 20-epoch fine-tune} \\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
             "Dataset & Starting model & " + " & ".join(["Wins"] + [f"vs L2, {mt.pct(x)}" for x in GAPS]) * 1
             + " & " + " & ".join(["Wins"] + [f"vs L2, {mt.pct(x)}" for x in GAPS]) + r" \\"]
    rows = {}
    for src in STARTS:
        with open(mt.RESULTS / src / "channel_runs.csv") as fh:
            rows[src] = list(csv.DictReader(fh))
    for ds, name in DATA.items():
        lines.append(r"\midrule")
        starts = STARTS if ds != "coco" else {"same_protocol": "COCO weights (both runs)"}
        for i, (src, lab) in enumerate(starts.items()):
            t = mt.channel_runs(rows[src], ds, MODELS, "")
            lines.append((name if i == 0 else "") + f" & {lab} & " + " & ".join(cells(t, "os") + cells(t, "ft")) + r" \\")
    mt.write("start_compare", lines + [r"\bottomrule", r"\end{tabular}"])
    print("written", mt.OUT / "start_compare.tex")
