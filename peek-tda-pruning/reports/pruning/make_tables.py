"""Builds the LaTeX tables for pruning_report.tex from the tina results CSVs (results/same_protocol/).

Writes tables/ once per architecture (YOLO26: no suffix; YOLOv5u: _v5; Viewsphere v2 versions add _vs2):<ds>_channel (one-shot and fine-tuned mAP of field H1 / H0 / H0+H1 / L2 / random, one per
dataset), recovery (pruning over training budgets, all datasets), block_gap (each block ranking -
TDA combined, H0, H1 scores and the L2 weight-norm baseline - minus random block choice), blocks (block rankings vs random orders, all datasets) and cost (GFLOPs and inference time of the channel-pruned models). The prose lives in pruning_report.tex; this file only turns numbers into tabulars.

    python make_tables.py [path to results/same_protocol; default: this repo's]
"""

import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# collected CSVs: tina/results/same_protocol (scripts/collect_same_protocol.py), or another copy as argv[1]
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / "results" / "same_protocol"
OUT = Path(__file__).resolve().parent / "tables"
DATA = {"wsd": "WSD", "viewsphere": "Viewsphere", "coco": "COCO"}
V2 = {"viewsphere2": "Viewsphere v2"}  # v2_wsd4, 17 satellites: written separately, with suffix _vs2
SHOW = ("0.1", "0.3", "0.5")  # ratios shown
CRIT = {"field_h1": "Field H1", "field_h0": "Field H0", "field_total": "Field H0+H1", "l2": "L2", "random": "Random"}
SCORED = ("field_h1", "field_h0", "field_total", "l2")  # one-shot masks from the seed-42 scores, BN seeds 43-51
# architecture -> (model ids, display name, table-name suffix, blocks eligible for replacement)
ARCHS = {"yolo26": (("n", "s"), "YOLO26", "", [2, 4, 6, 8, 13, 16, 19, 22]),   # C3k2 blocks
         "yolov5": (("v5n", "v5s"), "YOLOv5", "_v5", [2, 4, 6, 8, 13, 17, 20, 23])}  # C3 blocks (YOLOv5u)
MODELS, NAME, SFX, BLOCKS_ALL = ARCHS["yolo26"]  # the architecture being written (set in __main__)
N_RANDOM = 23


def load(name):
    with open(SRC / name) as fh:
        return list(csv.DictReader(fh))


def ratio(r):
    return f"{float(r['ratio']):.1f}"


def sd(v):
    return st.stdev(v) if len(v) > 1 else 0.0


def holds(a, b):
    """mean(a) > mean(b) by more than two standard errors (Welch)."""
    se = (sd(a) ** 2 / len(a) + sd(b) ** 2 / len(b)) ** 0.5
    return st.mean(a) - st.mean(b) > 2 * se


def write(name, lines):
    OUT.mkdir(exist_ok=True)
    (OUT / f"{name}{SFX}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def pct(x):
    return f"{float(x):.0%}".replace("%", r"\%")


def channel_runs(ch, ds, models, name):
    """t[(stage "os" / "ft", model, criterion, ratio)] -> mAP50-95 over seeds (also used by win_chart.py)."""
    t = defaultdict(list)
    for r in ch:
        if r["dataset"] != ds or r["basis"] != "peek":
            continue
        c, x = r["criterion"], ratio(r)
        if c in SCORED and (r["set"] == "oneshot_seeds" or (r["set"] == "grid" and r["seed"] == "42")):
            t[("os", r["model"], c, x)].append(float(r["map_bn"]))
        elif c == "random" and r["set"] == "oneshot":
            t[("os", r["model"], c, x)].append(float(r["map_bn"]))
        if r["set"] in ("grid", "seeds") and r["map_ft"] and c in ("none", *CRIT):
            t[("ft", r["model"], c, x)].append(float(r["map_ft"]))
        if r["set"] == "grid" and r["seed"] == "42" and c == "none":
            t[("os", r["model"], c, x)].append(float(r["map_bn"]))
    return t


def channel(ch, ds):
    """Rows: model x criterion; columns: one-shot and fine-tuned mAP at 10/30/50%. Best criterion in
    bold, with a dagger when it beats the runner-up by more than 2 standard errors."""
    t = channel_runs(ch, ds, MODELS, NAME)
    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r" & & \multicolumn{3}{c}{One-shot} & \multicolumn{3}{c}{After 20-epoch fine-tune} \\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
             "Model & Criterion & " + " & ".join(pct(x) for x in SHOW * 2) + r" \\", r"\midrule"]
    for m in MODELS:
        best = {}
        for s in ("os", "ft"):
            for x in SHOW:
                order = sorted(CRIT, key=lambda c: -st.mean(t[(s, m, c, x)]))
                best[(s, x)] = (order[0], holds(t[(s, m, order[0], x)], t[(s, m, order[1], x)]))
        lines.append(rf"{NAME}{m[-1]} & Unpruned & \multicolumn{{3}}{{c}}{{{st.mean(t[('os', m, 'none', '0.0')]):.3f}}}"
                     rf" & \multicolumn{{3}}{{c}}{{{st.mean(t[('ft', m, 'none', '0.0')]):.3f}}} \\")
        for c, lab in CRIT.items():
            cells = []
            for s in ("os", "ft"):
                for x in SHOW:
                    v = f"{st.mean(t[(s, m, c, x)]):.3f}"
                    b, sig = best[(s, x)]
                    cells.append(rf"\textbf{{{v}}}{'$^\\dagger$' if sig else ''}" if b == c else v)
            lines.append(f" & {lab} & " + " & ".join(cells) + r" \\")
        if m == MODELS[0]:
            lines.append(r"\midrule")
    write(f"{ds}_channel", lines + [r"\bottomrule", r"\end{tabular}"])


BUDGETS = (250, 500, 1000, 2500, 5000)
LONG = (25000, 50000, "full3", "full7")  # COCO only: 10k subset, then all of train2017 for 3 / 7 epochs


def budget_data(bud):
    """t[(ds, m, criterion, ratio, budget)] -> mAPs over seeds (channel pruning, matched warmup)."""
    t = defaultdict(list)
    for r in bud:
        if r["method"] == "channel" and r["variant"] in ("wblr", "wblr_full"):
            b = int(r["budget"]) if r["variant"] == "wblr" else f"full{r['epoch']}"
            t[(r["dataset"], r["model"], r["criterion"], f"{float(r['ratio']):.1f}", b)].append(float(r["map50_95"]))
    return {k: st.mean(v) for k, v in t.items()}


def budget_table(t, name, datasets, budgets, head):
    """Rows: unpruned + each criterion at 10/30/50%; columns: budgets for n, then s. Bold = best criterion."""
    nb = len(budgets)
    lines = [r"\begin{tabular}{ll" + "r" * 2 * nb + "}", r"\toprule",
             rf" & & \multicolumn{{{nb}}}{{c}}{{{NAME}n}} & \multicolumn{{{nb}}}{{c}}{{{NAME}s}} \\",
             rf"\cmidrule(lr){{3-{2 + nb}}}\cmidrule(lr){{{3 + nb}-{2 + 2 * nb}}}",
             "Data & Run & " + " & ".join(head * 2) + r" \\"]
    rows = [("none", "0.0", "Unpruned")] + [(c, x, f"{lab} {pct(x)}") for x in SHOW for c, lab in CRIT.items()]
    for ds in datasets:
        lines.append(r"\midrule")
        for i, (c, x, lab) in enumerate(rows):
            cells = []
            for m in MODELS:
                for b in budgets:
                    v = round(t[(ds, m, c, x, b)], 3)
                    best = c != "none" and v == max(round(t[(ds, m, cc, x, b)], 3) for cc in CRIT)
                    cells.append(rf"\textbf{{{v:.3f}}}" if best else f"{v:.3f}")
            if c == "field_h1" and x != SHOW[0]:
                lines.append(r"\addlinespace[2pt]")
            lines.append((DATA[ds] if i == 0 else "") + f" & {lab} & " + " & ".join(cells) + r" \\")
    write(name, lines + [r"\bottomrule", r"\end{tabular}"])


def recovery(bud):
    t = budget_data(bud)
    budget_table(t, "recovery", DATA, BUDGETS, [f"{b:,}" for b in BUDGETS])
    if "coco" in DATA:
        budget_table(t, "recovery_coco_long", ("coco",), LONG, ["25,000", "50,000", "full 3 ep", "full 7 ep"])
    recovery_speed(t)


def recovery_speed(t, frac=0.95):
    """Fewest training batches after which the pruned model reaches `frac` of the unpruned model trained
    for the same number of batches; 'never' = not within the budgets run. (An absolute 0.01 margin is
    reached almost nowhere, so it would not separate the criteria.)"""
    steps = {ds: list(BUDGETS) + (list(LONG) if ds == "coco" else []) for ds in DATA}
    label = lambda b: f"{b:,}" if isinstance(b, int) else f"{b[4:]} ep (full)"
    lines = [r"\begin{tabular}{ll" + "r" * 6 + "}", r"\toprule",
             rf" & & \multicolumn{{3}}{{c}}{{{NAME}n}} & \multicolumn{{3}}{{c}}{{{NAME}s}} \\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
             "Data & Criterion & " + " & ".join(pct(x) for x in SHOW * 2) + r" \\"]
    for ds, name in DATA.items():
        lines.append(r"\midrule")
        first = {}
        for m in MODELS:
            for x in SHOW:
                for c in CRIT:
                    hit = [i for i, b in enumerate(steps[ds]) if t[(ds, m, c, x, b)] >= frac * t[(ds, m, "none", "0.0", b)]]
                    first[(m, x, c)] = hit[0] if hit else len(steps[ds])
        for i, (c, lab) in enumerate(CRIT.items()):
            cells = []
            for m in MODELS:
                for x in SHOW:
                    j = first[(m, x, c)]
                    s = label(steps[ds][j]) if j < len(steps[ds]) else "never"
                    fastest = j < len(steps[ds]) and j == min(first[(m, x, cc)] for cc in CRIT)
                    cells.append(rf"\textbf{{{s}}}" if fastest else s)
            lines.append((name if i == 0 else "") + f" & {lab} & " + " & ".join(cells) + r" \\")
    write("recovery_speed", lines + [r"\bottomrule", r"\end{tabular}"])


def cost(ch, gpu, cpu):
    """Size and inference time of the channel-pruned models (identical for every criterion)."""
    size = {(r["dataset"], r["model"], ratio(r)): float(r["gflops"]) for r in ch
            if r["set"] == "grid" and r["seed"] == "42" and r["criterion"] in ("none", "field_h1")}
    tm = {(dev, r["dataset"], r["model"], f"{float(r['setting'] or 0):.1f}"): r
          for dev, rows in (("gpu", gpu), ("cpu", cpu)) for r in rows if r["method"] in ("unpruned", "channel")}
    head = ["GFLOPs", "GPU ms", "GPU img/s", "CPU ms"]
    lines = [r"\begin{tabular}{ll" + "rrrr" * 2 + "}", r"\toprule",
             rf" & & \multicolumn{{4}}{{c}}{{{NAME}n}} & \multicolumn{{4}}{{c}}{{{NAME}s}} \\",
             r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
             "Data & Pruned & " + " & ".join(head * 2) + r" \\"]
    xs = ("0.0", "0.1", "0.2", "0.3", "0.4", "0.5")
    # (format, value getter, better = lower?) per column; bold = best ratio for that dataset and model
    cols = [("{:.2f}", lambda ds, m, x: size[(ds, m, x)], True),
            ("{:.2f}", lambda ds, m, x: float(tm[("gpu", ds, m, x)]["b1_ms_median"]), True),
            ("{:,.0f}", lambda ds, m, x: float(tm[("gpu", ds, m, x)]["b32_img_per_s"]), False),
            ("{:.1f}", lambda ds, m, x: float(tm[("cpu", ds, m, x)]["b1_ms_median"]), True)]
    for ds, name in DATA.items():
        lines.append(r"\midrule")
        best = {(m, j): (min if low else max)(get(ds, m, x) for x in xs)
                for m in MODELS for j, (_, get, low) in enumerate(cols)}
        for i, x in enumerate(xs):
            cells = []
            for m in MODELS:
                for j, (fmt, get, _) in enumerate(cols):
                    v = get(ds, m, x)
                    cells.append(rf"\textbf{{{fmt.format(v)}}}" if v == best[(m, j)] else fmt.format(v))
            lines.append((name if i == 0 else "") + " & " + ("none" if x == "0.0" else pct(x)) + " & " + " & ".join(cells) + r" \\")
    write("cost", lines + [r"\bottomrule", r"\end{tabular}"])


def random_orders(k):
    return [frozenset(int(i) for i in np.random.default_rng(s).permutation(BLOCKS_ALL)[:k]) for s in range(N_RANDOM)]


LABS = {"head": "Head (raw)", "grid": "All, grid", "null": "All, null", "all": "All, raw"}
# block ranking scores: H0/H1 blend, H0 only, H1 only, L2 (conv-weight RMS baseline, added with the COCO-base
# re-run 2026-11-07; not run on COCO). L2 has rankings head and all (raw); the TDA scores head, grid and null.
SCORES = {"combined": "C", "h0": "H0", "h1": "H1", "l2": "L2"}
NA = "--"  # ranking not run (L2 on grid / null, TDA on all-raw, L2 on COCO)


def cut_runs(runs, ranked, ds, m, k, c, lab):
    """Seed -> mAPs of the cut that ranking picks, or None if that ranking was not run."""
    key = (ds, m, k, c, lab)
    return runs[(ds, m, ranked[key])] if key in ranked else None


def block_data(bl):
    """runs[(ds, m, cut)][seed] -> mAPs; ranked[(ds, m, k, score, ranking)] -> the cut set that ranking
    picks (the collector's `rankings` column; a set shared by rankings was trained once)."""
    runs, ranked = defaultdict(lambda: defaultdict(list)), {}
    for r in bl:
        cut = frozenset(int(x) for x in r["layers_cut"].split())
        runs[(r["dataset"], r["model"], cut)][r["seed"]].append(float(r["map50_95"]))
        for lab in r["rankings"].split():
            c, rk, k = lab.split("-")
            ranked[(r["dataset"], r["model"], int(k[1:]), c, rk)] = cut
    return runs, ranked


def block_header(first):
    n = len(SCORES)
    return [r"\begin{tabular}{ll" + "r" * 3 * n + "}", r"\toprule",
            rf"{first} & Ranking & " + " & ".join(rf"\multicolumn{{{n}}}{{c}}{{$k={k}$}}" for k in (1, 2, 3)) + r" \\",
            "".join(rf"\cmidrule(lr){{{3 + i * n}-{2 + (i + 1) * n}}}" for i in range(3)),
            " & & " +" & ".join(v for _ in (1, 2, 3) for v in SCORES.values()) + r" \\"]


def block_gap(runs, ranked):
    """Each ranking's cut minus the mean of the 23 random block orders; bold when the gap exceeds two
    standard errors (cut: 3 seeds, random: 23 orders)."""
    lines = block_header("Data, model")
    for ds, name in DATA.items():
        for m in MODELS:
            lines.append(r"\midrule")
            rnd = {k: [st.mean(runs[(ds, m, c)]["42"]) for c in random_orders(k)] for k in (1, 2, 3)}
            lines.append(f"{name} {NAME}{m[-1]} & Random (mean $\\pm$ sd) & "
                         + " & ".join(rf"\multicolumn{{{len(SCORES)}}}{{c}}{{{st.mean(rnd[k]):.3f} $\pm$ {sd(rnd[k]):.3f}}}" for k in (1, 2, 3)) + r" \\")
            for lab, title in LABS.items():
                cells = []
                for k in (1, 2, 3):
                    for c in SCORES:
                        cr = cut_runs(runs, ranked, ds, m, k, c, lab)
                        if cr is None:
                            cells.append(NA)
                            continue
                        v = [st.mean(x) for x in cr.values()]
                        d = st.mean(v) - st.mean(rnd[k])
                        s = f"{d:+.3f}".replace("-", "$-$")
                        cells.append(rf"\textbf{{{s}}}" if holds(v, rnd[k]) or holds(rnd[k], v) else s)
                lines.append(f" & {title} & " + " & ".join(cells) + r" \\")
    write("block_gap", lines + [r"\bottomrule", r"\end{tabular}"])


def blocks(runs, ranked):
    """mAP of each ranking's cut (mean of seeds 42-44) next to the 23 random block orders (seed 42)."""
    mean = lambda key: st.mean(st.mean(v) for v in runs[key].values())
    lines = block_header("Data, model (uncut)")
    for ds, name in DATA.items():
        for m in MODELS:
            lines.append(r"\midrule")
            rv = {k: [st.mean(runs[(ds, m, c)]["42"]) for c in random_orders(k)] for k in (1, 2, 3)}
            v = {(k, c, lab): mean((ds, m, ranked[(ds, m, k, c, lab)])) for k in (1, 2, 3) for c in SCORES for lab in LABS
                 if (ds, m, k, c, lab) in ranked}
            top = {k: max(x for (kk, _, _), x in v.items() if kk == k) for k in (1, 2, 3)}
            lines.append(f"{name} {NAME}{m[-1]} ({mean((ds, m, frozenset())):.3f}) & Random: median (range) & "
                         + " & ".join(rf"\multicolumn{{{len(SCORES)}}}{{c}}{{{st.median(rv[k]):.3f} ({min(rv[k]):.3f}--{max(rv[k]):.3f})}}"
                                      for k in (1, 2, 3)) + r" \\")
            for lab, title in LABS.items():
                cells = []
                for k in (1, 2, 3):
                    for c in SCORES:
                        if (k, c, lab) not in v:
                            cells.append(NA)
                            continue
                        x = v[(k, c, lab)]
                        s = f"{x:.3f}" + (r"$^\dagger$" if x > max(rv[k]) else "")
                        cells.append(rf"\textbf{{{s}}}" if x == top[k] else s)
                lines.append(f" & {title} & " + " & ".join(cells) + r" \\")
    write("blocks", lines + [r"\bottomrule", r"\end{tabular}"])


if __name__ == "__main__":
    ch, bud, bl = load("channel_runs.csv"), load("budget_runs.csv"), load("block_runs.csv")
    gpu, cpu = load("inference_gpu.csv"), load("inference_cpu.csv")
    for arch in ARCHS:
        MODELS, NAME, SFX, BLOCKS_ALL = ARCHS[arch]
        for ds in DATA:
            channel(ch, ds)
        recovery(bud)
        runs, ranked = block_data(bl)
        block_gap(runs, ranked)
        blocks(runs, ranked)
        if any(r["model"] in MODELS for r in gpu) and any(r["model"] in MODELS for r in cpu):
            cost(ch, gpu, cpu)
        else:
            print(f"no {NAME} inference timings yet: cost{SFX} not written")
        # Viewsphere v2 (17 satellites): its own channel / recovery / block tables, suffix _vs2
        MAIN = DATA
        DATA, SFX = V2, SFX + "_vs2"
        channel(ch, "viewsphere2")
        recovery(bud)
        block_gap(runs, ranked)
        blocks(runs, ranked)
        DATA = MAIN
    print("tables written to", OUT)
