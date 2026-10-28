"""README tables and figures for the recovery study and the inference timings.

Reads results/same_protocol/{curves,probe_curves,budget_runs,channel_runs,block_runs,inference_*}.csv
(made by scripts/collect_same_protocol.py), prints the markdown for the README sections "How fast pruned
models recover" and "Inference timing", and writes results/same_protocol/figures/{warmup,budgets}.png.
The 20-epoch comparison tables come from scripts/same_protocol_tables.py.

    python scripts/recovery_tables.py > /tmp/recovery.md
"""

import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "results" / "same_protocol"
DATA = {"wsd": "WSD", "coco": "COCO", "viewsphere": "Viewsphere"}
IPE = {"wsd": 63, "coco": 625, "viewsphere": 58}  # batches (of 16) per epoch: WSD 1,005 images, COCO 10k subset, viewsphere 925
FULL_IPE = 7393                 # COCO, all of train2017
BUDGETS = {"wsd": (250, 500, 1000, 2500, 5000), "coco": (250, 500, 1000, 2500, 5000, 25000, 50000),
           "viewsphere": (250, 500, 1000, 2500, 5000)}
FULL = (3 * FULL_IPE, 7 * FULL_IPE)
COLORS = {"field_h1": "#2a78d6", "l2": "#eb6834", "random": "#1baf7a", "none": "#52514e"}
CRIT = ("field_h1", "l2", "random")


def load(name: str) -> list[dict]:
    with open(HERE / name) as fh:
        return list(csv.DictReader(fh))


def label(r: dict) -> str:
    """Short row label: 'field_h1 10%', 'unpruned', 'grid [2 8]', 'uncut'."""
    if r["method"] == "channel":
        return "unpruned" if r["criterion"] == "none" else f"{r['criterion']} {float(r['ratio']):.0%}"
    if not r["layers_cut"]:
        return "uncut"
    return f"{'head' if r['scope'] == 'head' else r['norm']} [{r['layers_cut']}]"


def order(lab: str) -> tuple:
    if lab in ("unpruned", "uncut"):
        return (0, lab)
    if "[" in lab:
        return (2, len(lab.split("[")[1].split()), lab)
    c, p = lab.split()
    return (1, CRIT.index(c), int(p[:-1]))


def f3(x) -> str:
    return "—" if x is None else f"{x:.3f}"


# ---------------------------------------------------------------- recovery inside the 20-epoch runs
def epoch_recovery(cur: list[dict]) -> None:
    curves = defaultdict(dict)
    for r in cur:
        if r["basis"] == "peek" and r["random_order"] == "" and r["criterion"] in ("none", *CRIT, "combined"):
            curves[(r["dataset"], r["model"], label(r), r["seed"])][int(r["epoch"])] = float(r["map50_95"])
    print("mAP50-95 on the full val set during the 20-epoch fine-tunes of the comparison above (standard warmup "
          "on WSD/COCO, matched on viewsphere; "
          "channel rows = mean of seeds 42/43/44, blocks = seed 42). *Start* = before training (channel: after BN "
          "re-estimation). *→95%* = first epoch within 95% of the run's own epoch-20 value.\n")
    for ds in DATA:
        print(f"**{DATA[ds]}** — {IPE[ds]} batches per epoch\n")
        print("| Model | Run | Start | Epoch 1 | Epoch 5 | Epoch 20 | →95% (epoch) |")
        print("|---|---|---:|---:|---:|---:|---:|")
        for m in "ns":
            labs = sorted({k[2] for k in curves if k[:2] == (ds, m)}, key=order)
            for lab in labs:
                if lab in ("uncut",) or ("[" not in lab and lab != "unpruned" and lab.split()[1] not in ("10%", "50%")):
                    continue
                cs = [c for k, c in curves.items() if k[:3] == (ds, m, lab)]
                mean = lambda e: st.mean(c[e] for c in cs) if all(e in c for c in cs) else None
                hit = [next((e for e in sorted(c) if e > 0 and c[e] >= 0.95 * c[20]), None) for c in cs]
                print(f"| {m} | {lab} | {f3(mean(0))} | {f3(mean(1))} | {f3(mean(5))} | {f3(mean(20))} | "
                      f"{st.median(hit):.0f} |")
        print()


# ---------------------------------------------------------------- warmup probes
def warmup(pc: list[dict]) -> None:
    probe = defaultdict(dict)
    for r in pc:
        probe[(r["dataset"], r["model"], label(r), r["variant"])][int(r["batch"])] = float(r["map50_95"])
    show = ("unpruned", "field_h1 10%", "l2 10%", "field_h1 50%", "l2 50%")
    batches = (0, 50, 100, 250, 500, 1000, 2500)
    print("Every-50-batch evaluations of the same 20-epoch schedule (seed 42), with Ultralytics' default warmup "
          "(*0.1*: the bias lr starts at 0.1 and decays to lr0 over the first epoch) and with it started at lr0 "
          "(*lr0*). COCO is scored on a fixed 500-image val2017 slice for the first 2,500 batches; WSD and "
          "viewsphere on their full val sets (viewsphere: lr0 only).\n")
    print("| Data | Model | Run | Bias lr | " + " | ".join(f"batch {b}" for b in batches) + " |")
    print("|---|---|---|---|" + "---:|" * len(batches))
    for ds in DATA:
        for m in "ns":
            for lab in show:
                for v, vl in (("", "0.1"), ("wblr", "lr0")):
                    c = probe.get((ds, m, lab, v))
                    if c:
                        print(f"| {DATA[ds]} | {m} | {lab} | {vl} | " + " | ".join(f3(c.get(b)) for b in batches) + " |")
    print()
    return probe


# ---------------------------------------------------------------- budget sweep
def budget_table(bud: list[dict]) -> dict:
    t = defaultdict(dict)
    for r in bud:
        if r["variant"] in ("wblr", "wblr_full"):
            key = int(r["budget"]) if r["variant"] == "wblr" else f"full{r['epoch']}"
            t[(r["dataset"], r["model"], label(r))][key] = float(r["map50_95"])
    print("Separate fine-tunes of B training batches each (matched warmup, seed 42), mAP50-95 on the full val "
          "set. One epoch = 63 batches on WSD, 625 on the COCO 10k subset, 58 on viewsphere; *full 3 ep / 7 ep* = COCO fine-tuned on "
          "all of train2017 (7,393 batches per epoch, so 22k / 52k batches). Bold = best of the row.\n")
    for ds in DATA:
        for m in "ns":
            cols = list(BUDGETS[ds]) + (["full3", "full7"] if ds == "coco" else [])
            head = [f"{b:,}" for b in BUDGETS[ds]] + (["full 3 ep", "full 7 ep"] if ds == "coco" else [])
            print(f"**{DATA[ds]} YOLO26{m}**\n")
            print("| Run | " + " | ".join(head) + " |")
            print("|---|" + "---:|" * len(cols))
            for lab in sorted({k[2] for k in t if k[:2] == (ds, m)}, key=order):
                v = t[(ds, m, lab)]
                best = max(v.values())
                print(f"| {lab} | " + " | ".join("—" if c not in v else (f"**{v[c]:.3f}**" if v[c] == best else f"{v[c]:.3f}")
                                                 for c in cols) + " |")
            print()
    return t


def coco_recovery(t: dict) -> None:
    orig = {"n": 0.400, "s": 0.476}
    print("COCO, as a share of the unpruned model given the same fine-tune (matched warmup). *→90%* = smallest "
          f"10k-subset budget reaching 90% of that control; no run reaches 95% on the subset. The original "
          f"pretrained models score {orig['n']:.3f} / {orig['s']:.3f}; no fine-tune of the unpruned model gets back "
          "there (best 0.392 / 0.459), so the control is the fair reference.\n")
    print("| Model | Run | Subset 5k | Subset 50k | Full 3 ep | Full 7 ep | →90% (subset) |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for m in "ns":
        ctl = t[("coco", m, "unpruned")]
        for lab in sorted({k[2] for k in t if k[:2] == ("coco", m)}, key=order):
            if lab in ("unpruned", "uncut"):
                continue
            v = t[("coco", m, lab)]
            pct = lambda c: f"{100 * v[c] / ctl[c]:.0f}%" if c in v and c in ctl else "—"
            hit = next((b for b in BUDGETS["coco"] if b in v and v[b] >= 0.9 * ctl[b]), None)
            print(f"| {m} | {lab} | {pct(5000)} | {pct(50000)} | {pct('full3')} | {pct('full7')} | "
                  f"{f'{hit:,}' if hit else 'never'} |")
    print()


# ---------------------------------------------------------------- timing
def timing(ch: list[dict], bl: list[dict]) -> None:
    gpu, cpu = load("inference_gpu.csv"), load("inference_cpu.csv")
    gf = {}
    for r in ch:
        if r["set"] == "grid" and r["basis"] == "peek" and r["seed"] == "42" and r["criterion"] in ("none", "field_h1"):
            gf[(r["dataset"], r["model"], "unpruned" if r["criterion"] == "none" else f"{float(r['ratio']):.2f}")] = float(r["gflops"])
    for r in bl:
        if r["random_order"] == "" and r["criterion"] == "combined":
            lab = ("k0" if r["k"] == "0" else f"{'head' if r['scope'] == 'head' else r['norm']} k{r['k']}")
            gf[(r["dataset"], r["model"], "block " + lab)] = float(r["gflops"])
    rows = defaultdict(dict)
    for dev, data in (("gpu", gpu), ("cpu", cpu)):
        for r in data:
            key = (r["dataset"], r["model"], "unpruned" if r["method"] == "unpruned" else
                   (r["setting"] if r["method"] == "channel" else f"block {r['method'][6:]} {r['setting']}"))
            rows[key][dev] = r
    print(f"Forward pass only (no pre/post-processing), random 640×640 input, after conv+BN fusion. GPU: "
          f"{gpu[0]['hardware']}, FP16; CPU: Intel Xeon, {cpu[0]['hardware'].split(', ')[-1]}, FP32, on a "
          "shared node (absolute CPU times run high; read the relative ones, and expect up to ~±10% between "
          "near-identical models). All models are timed round-robin: 40 "
          "rounds, each running every model 5 times in a shuffled order, so drift on the node hits them alike (timing "
          "them one after another skewed some CPU results by 10-30%). Batch-1 latency = median of the 200 runs; "
          "throughput = batch 32. Δ = vs the unpruned model; GFLOPs Δ for channel pruning from Torch-Pruning, "
          "for blocks from Ultralytics (each against its own unpruned count). Channel rows = the PEEK field H1 "
          "model (every criterion gives the same widths at a ratio). WSD differs from COCO only in the head's "
          "class count; its timings are within a few % of these. Viewsphere models have WSD's architecture and "
          "are listed for their own block cuts. All four tables come from one round-robin run.\n")
    for ds, m in [(ds, m) for ds in ("coco", "viewsphere") for m in "ns" if (ds, m, "unpruned") in rows]:
        print(f"**{DATA[ds]} YOLO26{m}**\n")
        print("| Model | GFLOPs Δ | GPU batch 1 (ms) | GPU batch 32 (img/s) | CPU batch 1 (ms) |")
        print("|---|---:|---:|---:|---:|")
        base = rows[(ds, m, "unpruned")]
        keys = [k for k in rows if k[:2] == (ds, m)]
        keys.sort(key=lambda k: (k[2] != "unpruned", k[2].startswith("block"), k[2][-2:] if k[2].startswith("block") else k[2], k[2]))
        for k in keys:
            r = rows[k]
            def cell(dev, col):
                if dev not in r:
                    return "—"
                v, b = float(r[dev][col]), float(base[dev][col])
                return f"{v:.2f}" if k[2] == "unpruned" else f"{v:.2f} ({100 * (v / b - 1):+.0f}%)"
            g = gf.get(k) if not k[2].startswith("block") else gf.get((ds, m, k[2]))
            g0 = gf[(ds, m, "unpruned")] if not k[2].startswith("block") else gf.get((ds, m, "block k0"))
            gd = "—" if k[2] == "unpruned" or g is None or g0 is None else f"{100 * (g / g0 - 1):+.0f}%"
            name = k[2] if k[2] in ("unpruned",) or k[2].startswith("block") else f"channel {float(k[2]):.0%}"
            print(f"| {name} | {gd} | {cell('gpu', 'b1_ms_median')} | {cell('gpu', 'b32_img_per_s')} | {cell('cpu', 'b1_ms_median')} |")
        print()


# ---------------------------------------------------------------- figures
def figures(probe: dict, t: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = HERE / "figures"
    out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for ax, m in zip(axes, "ns"):
        for lab, c in (("unpruned", "none"), ("field_h1 10%", "field_h1"), ("field_h1 50%", "field_h1"), ("l2 50%", "l2")):
            for v, ls in (("", "--"), ("wblr", "-")):
                cur = probe.get(("coco", m, lab, v))
                if cur:
                    xs = sorted(cur)
                    ax.plot(xs, [cur[x] for x in xs], ls, color=COLORS[c], lw=1.6 if "50%" not in lab else 1.2,
                            label=f"{lab}, bias lr {'lr0' if v else '0.1'}")
        ax.set_title(f"COCO YOLO26{m}: first 2,500 batches")
        ax.set_xlabel("training batches")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("mAP50-95 (500-image val slice)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.12))
    fig.savefig(out / "warmup.png", dpi=150, bbox_inches="tight")

    fig, axes = plt.subplots(len(DATA), 2, figsize=(11, 4 * len(DATA)), constrained_layout=True)
    for i, ds in enumerate(DATA):
        for j, m in enumerate("ns"):
            ax = axes[i, j]
            for lab, c, ls in (("unpruned", "none", "-"), *((f"{k} {p}", k, "-" if p == "50%" else ":")
                                                          for k in CRIT for p in ("10%", "50%"))):
                v = t.get((ds, m, lab))
                if not v:
                    continue
                xs = [b for b in BUDGETS[ds] if b in v]
                ax.plot(xs, [v[b] for b in xs], ls, marker="o", ms=3, color=COLORS[c], lw=1.6, label=lab)
                if ds == "coco":
                    fx = [(b, v[f"full{e}"]) for b, e in zip(FULL, (3, 7)) if f"full{e}" in v]
                    if fx:
                        ax.plot(*zip(*fx), ls, marker="D", ms=5, mfc="white", color=COLORS[c], lw=1)
            ax.set_xscale("log")
            ax.set_xticks(BUDGETS[ds], [f"{x / 1000:g}k" if x >= 1000 else str(x) for x in BUDGETS[ds]], fontsize=7)
            ax.minorticks_off()
            ax.set_title(f"{DATA[ds]} YOLO26{m}" + (" (◇ = all of train2017: 22k, 52k batches)" if ds == "coco" else ""))
            ax.grid(alpha=0.3, which="both")
            if i == len(DATA) - 1:
                ax.set_xlabel("training batches (log)")
            if j == 0:
                ax.set_ylabel("mAP50-95")
    axes[0, 0].legend(fontsize=7, ncol=2, loc="lower right")
    fig.suptitle("Training-budget sweep, matched warmup (solid = 50% pruned, dotted = 10%)")
    fig.savefig(out / "budgets.png", dpi=150)


if __name__ == "__main__":
    ch, bl = load("channel_runs.csv"), load("block_runs.csv")
    print("#### Recovery within the 20-epoch fine-tune\n")
    epoch_recovery(load("curves.csv"))
    print("#### The warmup crash\n")
    probe = warmup(load("probe_curves.csv"))
    print("![Warmup bias lr on COCO](results/same_protocol/figures/warmup.png)\n")
    print("#### Training-budget sweep\n")
    t = budget_table(load("budget_runs.csv"))
    print("![Training-budget sweep](results/same_protocol/figures/budgets.png)\n")
    print("#### Does COCO recover?\n")
    coco_recovery(t)
    print("### Inference timing\n")
    timing(ch, bl)
    figures(probe, t)
