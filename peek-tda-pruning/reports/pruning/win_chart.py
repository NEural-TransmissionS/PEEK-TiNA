"""Bar charts: how often each channel-pruning criterion is the best one, stacked by dataset.

criterion_wins.png: a "setting" = dataset x model (YOLO26n/s, YOLOv5n/s) x pruning ratio (10-50%); the
criterion with the highest mean mAP50-95 over seeds wins it. One panel for one-shot (after BN
re-estimation), one for after the 20-epoch fine-tune (make_tables.channel_runs, as in the tables).
budget_wins.png: the same over the training-budget runs (make_tables.budget_data, as in the recovery
tables), one panel per budget, setting = dataset x model x ratio (10/30/50%).
Writes both into figures/ and prints the counts.

    python win_chart.py [path to results/same_protocol; default: this repo's]
"""

import statistics as st
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import make_tables as mt

DATASETS = {"wsd": "WSD", "viewsphere": "Viewsphere", "viewsphere2": "Viewsphere v2", "coco": "COCO"}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # dataviz categorical slots 1-4, fixed order
RATIOS = ("0.1", "0.2", "0.3", "0.4", "0.5")
STAGES = {"os": "One-shot (after BN re-estimation)", "ft": "After 20-epoch fine-tune"}
INK, MUTED, SURFACE = "#1f1f1e", "#6b6a64", "#fcfcfb"


def score(n, key, ds, means):
    """Credit the best of `means` {criterion: mean mAP} (ties: each) if every criterion was run."""
    if len(means) == len(mt.CRIT):
        top = max(means.values())
        for c, v in means.items():
            n[key][ds][c] += v == top


def empty(keys):
    return {k: {ds: dict.fromkeys(mt.CRIT, 0) for ds in DATASETS} for k in keys}


def wins():
    """n[stage][dataset][criterion] -> settings won (a tie counts for each tied criterion)."""
    ch = mt.load("channel_runs.csv")
    n = empty(STAGES)
    for models, name, _, _ in mt.ARCHS.values():
        for ds in DATASETS:
            t = mt.channel_runs(ch, ds, models, name)
            for s in STAGES:
                for m in models:
                    for x in RATIOS:
                        score(n, s, ds, {c: st.mean(t[(s, m, c, x)]) for c in mt.CRIT if t.get((s, m, c, x))})
    return n


def budget_wins():
    """n[budget][dataset][criterion] -> settings won, over the matched-warmup budget runs."""
    t = mt.budget_data(mt.load("budget_runs.csv"))
    n = empty(mt.BUDGETS)
    models = [m for ms, *_ in mt.ARCHS.values() for m in ms]
    for b in mt.BUDGETS:
        for ds in DATASETS:
            for m in models:
                for x in mt.SHOW:
                    score(n, b, ds, {c: t[(ds, m, c, x, b)] for c in mt.CRIT if (ds, m, c, x, b) in t})
    return n


def plot(n, panels, title, note, out, labels=None, width=11, only=None, ytop=None):
    """panels {key in n: panel title}; labels = x tick text per criterion (default: full names).
    only = one dataset key: just that dataset, in its colour from the combined chart, y axis fixed at
    `ytop` (the settings per panel) so the per-dataset charts compare; no legend (the title names it)."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": MUTED})
    fig, axes = plt.subplots(1, len(panels), figsize=(width, 4.8), sharey=True, facecolor=SURFACE)
    crit = list(mt.CRIT)
    shown = [(ds, lab, col) for (ds, lab), col in zip(DATASETS.items(), COLORS) if only in (None, ds)]
    ymax = ytop or max(sum(n[s][ds][c] for ds, _, _ in shown) for s in panels for c in crit)
    for ax, (s, sub) in zip(axes, panels.items()):
        ax.set_facecolor(SURFACE)
        bottom = [0] * len(crit)
        for ds, lab, col in shown:
            h = [n[s][ds][c] for c in crit]
            ax.bar(range(len(crit)), h, 0.62, bottom=bottom, color=col, edgecolor=SURFACE, linewidth=2,
                   label=lab, zorder=3)
            for i, (b, v) in enumerate(zip(bottom, h)):
                if v >= 3 and len(shown) > 1:  # count inside the segment when it fits (stacked charts only)
                    ax.text(i, b + v / 2, str(v), ha="center", va="center", fontsize=8.5, color=INK, zorder=4)
            bottom = [b + v for b, v in zip(bottom, h)]
        for i, b in enumerate(bottom):
            ax.text(i, b + ymax * 0.015, str(b), ha="center", va="bottom", fontsize=10, fontweight="bold", color=INK)
        ax.set_xticks(range(len(crit)), labels or [mt.CRIT[c] for c in crit])
        ax.set_title(sub, color=INK, fontsize=11, loc="left")
        ax.grid(axis="y", color="#e4e3df", linewidth=0.8, zorder=0)
        ax.set_ylim(0, ymax * 1.12)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # counts: whole-number ticks
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="both", length=0)
    axes[0].set_ylabel("Settings won (best mean mAP50-95)")
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 0.905, note, ha="left", fontsize=9, color=MUTED)
    if len(shown) > 1:
        fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right", ncol=4, frameon=False,
                   bbox_to_anchor=(0.995, 0.995), fontsize=9.5, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURFACE)


def report(n):
    for s in n:
        print(s, {mt.CRIT[c]: sum(n[s][ds][c] for ds in DATASETS) for c in mt.CRIT})
        for ds in DATASETS:
            print("  ", ds, list(n[s][ds].values()))


if __name__ == "__main__":
    FIG = Path(__file__).resolve().parent / "figures"
    n = wins()
    report(n)
    for ds, lab in DATASETS.items():  # one per dataset (criterion_wins_<ds>.png), then all of them together
        plot(n, STAGES, f"{lab}: which channel-pruning criterion wins most often",
             "20 settings (4 models x 5 pruning ratios, 10-50%); ties count for each tied criterion."
             + (" One-shot results are unreliable here (BN re-estimation)." if ds == "viewsphere2" else ""),
             FIG / f"criterion_wins_{ds}.png", only=ds, ytop=20)
    plot(n, STAGES, "All datasets: which channel-pruning criterion wins most often",
         "20 settings per dataset (4 models x 5 pruning ratios, 10-50%); ties count for each tied criterion. "
         "Viewsphere v2 one-shot results are unreliable (BN re-estimation).", FIG / "criterion_wins.png")
    n = budget_wins()
    report(n)
    panels, short = {b: f"{b:,} batches" for b in mt.BUDGETS}, ["H1", "H0", "H0+H1", "L2", "Rand."]
    for ds, lab in DATASETS.items():  # one per dataset (budget_wins_<ds>.png), then all of them together
        plot(n, panels, f"{lab}: which criterion wins at each training budget",
             "12 settings per budget (4 models x pruning ratios 10/30/50%); separate fine-tunes of B batches, "
             "matched warmup; ties count for each tied criterion.", FIG / f"budget_wins_{ds}.png",
             labels=short, width=14, only=ds, ytop=12)
    plot(n, panels, "All datasets: which criterion wins at each training budget",
         "12 settings per dataset and budget (4 models x pruning ratios 10/30/50%); separate fine-tunes of B batches, "
         "matched warmup; ties count for each tied criterion.", FIG / "budget_wins.png", labels=short, width=14)
    print("wrote figures/criterion_wins.png, figures/budget_wins_<dataset>.png and figures/budget_wins.png")
