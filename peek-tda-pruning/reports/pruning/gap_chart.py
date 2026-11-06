"""Line chart: how close the persistence scores (field H1, H0, H0+H1) are to L2.

Each point = criterion mean mAP50-95 minus L2's at the same dataset, model and pruning ratio, averaged
over the four models (YOLO26n/s, YOLOv5n/s); 0 = L2. Rows: one-shot (after BN re-estimation) and
after the 20-epoch fine-tune; columns: datasets. Means come from make_tables.channel_runs (the same
numbers as the tables); dataset names and colours of the ink/surface from win_chart.py.
Writes figures/gap_to_l2.png and prints the gaps.

    python gap_chart.py [path to results/same_protocol; default: this repo's]
"""

import statistics as st
from pathlib import Path

import matplotlib.pyplot as plt

import make_tables as mt
from win_chart import DATASETS, INK, MUTED, RATIOS, STAGES, SURFACE

SCORES = {"field_h1": "Field H1", "field_h0": "Field H0", "field_total": "Field H0+H1"}
COLORS = ["#e87ba4", "#008300", "#6250d6"]  # dataviz categorical slots 5-7 (1-4 are the datasets)


def gaps():
    """g[(stage, dataset, criterion)] -> [mean over models of (criterion - L2)] per ratio."""
    ch = mt.load("channel_runs.csv")
    d = {}
    for models, name, _, _ in mt.ARCHS.values():
        for ds in DATASETS:
            t = mt.channel_runs(ch, ds, models, name)
            for s in STAGES:
                for c in SCORES:
                    for m in models:
                        for x in RATIOS:
                            d.setdefault((s, ds, c, x), []).append(st.mean(t[(s, m, c, x)]) - st.mean(t[(s, m, "l2", x)]))
    return {(s, ds, c): [st.mean(d[(s, ds, c, x)]) for x in RATIOS] for s in STAGES for ds in DATASETS for c in SCORES}


def plot(g, out):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(2, len(DATASETS), figsize=(14, 7.2), sharex=True, sharey="row", facecolor=SURFACE)
    xs = [float(x) * 100 for x in RATIOS]
    for row, (s, stage) in zip(axes, STAGES.items()):
        lim = max(abs(v) for ds in DATASETS for c in SCORES for v in g[(s, ds, c)]) * 1.15
        for ax, (ds, lab) in zip(row, DATASETS.items()):
            ax.set_facecolor(SURFACE)
            ax.axhline(0, color=INK, linewidth=1.2, zorder=2)
            ax.text(xs[-1] + 2.5, 0, "L2", ha="left", va="center", fontsize=8.5, color=INK)
            for (c, name), col in zip(SCORES.items(), COLORS):
                ax.plot(xs, g[(s, ds, c)], color=col, linewidth=2, solid_capstyle="round", zorder=3, label=name)
                ax.scatter(xs, g[(s, ds, c)], s=34, color=col, edgecolor=SURFACE, linewidth=1.5, zorder=4)
            ax.set_ylim(-lim, lim)
            ax.set_xlim(xs[0] - 3, xs[-1] + 6)
            ax.set_xticks(xs, [f"{x:.0f}%" for x in xs])
            ax.grid(axis="y", color="#e4e3df", linewidth=0.8, zorder=0)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.tick_params(axis="both", length=0)
            ax.set_title(f"{lab}", color=INK, fontsize=11, loc="left")
        row[0].set_ylabel(f"{stage}\nmAP50-95 minus L2", color=INK)
    for ax in axes[1]:
        ax.set_xlabel("Channels pruned", color=MUTED)
    fig.suptitle("How close the persistence scores are to L2", x=0.01, ha="left", fontsize=13,
                 fontweight="bold", color=INK)
    fig.text(0.01, 0.935, "Criterion minus L2 at the same dataset, model and pruning ratio, averaged over YOLO26n/s "
             "and YOLOv5n/s; above 0 = better than L2. Viewsphere v2 one-shot results are unreliable (BN re-estimation).",
             ha="left", fontsize=9, color=MUTED)
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="upper right", ncol=3, frameon=False,
               bbox_to_anchor=(0.995, 0.995), fontsize=9.5, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURFACE)


if __name__ == "__main__":
    g = gaps()
    for k, v in g.items():
        print(*k, " ".join(f"{x:+.3f}" for x in v))
    out = Path(__file__).resolve().parent / "figures" / "gap_to_l2.png"
    plot(g, out)
    print("wrote", out)
