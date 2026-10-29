"""README tables and figures for WSD vs COCO vs viewsphere on the same protocol.

Reads results/same_protocol/{channel_runs,block_runs}.csv (made by scripts/collect_same_protocol.py)
and prints the markdown for the README section "WSD, COCO and viewsphere on the same protocol"; writes
results/same_protocol/figures/{channel,blocks}.png. The older WSD/shapes tables come from
scripts/make_results_tables.py.

    python scripts/same_protocol_tables.py > /tmp/same_protocol.md
"""

import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1] / "results" / "same_protocol"
RATIOS = ("0.1", "0.2", "0.3", "0.4", "0.5")
SEEDS = ("42", "43", "44")
DATA = {"wsd": "WSD", "coco": "COCO", "viewsphere": "Viewsphere"}
BLOCKS_ALL = [2, 4, 6, 8, 13, 16, 19, 22]  # C3k2 blocks eligible for replacement (scope "all")
N_RANDOM = 23  # random block orders 0-22
COLORS = {"field_h1": "#2a78d6", "l2": "#eb6834", "random": "#1baf7a",
          "head": "#eb6834", "grid": "#2a78d6", "null": "#1baf7a"}


def load(name: str) -> list[dict]:
    with open(HERE / name) as fh:
        return list(csv.DictReader(fh))


def ratio(r: dict) -> str:
    return f"{float(r['ratio']):.1f}"


def seed_table(ch: list[dict]) -> dict:
    """(dataset, model, criterion, ratio) -> list of after-fine-tune mAP over seeds 42/43/44, PEEK basis."""
    t = defaultdict(list)
    for r in ch:
        if r["basis"] == "peek" and r["set"] in ("grid", "seeds") and r["seed"] in SEEDS and r["map_ft"]:
            if r["criterion"] in ("none", "field_h1", "l2", "random"):
                t[(r["dataset"], r["model"], r["criterion"], ratio(r))].append(float(r["map_ft"]))
    return t


def channel_seeds(ch: list[dict]) -> None:
    t = seed_table(ch)
    print("mAP50-95 after the 20-epoch fine-tune, mean ± sd over seeds 42/43/44 (PEEK basis). "
          "**Bold** = best at that ratio.\n")
    for ds in DATA:
        for m in "ns":
            none = t[(ds, m, "none", "0.0")]
            print(f"**{DATA[ds]} YOLO26{m}** — unpruned + fine-tune {st.mean(none):.3f}\n")
            print("| Criterion | " + " | ".join(f"{float(x):.0%}" for x in RATIOS) + " | Mean |")
            print("|---|" + "---:|" * (len(RATIOS) + 1))
            best = {x: max(st.mean(t[(ds, m, c, x)]) for c in ("field_h1", "l2", "random")) for x in RATIOS}
            for c in ("field_h1", "l2", "random"):
                cells = []
                for x in RATIOS:
                    v = t[(ds, m, c, x)]
                    s = f"{st.mean(v):.3f} ± {st.stdev(v):.3f}"
                    cells.append(f"**{s}**" if st.mean(v) == best[x] else s)
                mean = st.mean(st.mean(t[(ds, m, c, x)]) for x in RATIOS)
                print(f"| {c} | " + " | ".join(cells) + f" | {mean:.3f} |")
            print()


def grid(ch: list[dict], ds: str) -> None:
    g = {(r["model"], r["basis"], r["criterion"], ratio(r)): float(r["map_ft"])
         for r in ch if r["dataset"] == ds and r["set"] == "grid" and r["seed"] == "42"}
    crits = ["field_h1", "field_total", "field_h0", "h0", "h1", "total", "l2", "random"]
    print("Seed 42, mAP50-95 after the fine-tune. *Activation* = the same TDA criterion computed on raw "
          "activations instead of PEEK maps (mean over the five ratios).\n")
    for m in "ns":
        print(f"**{DATA[ds]} YOLO26{m}** — unpruned + fine-tune {g[(m, 'peek', 'none', '0.0')]:.3f}\n")
        print("| Criterion | " + " | ".join(f"{float(x):.0%}" for x in RATIOS) + " | Mean | Activation mean |")
        print("|---|" + "---:|" * (len(RATIOS) + 2))
        for c in sorted(crits, key=lambda c: -st.mean(g[(m, "peek", c, x)] for x in RATIOS)):
            v = [g[(m, "peek", c, x)] for x in RATIOS]
            a = [g.get((m, "activation", c, x)) for x in RATIOS]
            am = f"{st.mean(a):.3f}" if None not in a else "—"
            print(f"| {c} | " + " | ".join(f"{x:.3f}" for x in v) + f" | {st.mean(v):.3f} | {am} |")
        print()


def oneshot(ch: list[dict], ds: str) -> None:
    """mAP50-95 right after pruning + BN re-estimation (no fine-tune): seed-42 grid vs 10 random masks."""
    g = {(r["model"], r["criterion"], ratio(r)): float(r["map_bn"])
         for r in ch if r["dataset"] == ds and r["set"] == "grid" and r["basis"] == "peek" and r["seed"] == "42"}
    rnd = defaultdict(list)
    for r in ch:
        if r["dataset"] == ds and r["set"] == "oneshot" and r["criterion"] == "random":
            rnd[(r["model"], ratio(r))].append(float(r["map_bn"]))
    print("mAP50-95 right after pruning and BN re-estimation, before any fine-tune (seed 42). *Random* = mean "
          "over 10 random masks (seeds 0-9) [best mask]. **Bold** = best at that ratio.\n")
    crits = ["field_h1", "field_total", "field_h0", "h1", "l2"]
    for m in "ns":
        print(f"**{DATA[ds]} YOLO26{m}**\n")
        print("| Criterion | " + " | ".join(f"{float(x):.0%}" for x in RATIOS) + " |")
        print("|---|" + "---:|" * len(RATIOS))
        best = {x: max([g[(m, c, x)] for c in crits] + [st.mean(rnd[(m, x)])]) for x in RATIOS}
        for c in crits:
            print(f"| {c} | " + " | ".join(f"**{g[(m, c, x)]:.3f}**" if g[(m, c, x)] == best[x] else f"{g[(m, c, x)]:.3f}"
                                           for x in RATIOS) + " |")
        print("| random (10 masks) | " + " | ".join(f"{st.mean(rnd[(m, x)]):.3f} [{max(rnd[(m, x)]):.3f}]"
                                                    for x in RATIOS) + " |")
        print()


def random_orders(k: int) -> list[frozenset]:
    return [frozenset(int(i) for i in np.random.default_rng(s).permutation(BLOCKS_ALL)[:k]) for s in range(N_RANDOM)]


def block_sets(bl: list[dict]) -> tuple[dict, dict]:
    """(ds, model) -> {cut set: mAP}, and (ds, model, k, ranking) -> (cut, mAP) for the ranked cuts."""
    by_cut, ranked = defaultdict(dict), {}
    for r in bl:
        if r["seed"] != "42":  # noise seeds 43/44: scripts/noise_tables.py
            continue
        cut = frozenset(int(x) for x in r["layers_cut"].split())
        by_cut[(r["dataset"], r["model"])].setdefault(cut, []).append(float(r["map50_95"]))
        if r["random_order"] == "" and r["criterion"] == "combined" and r["k"] != "0":
            lab = "head" if r["scope"] == "head" and r["norm"] == "raw" else (r["norm"] if r["scope"] == "all" else None)
            if lab:
                ranked[(r["dataset"], r["model"], int(r["k"]), lab)] = cut
    by_cut = {key: {c: st.mean(v) for c, v in d.items()} for key, d in by_cut.items()}
    return by_cut, ranked


def blocks(bl: list[dict]) -> None:
    by_cut, ranked = block_sets(bl)
    print(f"Cut + fine-tune from the trained model (20 epochs). Each cell: blocks replaced → mAP50-95, and the "
          f"share of {N_RANDOM} random block orders it beats. *Head* = raw ranking over the head blocks; "
          "*grid* / *null* = ranking over all eight C3k2 blocks with the map-size normalizations.\n")
    print("| Data | Model | k | Random orders: median [min–max] | Head (raw) | All, grid | All, null |")
    print("|---|---|---:|---|---|---|---|")
    for ds in DATA:
        for m in "ns":
            res = by_cut[(ds, m)]
            k0 = res[frozenset()]
            for k in (1, 2, 3):
                rnd = [res[c] for c in random_orders(k)]
                cells = []
                for lab in ("head", "grid", "null"):
                    cut = ranked.get((ds, m, k, lab))
                    if cut is None:  # cut set already trained under another ranking
                        cells.append("—")
                        continue
                    v = res[cut]
                    beat = sum(v > x for x in rnd) / len(rnd)
                    cells.append(f"{sorted(cut)} → {v:.3f} ({beat:.0%})")
                print(f"| {DATA[ds]} | {m} (full {k0:.3f}) | {k} | {st.median(rnd):.3f} [{min(rnd):.3f}–{max(rnd):.3f}] | "
                      + " | ".join(cells) + " |")
    print()


def lr_pilot(ch: list[dict]) -> None:
    p = defaultdict(dict)
    for r in ch:
        if r["dataset"] == "coco" and r["model"] == "n" and r["seed"] == "42" and r["map_ft"] and (
                r["set"] == "lrpilot") and (r["criterion"] == "none" or ratio(r) == "0.2"):
            p[float(r["lr0"])][r["criterion"]] = float(r["map_ft"])
    print("COCO YOLO26n, seed 42, 20% pruning (unpruned for *none*), mAP50-95 after the fine-tune.\n")
    print("| lr0 | none | field_h1 | L2 | h1 |")
    print("|---:|---:|---:|---:|---:|")
    for lr in sorted(p, reverse=True):
        print(f"| {lr:g} | " + " | ".join(f"{p[lr][c]:.3f}" if c in p[lr] else "—"
                                          for c in ("none", "field_h1", "l2", "h1")) + " |")
    print()


def figures(ch: list[dict], bl: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = HERE / "figures"
    out.mkdir(exist_ok=True)
    t = seed_table(ch)
    x = [float(r) for r in RATIOS]
    fig, axes = plt.subplots(len(DATA), 2, figsize=(10, 3.75 * len(DATA)), constrained_layout=True)
    for i, ds in enumerate(DATA):
        for j, m in enumerate("ns"):
            ax = axes[i, j]
            for c, lab in (("field_h1", "PEEK field H1"), ("l2", "L2 magnitude"), ("random", "random")):
                mu = np.array([st.mean(t[(ds, m, c, r)]) for r in RATIOS])
                sd = np.array([st.stdev(t[(ds, m, c, r)]) for r in RATIOS])
                ax.plot(x, mu, "-o", color=COLORS[c], lw=2, ms=5, label=lab)
                ax.fill_between(x, mu - sd, mu + sd, color=COLORS[c], alpha=0.2, lw=0)
            ax.axhline(st.mean(t[(ds, m, "none", "0.0")]), color="#52514e", ls="--", lw=1, label="unpruned + fine-tune")
            ax.set_title(f"{DATA[ds]} YOLO26{m}")
            ax.set_xticks(x, [f"{v:.0%}" for v in x])
            ax.grid(alpha=0.3)
            if i == len(DATA) - 1:
                ax.set_xlabel("channels pruned per layer")
            if j == 0:
                ax.set_ylabel("mAP50-95 after fine-tune")
    axes[0, 0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Structural channel pruning, 3 seeds (mean ± sd)")
    fig.savefig(out / "channel.png", dpi=150)

    by_cut, ranked = block_sets(bl)
    fig, axes = plt.subplots(len(DATA), 2, figsize=(10, 3.75 * len(DATA)), constrained_layout=True)
    rng = np.random.default_rng(0)
    for i, ds in enumerate(DATA):
        for j, m in enumerate("ns"):
            ax, res = axes[i, j], by_cut[(ds, m)]
            for k in (1, 2, 3):
                rnd = [res[c] for c in random_orders(k)]
                ax.scatter(k + rng.uniform(-0.12, 0.12, len(rnd)), rnd, s=14, color="#898781", alpha=0.6,
                           label="random order" if k == 1 else None)
                for off, lab in ((-0.25, "head"), (0.25, "grid"), (0.32, "null")):
                    cut = ranked.get((ds, m, k, lab))
                    if cut is not None:
                        ax.scatter(k + off, res[cut], s=60, marker="D", color=COLORS[lab], edgecolor="white",
                                   zorder=3, label={"head": "head (raw)", "grid": "all, grid", "null": "all, null"}[lab]
                                   if k == 1 or (ds, m, 1, lab) not in ranked else None)
            ax.axhline(res[frozenset()], color="#52514e", ls="--", lw=1, label="no blocks cut")
            ax.set_title(f"{DATA[ds]} YOLO26{m}")
            ax.set_xticks([1, 2, 3])
            ax.grid(alpha=0.3)
            if i == len(DATA) - 1:
                ax.set_xlabel("blocks replaced (k)")
            if j == 0:
                ax.set_ylabel("mAP50-95 after fine-tune")
    axes[0, 0].legend(loc="lower left", fontsize=8)
    fig.suptitle(f"Block replacement: PEEK rankings vs {N_RANDOM} random orders (cut + fine-tune)")
    fig.savefig(out / "blocks.png", dpi=150)


if __name__ == "__main__":
    ch, bl = load("channel_runs.csv"), load("block_runs.csv")
    print("#### Structural channel pruning, 3 seeds\n")
    channel_seeds(ch)
    print("![Channel pruning, 3 seeds](results/same_protocol/figures/channel.png)\n")
    for ds in ("coco", "viewsphere"):
        print(f"#### {DATA[ds]}: every criterion\n")
        grid(ch, ds)
        print(f"#### {DATA[ds]}: one-shot\n")
        oneshot(ch, ds)
    print("#### Block replacement vs random block orders\n")
    blocks(bl)
    print("![Block replacement vs random orders](results/same_protocol/figures/blocks.png)\n")
    print("#### COCO fine-tune learning rate\n")
    lr_pilot(ch)
    figures(ch, bl)
