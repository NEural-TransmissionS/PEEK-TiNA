"""Seed-noise check: which same-protocol and recovery differences are larger than seed-to-seed spread.

Reads results/same_protocol/{channel_runs,block_runs,budget_runs}.csv (scripts/collect_same_protocol.py) and
prints markdown. Seeds (structural_pruning/scripts/slurm/same_protocol/submit_noise_seeds.sh): channel fine-tune 42-46, one-shot
BN re-estimation 42-51 (masks fixed), block cut + fine-tune and recovery budgets 42-44. A gap is marked
**bold** when it exceeds twice its standard error (Welch, roughly 95%); plain = inside noise. The seed-42
tables themselves come from scripts/same_protocol_tables.py and scripts/recovery_tables.py.

    python scripts/noise_tables.py > /tmp/noise.md
"""

import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

from same_protocol_tables import random_orders

HERE = Path(__file__).resolve().parents[1] / "results" / "same_protocol"
DATA = {"wsd": "WSD", "coco": "COCO", "viewsphere": "Viewsphere"}
RATIOS = ("0.1", "0.2", "0.3", "0.4", "0.5")
CRIT = ("field_h1", "l2", "random")
BUDGETS = (250, 500, 1000, 2500, 5000)


def load(name: str) -> list[dict]:
    with open(HERE / name) as fh:
        return list(csv.DictReader(fh))


def ratio(r: dict) -> str:
    return f"{float(r['ratio']):.1f}"


def sd(v: list[float]) -> float:
    return st.stdev(v) if len(v) > 1 else 0.0


def gap(a: list[float], b: list[float]) -> str:
    """mean(a) - mean(b), bold when |gap| > 2 standard errors."""
    d = st.mean(a) - st.mean(b)
    se = (sd(a) ** 2 / len(a) + sd(b) ** 2 / len(b)) ** 0.5
    s = f"{d:+.3f}"
    return f"**{s}**" if abs(d) > 2 * se else s


def ms(v: list[float]) -> str:
    return f"{st.mean(v):.3f} ± {sd(v):.3f}"


# ---------------------------------------------------------------- channel pruning + 20-epoch fine-tune
def channel(ch: list[dict]) -> None:
    t = defaultdict(list)
    for r in ch:
        if r["basis"] == "peek" and r["set"] in ("grid", "seeds") and r["map_ft"] and r["criterion"] in ("none", *CRIT):
            t[(r["dataset"], r["model"], r["criterion"], ratio(r))].append(float(r["map_ft"]))
    print("**Channel pruning + 20-epoch fine-tune**, seeds 42-46: mAP50-95 mean ± sd, and the gaps between "
          "criteria (bold = more than 2 standard errors).\n")
    print("| Data | Model | Ratio | field H1 | L2 | random | field − L2 | field − random | L2 − random |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for ds in DATA:
        for m in "ns":
            none = t[(ds, m, "none", "0.0")]
            print(f"| {DATA[ds]} | {m} | unpruned | {ms(none)} (n={len(none)}) | | | | | |")
            for x in RATIOS:
                f, l, rd = (t[(ds, m, c, x)] for c in CRIT)
                print(f"| {DATA[ds]} | {m} | {float(x):.0%} | {ms(f)} | {ms(l)} | {ms(rd)} | {gap(f, l)} | "
                      f"{gap(f, rd)} | {gap(l, rd)} |")
    print()


# ---------------------------------------------------------------- one-shot (BN re-estimation, no fine-tune)
def oneshot(ch: list[dict]) -> None:
    t, rnd = defaultdict(list), defaultdict(list)
    for r in ch:
        if r["basis"] != "peek":
            continue
        if r["criterion"] in ("field_h1", "l2") and (r["set"] == "oneshot_seeds" or (r["set"] == "grid" and r["seed"] == "42")):
            t[(r["dataset"], r["model"], r["criterion"], ratio(r))].append(float(r["map_bn"]))
        elif r["criterion"] == "random" and r["set"] == "oneshot":
            rnd[(r["dataset"], r["model"], ratio(r))].append(float(r["map_bn"]))
    print("**One-shot** (prune + BN re-estimation, no fine-tune): field H1 and L2 over BN-sample seeds 42-51 "
          "(the mask is fixed), random over 10 masks (seeds 0-9, so mask choice is included).\n")
    print("| Data | Model | Ratio | field H1 | L2 | random masks | field − L2 | field − random |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for ds in DATA:
        for m in "ns":
            for x in RATIOS:
                f, l, rd = t[(ds, m, "field_h1", x)], t[(ds, m, "l2", x)], rnd.get((ds, m, x))
                print(f"| {DATA[ds]} | {m} | {float(x):.0%} | {ms(f)} | {ms(l)} | {ms(rd) if rd else '—'} | "
                      f"{gap(f, l)} | {gap(f, rd) if rd else '—'} |")
    print()


# ---------------------------------------------------------------- block cut + fine-tune
def blocks(bl: list[dict]) -> None:
    runs, ranked = defaultdict(lambda: defaultdict(list)), {}
    for r in bl:
        cut = frozenset(int(x) for x in r["layers_cut"].split())
        runs[(r["dataset"], r["model"], cut)][r["seed"]].append(float(r["map50_95"]))
        if r["random_order"] == "" and r["criterion"] == "combined" and r["k"] != "0":
            lab = "head" if r["scope"] == "head" and r["norm"] == "raw" else (r["norm"] if r["scope"] == "all" else None)
            if lab:
                ranked[(r["dataset"], r["model"], int(r["k"]), lab)] = cut
    seeds = lambda ds, m, cut: [st.mean(v) for v in runs[(ds, m, cut)].values()]  # one value per seed
    print("**Block cut + fine-tune**, seeds 42-44: mAP50-95 mean ± sd for each ranked cut, against the random "
          "block orders (seed 42 only, so their spread is mostly *which* blocks). *vs random* = cut mean − mean of "
          "the random orders, bold when larger than 2 standard errors of both; *beats* = share of random orders "
          "below the 3-seed mean (seed-42 value in brackets).\n")
    print("| Data | Model | k | Random orders | Head (raw) | All, grid | All, null |")
    print("|---|---|---:|---|---|---|---|")
    for ds in DATA:
        for m in "ns":
            k0 = seeds(ds, m, frozenset())
            print(f"| {DATA[ds]} | {m} | 0 | — | uncut {ms(k0)} | | |")
            for k in (1, 2, 3):
                rv = [st.mean(runs[(ds, m, c)]["42"]) for c in random_orders(k)]  # 23 orders, as in the seed-42 table
                cells = []
                for lab in ("head", "grid", "null"):
                    cut = ranked.get((ds, m, k, lab))
                    if cut is None:
                        cells.append("—")
                        continue
                    v = seeds(ds, m, cut)
                    s42 = st.mean(runs[(ds, m, cut)]["42"])
                    beat, beat42 = (sum(x < st.mean(v) for x in rv) / len(rv), sum(x < s42 for x in rv) / len(rv))
                    cells.append(f"{sorted(cut)} {ms(v)}, vs random {gap(v, rv)}, beats {beat:.0%} ({beat42:.0%})")
                print(f"| {DATA[ds]} | {m} | {k} | {ms(rv)} (n={len(rv)}) | " + " | ".join(cells) + " |")
    print()
    # pooled seed sd: how far a single 20-epoch block run moves with the seed
    print("Seed-to-seed sd of one block run (pooled over every cut with 3 seeds): " + ", ".join(
        f"{DATA[ds]} {m} {st.mean([sd(seeds(d, mm, c)) for (d, mm, c) in runs if (d, mm) == (ds, m) and len(runs[(d, mm, c)]) == 3]):.3f}"
        for ds in DATA for m in "ns") + ".\n")


# ---------------------------------------------------------------- recovery budgets
def label(r: dict) -> str:
    if r["method"] == "channel":
        return "unpruned" if r["criterion"] == "none" else f"{r['criterion']} {float(r['ratio']):.0%}"
    return "uncut" if not r["layers_cut"] else f"{'head' if r['scope'] == 'head' else r['norm']} [{r['layers_cut']}]"


def budgets(bud: list[dict]) -> None:
    t = defaultdict(list)
    for r in bud:
        if r["variant"] == "wblr" and int(r["budget"]) in BUDGETS:
            t[(r["dataset"], r["model"], label(r), int(r["budget"]))].append(float(r["map50_95"]))
    have = {k[:2] for k, v in t.items() if len(v) == 3}
    print("**Recovery budgets** (separate fine-tunes of B batches, matched warmup), seeds 42-44. First the "
          "typical seed sd per budget, then gaps that the seed-42 tables read as differences.\n")
    print("| Data | Model | " + " | ".join(f"sd @ {b:,}" for b in BUDGETS) + " |")
    print("|---|---|" + "---:|" * len(BUDGETS))
    for ds, m in sorted(have, key=lambda k: (list(DATA).index(k[0]), k[1])):
        print(f"| {DATA[ds]} | {m} | " + " | ".join(
            f"{st.median([sd(v) for k, v in t.items() if k[:2] == (ds, m) and k[3] == b and len(v) == 3]):.3f}"
            for b in BUDGETS) + " |")
    print()
    print("| Data | Model | Comparison | " + " | ".join(f"{b:,}" for b in BUDGETS) + " |")
    print("|---|---|---|" + "---:|" * len(BUDGETS))
    comps = [("field_h1 50%", "l2 50%"), ("field_h1 50%", "random 50%"), ("l2 50%", "random 50%"),
             ("field_h1 10%", "unpruned"), ("field_h1 30%", "unpruned"), ("field_h1 50%", "unpruned")]
    for ds, m in sorted(have, key=lambda k: (list(DATA).index(k[0]), k[1])):
        labs = {k[2] for k in t if k[:2] == (ds, m)}
        cuts = sorted((l for l in labs if "[" in l), key=lambda l: (len(l.split("[")[1].split()), l))
        for a, b in comps + [(c, "uncut") for c in cuts]:
            row = [gap(t[(ds, m, a, B)], t[(ds, m, b, B)]) if t.get((ds, m, a, B)) and t.get((ds, m, b, B)) else "—"
                   for B in BUDGETS]
            print(f"| {DATA[ds]} | {m} | {a} − {b} | " + " | ".join(row) + " |")
    print()


if __name__ == "__main__":
    ch = load("channel_runs.csv")
    print("#### Seed noise\n")
    channel(ch)
    oneshot(ch)
    blocks(load("block_runs.csv"))
    budgets(load("budget_runs.csv"))
