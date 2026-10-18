#!/usr/bin/env python3
"""Print the README's results tables (Markdown) from the committed results/ files.

    python scripts/make_results_tables.py > /tmp/tables.md

Both TDA bases share one setup; l2 / random / none rows come from the PEEK grid.
"""

from __future__ import annotations

import csv
import statistics as st
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / "results"
SP = RES / "structural_pruning"
RATIOS = ["0.1", "0.2", "0.3", "0.4", "0.5"]
TDA = {"field_h0": "field H0", "field_h1": "field H1", "field_total": "field H0+H1",
       "h0": "per-channel H0", "h1": "per-channel H1", "total": "per-channel H0+H1"}
BASES = {"peek": "PEEK", "activation": "Activation"}
ROWS = {(b, c): f"{name} {TDA[c]}" for b, name in BASES.items() for c in TDA}
ROWS[("peek", "l2")] = "L2 magnitude"
ROWS[("peek", "random")] = "Random (seed 42)"
STAGES = {"after_bn_recal": "before fine-tuning (pruned + BN re-estimated)",
          "after_finetune": "after 20-epoch fine-tune"}
SHAPES = [f"shapes_{d}_{m}" for d in ("circles", "triangles", "squares", "mixed") for m in "ns"]
WSD = ["wsd_n", "wsd_s"]


def load(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open()))


def grids(tag: str) -> dict:
    """(basis, criterion, ratio) -> row, for both bases of one model."""
    out = {}
    for basis in BASES:
        for r in load(SP / basis / tag / "results.csv"):
            out[(basis, r["criterion"], r["ratio"])] = r
    return out


def val(g: dict, row: tuple, ratio: str, stage: str, key: str = "map50_95") -> float:
    return float(g[(*row, ratio)][f"{key}_{stage}"])


def random_seeds(tag: str, stage: str) -> dict[str, list[float]]:
    """mAP50-95 of every random mask at each ratio: 10 one-shot masks before fine-tuning,
    seeds 42/43/44 after the 20-epoch fine-tune."""
    if stage == "after_bn_recal":
        rows = load(SP / "random_spread" / f"{tag}_oneshot.csv")
    else:
        rows = load(SP / "random_spread" / f"{tag}_ft20_random.csv") + load(SP / "peek" / tag / "results.csv")
    out = {r: [] for r in RATIOS}
    for row in rows:
        if row["criterion"] == "random" and row["ratio"] in out:
            out[row["ratio"]].append(float(row[f"map50_95_{stage}"]))
    return out


def beats_random(gs: dict, rnd: dict, row: tuple, tags: list[str], stage: str) -> tuple[int, int, float]:
    """Over the (model, ratio) pairs: how often above the random mean, how often above
    every random seed, and the mean margin over the random mean."""
    above_mean = above_all = 0
    margins = []
    for t in tags:
        for r in RATIOS:
            x, seeds = val(gs[t], row, r, stage), rnd[(t, stage)][r]
            above_mean += x > st.mean(seeds)
            above_all += x > max(seeds)
            margins.append(x - st.mean(seeds))
    return above_mean, above_all, st.mean(margins)


def structural_wsd(model: str) -> None:
    g = grids(f"wsd_{model}")
    base = g[("peek", "none", "0.0")]
    gfl = " / ".join(f"{float(g[('peek', 'l2', r)]['gflops']):.2f}" for r in RATIOS)
    print(f"#### YOLO26{model} — unpruned {float(base['gflops']):.2f} GFLOPs; mAP50-95 "
          f"{float(base['map50_95_after_bn_recal']):.3f} (BN re-estimated), "
          f"{float(base['map50_95_after_finetune']):.3f} (20-epoch fine-tune)\n")
    print(f"GFLOPs at 10 / 20 / 30 / 40 / 50 % per-layer pruning: {gfl}\n")
    for stage, title in STAGES.items():
        v = lambda row, r, k="map50_95": val(g, row, r, stage, k)  # noqa: E731
        mean = lambda row, k="map50_95": st.mean(v(row, r, k) for r in RATIOS)  # noqa: E731
        rnd = random_seeds(f"wsd_{model}", stage)
        rnd_mean = st.mean(st.mean(x) for x in rnd.values())
        l2 = ("peek", "l2")
        best = {r: max(v(row, r) for row in ROWS) for r in RATIOS}
        print(f"**mAP50-95, {title}**\n")
        print("| Criterion | 10% | 20% | 30% | 40% | 50% | Mean | Δ vs L2 | Δ vs random mean | Mean mAP50 |")
        print("|---|" + "---:|" * 9)
        for row, name in ROWS.items():
            cells = []
            for r in RATIOS:
                cell = f"{v(row, r):.3f}" + ("†" if row[1] != "random" and v(row, r) > max(rnd[r]) else "")
                cells.append(f"**{cell}**" if v(row, r) == best[r] else cell)
            dl = "—" if row == l2 else f"{mean(row) - mean(l2):+.3f}"
            print(f"| {name} | " + " | ".join(cells) + f" | {mean(row):.3f} | {dl} | "
                  f"{mean(row) - rnd_mean:+.3f} | {mean(row, 'map50'):.3f} |")
        n = len(rnd[RATIOS[0]])
        cells = [f"{st.mean(rnd[r]):.3f} ({min(rnd[r]):.3f}–{max(rnd[r]):.3f})" for r in RATIOS]
        print(f"| Random, {n} seeds: mean (min–max) | " + " | ".join(cells)
              + f" | {rnd_mean:.3f} | {rnd_mean - mean(l2):+.3f} | — | — |")
        print()


def structural_shapes() -> None:
    gs = {t: grids(t) for t in SHAPES}
    rnd = {(t, s): random_seeds(t, s) for t in SHAPES for s in STAGES}
    base = st.mean(float(g[("peek", "none", "0.0")]["map50_95_after_finetune"]) for g in gs.values())
    print(f"Mean over the 8 shape models (4 datasets × n/s); unpruned mean mAP50-95 {base:.3f} "
          "after the same 20-epoch fine-tune. Counts are over the 40 (model, ratio) pairs; random "
          "uses the same seeds as WSD (10 one-shot masks before fine-tuning, seeds 42/43/44 after).\n")
    pairs = [(t, r) for t in SHAPES for r in RATIOS]
    for stage, title in STAGES.items():
        v = lambda row, t, r: val(gs[t], row, r, stage)  # noqa: E731
        print(f"**mAP50-95, {title}**\n")
        print("| Criterion | 10% | 20% | 30% | 40% | 50% | Mean | > random mean | > every random seed | Wins vs L2 |")
        print("|---|" + "---:|" * 9)
        for row, name in ROWS.items():
            cells = [st.mean(v(row, t, r) for t in SHAPES) for r in RATIOS]
            am, aa, _ = beats_random(gs, rnd, row, SHAPES, stage)
            wr = "— | —" if row[1] == "random" else f"{am}/40 | {aa}/40"
            wl = "—" if row[1] == "l2" else f"{sum(v(row, t, r) > v(('peek', 'l2'), t, r) for t, r in pairs)}/40"
            print(f"| {name} | " + " | ".join(f"{x:.3f}" for x in cells)
                  + f" | {st.mean(cells):.3f} | {wr} | {wl} |")
        n = len(rnd[(SHAPES[0], stage)][RATIOS[0]])
        cells = [st.mean(st.mean(rnd[(t, stage)][r]) for t in SHAPES) for r in RATIOS]
        print(f"| Random, {n} seeds (mean) | " + " | ".join(f"{x:.3f}" for x in cells)
              + f" | {st.mean(cells):.3f} | — | — | — |")
        print()


def basis_comparison() -> None:
    """PEEK minus activation, criterion by criterion, on identical runs otherwise."""
    gs = {t: grids(t) for t in WSD + SHAPES}
    print("Mean Δ mAP50-95 (PEEK − activation) and how often PEEK is higher, over every "
          "(model, ratio) pair: 10 for WSD (n/s × 5 ratios), 40 for the shapes.\n")
    head = [f"{s} {'before' if st_ == 'after_bn_recal' else 'after'} FT" for s in ("WSD", "Shapes") for st_ in STAGES]
    print("| Criterion | " + " | ".join(head) + " |")
    print("|---|" + "---:|" * len(head))
    for c, name in TDA.items():
        cells = []
        for models in (WSD, SHAPES):
            for stage in STAGES:
                d = [val(gs[t], ("peek", c), r, stage) - val(gs[t], ("activation", c), r, stage)
                     for t in models for r in RATIOS]
                cells.append(f"{st.mean(d):+.3f} ({sum(x > 0 for x in d)}/{len(d)})")
        print(f"| {name} | " + " | ".join(cells) + " |")
    print()


def h1_vs_random() -> None:
    """How often the H1 criteria beat random pruning, per basis and stage."""
    gs = {t: grids(t) for t in WSD + SHAPES}
    rnd = {(t, s): random_seeds(t, s) for t in WSD + SHAPES for s in STAGES}
    print("Over the (model, ratio) pairs — 10 for WSD, 40 for the shapes — how often the "
          "criterion is above the mean of the random seeds and above every random seed (10 "
          "one-shot masks before fine-tuning, seeds 42/43/44 after), and the mean margin over "
          "the random mean.\n")
    head = [f"{g} {h}" for g in ("WSD", "Shapes") for h in ("> random mean", "> every seed", "Δ vs random mean")]
    print("| Criterion | Stage | " + " | ".join(head) + " |")
    print("|---|---|" + "---:|" * len(head))
    for c in ("field_h1", "h1"):
        for b, bname in BASES.items():
            for stage in STAGES:
                cells = []
                for tags in (WSD, SHAPES):
                    am, aa, mg = beats_random(gs, rnd, (b, c), tags, stage)
                    n = len(tags) * len(RATIOS)
                    cells += [f"{am}/{n}", f"{aa}/{n}", f"{mg:+.3f}"]
                when = "before FT" if stage == "after_bn_recal" else "after FT"
                print(f"| {bname} {TDA[c]} | {when} | " + " | ".join(cells) + " |")
    print()


EPOCHS = [1, 5, 10, 15, 20]
GROUPS = {"WSD n": ["wsd_n"], "WSD s": ["wsd_s"], "Shapes (8 models)": SHAPES}


def curve_sets(tag: str) -> dict:
    """(basis, criterion, ratio) -> per-epoch val mAP50-95 lists, one per seed."""
    out: dict = {}
    for basis, path in (("peek", SP / "peek" / tag / "curves.csv"),
                        ("activation", SP / "activation" / tag / "curves.csv"),
                        ("peek", SP / "random_spread" / f"{tag}_ft20_curves.csv")):
        runs: dict = {}
        for r in load(path):
            runs.setdefault((r["criterion"], r["ratio"], r["seed"]), {})[int(r["epoch"])] = float(r["map50_95"])
        for (c, ratio, _), ep in runs.items():
            out.setdefault((basis, c, ratio), []).append([ep[e] for e in sorted(ep)])
    return out


def recovery(frac: float = 0.95) -> None:
    cs = {t: curve_sets(t) for t in WSD + SHAPES}
    base = {t: float(grids(t)[("peek", "none", "0.0")]["map50_95_after_bn_recal"]) for t in cs}
    rows = {**{row: name for row, name in ROWS.items() if row[1] != "random"},
            ("peek", "random"): "Random (seeds 42/43/44)"}
    print(f"**Epochs to recover {frac:.0%} of the unpruned mAP50-95 before fine-tuning** — "
          "mean first epoch over the runs that recover; (never) = runs still below after 20 "
          "epochs. Random counts all three seeds. On the shapes the unpruned models reach only "
          f"{st.mean(base[t] for t in SHAPES):.3f} before fine-tuning (0.98 after), so every "
          "criterion clears this bar within a few epochs.\n")
    print("| Criterion | " + " | ".join(GROUPS) + " |")
    print("|---|" + "---:|" * len(GROUPS))
    for row, name in rows.items():
        cells = []
        for tags in GROUPS.values():
            hits = [next((i + 1 for i, y in enumerate(c) if y >= frac * base[t]), None)
                    for t in tags for r in RATIOS for c in cs[t][(*row, r)]]
            ok = [h for h in hits if h is not None]
            cells.append((f"{st.mean(ok):.1f}" if ok else "—") + f" ({len(hits) - len(ok)}/{len(hits)} never)")
        print(f"| {name} | " + " | ".join(cells) + " |")
    print()
    rows[("peek", "none")] = "Unpruned (control)"
    for group, tags in GROUPS.items():
        b = f"{base[tags[0]]:.3f}" if len(tags) == 1 else f"mean {st.mean(base[t] for t in tags):.3f}"
        print(f"**{group}: val mAP50-95 during the 20-epoch fine-tune** (mean over 10–50% pruning "
              f"and seeds; unpruned before fine-tuning: {b})\n")
        print("| Criterion | " + " | ".join(f"Epoch {e}" for e in EPOCHS) + " |")
        print("|---|" + "---:|" * len(EPOCHS))
        for row, name in rows.items():
            rs = ["0.0"] if row[1] == "none" else RATIOS
            cells = [st.mean(c[e - 1] for t in tags for r in rs for c in cs[t][(*row, r)]) for e in EPOCHS]
            print(f"| {name} | " + " | ".join(f"{x:.3f}" for x in cells) + " |")
        print()


def wsd_methods() -> None:
    """Channel pruning (PEEK field H1) next to block replacement (PEEK ranking) on WSD."""
    blocks = [r for r in load(RES / "block_pruning" / "peek" / "results.csv")
              if r["dataset"] == "wsd" and r["criterion"] == "combined"]
    print("Channel-pruned models are fine-tuned 20 epochs; block-cut models are retrained from "
          "scratch (so no before-fine-tune score). Each method is measured against its own "
          "unpruned reference, and the two pipelines count GFLOPs / parameters with different "
          "tools, so compare the Δ% columns, not absolute counts. L2 and random use the same "
          "channel pruning; random is the mean of seeds 42/43/44.\n")
    pct = lambda a, b: f"{100 * (float(a) / float(b) - 1):+.1f}%"  # noqa: E731
    for m in "ns":
        g = grids(f"wsd_{m}")
        rnd = random_seeds(f"wsd_{m}", "after_finetune")
        ref = g[("peek", "none", "0.0")]
        ft = float(ref["map50_95_after_finetune"])
        print(f"#### YOLO26{m}\n")
        print("| Method | Setting | GFLOPs Δ | Params Δ | mAP50-95 before FT | mAP50-95 final | "
              "Δ final vs unpruned | L2 final | Random final |")
        print("|---|---|" + "---:|" * 7)
        print(f"| Unpruned (channel reference) | — | — | — | {float(ref['map50_95_after_bn_recal']):.3f} | "
              f"{ft:.3f} | — | — | — |")
        for i, r in enumerate(RATIOS):
            row = g[("peek", "field_h1", r)]
            name = "Channel pruning, PEEK field H1" if i == 0 else ""
            print(f"| {name} | {float(r):.0%} | {pct(row['gflops'], ref['gflops'])} | "
                  f"{pct(row['params'], ref['params'])} | {float(row['map50_95_after_bn_recal']):.3f} | "
                  f"{float(row['map50_95_after_finetune']):.3f} | {float(row['map50_95_after_finetune']) - ft:+.3f} | "
                  f"{val(g, ('peek', 'l2'), r, 'after_finetune'):.3f} | {st.mean(rnd[r]):.3f} |")
        full = next(b for b in blocks if b["size"] == m and b["k"] == "0")
        print(f"| Unpruned (block reference) | — | — | — | — | {float(full['mAP50-95']):.3f} | — | — | — |")
        for b in (b for b in blocks if b["size"] == m and b["k"] != "0"):
            name = "Block replacement (PEEK ranking)" if b["k"] == "1" else ""
            print(f"| {name} | cut {b['layers_cut'].strip('[]')} | {float(b['gflops_vs_full_%']):+.1f}% | "
                  f"{pct(b['params'], full['params'])} | — | {float(b['mAP50-95']):.3f} | "
                  f"{float(b['mAP_delta_vs_full']):+.3f} | — | — |")
        print()


def _channel_runs(tag: str) -> dict:
    """(criterion, ratio, seed) -> [(epoch, mAP50-95, seconds)] for every fine-tune of a model."""
    runs: dict = {}
    for path in (SP / "peek" / tag / "curves.csv", SP / "random_spread" / f"{tag}_ft20_curves.csv"):
        for r in load(path):
            runs.setdefault((r["criterion"], r["ratio"], r["seed"]), []).append(
                (int(r["epoch"]), float(r["map50_95"]), float(r["time_s"])))
    return {k: sorted(v) for k, v in runs.items()}


def _block_run(name: str) -> list:
    rows = [{k.strip(): v for k, v in r.items()} for r in load(RES / "block_pruning" / "curves" / f"{name}.csv")]
    return [(int(r["epoch"]), float(r["metrics/mAP50-95(B)"]), float(r["time"])) for r in rows]


def convergence(frac: float = 0.95) -> None:
    """Epochs and wall-clock time until each method reaches frac x its unpruned reference."""
    blocks = [r for r in load(RES / "block_pruning" / "peek" / "results.csv")
              if r["dataset"] == "wsd" and r["criterion"] == "combined"]
    print(f"First epoch whose val mAP50-95 reaches {frac:.0%} of the method's unpruned reference: "
          "the unpruned model after the same 20-epoch fine-tune (channel pruning) or the full "
          "model trained from scratch (block replacement). Channel rows average the 10–50% ratios "
          "(random: and seeds 42/43/44); never reached = runs still below after 20 epochs. Block "
          "runs train with patience 100, so their totals run well past that point. Times are per "
          "run on one A100, including per-epoch validation.\n")
    for m in "ns":
        g = grids(f"wsd_{m}")
        runs = _channel_runs(f"wsd_{m}")
        thr = frac * float(g[("peek", "none", "0.0")]["map50_95_after_finetune"])
        print(f"#### YOLO26{m}\n")
        print(f"| Method | Setting | Epochs to {frac:.0%} | Time to {frac:.0%} | Never reached | "
              "Total epochs | Total time |")
        print("|---|---|" + "---:|" * 5)
        for crit, name in (("field_h1", "Channel pruning, PEEK field H1"), ("l2", "Channel pruning, L2"),
                           ("random", "Channel pruning, random (3 seeds)"),
                           ("none", "Unpruned, fine-tuned (reference)")):
            curves = [v for (c, _, _), v in runs.items() if c == crit]
            hits = [next(((e, t) for e, y, t in c if y >= thr), None) for c in curves]
            ok = [h for h in hits if h]
            setting = "—" if crit == "none" else "mean of 10–50%"
            print(f"| {name} | {setting} | {st.mean(h[0] for h in ok):.1f} | "
                  f"{st.mean(h[1] for h in ok) / 60:.1f} min | {len(hits) - len(ok)}/{len(hits)} | "
                  f"{st.mean(len(c) for c in curves):.0f} | {st.mean(c[-1][2] for c in curves) / 60:.1f} min |")
        full = next(b for b in blocks if b["size"] == m and b["k"] == "0")
        bthr = frac * float(full["mAP50-95"])
        for b in (b for b in blocks if b["size"] == m):
            c = _block_run(b["source_run"])
            e, t = next(((e, t) for e, y, t in c if y >= bthr), (None, None))
            name = ("Block replacement, full model (reference)" if b["k"] == "0"
                    else "Block replacement" if b["k"] == "1" else "")
            setting = "none cut" if b["k"] == "0" else f"cut {b['layers_cut'].strip('[]')}"
            reach = f"{e} | {t / 60:.0f} min | —" if e else "— | — | yes"
            print(f"| {name} | {setting} | {reach} | {len(c)} | {c[-1][2] / 60:.0f} min |")
        print()


def block() -> None:
    rows = {b: load(RES / "block_pruning" / b / "results.csv") for b in BASES}
    orders = {(b, r["size"], r["criterion"]): r["layers_cut"] for b in BASES for r in rows[b]
              if r["dataset"] == "wsd" and r["k"] == "3"}
    same = len(set(orders.values())) == 1
    print("**WSD (full 1000-epoch scratch protocol)** — "
          + ("both bases and all three criteria (combined, H0-only, H1-only) cut "
             f"{' → '.join(next(iter(orders.values())).strip('[]').split(', '))} on both models, "
             "so the table is the same for every ranking\n"
             if same else "cut orders differ by basis/criterion; see results.csv\n"))
    print("| Model | Blocks cut | mAP50-95 | Δ vs full | mAP50 | GFLOPs | Δ GFLOPs |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for r in (r for r in rows["peek"] if r["dataset"] == "wsd" and r["criterion"] == "combined"):
        cut = r["layers_cut"].strip("[]") or "none"
        print(f"| {r['size']} | {cut} | {float(r['mAP50-95']):.3f} | {float(r['mAP_delta_vs_full']):+.3f} | "
              f"{float(r['mAP50']):.3f} | {float(r['gflops']):.2f} | {float(r['gflops_vs_full_%']):+.1f}% |")
    print("\n**Synthetic shapes (60 epochs; mean Δ mAP50-95 vs the full model over "
          "circles/triangles/squares/mixed × n/s, worst in brackets)**\n")
    head = [f"{BASES[b]} {c}" for b in BASES for c in ("combined", "H0", "H1")]
    print("| Blocks cut | GFLOPs Δ | " + " | ".join(head) + " |")
    print("|---|" + "---:|" * (1 + len(head)))
    for k in "123":
        cells = []
        for b in BASES:
            shapes = [r for r in rows[b] if r["dataset"] != "wsd" and r["k"] == k]
            for c in ("combined", "h0", "h1"):
                d = [float(r["mAP_delta_vs_full"]) for r in shapes if r["criterion"] == c]
                cells.append(f"{st.mean(d):+.4f} ({min(d):+.3f})")
        g = st.mean(float(r["gflops_vs_full_%"]) for r in rows["peek"] if r["dataset"] != "wsd" and r["k"] == k)
        print(f"| {k} | {g:.1f}% | " + " | ".join(cells) + " |")
    print()


if __name__ == "__main__":
    print("### Structural channel pruning (WSD)\n")
    print("† = above every random mask at that ratio (10 one-shot masks before fine-tuning; "
          "seeds 42/43/44 after).\n")
    for m in "ns":
        structural_wsd(m)
    print("### Structural channel pruning (synthetic shapes)\n")
    structural_shapes()
    print("### PEEK vs activation basis\n")
    basis_comparison()
    print("### H1 criteria vs random pruning\n")
    h1_vs_random()
    print("### Recovery during the fine-tune\n")
    recovery()
    print("### Block replacement\n")
    block()
    print("### Channel pruning vs block replacement (WSD)\n")
    wsd_methods()
    print("### Convergence: channel pruning vs block replacement (WSD)\n")
    convergence()
