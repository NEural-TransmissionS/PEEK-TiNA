#!/usr/bin/env python3
"""Collect ``results/*.json`` into ``results.csv`` + ``summary.png``.

Two panels: mAP50-95 vs blocks-cut and GFLOPs vs blocks-cut, one line per
(dataset, size, criterion). k = 0 is the full model, shared by every criterion.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

from matplotlib import pyplot as plt

HERE = Path(__file__).resolve().parent
COLS = ["dataset", "size", "criterion", "k", "layers_cut", "cut_persistence",
        "mAP50-95", "mAP50", "gflops", "params", "epochs"]


def main() -> None:
    rows = [json.loads(p.read_text()) for p in sorted((HERE / "results").glob("*.json"))]
    for r in rows:
        r.setdefault("criterion", "combined")  # pre-criterion results
    if not rows:
        raise SystemExit("no results/*.json yet")
    rows.sort(key=lambda r: (r["dataset"], r["size"], r["criterion"], r["k"]))
    base = {(r["dataset"], r["size"]): r for r in rows if r["k"] == 0}

    with (HERE / "results.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([*COLS, "gflops_vs_full_%", "mAP_delta_vs_full"])
        for r in rows:
            b = base.get((r["dataset"], r["size"]), r)
            writer.writerow([
                *[r.get(c, "") for c in COLS],
                round(100 * (r["gflops"] - b["gflops"]) / b["gflops"], 2) if b["gflops"] else "",
                round(r["mAP50-95"] - b["mAP50-95"], 4),
            ])

    keys = sorted({(r["dataset"], r["size"], r["criterion"]) for r in rows if r["k"]})
    styles = {"combined": "o-", "h0": "s--", "h1": "^:"}
    fig, (ax_map, ax_flop) = plt.subplots(1, 2, figsize=(11, 4.2))
    for dataset, size, crit in keys:
        seq = [base[(dataset, size)]] if (dataset, size) in base else []
        seq += sorted((r for r in rows if (r["dataset"], r["size"], r["criterion"]) ==
                       (dataset, size, crit) and r["k"]), key=lambda r: r["k"])
        ks = [r["k"] for r in seq]
        label, fmt = f"{dataset}_{size}_{crit}", styles.get(crit, "o-")
        ax_map.plot(ks, [r["mAP50-95"] for r in seq], fmt, label=label)
        ax_flop.plot(ks, [r["gflops"] for r in seq], fmt, label=label)
    ax_map.set(xlabel="blocks cut (lowest persistence first)", ylabel="mAP50-95",
               title="accuracy vs pruning", ylim=(0, 1))
    ax_flop.set(xlabel="blocks cut", ylabel="GFLOPs", title="compute vs pruning")
    for ax in (ax_map, ax_flop):
        ax.set_xticks([0, 1, 2, 3])
        ax.grid(alpha=0.3)
    ax_flop.legend(fontsize=5, ncol=3)
    fig.tight_layout()
    fig.savefig(HERE / "summary.png", dpi=140)
    print(f"wrote {HERE / 'results.csv'} and {HERE / 'summary.png'}  ({len(rows)} runs)")


if __name__ == "__main__":
    main()
