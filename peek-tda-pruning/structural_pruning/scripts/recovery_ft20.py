#!/usr/bin/env python3
"""Fine-tune recovery from the per-epoch training logs of the 20-epoch grids.

Recovery epoch = first epoch whose val mAP50-95 reaches FRAC x the unpruned model's
mAP50-95 before fine-tuning (pruned-free, BN re-estimated: the `none` row). Runs that
never reach it within the fine-tune are counted, not averaged.

    python scripts/recovery_ft20.py [--frac 0.95]
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
from pathlib import Path

OUT = Path(__file__).absolute().parent.parent / "outputs" / "yolo26_tda"
RATIOS = ["0.10", "0.20", "0.30", "0.40", "0.50"]
TDA = {"field_h0": "field H0", "field_h1": "field H1", "field_total": "field H0+H1",
       "h0": "per-channel H0", "h1": "per-channel H1", "total": "per-channel H0+H1"}
ROWS = [(b, c, f"{n} {TDA[c]}") for b, n in (("peek", "PEEK"), ("activation", "Activation")) for c in TDA]
ROWS += [("peek", "l2", "L2 magnitude"), ("peek", "random", "Random (seed 42)"), ("peek", "none", "Unpruned (control)")]
GROUPS = {"WSD n": ["wsd_n"], "WSD s": ["wsd_s"],
          "Shapes (8 models)": [f"shapes_{d}_{m}" for d in ("circles", "triangles", "squares", "mixed") for m in "ns"]}
EPOCHS = [1, 5, 10, 15, 20]


def curve(basis: str, model: str, crit: str, ratio: str) -> list[float]:
    tag = "none_r0.00_s42" if crit == "none" else f"{crit}_r{ratio}_s42"
    rows = csv.DictReader((OUT / f"{model}_protect6_{basis}_ft20" / "runs" / tag / "finetune" / "results.csv").open())
    return [float({k.strip(): v for k, v in r.items()}["metrics/mAP50-95(B)"]) for r in rows]


def base(model: str) -> float:
    for r in csv.DictReader((OUT / f"{model}_protect6_peek_ft20" / "results.csv").open()):
        if r["criterion"] == "none":
            return float(r["map50_95_after_bn_recal"])
    raise SystemExit(f"no none row for {model}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frac", type=float, default=0.95)
    args = ap.parse_args()
    curves = {(b, c, m, r): curve(b, m, c, r) for b, c, _ in ROWS for ms in GROUPS.values()
              for m in ms for r in (RATIOS[:1] if c == "none" else RATIOS)}
    bases = {m: base(m) for ms in GROUPS.values() for m in ms}

    print(f"#### Epochs to recover {args.frac:.0%} of the unpruned mAP50-95 (before fine-tuning)\n")
    print("Mean first epoch over the runs that recover; (never) = runs still below after 20 epochs.\n")
    print("| Criterion | " + " | ".join(GROUPS) + " |")
    print("|---|" + "---:|" * len(GROUPS))
    for b, c, name in ROWS:
        if c == "none":
            continue
        cells = []
        for ms in GROUPS.values():
            hits = []
            for m in ms:
                for r in RATIOS:
                    y = curves[(b, c, m, r)]
                    hits.append(next((i + 1 for i, v in enumerate(y) if v >= args.frac * bases[m]), None))
            ok = [h for h in hits if h is not None]
            never = len(hits) - len(ok)
            cells.append((f"{st.mean(ok):.1f}" if ok else "—") + f" ({never}/{len(hits)} never)")
        print(f"| {name} | " + " | ".join(cells) + " |")
    print()

    for group, ms in GROUPS.items():
        thr = f"{bases[ms[0]]:.3f}" if len(ms) == 1 else f"mean {st.mean(bases[m] for m in ms):.3f}"
        print(f"#### {group}: val mAP50-95 during the 20-epoch fine-tune (mean over 10–50% pruning)\n")
        print(f"Unpruned before fine-tuning: {thr}\n")
        print("| Criterion | " + " | ".join(f"Epoch {e}" for e in EPOCHS) + " |")
        print("|---|" + "---:|" * len(EPOCHS))
        for b, c, name in ROWS:
            rs = RATIOS[:1] if c == "none" else RATIOS
            cells = [st.mean(curves[(b, c, m, r)][e - 1] for m in ms for r in rs) for e in EPOCHS]
            print(f"| {name} | " + " | ".join(f"{x:.3f}" for x in cells) + " |")
        print()


if __name__ == "__main__":
    main()
