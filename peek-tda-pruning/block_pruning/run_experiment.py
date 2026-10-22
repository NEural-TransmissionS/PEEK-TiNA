#!/usr/bin/env python3
"""One (dataset, size): rank the candidate blocks once, then train the full model
and the k = 1 / 2 / 3 pruned models -- each in its own subprocess.

One SLURM array task per (dataset, size); 8 tasks total.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from topo_prune import CRITERIA, SCOPES, rank_layers, ranking_path

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--size", required=True, choices=["n", "s"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--n-rank", type=int, default=40)
    ap.add_argument("--rank-split", default="val", help="images/<split> used for ranking")
    ap.add_argument("--ks", type=int, nargs="*", default=[0, 1, 2, 3],
                    help="blocks to cut per run; pass none to rank only")
    ap.add_argument("--device", default="0")
    ap.add_argument("--criterion", default="combined", choices=list(CRITERIA))
    ap.add_argument("--basis", default="peek", choices=["activation", "peek"],
                    help="map the block topology is read from; peek rankings go to ranking_peek/")
    ap.add_argument("--scope", default="head", choices=list(SCOPES),
                    help="candidate blocks: the 4 head C3k2 (default) or all 8 C3k2 incl. backbone")
    ap.add_argument("--norms", action="store_true",
                    help="also store map-size-normalised rankings (grid, null); file gets _norm")
    args, train_extra = ap.parse_known_args()  # extras go to _train_one

    data_dir = HERE / "data" / args.dataset
    if not (data_dir / "data.yaml").is_file():
        subprocess.run([sys.executable, str(HERE / "make_shapes.py")], check=True)

    rank_path = ranking_path(args.dataset, args.size, args.basis, args.scope, "norm" if args.norms else "")
    rank_path.parent.mkdir(exist_ok=True)
    if not rank_path.is_file():
        ranking = rank_layers(
            str(HERE / "weights" / f"{args.dataset}_{args.size}.pt"),
            data_dir / "images" / args.rank_split,
            args.n_rank,
            candidates=SCOPES[args.scope],
            basis=args.basis,
            norms=args.norms,
        )
        rank_path.write_text(json.dumps(ranking, indent=2))
        print(f"[{args.dataset}_{args.size}] cut order (low persistence first): "
              f"{ranking['cut_order_ascending']}  combined={ranking['combined']}")
        for v in ("grid", "null"):
            if v in ranking:
                print(f"[{args.dataset}_{args.size}] {v}: {ranking[v]['cut_order_ascending']}  "
                      f"combined={ranking[v]['combined']}")

    for k in args.ks:
        subprocess.run(
            [sys.executable, str(HERE / "_train_one.py"),
             "--dataset", args.dataset, "--size", args.size, "--k", str(k),
             "--epochs", str(args.epochs), "--device", args.device,
             "--criterion", args.criterion, "--basis", args.basis, "--scope", args.scope, *train_extra],
            check=True,
        )


if __name__ == "__main__":
    main()
