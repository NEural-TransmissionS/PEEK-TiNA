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

from topo_prune import CRITERIA, rank_layers

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
    args, train_extra = ap.parse_known_args()  # extras go to _train_one

    data_dir = HERE / "data" / args.dataset
    if not (data_dir / "data.yaml").is_file():
        subprocess.run([sys.executable, str(HERE / "make_shapes.py")], check=True)

    rank_dir = "ranking" if args.basis == "activation" else f"ranking_{args.basis}"
    rank_path = HERE / rank_dir / f"{args.dataset}_{args.size}.json"
    rank_path.parent.mkdir(exist_ok=True)
    if not rank_path.is_file():
        ranking = rank_layers(
            str(HERE / "weights" / f"{args.dataset}_{args.size}.pt"),
            data_dir / "images" / args.rank_split,
            args.n_rank,
            basis=args.basis,
        )
        rank_path.write_text(json.dumps(ranking, indent=2))
        print(f"[{args.dataset}_{args.size}] cut order (low persistence first): "
              f"{ranking['cut_order_ascending']}  combined={ranking['combined']}")

    for k in args.ks:
        subprocess.run(
            [sys.executable, str(HERE / "_train_one.py"),
             "--dataset", args.dataset, "--size", args.size, "--k", str(k),
             "--epochs", str(args.epochs), "--device", args.device,
             "--criterion", args.criterion, "--basis", args.basis, *train_extra],
            check=True,
        )


if __name__ == "__main__":
    main()
