#!/usr/bin/env python3
"""Train ONE model (full or pruned) from scratch on one shape dataset.

Runs in its own process on purpose: chaining several ``YOLO().train()`` calls in
a single interpreter reliably trips ``CUDNN_STATUS_INTERNAL_ERROR`` after a few
runs. ``run_experiment.py`` spawns one of these per ``k``.

Writes ``results/<dataset>_<size>[_<criterion>]_k<k>.json`` (no suffix for ``combined``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from topo_prune import CRITERIA, build_pruned_yaml, cut_order, model_stats

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--size", required=True, choices=["n", "s"])
    ap.add_argument("--k", type=int, required=True, help="blocks to cut; 0 = full model")
    ap.add_argument("--criterion", default="combined", choices=list(CRITERIA),
                    help="persistence score that orders the cuts")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--patience", type=int, default=100)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--device", default="0")
    ap.add_argument("--basis", default="peek", choices=["activation", "peek"],
                    help="which ranking orders the cuts (peek -> ranking_peek/, run names get _peek)")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    if str(args.device).lower() != "cpu" and not torch.cuda.is_available():
        raise SystemExit(
            f"torch.cuda.is_available() is False (torch {torch.__version__}, "
            f"built for CUDA {torch.version.cuda}). The GPU node's driver is likely "
            "too old for this torch build -- see README. Pass --device cpu to force CPU."
        )

    data_yaml = HERE / "data" / args.dataset / "data.yaml"
    rank_dir = "ranking" if args.basis == "activation" else f"ranking_{args.basis}"
    ranking = json.loads((HERE / rank_dir / f"{args.dataset}_{args.size}.json").read_text())
    order, scores = cut_order(ranking, args.criterion)
    cut = order[: args.k]
    cfg = build_pruned_yaml(args.size, cut, HERE / "runs" / "_cfg")

    model = YOLO(str(cfg))  # random init -> trains from scratch
    stats = model_stats(model)

    tag = "" if args.criterion == "combined" else f"_{args.criterion}"
    name = f"{args.dataset}_{args.size}{tag}_k{args.k}" + ("" if args.basis == "activation" else f"_{args.basis}")
    model.train(
        data=str(data_yaml), epochs=args.epochs, batch=args.batch, imgsz=640, seed=42,
        patience=args.patience, deterministic=args.deterministic, amp=args.amp,
        device=args.device,
        project=str(HERE / "runs"), name=name, exist_ok=True, plots=False, verbose=False,
    )
    metrics = model.val(
        data=str(data_yaml), split="val", verbose=False,
        project=str(HERE / "runs"), name=f"{name}_val", exist_ok=True, plots=False,
    )

    out = HERE / "results" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "dataset": args.dataset, "size": args.size, "criterion": args.criterion, "k": args.k,
        "basis": args.basis,
        "layers_cut": cut,
        "cut_persistence": [scores[str(i)] for i in cut],
        "mAP50-95": round(float(metrics.box.map), 4),
        "mAP50": round(float(metrics.box.map50), 4),
        "epochs": args.epochs,
        "epochs_run": model.trainer.epoch + 1,  # < epochs if patience stopped it
        **stats,
    }, indent=2))
    print(f"{name}: mAP50-95={float(metrics.box.map):.4f}  "
          f"GFLOPs={stats['gflops']}  params={stats['params']}  cut={cut}")


if __name__ == "__main__":
    main()
