#!/usr/bin/env python3
"""Train ONE model (full or pruned): from scratch, or with ``--init-weights`` cut from a
pretrained model and fine-tuned (that protocol is in ``finetune_init.py``).

Runs in its own process on purpose: chaining several ``YOLO().train()`` calls in
a single interpreter reliably trips ``CUDNN_STATUS_INTERNAL_ERROR`` after a few
runs. ``run_experiment.py`` spawns one of these per ``k``.

Writes ``results/<dataset>_<size>[_<criterion>]_k<k>.json`` (no suffix for ``combined``;
fine-tuned runs get ``_ft``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from topo_prune import CRITERIA, SCOPES, build_pruned_yaml, cut_order, model_stats, ranking_path

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
    ap.add_argument("--scope", default="head", choices=list(SCOPES),
                    help="ranking over the 4 head C3k2 or all 8 C3k2 (run names get _all)")
    ap.add_argument("--norm", default="raw", choices=["raw", "grid", "null"],
                    help="ranking variant from a --norms ranking file (*_norm.json); run names get _<norm>")
    ap.add_argument("--random-order", type=int, default=None, metavar="SEED",
                    help="baseline: cut the first k of a seeded random order of the scope's blocks "
                         "instead of a ranking (run names get _rand<SEED>)")
    ap.add_argument("--init-weights", default=None,
                    help="cut this pretrained model and fine-tune it instead of training from scratch")
    ap.add_argument("--data", default=None, help="training/val dataset yaml (default data/<dataset>/data.yaml)")
    ap.add_argument("--optimizer", default="auto")
    ap.add_argument("--lr0", type=float, default=None)
    ap.add_argument("--momentum", type=float, default=None)
    ap.add_argument("--warmup-epochs", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=42)
    # recovery studies (iter_probe.py lives with the structural-pruning code)
    ap.add_argument("--fraction", type=float, default=None, help="fraction of the train set per epoch")
    ap.add_argument("--close-mosaic", type=int, default=None, help="epochs at the end without mosaic (default 10)")
    ap.add_argument("--warmup-bias-lr", type=float, default=None,
                    help="starting lr of the bias group during warmup (Ultralytics default 0.1, whatever lr0 is)")
    ap.add_argument("--probe-every", type=int, default=None, help="evaluate every N training batches (probe.csv)")
    ap.add_argument("--probe-max-batches", type=int, default=None, help="stop the fine-tune after N batches")
    ap.add_argument("--tag", default="", help="suffix for the run name (e.g. _probe, _b500)")
    ap.add_argument("--skip-existing", action="store_true", help="exit if results/<name>.json already exists")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    if str(args.device).lower() != "cpu" and not torch.cuda.is_available():
        raise SystemExit(
            f"torch.cuda.is_available() is False (torch {torch.__version__}, "
            f"built for CUDA {torch.version.cuda}). The GPU node's driver is likely "
            "too old for this torch build -- see README. Pass --device cpu to force CPU."
        )

    data_yaml = Path(args.data) if args.data else HERE / "data" / args.dataset / "data.yaml"
    if args.random_order is not None:
        import numpy as np

        order = [int(i) for i in np.random.default_rng(args.random_order).permutation(SCOPES[args.scope])]
        scores = {}
    else:
        rank_tag = "" if args.norm == "raw" else "norm"
        ranking = json.loads(ranking_path(args.dataset, args.size, args.basis, args.scope, rank_tag).read_text())
        if args.norm != "raw":
            ranking = {**ranking, **ranking[args.norm]}  # that variant's scores, same candidates
        order, scores = cut_order(ranking, args.criterion)
    cut = order[: args.k]
    cfg = build_pruned_yaml(args.size, cut, HERE / "runs" / "_cfg")

    model = YOLO(str(cfg))  # random init -> trains from scratch
    extra = {}
    if args.init_weights:
        from finetune_init import init_from_pretrained, trainer_for

        net, copied, total = init_from_pretrained(cfg, args.init_weights, data_yaml)
        print(f"init from {Path(args.init_weights).name}: copied {copied}/{total} tensors (cut blocks start fresh)")
        model.model = net
        extra["trainer"] = trainer_for(net, args.seed)
    extra.update({k: v for k, v in (("lr0", args.lr0), ("momentum", args.momentum)) if v is not None})
    stats = model_stats(model)

    if args.random_order is not None:
        name = f"{args.dataset}_{args.size}_k{args.k}_{args.scope}_rand{args.random_order}"
    else:
        tag = "" if args.criterion == "combined" else f"_{args.criterion}"
        name = (f"{args.dataset}_{args.size}{tag}_k{args.k}" + ("" if args.basis == "activation" else f"_{args.basis}")
                + ("" if args.scope == "head" else f"_{args.scope}") + ("" if args.norm == "raw" else f"_{args.norm}"))
    if args.init_weights:
        name += "_ft" + ("" if args.seed == 42 else f"_s{args.seed}")
    name += args.tag
    if args.skip_existing and (HERE / "results" / f"{name}.json").exists():
        print(f"{name}: already done")
        return
    extra.update({k: v for k, v in (("fraction", args.fraction), ("close_mosaic", args.close_mosaic), ("warmup_bias_lr", args.warmup_bias_lr)) if v is not None})
    stopped = None
    if args.probe_every:
        import sys
        for d in (HERE.parent / "PEEK-rvc-pruning", HERE.parent / "structural_pruning"):  # server / tina layout
            sys.path.insert(0, str(d))
        from peek_rvc_pruning import iter_probe
        iter_probe.attach(model, HERE / "runs" / name / "probe.csv", args.probe_every, args.probe_max_batches)
    try:
        model.train(
            data=str(data_yaml), epochs=args.epochs, batch=args.batch, imgsz=640, seed=args.seed,
            patience=args.patience, deterministic=args.deterministic, amp=args.amp,
            device=args.device, optimizer=args.optimizer, warmup_epochs=args.warmup_epochs,
            project=str(HERE / "runs"), name=name, exist_ok=True, plots=False, verbose=False, **extra,
        )
    except Exception as e:  # iter_probe.StopProbe: probe run cut short on purpose, the curve is the result
        if type(e).__name__ != "StopProbe":
            raise
        stopped = str(e)
    metrics = None if stopped else model.val(
        data=str(data_yaml), split="val", verbose=False,
        project=str(HERE / "runs"), name=f"{name}_val", exist_ok=True, plots=False,
    )

    out = HERE / "results" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "dataset": args.dataset, "size": args.size, "criterion": args.criterion, "k": args.k,
        "basis": args.basis, "scope": args.scope, "norm": args.norm, "random_order": args.random_order,
        "init_weights": Path(args.init_weights).name if args.init_weights else None, "seed": args.seed,
        "optimizer": args.optimizer, "lr0": args.lr0, "data": data_yaml.name,
        "layers_cut": cut,
        "cut_persistence": [scores[str(i)] for i in cut] if scores else [],
        "mAP50-95": round(float(metrics.box.map), 4) if metrics else None,
        "mAP50": round(float(metrics.box.map50), 4) if metrics else None,
        "epochs": args.epochs,
        "epochs_run": model.trainer.epoch + 1,  # < epochs if patience stopped it
        **{k: getattr(args, k) for k in ("fraction", "close_mosaic", "warmup_bias_lr", "probe_every", "probe_max_batches")
           if getattr(args, k) is not None},
        **({"probe_stopped": stopped} if stopped else {}),
        **stats,
    }, indent=2))
    print(f"{name}: mAP50-95={float(metrics.box.map) if metrics else float('nan'):.4f}  "
          f"GFLOPs={stats['gflops']}  params={stats['params']}  cut={cut}")


if __name__ == "__main__":
    main()
