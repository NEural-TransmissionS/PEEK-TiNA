"""
YOLO26 channel pruning by persistent-homology (TDA) criteria.

Stages (each is its own SLURM step; see scripts/slurm/submit_yolo26_tda.sh):

    baseline   verify the unpruned official weights reproduce the published mAP
    score      baseline check, then per-channel H0/H1/total persistence, cached
    score-field  baseline check, then per-channel layer-field drift (field_h0/h1/total), cached
    prune      one (criterion, ratio) run: prune -> eval -> fine-tune -> eval
    aggregate  collect every run into results.csv + plots

Criteria: h0 / h1 / total (lowest average persistence pruned first, ranked
within each layer), plus l2-magnitude and random baselines, plus an unpruned
"none" control fine-tuned for the same number of epochs.

    python scripts/run_yolo26_tda_pruning.py score
    python scripts/run_yolo26_tda_pruning.py prune --grid-index 3
    python scripts/run_yolo26_tda_pruning.py aggregate
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.data.augment import LetterBox  # noqa: E402
from ultralytics.data.utils import IMG_FORMATS, check_det_dataset  # noqa: E402
from ultralytics.models.yolo.detect import DetectionTrainer  # noqa: E402

from peek_rvc_pruning import yolo_tda  # noqa: E402

CRITERIA = (*yolo_tda.TDA_CRITERIA, *yolo_tda.BASELINE_CRITERIA)  # default grid (keeps old indices)
ALL_CRITERIA = (*CRITERIA, *yolo_tda.FIELD_CRITERIA)
STAGES = ("before_finetune", "after_bn_recal", "after_finetune")
# Ultralytics-published COCO val2017 mAP50-95 at 640 px, NMS-free end-to-end head -- the head
# `yolo val` uses by default for YOLO26 (docs.ultralytics.com/models/yolo26; the NMS-head
# column is 40.9/48.6/53.1/55.0/57.5). Local check: yolo26n.pt -> 40.22.
PUBLISHED_COCO_MAP = {"yolo26n": 40.1, "yolo26s": 47.8, "yolo26m": 52.5, "yolo26l": 54.4, "yolo26x": 56.9}


def build_grid(ratios: list[float], criteria=CRITERIA) -> list[tuple[str, float]]:
    return [("none", 0.0)] + [(c, r) for c in criteria for r in ratios]


def load_prunable(weights: str) -> torch.nn.Module:
    return yolo_tda.make_prunable(YOLO(weights).model.float())


def calibration_images(data: str, n: int, imgsz: int, seed: int) -> torch.Tensor:
    """`n` letterboxed training images (train split, so val stays untouched)."""
    info = check_det_dataset(data)
    files: list[Path] = []
    for src in info["train"] if isinstance(info["train"], list) else [info["train"]]:
        src = Path(src)
        if src.suffix == ".txt":
            files += [src.parent / line.strip().lstrip("./") for line in src.read_text().splitlines() if line.strip()]
        else:
            files += sorted(f for f in src.rglob("*") if f.suffix[1:].lower() in IMG_FORMATS)
    files = random.Random(seed).sample(files, min(n, len(files)))
    letterbox = LetterBox((imgsz, imgsz), auto=False)
    batch = [letterbox(image=cv2.imread(str(f)))[:, :, ::-1].transpose(2, 0, 1) for f in files]
    return torch.from_numpy(np.ascontiguousarray(np.stack(batch))).float() / 255.0


def model_cost(net: torch.nn.Module, imgsz: int) -> dict[str, float]:
    net = copy.deepcopy(net).cpu().train()
    macs, params = yolo_tda.tp.utils.count_ops_and_params(net, torch.zeros(1, 3, imgsz, imgsz))
    return {"params": int(params), "gflops": 2 * macs / 1e9}


def evaluate(weights: str, net: torch.nn.Module, args, run_dir: Path, name: str) -> dict[str, float]:
    yolo = YOLO(weights)
    yolo.model = copy.deepcopy(net)  # val fuses conv+bn in place
    m = yolo.val(
        data=args.data, imgsz=args.imgsz, batch=args.batch, device=args.device,
        project=str(run_dir.resolve()), name=name, exist_ok=True, plots=False, verbose=False,
    )
    return {"map50_95": float(m.box.map), "map50": float(m.box.map50)}


def make_trainer(net: torch.nn.Module, seed: int = 42):
    class PrunedTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            return net  # keep the pruned graph; the default rebuilds from yaml

        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
            loader = super().get_dataloader(dataset_path, batch_size, rank, mode)
            if mode == "train" and seed != 42:
                # Ultralytics seeds the train loader's shuffle + augmentation stream with a fixed
                # constant, so --seed alone gives identical fine-tunes. Offset it by the run seed;
                # seed 42 keeps the original stream, so earlier runs stay reproducible.
                loader.generator.manual_seed(loader.generator.initial_seed() + seed - 42)
                loader.reset()
            return loader

    return PrunedTrainer


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------

def cmd_baseline(args) -> None:
    """Evaluate the untouched official weights and check them against the
    published COCO number, so every pruned run is measured from a known-good
    starting point. Fails (non-zero exit -> SLURM chain stops) on a mismatch."""
    out = args.output_dir / "baseline.json"
    if out.exists() and not args.force:
        print(out.read_text())
        return
    m = YOLO(args.weights).val(
        data=args.data, imgsz=args.imgsz, batch=args.batch, device=args.device,
        project=str((args.output_dir / "baseline").resolve()), name="val", exist_ok=True, plots=False,
    )
    result = {"weights": args.weights, "data": args.data, "imgsz": args.imgsz,
              "map50_95": round(100 * float(m.box.map), 2), "map50": round(100 * float(m.box.map50), 2)}
    published = PUBLISHED_COCO_MAP.get(Path(args.weights).stem)
    if Path(args.data).stem == "coco" and args.imgsz == 640 and published is not None:
        result["published_map50_95"] = published
        result["verified"] = abs(result["map50_95"] - published) <= args.baseline_tolerance
    else:
        result["verified"] = None  # no published reference for this model/dataset/size
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if result["verified"] is False:
        out.unlink()  # don't let a resumed chain skip past a failed check
        sys.exit(f"baseline {result['map50_95']} differs from published {published} by more than "
                 f"{args.baseline_tolerance}; check weights/dataset before pruning")


def cmd_score(args, field: bool = False) -> None:
    cmd_baseline(args)
    out = args.output_dir / ("field_scores.pt" if field else "tda_scores.pt")
    if out.exists() and not args.force:
        cached = torch.load(out)["config"].get("basis", "activation")
        if cached != args.basis:
            sys.exit(f"{out} was scored on the {cached} basis, not {args.basis}; use another --output-dir")
        print(f"{out} exists; pass --force to recompute")
        return
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    net = load_prunable(args.weights).to(device)
    images = calibration_images(args.data, args.calib_images, args.imgsz, args.seed)
    scores = yolo_tda.compute_persistence_scores(
        net, images, batch_size=args.batch_size_calib, max_side=args.max_side, workers=args.workers,
        field=field, basis=args.basis,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"scores": scores, "config": {k: str(v) for k, v in vars(args).items()}}, out)

    # layer-level view: average channel persistence per conv
    keys = yolo_tda.FIELD_CRITERIA if field else yolo_tda.TDA_CRITERIA
    with (args.output_dir / ("layer_field_drift.csv" if field else "layer_persistence.csv")).open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["layer", "channels", *(f"mean_{k}" for k in keys)])
        for name, s in scores.items():
            writer.writerow([name, len(s[keys[0]]), *(f"{s[k].mean().item():.6g}" for k in keys)])
    print(f"scored {len(scores)} conv layers on {len(images)} images -> {out}")


def cmd_prune(args) -> None:
    grid = build_grid(args.ratios, args.criteria)
    criterion, ratio = (args.criterion, args.ratio) if args.grid_index is None else grid[args.grid_index]
    tag = f"{criterion}_r{ratio:.2f}_s{args.seed}"
    run_dir = args.output_dir / "runs" / tag
    result_path = run_dir / "result.json"
    if result_path.exists() and not args.force:
        print(f"{tag} already done")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    net = load_prunable(args.weights)
    result = {
        "criterion": criterion, "ratio": ratio, "seed": args.seed, "epochs": args.epochs,
        "warmup_epochs": args.warmup_epochs, "optimizer": args.optimizer, "lr0": args.lr0, "momentum": args.momentum,
        "finetune_data": Path(args.finetune_data).name if args.finetune_data else None,
        "protect_layers": args.protect_layers,
    }
    result["dense"] = model_cost(net, args.imgsz)
    if criterion != "none":
        score_file = {**dict.fromkeys(yolo_tda.TDA_CRITERIA, "tda_scores.pt"),
                      **dict.fromkeys(yolo_tda.FIELD_CRITERIA, "field_scores.pt")}.get(criterion)
        scores = torch.load(args.output_dir / score_file)["scores"] if score_file else None
        importance = yolo_tda.build_importance(criterion, net, scores)
        yolo_tda.prune(net, importance, ratio, torch.zeros(1, 3, args.imgsz, args.imgsz), args.protect_layers)
    result["pruned"] = model_cost(net, args.imgsz)
    result["before_finetune"] = evaluate(args.weights, net, args, run_dir, "val_before_finetune")
    # all runs (incl. "none") get the same BN recalibration, and fine-tuning starts from it
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    images = calibration_images(args.data, args.calib_images, args.imgsz, args.seed + 1)
    yolo_tda.recalibrate_bn(net.to(device), images).cpu()
    result["after_bn_recal"] = evaluate(args.weights, net, args, run_dir, "val_after_bn_recal")
    print(json.dumps(result, indent=2))

    if args.epochs > 0:
        yolo = YOLO(args.weights)
        yolo.train(
            trainer=make_trainer(net, args.seed),
            data=args.finetune_data or args.data,  # e.g. a fixed COCO train subset; val stays the same
            epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            optimizer=args.optimizer,
            # auto picks its own lr/momentum by run length (AdamW 0.00125 on WSD, MuSGD 0.01 on
            # COCO); pin them to compare datasets on one recipe
            **{k: v for k, v in (("lr0", args.lr0), ("momentum", args.momentum)) if v is not None},
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            seed=args.seed,
            project=str(run_dir.resolve()),
            name="finetune",
            exist_ok=True,
            plots=False,
        )
        best = run_dir / "finetune" / "weights" / "best.pt"
        result["after_finetune"] = evaluate(args.weights, YOLO(best).model.float(), args, run_dir, "val_after_finetune")
    result_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def cmd_aggregate(args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for path in sorted((args.output_dir / "runs").glob("*/result.json")):
        r = json.loads(path.read_text())
        row = {k: r.get(k) for k in ("criterion", "ratio", "seed", "epochs", "protect_layers")}
        row["params"], row["gflops"] = r["pruned"]["params"], round(r["pruned"]["gflops"], 3)
        row["params_kept"] = round(r["pruned"]["params"] / r["dense"]["params"], 4)
        for stage in STAGES:
            for k, v in r.get(stage, {}).items():
                row[f"{k}_{stage}"] = round(v, 4)
        rows.append(row)
    if not rows:
        sys.exit(f"no results under {args.output_dir / 'runs'}")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (args.output_dir / "results.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # categorical slots 1-5 of the dataviz reference palette, in fixed order
    style = {
        "h0": ("#2a78d6", "o"), "h1": ("#eb6834", "s"), "total": ("#1baf7a", "D"),
        "l2": ("#eda100", "^"), "random": ("#e87ba4", "v"),
        # layer-field variants: same hue as their per-channel counterpart, dashed
        "field_h0": ("#2a78d6", "o"), "field_h1": ("#eb6834", "s"), "field_total": ("#1baf7a", "D"),
    }
    stages = [s for s in STAGES if f"map50_95_{s}" in fields]
    fig, axes = plt.subplots(1, len(stages), figsize=(5 * len(stages), 4.2), squeeze=False, sharey=True)
    for ax, stage in zip(axes[0], stages):
        key = f"map50_95_{stage}"
        base = [r[key] for r in rows if r["criterion"] == "none" and key in r]
        if base:
            ax.axhline(np.mean(base), color="#898781", lw=1, ls="--", label="unpruned")
        for crit, (color, marker) in style.items():
            by_ratio: dict[float, list[float]] = {}
            for r in rows:
                if r["criterion"] == crit and key in r:
                    by_ratio.setdefault(r["ratio"], []).append(r[key])
            if by_ratio:  # mean over seeds
                xs = sorted(by_ratio)
                ys = [np.mean(by_ratio[x]) for x in xs]
                ax.plot(xs, ys, color=color, marker=marker, ms=7, lw=2, label=crit,
                        ls="--" if crit.startswith("field_") else "-", mfc="white" if crit.startswith("field_") else color)
        ax.set_title(stage.replace("_", " "), loc="left", fontsize=11)
        ax.set_xlabel("per-layer pruning ratio")
        ax.grid(axis="y", color="#e5e4de", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel("mAP50-95")
    axes[0][-1].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(args.output_dir / "map_vs_ratio.png", dpi=160)
    print(f"{len(rows)} runs -> {args.output_dir / 'results.csv'}, map_vs_ratio.png")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["baseline", "score", "score-field", "prune", "aggregate", "grid"])
    p.add_argument("--weights", default="yolo26n.pt")
    p.add_argument("--data", default="coco.yaml")
    p.add_argument("--finetune-data", default=None,
                   help="dataset yaml for the fine-tune only (default --data); scoring, BN re-estimation "
                        "and evaluation always use --data")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "yolo26_tda")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=8, help="dataloader / persistence worker processes")
    p.add_argument("--force", action="store_true")
    p.add_argument("--baseline-tolerance", type=float, default=1.0, help="max |mAP50-95 - published|, in points")
    # score
    p.add_argument("--calib-images", type=int, default=128, help="images for TDA scoring and BN recalibration")
    p.add_argument("--batch-size-calib", type=int, default=8)
    p.add_argument("--max-side", type=int, default=64, help="activation maps are avg-pooled to at most this side")
    p.add_argument("--basis", choices=["activation", "peek"], default="peek",
                   help="topology of raw activations, or of PEEK maps (computed full-res, then pooled)")
    # prune
    p.add_argument("--ratios", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5])
    p.add_argument("--grid-index", type=int, help="index into the (criterion, ratio) grid; = SLURM_ARRAY_TASK_ID")
    p.add_argument("--criterion", choices=["none", *ALL_CRITERIA], default="h0")
    p.add_argument("--criteria", nargs="+", choices=ALL_CRITERIA, default=list(CRITERIA),
                   help="criteria in the --grid-index grid (default keeps the original grid's indices)")
    p.add_argument("--ratio", type=float, default=0.3)
    p.add_argument(
        "--protect-layers", type=int, default=yolo_tda.PROTECT_EARLY,
        help="never prune top-level layers [0, N) (early edge/shape layers); 0 = prune all",
    )
    p.add_argument("--epochs", type=int, default=10, help="fine-tune epochs after pruning (0 = none)")
    p.add_argument("--warmup-epochs", type=float, default=3.0, help="Ultralytics LR warmup (default 3)")
    p.add_argument("--optimizer", default="auto", help="Ultralytics optimizer for the fine-tune (auto, AdamW, SGD, ...)")
    p.add_argument("--lr0", type=float, default=None, help="fine-tune lr; needed when --optimizer is not auto")
    p.add_argument("--momentum", type=float, default=None, help="fine-tune momentum / AdamW beta1")
    p.add_argument("--batch", type=int, default=32)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.stage == "grid":  # print grid size (for the SLURM array range) and contents
        grid = build_grid(args.ratios, args.criteria)
        for i, (c, r) in enumerate(grid):
            print(i, c, r, file=sys.stderr)
        print(len(grid))
    else:
        {"baseline": cmd_baseline, "score": cmd_score, "score-field": lambda a: cmd_score(a, field=True),
         "prune": cmd_prune, "aggregate": cmd_aggregate}[args.stage](args)
