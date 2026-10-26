"""Evaluate a fine-tune every N training batches (recovery curves finer than one epoch).

Shared by the channel-pruning fine-tune (scripts/run_yolo26_tda_pruning.py) and the block cut +
fine-tune (block_pruning/_train_one.py). attach() adds Ultralytics callbacks to a YOLO object before
.train(): the EMA model the trainer would validate at epoch end is scored on the trainer's val set
(the fine-tune yaml's `val`, e.g. a small fixed slice) at batch 0 and every `every` batches, one CSV
row per evaluation. With max_batches, training stops there by raising StopProbe, which the caller
catches: the schedule up to that point is the full run's, so the curve is its exact prefix.
"""

import csv
import time
from pathlib import Path


class StopProbe(Exception):
    """Raised from the batch callback once max_batches is reached."""


def attach(yolo, out_csv: Path, every: int, max_batches: int | None = None) -> None:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    state = {"batch": 0, "t0": None, "eval_s": 0.0, "rows": []}

    def evaluate(trainer):
        t = time.time()
        if not hasattr(trainer, "loss_items"):  # batch 0: the validator sizes its loss buffer from this
            import torch
            trainer.loss_items = torch.zeros(len(trainer.loss_names), device=trainer.device)
        m = trainer.validator(trainer)  # not trainer.validate(): that would move best_fitness / best.pt
        trainer.model.train()
        row = {
            "batch": state["batch"], "images": state["batch"] * trainer.batch_size, "epoch": trainer.epoch,
            "lr": round(trainer.optimizer.param_groups[0]["lr"], 8),
            "map50_95": round(float(m["metrics/mAP50-95(B)"]), 5), "map50": round(float(m["metrics/mAP50(B)"]), 5),
            "train_s": round(t - state["t0"] - state["eval_s"], 1),  # wall time training, excl. these evals
        }
        state["eval_s"] += time.time() - t
        state["rows"].append(row)
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row))
            w.writeheader()
            w.writerows(state["rows"])

    def on_batch_start(trainer):
        if state["t0"] is None:
            state["t0"] = time.time()
            evaluate(trainer)  # batch 0: the model the fine-tune starts from

    def on_batch_end(trainer):
        state["batch"] += 1
        if state["batch"] % every == 0:
            evaluate(trainer)
        if max_batches and state["batch"] >= max_batches:
            raise StopProbe(f"stopped after {state['batch']} batches")

    yolo.add_callback("on_train_batch_start", on_batch_start)
    yolo.add_callback("on_train_batch_end", on_batch_end)
