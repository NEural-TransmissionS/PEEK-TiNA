# Seed confirmation plan for the tuned YOLO26n condition

This is the concrete plan for the multi-seed confirmation step required by the frozen protocol (`experiment-protocol.md`, "Detector uncertainty"): repeat the selected condition at deterministic seeds 42-46, evaluate each checkpoint once on the untouched test split, and summarize precision/recall/mAP@0.5/mAP@0.5:0.95 across seeds with the arithmetic mean, sample standard deviation, and a two-sided 95% Student-t interval. At least three completed seeds are required for a reportable row; five are preferred.

## Selected condition

Pretrained YOLO26n, seed 42, `cls=0.75, dfl=1.75` (box left at the native 7.5) — the best validation result across every completed search, from `runs/tuning_yolo26_seed42_followup/yolo26n_pretrained_cls075_dfl175_seed42` (val mAP50-95 0.4387 at epoch 627 of 727; see [`runs-summary.md`](runs-summary.md)). Optimizer stays `auto`, augmentation stays native — only the classification and DFL loss weights change from candidate zero.

## Tooling gap to close first

The existing confirmation harness (`scripts/run_confirmation_queue.py`, `configs/confirmation_yolo26_cls075.yaml`) predates the seed42-followup grid. It only knows two trials, `native` and `cls075` (cls=0.75, dfl left at native 1.5) — neither is the actual current winner. Before launching seeds 43-46, `configs/confirmation_yolo26_cls075.yaml` needs a third trial added:

```yaml
  - name: cls075_dfl175
    yolov5: {}
    yolo26:
      optimizer: auto
      cls: 0.75
      dfl: 1.75
```

and `scripts/run_confirmation_queue.py`'s `--trial` choices need `cls075_dfl175` added alongside `native`/`cls075`. Without this, `run_confirmation_queue.py` cannot target the winning condition. Alternatively, `scripts/tune_baselines.py` can be invoked directly against `configs/tuning_yolo26_seed42_followup.yaml` (which already defines the `cls075_dfl175` trial) with a `--seed` override, without touching the confirmation config — see launch commands below. Either path is acceptable; the confirmation-config route keeps output under the `runs/confirmation_yolo26_cls075` naming this project already uses for paired native/tuned comparisons.

Separately, `configs/confirmation_seeds.yaml`'s `selection: configs/tuning_nano_selection.json` pointer is stale — that file does not exist. It dates from before the error-driven pivot documented in the README. It should be repointed at the seed42-followup result (or removed in favor of the hardcoded condition above) before this config is treated as authoritative.

## Seeds

| Seed | Status | Source |
|---|---|---|
| 42 | Complete | `runs/tuning_yolo26_seed42_followup/yolo26n_pretrained_cls075_dfl175_seed42` — reuse, do not rerun |
| 43 | Not started | Prior attempt (`native` and `cls075` trials, not `cls075_dfl175`) was cancelled after 14 epochs — see `runs/interrupted_tuning/yolo26n_pretrained_{native,cls075}_seed43_cancelled_seed42_only`, logs in `runs/logs/yolo26-confirm-{native,cls075}.log`. Cause of cancellation was never confirmed; check the log tails before relaunch rather than assuming a clean restart is safe. |
| 44 | Not started | — |
| 45 | Not started | — |
| 46 | Not started | — |

## Protocol (unchanged from the frozen search)

WSD v71, `data/wsd.yaml`, 640x640, batch 16, workers 8, epoch ceiling 1000, patience 100, deterministic kernels on. Only the seed varies across runs — no re-tuning.

## Launch

Using the confirmation harness, once the trial above is added:

```bash
source .venv/bin/activate
python scripts/run_confirmation_queue.py \
  --trial cls075_dfl175 --seeds 43 44 45 46 \
  --device 0 --data data/wsd.yaml
```

Or directly against the followup config, without modifying the confirmation config:

```bash
source .venv/bin/activate
for seed in 43 44 45 46; do
  python scripts/tune_baselines.py \
    --config configs/tuning_yolo26_seed42_followup.yaml \
    --family yolo26 --initialization pretrained \
    --trial cls075_dfl175 --seed "$seed" --device 0
done
```

Run in `tmux`, not `nohup`, so a dropped session doesn't kill an in-progress seed. `runs/invalid_shared_gpu` shows this project has already lost runs to GPU contention — use a device not shared with another job, and run the four seeds sequentially on one GPU rather than packing them concurrently, unless a second dedicated GPU is confirmed free.

Budget per seed: the seed-42 run took 727 epochs to exhaust patience past its best checkpoint at epoch 627. Expect similar order-of-magnitude wall time for each of the remaining four.

## After training: test evaluation and summary

Per the frozen protocol, each of the five seed checkpoints gets exactly one evaluation on the held-out test split (`configs/evaluation.yaml`: conf 0.001, NMS IoU 0.60, max-det 300, 640px), analogous to `scripts/evaluate_baselines.py` but pointed at each `yolo26n_pretrained_cls075_dfl175_seed{42..46}/weights/best.pt`. Test access is otherwise prohibited during selection — that prohibition ends only once all five seeds have finished training and the condition is frozen.

Then:

```bash
python scripts/collect_seed_uncertainty.py \
  --config configs/confirmation_seeds.yaml \
  --runs runs/<test-eval-output-dir> \
  --output docs/final-seed-uncertainty.csv
```

`collect_seed_uncertainty.py` expects one `metrics.json` per seed under directories named `..._seed<N>`, and writes mean/standard-deviation/95%-CI/reportable rows per (condition, class, metric) to `docs/final-seed-uncertainty.csv` — currently header-only, since no confirmation run has completed test evaluation. `--runs` must point at wherever the five test evaluations actually land; nothing currently defaults there for the `cls075_dfl175` condition, since `confirmation_seeds.yaml`'s `train_output`/`test_output` (`runs/final_nano`/`runs/final_nano_test`) were never used by any completed run either.

## Deliverable

A `docs/final-seed-uncertainty.csv` row set for `cls075_dfl175`, overall and per-class, each with `count >= 3` (reportable) and ideally `count = 5`. This is the number that should replace the single-seed 0.4387 val figure in any manuscript-facing table, with the explicit caveat (already noted in `experiment-protocol.md`) that it measures training variability across seeds, not sampling uncertainty from the 40-image test set.
