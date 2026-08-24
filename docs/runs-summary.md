# Training and tuning runs summary

This is a status snapshot of every entry under `runs/` as of 2026-08-24. It exists to orient a reader across the full run history without opening sixteen directories individually. Numbers are read directly from each run's `results.csv`/`metrics.json`; "best" always means the epoch with the highest validation mAP@0.5:0.95, matching the selection rule in the tuning configs. None of the tuning-search numbers below are test-split results — test access is prohibited during selection per the frozen protocol.

## Frozen nano baseline (`runs/baseline_nano`, `runs/baseline_nano_test`)

Complete, official. See [`baseline-nano-results.csv`](baseline-nano-results.csv) for the canonical table. All four runs (YOLOv5n/YOLO26n × pretrained/scratch) ran to early stopping.

| Model | Init | Stop epoch | Best epoch | Val mAP50 | Val mAP50-95 | Test mAP50-95 |
|---|---|---:|---:|---:|---:|---:|
| YOLO26n | Pretrained | 214 | 114 | 0.564 | 0.422 | 0.312 |
| YOLO26n | Scratch | 273 | 173 | 0.482 | 0.348 | 0.245 |
| YOLOv5n | Pretrained | 434 | 432 | 0.556 | 0.389 | 0.308 |
| YOLOv5n | Scratch | 600 | 500 | 0.514 | 0.334 | 0.262 |

Pretrained YOLO26n is the leading condition and anchors every later search. `baseline_nano_test` holds the corresponding one-shot test-split `metrics.json` for each of the four checkpoints; it contains no training curves.

## Infrastructure checks (`runs/smoke`, `runs/smoke_device_isolation`, `runs/invalid_shared_gpu`)

Not accuracy results. `smoke` and `smoke_device_isolation` are one-epoch runs that validate the harness and multi-GPU device isolation; near-zero mAP is expected and uninformative. `invalid_shared_gpu` holds a pretrained YOLO26n and YOLOv5n run each killed early (64 and 130 epochs) by GPU contention from a concurrent job; both are excluded from selection.

## Validation-only hyperparameter search (`runs/tuning_nano`)

Ten runs: three candidates (low-LR + light geometry, AdamW + light geometry, SGD strong regularization) crossed with pretrained/scratch, for both model families.

| Family | Init | Candidate | Best val mAP50-95 | Best epoch |
|---|---|---|---:|---:|
| YOLO26n | Pretrained | AdamW, light geometry | 0.4202 | 335 |
| YOLO26n | Pretrained | SGD strong regularization | 0.4122 | 196 |
| YOLO26n | Pretrained | Low-LR, light geometry | 0.4045 | 98 |
| YOLO26n | Scratch | Low-LR, light geometry | 0.3490 | 538 |
| YOLOv5n | Pretrained | AdamW, light geometry | 0.3778 | 464 |
| YOLOv5n | Pretrained | SGD strong regularization | 0.3681 | 77 |
| YOLOv5n | Pretrained | Low-LR, light geometry | 0.3668 | 193 |
| YOLOv5n | Scratch | SGD strong regularization | 0.3347 | 428 |
| YOLOv5n | Scratch | Low-LR, light geometry | 0.3135 | 888 |
| YOLOv5n | Scratch | AdamW, light geometry | 0.3115 | 519 |

YOLOv5n does not exceed 0.378 val mAP50-95 in any configuration tested, which is why every subsequent search is restricted to pretrained YOLO26n.

## Error-driven search (`runs/tuning_yolo26_error_driven`, `runs/tuning_yolo26_error_driven_node2`)

Four hypotheses drawn from [`yolo26-validation-error-analysis.md`](yolo26-validation-error-analysis.md), which identified low-confidence classification and sub-1%-area objects as the dominant failure modes. Some original attempts stalled and were rerun on a second host (`node2`, `attempt2`); the stalled originals remain as `args.yaml`/`tuning-manifest.json` stubs with no training data.

| Trial | Best val mAP50-95 | Best epoch |
|---|---:|---:|
| Classification, mild (node2, attempt2) | 0.4348 | 457 |
| Multiscale + classification (node2, attempt2) | 0.4205 | 409 |
| Multiscale + localization (node2, attempt2) | 0.4186 | 159 |
| Localization emphasis (node2, attempt2) | 0.4161 | 254 |
| Classification emphasis (original run) | 0.4151 | 227 |
| Multiscale, small objects (original run) | 0.4131 | 203 |

## Architecture search: YOLO26n-P2 (`runs/tuning_yolo26_p2_error_driven`)

Complete, four runs. Tests the upstream P2/4 detection head against the same error-driven hypotheses, at higher FLOPs.

| Trial | Best val mAP50-95 | Best epoch |
|---|---:|---:|
| P2, classification mild | 0.4235 | 499 |
| P2, native | 0.4227 | 294 |
| P2, moderate multiscale | 0.4161 | 301 |
| P2, reduced scale jitter | 0.4137 | 286 |

Roughly matches the plain-architecture classification-emphasis result at increased compute cost; must be reported as an architecture/accuracy tradeoff, not a free win.

## Seed-42 followup grid (`runs/tuning_yolo26_seed42_followup`)

Twenty runs, pretrained YOLO26n only, seed 42. Fine grid over classification/box/DFL loss weights plus individual augmentation knobs (mosaic, cutmix, HSV, translate, scale, multiscale, close-mosaic).

| Trial | Best val mAP50-95 | Best epoch |
|---|---:|---:|
| cls 0.75, dfl 1.75 | **0.4387** | 627 |
| cls 0.80, box 8.5 | 0.4342 | 380 |
| cls 0.875, box 8.5 | 0.4329 | 289 |
| cls 0.75, scale 0.25 | 0.4307 | 369 |
| cls 0.80 | 0.4302 | 204 |
| cls 0.75, translate 0.05 | 0.4271 | 436 |
| cls 0.70 | 0.4270 | 235 |
| cls 0.625, box 8.5 | 0.4268 | 325 |
| cls 0.875 | 0.4266 | 191 |
| cls 0.75, mosaic 0.75 | 0.4237 | 124 |
| cls 0.625 | 0.4225 | 216 |
| cls 0.75, HSV mild | 0.4229 | 154 |
| cls 0.70, box 8.5 | 0.4196 | 298 |
| cls 0.75, close-mosaic 50 | 0.4200 | 146 |
| cls 0.75, box 8.5 | 0.4214 | 237 |
| cls 0.75, cutmix 0.1 | 0.4178 | 192 |
| cls 0.75, multiscale 0.25 | 0.4141 | 277 |
| cls 0.55 | 0.4107 | 115 |
| cls 0.75, mosaic 0.5 | 0.4073 | 111 |
| cls 0.75, box 9, dfl 1.75 | 0.4013 | 53 |

This is the best result in the run history: `cls=0.75, dfl=1.75` (box left at native 7.5) reaches 0.4387 val mAP50-95, about 4% relative above the 0.4218 native-pretrained baseline. Raising `cls` alone from 0.55 to 0.875 is consistently positive; individual augmentation changes gave smaller, mixed effects on top of `cls=0.75`.

## Interrupted, cancelled, and superseded attempts (`runs/interrupted_tuning`)

Eleven directories kept for provenance; none feed into selection.

- Two P2 stubs (`p2_native`, `p2_reduced_scale_jitter`) killed at zero epochs by a bad remote dataset path.
- Two seed-43 confirmation runs (`native`, `cls075`) cancelled after 14 epochs. These are the only trace of the multi-seed confirmation protocol (`configs/confirmation_seeds.yaml`, seeds 42-46) being attempted; [`final-seed-uncertainty.csv`](final-seed-uncertainty.csv) is still header-only, so no multi-seed uncertainty interval has been produced.
- Four dated (`_20260803`) early attempts at localization/multiscale-classification emphasis, superseded by the node2/attempt2 reruns in `tuning_yolo26_error_driven_node2`.
- Two scratch-init side experiments, `adamw_light_geometry` (0.3461 best) and `sgd_strong_regularization_error_analysis_pivot` (0.2751 best), that were not carried forward.
- One cls075 seed-43 run cancelled at 14 epochs (paired with the native cancellation above).

## Error analysis, not training (`runs/validation_error_analysis`, `runs/invalid_validation_error_analysis`)

Per-image true/false positive/negative breakdowns for the leading pretrained checkpoints (`error-analysis.json`, `per-image-errors.csv`), not accuracy curves. These are the source data behind [`yolo26-validation-error-analysis.md`](yolo26-validation-error-analysis.md): pretrained YOLO26n totals 615 TP / 275 FP / 394 FN at confidence 0.25, with thruster recall near 5%. The `invalid_` copy was redone after a relative-path bug in the original run.

## Open items

- Multi-seed confirmation (seeds 42-46) for the leading tuned condition has not been completed; only the two cancelled seed-43 attempts above exist.
- No frozen test-split evaluation exists yet for any tuned condition — only the four original baselines have test numbers.
- Pruning itself has not started; every run above is pre-pruning baseline/tuning work.
