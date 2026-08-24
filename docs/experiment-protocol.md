# Frozen experiment details

## Preprocessing and augmentation equivalence

Both detector families consume the same WSD YAML, unchanged split files, 640-pixel maximum input dimension, batch size 16, and seed 42. Both use aspect-ratio-preserving letterboxing, RGB inputs scaled to `[0,1]`, horizontal flip probability 0.5, HSV gains `(0.015, 0.7, 0.4)`, zero rotation/shear/perspective/vertical flip, and the declared translate, scale, mosaic, and mixup values in each trial.

Implementation equivalence is intentionally not claimed. YOLOv5 uses its pinned dataloader, anchor-based head, SGD default, three warm-up epochs, and YOLOv5 loss scaling. YOLO26 uses the pinned Ultralytics dataloader, anchor-free end-to-end head, its own loss/assignment behavior, closes mosaic for the final ten epochs, and may choose an optimizer only when `optimizer: auto` is declared. Native rectangular padding, interpolation, batch augmentation ordering, and loss hyperparameters therefore differ. Every effective argument is retained in `opt.yaml` or `args.yaml`; cross-family comparisons are matched protocols, not identical training algorithms.

## PEEK calibration

The calibration population is 128 WSD training images selected before pruning experiments. Selection round-robins over the frozen robustness slice and rarest-present-class strata, with SHA-256 `(seed:image)` ordering within strata. It does not condition on correct detections. Empty scenes remain eligible through a background stratum. The exact list and image hashes are in `wsd-v71-peek-calibration.json`.

PEEK maps are computed per image and module—not per detection. Batches must be separated along the batch dimension before calling official PEEK. Captures use float32. Modules `[4, 6, 10, 16, 19, 22]` were frozen before pruning outcomes: three backbone outputs and the corresponding three neck outputs at strides approximately 8, 16, and 32. Hook semantics and observed shapes are recorded in `yolo26n-peek-module-inventory.json`.

## Filtration and aggregation

PEEK maps are min-max normalized per image/module. Constant maps become zero maps. Superlevel filtration is frozen because larger PEEK values are the intended high-information regions; the cubical lower-star implementation receives `1-normalized_map`, causing high-PEEK cells to enter first. Dimensions 0 and 1 retain complete finite persistence intervals plus essential births. Candidate/reference drift reports bottleneck and 2-Wasserstein distances for each dimension.

Image-level values are the independent aggregation unit. For each module, report median and interquartile range plus the mean and a 95% percentile bootstrap confidence interval using 10,000 image resamples and seed 42. Aggregate across modules only after reporting module-level values; use an equal-weight mean across the six predeclared modules. Do not treat modules or persistence features as independent replicates.

## Numerical behavior

Official PEEK converts inputs to float32. The adapter accepts one CHW/HWC activation or a singleton BCHW activation, rejects multi-item batches and non-3D inputs, and the topology layer rejects non-finite or empty maps. Tests cover float16 constant activations, singleton-batch behavior, constant maps, non-finite maps, and zero self-distance for diagrams.

## Detector uncertainty

Hyperparameters are selected once using seed 42 validation results. After selection is frozen, the selected conditions are repeated with deterministic seeds 42, 43, 44, 45, and 46. Each resulting checkpoint is evaluated once on the unchanged test protocol. Overall and per-class precision, recall, AP@0.5, and AP@0.5:0.95 are summarized across independent training seeds using the arithmetic mean, sample standard deviation, and two-sided 95% Student-t confidence interval.

One seed is always labeled a point estimate and receives no standard deviation or interval. At least three completed seeds are required for an uncertainty row to be marked reportable; all five are preferred for manuscript tables. This across-seed interval measures training variability, not uncertainty from the finite 40-image test sample, and that limitation must accompany the results.
