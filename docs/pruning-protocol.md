# YOLO26n topology-guided pruning protocol

This protocol is frozen before pruning outcomes and remains exploratory until confirmation seeds and the one-time test evaluation are complete. The selected YOLO26n pretrained checkpoint is intentionally unresolved; it will be supplied after validation-only selection.

## Structural unit and dependency safety

The pruning unit is a dependency group rooted at convolution output channels. Removing a channel must physically remove its parameters and every coupled normalization entry, propagate the same indices into downstream convolution inputs, and preserve the tensor contracts of residual additions and concatenations. A zero mask is not a structurally pruned candidate.

Prediction outputs in `Detect`, unmatched residual branches, standalone normalization channels, and attention embeddings without complete dependency propagation are forbidden roots. Concatenation branches may be pruned only when downstream offsets are updated. Residual branches must retain equal channel dimensions. Every generated candidate must pass a raw forward at 640 px and report physical parameter and FLOP reductions before training.

The checkpoint-dependent graph surgery is not implemented speculatively. It must be integrated and tested against the exact selected checkpoint because YOLO module dependencies and fused/unfused state are part of the serialized model.

## Matched-compute ablations

At each measured FLOP-reduction target of 10%, 20%, 30%, 40%, and 50%, generate four candidates using identical eligible dependency groups:

1. `magnitude`: smallest group-normalized weight magnitude.
2. `random`: seed-42 random ordering, retained as a negative control.
3. `peek_variance`: smallest retained calibration-set PEEK variance contribution.
4. `topology`: smallest increase in calibration-set topology drift.

Actual raw-forward FLOP reduction must be within one absolute percentage point of its target. Methods are compared only inside the same measured-compute budget.

## PEEK and topology execution

`scripts/capture_peek_maps.py` validates every calibration image SHA-256, reads only the frozen train calibration set, hooks modules `[4, 6, 10, 16, 19, 22]`, runs official `peek.PEEK`, and writes one float32 map per image/module. It refuses partial overwrites by default.

`scripts/compare_peek_topology.py` requires matched map trees with this layout:

```text
<map-root>/<image-stem>/module_4.npy
<map-root>/<image-stem>/module_6.npy
...
```

Each map receives the frozen min-max normalization and superlevel cubical filtration. Complete finite persistence diagrams in H0 and H1 are compared by bottleneck and 2-Wasserstein distances. Coverage must include all 128 images, six modules, and two dimensions. Images are the independent units: module/dimension summaries report median, IQR, mean, and a seed-42 10,000-resample percentile-bootstrap interval. Overall drift first averages modules and dimensions within each image, then aggregates across images with equal module weight.

## Acceptance and scoring

A candidate is rejected if it lacks physical structure removal, misses its compute target, loses more than 0.02 absolute validation mAP@0.5:0.95, exceeds mean bottleneck 0.10 or mean 2-Wasserstein 0.15, or retains less than 80% of reference PEEK variance. These are protocol gates, not claims that the thresholds are universally optimal.

Passing candidates receive a lower-is-better diagnostic score with weights 0.40 validation accuracy, 0.20 bottleneck drift, 0.15 Wasserstein drift, 0.15 PEEK-variance loss, and 0.10 compute mismatch. Each term is normalized by its frozen gate. `scripts/collect_pruning_results.py` validates candidate JSON records and writes record-driven CSV and JSON selections. Selection reads validation only; the test split remains prohibited until the final condition is frozen.

## Fine-tuning and reporting

Fine-tuning retains the YOLO26 pretrained protocol: seed 42, deterministic mode, 640 px, batch 16, eight workers, native optimizer behavior, at most 1,000 epochs, and patience 100. Candidates are selected by validation mAP@0.5:0.95 subject to the gates above. Selected final conditions must later run across seeds 42–46 before uncertainty reporting and receive only the separately frozen one-time test evaluation.
