# Topology-Guided PEEK Pruning

Research scaffold for testing whether structured pruning of YOLOv5 and YOLO26 can preserve the topology of internal PEEK maps while reducing the compute and memory cost of spacecraft-component detection.

This repository implements only the parts of the draft that are currently well specified: reproducible dependency pins, WSD dataset validation, canonical PEEK integration, and persistent-homology summaries. It does **not** present the draft's planned experiments as completed work or silently choose an unvalidated pruning rule.

## Repository decisions

- **Detectors:** YOLOv5 and YOLO26 from pinned upstream submodules. Both are first-class WSD baselines because both are promised in the draft.
- **Explanation method:** the official [PEEK repository](https://github.com/Kenzie-Meni/PEEK), also pinned as a submodule. PEEK's equation and latent extraction are not reimplemented here.
- **Primary dataset:** the authoritative WSD v71 release, internally exported as Roboflow `satellite_components` version 71 (`antenna`, `body`, `solar`, `thruster`). WSD consists of distinct still images, not extracted video frames. It is private project data, available only to authorized collaborators, and no images or labels are committed to Git.
- **Pretraining data:** COCO may be inherited through pretrained detector weights, but it is not currently an evaluation dataset or a committed experimental branch.
- **Topology:** lower-star cubical filtrations of normalized two-dimensional PEEK maps, summarized in homology dimensions 0 and 1 with GUDHI.
- **Pruning:** intentionally pending protocol selection. The config records proposed sparsity levels but no result-producing command claims that the selection criterion has been settled.

## Setup

Python 3.10+ is required.

```bash
git clone --recurse-submodules <this-repository>
cd topology-guided-pruning
python -m venv .venv
source .venv/bin/activate
make setup
make test
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

The pins are:

| Dependency | Location | Commit |
|---|---|---|
| Ultralytics YOLOv5 | `third_party/yolov5` | `84ef1e59e3ef0e35ae8e4fe8aef9e0ee16c852c1` |
| Ultralytics / YOLO26 | `third_party/ultralytics` | `40eb41ac3032958ef54bece0d8b262e196af3259` |
| Official PEEK | `third_party/PEEK` | `49d8531fdade2ebed425745d7928a2049f02c88e` |

## WSD dataset mapping

Set `WSD_ROOT` to a WSD export containing `train/images`, `train/labels`, `valid/images`, and so on, then audit it and generate the machine-local Ultralytics YAML:

```bash
export WSD_ROOT=/path/to/satellite_components-71
make audit-wsd
```

The generated `data/wsd.yaml` is ignored by Git so absolute workstation paths cannot leak into the repository. The local Blake export found during repository construction contained:

| Split | Images | Labels |
|---|---:|---:|
| Train | 1,004 | 1,004 |
| Validation | 186 | 186 |
| Test | 40 | 40 |

WSD v71 is the authoritative dataset for this study. It is associated with:

> Trupti Mahendrakar, Ryan T. White, Madhur Tiwari, and Markus Wilde, “Unknown Non-Cooperative Spacecraft Characterization with Lightweight Convolutional Neural Networks,” *Journal of Aerospace Information Systems*, vol. 21, no. 5, pp. 455–460, May 2024. [doi:10.2514/1.I011343](https://doi.org/10.2514/1.I011343).

The completed [duplicate review](docs/wsd-v71-duplicate-review.json) found **zero byte-identical image groups** among all 1,230 images using SHA-256 content hashes. A 64-bit difference-hash screen at Hamming distance ≤2 produced 16 cross-split candidates; none passed the frozen confirmation threshold of 0.98 normalized grayscale correlation at 128×128. The conclusion is zero exact or confirmed near-duplicate cross-split pairs. The screen and all candidate scores are retained in the JSON artifact. WSD contains one spacecraft per distinct still image, and spacecraft renders do not cross split boundaries.

The source paper above is the authoritative reference for WSD acquisition and synthesis. The machine-readable [quality audit](docs/wsd-v71-quality-audit.json) additionally confirms that all 1,230 images decode; all label rows have five finite fields, valid class IDs, positive box dimensions, and normalized boxes within image bounds; and no image/label pairing errors were found. Seven train images are valid backgrounds, while validation and test contain none. Across the complete dataset there are 2,507 antenna, 1,028 body, 2,481 solar, and 1,954 thruster instances (largest/smallest ratio 2.44). The test set's 25 thruster instances make that class's reported AP particularly sample-sensitive.

The accompanying [robustness metalabels](docs/wsd-v71-robustness-slices.csv) assign every image to one of three deterministic seed-42 clusters using standardized brightness, contrast, saturation, annotation count, normalized annotation area, and object extent. Box count is `log1p` transformed during clustering to prevent unusually dense images from forming a tiny outlier group. The resulting global slices are bright/large/many-part (389 images), bright/large/few-part (149), and dark/small/many-part (692). These are descriptive metadata slices—not spacecraft-identity labels—and must remain frozen before model comparisons. Per-split counts and centroids are recorded in the audit JSON.

Recreate the review with:

```bash
export WSD_ROOT=/path/to/satellite_components-71
make review-wsd
make audit-wsd-quality
```

## Frozen nano baseline experiment

The initial baseline matrix compares YOLOv5n and YOLO26n under both standard pretrained initialization and training from scratch. The frozen protocol is in [`configs/baseline_nano.yaml`](configs/baseline_nano.yaml):

| Setting | Value |
|---|---|
| Dataset | WSD v71, unchanged train/validation/test splits |
| Seed | 42 |
| Deterministic kernels | Enabled |
| Input size | 640×640 |
| Batch size | 16 |
| Workers | 8 |
| Epoch ceiling | 1,000 |
| Early-stopping patience | 100 validation epochs |
| Initializations | Standard pretrained weights and random initialization |
| Evaluation during training | WSD validation split |
| Final reporting | Best checkpoint on WSD validation, then one WSD test evaluation |
| Test confidence threshold | 0.001 |
| Test NMS IoU threshold | 0.60 |
| Test maximum detections | 300 |

Pretrained runs may inherit COCO knowledge through their upstream weights; scratch runs do not. Native optimizer, loss, augmentation, and architecture behavior are retained for each family and captured in the run artifacts, since forcing one implementation's defaults onto the other would not reproduce either standard baseline.

Run the full matrix with two GPUs:

```bash
python scripts/train_baselines.py --family yolov5 --initialization all --device 0
python scripts/train_baselines.py --family yolo26 --initialization all --device 1
```

The commands are intended to run concurrently. Each family runs pretrained then scratch sequentially on its assigned GPU. Run names encode architecture, initialization, and seed. A manifest records the effective protocol, Python/PyTorch/CUDA versions, GPUs, and repository/submodule revisions before every run.

After all four runs finish, evaluate each best checkpoint once on the held-out test split and collect a chart-ready table:

```bash
python scripts/evaluate_baselines.py --family all --device 0
python scripts/collect_baseline_results.py
```

### Seed-42 baseline results

All four runs completed through early stopping. The machine-readable table is [`docs/baseline-nano-results.csv`](docs/baseline-nano-results.csv).

| Model | Initialization | Stop epoch | Best val epoch | Val mAP@0.5 | Val mAP@0.5:0.95 | Test mAP@0.5 | Test mAP@0.5:0.95 | GFLOPs | Weights (MB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOv5n | Pretrained | 434 | 432 | 0.556 | 0.389 | 0.455 | 0.308 | 4.2 | 3.90 |
| YOLOv5n | Scratch | 600 | 500 | 0.514 | 0.334 | 0.426 | 0.262 | 4.2 | 3.90 |
| YOLO26n | Pretrained | 214 | 114 | 0.564 | 0.422 | 0.445 | 0.312 | 5.8 | 5.41 |
| YOLO26n | Scratch | 273 | 173 | 0.482 | 0.348 | 0.382 | 0.245 | 5.8 | 5.43 |

These values do **not** recreate the draft table’s earlier YOLOv5n value of 0.808/0.570. That row must not be silently replaced or described as reproduced. Relevant evidence:

- The new results use the authoritative v71 split fingerprints and a separately held-out 40-image test set with 283 component instances.
- The test set is substantially harder than validation, especially for thrusters. The pretrained test AP@0.5:0.95 values for thruster were approximately 0.009 (YOLOv5n) and 0.003 (YOLO26n).
- Blake’s archived v71 YOLOv5s experiments using `satellite_detection_weights.pt` peak around 0.585/0.423 on validation, much closer to the new validation range than to 0.808/0.570.
- The draft row may come from another dataset version/split, an older specialized checkpoint, a different evaluation set, or copied prior-study results. Its originating checkpoint and evaluation command still need to be identified.

Pretraining improved held-out mAP@0.5:0.95 by about 0.045 for YOLOv5n and 0.067 for YOLO26n in this seed-42 run. Because this is one fixed seed, these are baseline measurements, not uncertainty estimates.

Structured per-class test metrics for every completed model are collected in [`docs/per-class-test-results.csv`](docs/per-class-test-results.csv). The evaluator records precision, recall, AP@0.5, and AP@0.5:0.95 for each class; rerun the collector after tuned winners are evaluated to extend the table. Current AP@0.5:0.95 values are:

| Model | Init. | Antenna | Body | Solar | Thruster |
|---|---|---:|---:|---:|---:|
| YOLOv5n | Pretrained | 0.294 | 0.425 | 0.502 | 0.009 |
| YOLOv5n | Scratch | 0.220 | 0.351 | 0.476 | 0.002 |
| YOLO26n | Pretrained | 0.287 | 0.426 | 0.531 | 0.003 |
| YOLO26n | Scratch | 0.207 | 0.299 | 0.453 | 0.020 |

These are single-seed point estimates on only 40 test images. In particular, the 25 thruster instances are insufficient for a stable fine-grained conclusion.

## Validation-only nano tuning

The frozen tuning plan is in [`configs/tuning_nano.yaml`](configs/tuning_nano.yaml). It keeps the dataset, seed, deterministic mode, image size, batch size, 1,000-epoch ceiling, and patience 100 fixed. The completed native-default baseline is candidate zero. Three additional candidates test a lower learning rate with lighter geometry, AdamW with lighter geometry, and stronger SGD regularization. Both pretrained and scratch conditions are tuned separately.

The search reads only the training and validation splits. Selection is by validation mAP@0.5:0.95; do not inspect or repeatedly evaluate the test split while choosing hyperparameters. Launch one family per physical GPU:

```bash
python scripts/tune_baselines.py --family yolov5 --device 0
python scripts/tune_baselines.py --family yolo26 --device 1
python scripts/collect_tuning_results.py
```

The tuning runner writes an explicit completion marker and skips completed runs. Each run stores the exact candidate, protocol, environment, GPU, and pinned repository revisions. The collector writes a validation-only table and a machine-readable winner file. Only after all candidates finish should each selected checkpoint receive one frozen test evaluation. Additional seeds are still required to estimate uncertainty; seed 42 tuning scores alone are not uncertainty estimates.

Stage-one native pretrained YOLO26n remains the leading condition. The completed [validation-only error analysis](docs/yolo26-validation-error-analysis.md) identifies low-confidence classification and sub-1%-area objects as the dominant failures. The next search is therefore narrowed to YOLO26 pretrained only in [`configs/tuning_yolo26_error_driven.yaml`](configs/tuning_yolo26_error_driven.yaml): classification emphasis, localization emphasis, multiscale small-object training, and a combined condition. The earlier optimizer-grid refinement config is retained only as provenance and should not be launched.

An additional architecture-focused search in [`configs/tuning_yolo26_p2_error_driven.yaml`](configs/tuning_yolo26_p2_error_driven.yaml) uses upstream YOLO26n-P2, whose P2/4 detection scale directly targets the observed small-object failure. Its increased FLOPs must be reported explicitly.

The multi-seed confirmation configuration is retained but deferred. At the owner's direction, the active follow-up remains on seed 42 and runs new validation-only experiments from [`configs/tuning_yolo26_seed42_followup.yaml`](configs/tuning_yolo26_seed42_followup.yaml). The four GPUs across two hosts receive disjoint trial names to prevent shared-NAS collisions. Test access remains prohibited.

```bash
python scripts/tune_baselines.py --config configs/tuning_pretrained_refinement.yaml --family yolov5 --device 0
python scripts/tune_baselines.py --config configs/tuning_pretrained_refinement.yaml --family yolo26 --device 1
```

## Current analysis path

Use the vendored PEEK hooks to capture YOLO26 activations, following `third_party/PEEK/notebooks/Ultralytics_Demo.ipynb`. Then analyze a selected module directly:

```bash
topoprun feature_maps/example.pkl --module 12 --output artifacts/example-topology.json
```

The command converts the activation with `peek.core.PEEK`, constructs the cubical complex, and records the full finite persistence diagrams, essential births, and summary statistics. A precomputed two-dimensional PEEK map may be supplied as `.npy` without `--module`. Add `--visualization-dir artifacts/example-frames` for a deterministic heatmap and five filtration frames.

The frozen details for augmentation equivalence, PEEK calibration, filtration, numerical behavior, and bootstrap aggregation are in [`docs/experiment-protocol.md`](docs/experiment-protocol.md). Diagram comparison reports bottleneck and 2-Wasserstein distances; compact-summary drift remains available only as a diagnostic.

The checkpoint-ready topology pipeline captures canonical maps for the frozen train-only calibration set and compares a structurally pruned candidate with its reference:

```bash
python scripts/capture_peek_maps.py --weights /path/to/reference.pt --data data/wsd.yaml \
  --output-root artifacts/peek/reference --device 0
python scripts/capture_peek_maps.py --weights /path/to/candidate.pt --data data/wsd.yaml \
  --output-root artifacts/peek/candidate --device 0
python scripts/compare_peek_topology.py --reference-root artifacts/peek/reference \
  --candidate-root artifacts/peek/candidate --output artifacts/topology/candidate.json
```

Capture validates the 128 frozen image hashes, reads only the train calibration set, and refuses to overwrite existing map groups. Comparison requires complete coverage for modules `[4, 6, 10, 16, 19, 22]` and dimensions 0/1, then reports image-level bootstrap intervals and equal-weight overall drift. The dependency rules, matched-compute ablations, acceptance gates, and record schema are frozen in [`configs/pruning_yolo26.yaml`](configs/pruning_yolo26.yaml) and [`docs/pruning-protocol.md`](docs/pruning-protocol.md). Checkpoint-dependent graph surgery remains pending the selected model checkpoint.

## Proposed experiment flow

1. Freeze WSD splits and matched YOLOv5/YOLO26 baselines.
2. Select PEEK hook modules and a deterministic calibration subset.
3. Capture baseline PEEK maps and persistence diagrams.
4. Generate structured pruning candidates at fixed compute budgets.
5. Measure topology drift, PEEK variance, WSD detection quality, and resource cost.
6. Fine-tune accepted candidates under one frozen schedule.
7. Re-evaluate on WSD test data and robustness slices.
8. Down-select models for Jetson Orin Nano benchmarking.

## To-do list

### Dataset and evaluation protocol

- [x] Identify WSD v71 as the authoritative release and record its four classes and split counts.
- [x] Record the JAIS 2024 source publication and DOI for WSD.
- [x] Record that WSD contains distinct still images rather than video-derived frames.
- [x] Add a portable WSD manifest generator with image/label pairing and exact SHA-256 duplicate checks.
- [x] Reference the JAIS 2024 source paper as the authoritative description of WSD acquisition and synthesis.
- [x] Document WSD as private project data that is intentionally excluded from Git and not publicly distributed.
- [x] Audit corrupt images, invalid/empty boxes, class IDs, box bounds, and class imbalance; archive the machine-readable result.
- [x] Check for exact image duplicates across all splits: zero SHA-256 duplicate groups in v71.
- [x] Complete and archive a two-stage cross-split perceptual review: 16 screened candidates and zero confirmed near-duplicate pairs.
- [x] Confirm dataset semantics with the owner: one spacecraft per distinct image and spacecraft renders do not cross split boundaries.
- [x] Freeze metadata-driven robustness slices for brightness/appearance and annotation scale/density/extent.
- [ ] Add owner-provided occlusion, viewpoint, background, and target-identity metadata if those factors cannot be inferred reliably from pixels/boxes.
- [x] Keep COCO out of the current evaluation matrix; it may only enter indirectly through pretrained weights unless the protocol is deliberately expanded later.
- [ ] Select any external spacecraft dataset needed to test domain shift; do not mix it into WSD until label mappings and licenses are explicit.
- [x] Freeze the initial nano baseline seed, deterministic mode, WSD splits, image size, batch size, workers, epoch ceiling, patience, and initialization conditions.
- [x] Record an ordered image-and-label SHA-256 fingerprint for every WSD split in the duplicate-review artifact.
- [x] Freeze standalone test evaluation at confidence 0.001, NMS IoU 0.60, 300 maximum detections, and 640 px.
- [x] Report per-class precision, recall, mAP@0.5, and mAP@0.5:0.95 for every currently completed model; rerun the collector as tuned models are evaluated.
- [x] Wire overall and per-class across-seed uncertainty reporting with explicit single-seed safeguards.
- [ ] Populate uncertainty intervals after the frozen tuned conditions complete across at least three seeds (five preferred).

### Baseline model

- [x] Vendor and pin the current YOLOv5 revision used by PEEK.
- [x] Vendor and pin the current YOLO26-capable Ultralytics revision used by PEEK.
- [x] Vendor and pin the official PEEK repository.
- [x] Freeze the first matched model matrix as YOLOv5n and YOLO26n, each pretrained and from scratch.
- [x] Define matched preprocessing and augmentation settings across v5 and v26 and document unavoidable native implementation differences.
- [x] Complete four unpruned WSD nano baselines (v5/v26 × pretrained/scratch) and archive weights, configs, environments, seeds, and metrics.
- [x] Evaluate each best checkpoint once on the frozen WSD test protocol and generate a chart-ready CSV.
- [x] Freeze a validation-only hyperparameter search with the native baseline plus three declared candidates per initialization.
- [x] Record, but defer, a local pretrained-only refinement until validation error analysis can narrow its hypotheses.
- [x] Complete validation-only YOLO26 pretrained FP/FN, confidence, confusion, size, and robustness-slice error analysis.
- [x] Replace blind optimizer refinement with four error-driven YOLO26 pretrained hypotheses.
- [ ] Complete the four tuning searches, select by validation mAP@0.5:0.95, and evaluate each winner once on the frozen test split.
- [ ] Identify the checkpoint, dataset fingerprint, split, and evaluation command behind the draft’s YOLOv5n 0.808/0.570 row; the reproducible v71 run did not match it.
- [ ] Repeat the selected final conditions over frozen seeds 42–46 before reporting preferred uncertainty intervals.
- [ ] Decide whether the ResNet50-YOLO branch remains in scope; the draft currently asserts it without an implementation.

### PEEK calibration

- [x] Route activation-to-PEEK conversion through the official package rather than copying its formula.
- [ ] Refresh and re-pin the upstream PEEK submodule after the announced push, then confirm its commit and API before freezing experiments.
- [x] Enumerate valid YOLO26 hook modules and document their spatial shapes and post-block semantics.
- [x] Freeze modules `[4, 6, 10, 16, 19, 22]` before looking at pruning outcomes.
- [x] Freeze a hash-stable 128-image, metadata/class-stratified train calibration manifest, including correct-only and empty-scene policies.
- [x] Define maps per image/module and require batches to be separated before official PEEK conversion.
- [x] Validate numerical behavior for mixed precision, singleton batches, constant maps, and non-finite inputs.
- [x] Add deterministic PEEK-map visualization and superlevel filtration-frame export.

### Topology and pruning method

- [x] Implement normalized cubical persistent-homology summaries for dimensions 0 and 1.
- [x] Add a CLI for official-PEEK activation pickles and precomputed maps.
- [x] Freeze superlevel filtration so high-information PEEK regions enter first and document its lower-star implementation.
- [x] Retain finite persistence diagrams and essential births; compare bottleneck and 2-Wasserstein distances.
- [x] Freeze image-level/module-level aggregation and a seed-42, 10,000-resample bootstrap confidence-interval protocol.
- [x] Specify dependency-grouped output channels as the pruning unit and freeze residual, concatenation, normalization, attention, and detection-head safety rules.
- [ ] Implement dependency-aware structural removal so parameter/FLOP savings are physical, not zero masks alone.
- [x] Define validated candidate records, acceptance gates, and lower-is-better scoring roles for PEEK variance, topology drift, accuracy, and compute matching.
- [x] Freeze magnitude-only, random, PEEK-variance-only, and topology-only ablations at matched measured-FLOP budgets.
- [x] Freeze accept/reject criteria and the validation-only fine-tuning schedule before running the final test set.
- [ ] Measure whether topology provides signal beyond output accuracy rather than assuming that it does.

### Deployment and reporting

- [x] Add a raw-forward profiler for parameter count, serialized size, FLOPs/MACs, peak CPU/GPU memory proxies, and host latency percentiles.
- [x] Freeze desktop and Jetson warm-up, batch size, precision, TensorRT/ONNX metadata, power mode, clocks, and trial count requirements.
- [ ] Benchmark the down-selected models on the Jetson Orin Nano; report median and tail latency, FPS, memory, temperature, and power.
- [x] Freeze separate desktop-proxy and physical-Jetson result sections and prohibit unlabeled cross-runtime comparisons.
- [x] Export current dataset and baseline experiment records to CSV/JSON and generate baseline tables from those records.
- [ ] Extend record-driven tables/figures to pruning and deployment results once those artifacts exist.
- [ ] Replace every red/TBD statement in the draft only after its corresponding artifact exists.
- [x] Add contributor guidance and a reproducibility release checklist.
- [ ] Add a repository license and citation metadata after confirming distribution terms and the final author list.

## Scope warning

The manuscript is a useful hypothesis and work plan, but it currently mixes prior results, planned experiments, and architectural possibilities. This repository therefore favors traceability over filling gaps with assumptions. Until the unchecked protocol items above are resolved, outputs should be labeled exploratory and should not populate the final results table.
