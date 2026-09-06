# REPO_MAP.md

Discovery pass for **P0.0** of `TOPO_PRUNE_SAT_PLAN.md` (the plan lives in the sibling
`../TDA/` working folder; this repo, `PEEK-TiNA`, is the "existing repo" the plan extends).

Produced against a fresh clone at commit `6ee76ac` with **submodules uninitialized and no
dataset / checkpoints / `runs/` present locally**. Everything below is read from committed
source, configs, and `docs/`.

---

## ⛔ Blocking findings — read before anything else (`[HUMAN]`)

### B1. The plan assumes segmentation. This repo is a detector.

The entire plan is built on a **segmentation** model:

- §1 "Why segmentation makes this tractable" — the output is "a function on a pixel grid;
  its superlevel sets have genuine homology".
- §3 "The certificate" — objects are a per-class **logit map** `f_c : Ω → ℝ`, a threshold
  `τ`, a predicted mask `S_c = {f_c ≥ τ}`, and the cubical persistence diagram of that
  logit map. The falsifiable claim (§3.4) is a statement about `β_k` of the predicted
  **mask**.

This repo trains **YOLOv5n / YOLO26n object detectors** on WSD (4 classes: `antenna`,
`body`, `solar`, `thruster`). The `Detect` head output is `[1, 300, 6]` decoded
box predictions (`docs/yolo26n-peek-module-inventory.json:284`), explicitly marked
`peek_compatible: false` because it "isn't a spatial map" (`README.md:234`). **There is no
logit map and no predicted segmentation mask anywhere in the forward pass.** The plan's
central object does not exist here.

What this repo actually computes topology on: **PEEK maps** — a third-party explainability
method (`third_party/PEEK`, pinned) reduces a `C×H×W` internal activation to a single
normalized `H×W` scalar "importance" map (`src/topoprun/peek_adapter.py:10`), and cubical
persistent homology (GUDHI, dims 0/1, superlevel) runs on *that*
(`src/topoprun/topology.py:53`). This is topology **of the representation**, which is
exactly the correlational quadrant the plan's §1 and §2 (Route 1, P0 kill criterion) say
the project must escape — "Topology of the representation is a hypothesis" in the plan's
own comparison table.

**Consequence:** P1–P3 as written (the certificate, the Betti-preservation guarantee, the
Betti-error primary metric) cannot be executed against this repo without either (a)
swapping in a segmentation model + component-mask labels, or (b) rewriting the plan's
certificate around a different output object. This is a fork in the project and needs
Matas's decision. Options in §"Recommended decisions" below.

### B2. No baseline checkpoint, dataset, or submodules in the clone.

- `git submodule status` shows all three (`ultralytics`, `yolov5`, `PEEK`) **not
  initialized**. `make setup` initializes and `pip install -e`s them.
- `../datasets/satellite_components-71/` (the conventional WSD location, `README.md:46`)
  is **absent**. WSD v71 is private Roboflow data, not committed. `[HUMAN]` — Matas must
  supply the export.
- No `runs/` directory. `docs/runs-summary.md` describes 16 completed training/tuning runs
  and four baseline checkpoints that exist **on the training machines, not in Git**. The
  leading checkpoint (`runs/baseline_nano/yolo26n_pretrained_seed42/weights/best.pt`,
  referenced in `docs/yolo26n-peek-module-inventory.json:2`) must be transferred.

### B3. The plan's P0.1 target number is already known not to reproduce.

Plan P0.1: "reproduce the SCITECH numbers within noise. If it does not reproduce, stop."
`README.md:143` records that the draft's YOLOv5n **0.808 / 0.570** mAP row **could not be
reproduced** on the authoritative v71 split — the clean re-run gets **0.455 / 0.308** on
test. The repo maintainers flag the draft number's provenance as unknown (`README.md:290`).
Whatever "the SCITECH numbers" means for P0.1 needs to be pinned to a specific artifact
before P0.1 can have a pass/fail.

---

## The seven P0.0 questions

### 1. Segmentation architecture — encoder / decoder / head / norm / activations

**There is no segmentation architecture.** The models are detectors:

| | |
|---|---|
| Families | YOLOv5n (`third_party/yolov5`, pin `84ef1e5…`) and YOLO26n (`third_party/ultralytics`, pin `40eb41a…`) |
| Leading model | **YOLO26n, pretrained, seed 42** — anchors every downstream experiment (`docs/runs-summary.md:16`) |
| YOLO26n structure | 24 top-level modules (`docs/yolo26n-peek-module-inventory.json`). Backbone: `Conv` stem → `C3k2` blocks → `Conv` downsamples → `SPPF` (idx 9) → `C2PSA` attention (idx 10). Neck: `Upsample`/`Concat`/`C3k2` (idx 11–22), PAN-style. Head: `Detect` (idx 23), anchor-free end-to-end, output `[1,300,6]`. |
| Norm / activation | Standard Ultralytics `Conv` = Conv2d + BatchNorm + SiLU. `[VERIFY]` against the pinned `ultralytics` commit once submodules are initialized. |
| Input | 640×640 letterboxed, RGB, `[0,1]` (`docs/experiment-protocol.md:5`) |
| P2 variant | An optional YOLO26n-P2 head (extra P2/4 scale) was tried for small objects at higher FLOPs (`docs/runs-summary.md:54`) |
| ResNet50-YOLO | Mentioned in the draft, **not implemented** — README to-do `README.md:292` flags it as an open scope question |

### 2. Where the forward pass produces the raw logit map (before argmax/softmax)

**No such tensor exists.** The plan says this tensor "is the object the entire project is
built on"; the closest analogues here are:

- **`Detect` head output** (`model.model.model[23]`): `[1, 300, 6]` — decoded
  `(x,y,w,h,conf,cls)` predictions after NMS-style top-k. Not spatial, no homology.
- **PEEK maps**: the project's actual topological object. Hook point is
  `output after top-level model.model.model[index]`
  (`docs/yolo26n-peek-module-inventory.json:6`), captured by
  `peek.extractors.hooks.LatentExtractor` (`scripts/capture_peek_maps.py:34,47`) for the
  frozen module set `[4, 6, 10, 16, 19, 22]` (three backbone + three neck outputs at
  strides ≈8/16/32), then reduced to `H×W` by `peek.core.PEEK()` via
  `topoprun.peek_adapter.to_peek_map` (`src/topoprun/peek_adapter.py:41`).
- **Pre-`Detect` neck feature maps** (modules 16/19/22): the raw per-scale classification
  logits *before* decode are internal to the `Detect` module. If the plan is retargeted to
  "class-logit heatmap per scale", **this** is where a hook would go — `[VERIFY]` the exact
  sub-tensor in the pinned `ultralytics` `Detect.forward`.

### 3. Dataset, classes, resolution, split protocol

| | |
|---|---|
| Dataset | **WSD v71** — Roboflow project `satellite_components` v71, private (`README.md:11`). Distinct still renders, **not** video frames. |
| Source paper | Mahendrakar, White, Tiwari & Wilde, *JAIS* 21(5):455–460, 2024, [doi:10.2514/1.I011343](https://doi.org/10.2514/1.I011343) |
| Classes (4) | `antenna`, `body`, `solar`, `thruster` (`src/topoprun/datasets.py:13`) |
| Counts | train 1004 / val 186 / test 40 images; instance counts 2507 / 1028 / 2481 / 1954; test has only **25 thruster instances** (sample-sensitive) (`README.md:82`) |
| Resolution | 640 px, batch 16, seed 42, deterministic (`README.md:98`) |
| Split protocol | Frozen train/valid/test from the Roboflow export. Duplicate review: **0** SHA-256 dupes, 0 confirmed near-dupes across splits (`README.md:80`). Owner-confirmed: **one spacecraft per distinct image; renders do not cross split boundaries** (`README.md:264`). |
| Identity-disjoint split? | `[HUMAN]` — the plan's P3.4 wants an *identity-disjoint* split. "Renders do not cross split boundaries" is suggestive but there is **no per-spacecraft-identity label** in the repo. Robustness slices (`docs/wsd-v71-robustness-slices.csv`) are brightness/scale/density clusters, explicitly "**not** spacecraft-identity labels" (`README.md:84`). Need Matas to confirm whether identity metadata exists. SPARK (plan Appendix C) is the fallback dataset, `[VERIFY]` component masks. |

### 4. The existing topological criterion

| Aspect | Value | Location |
|---|---|---|
| Input tensor | PEEK map (normalized `H×W` reduction of a hooked activation), **not** a logit/output map | `src/topoprun/peek_adapter.py` |
| Filtration | Cubical, **superlevel** (implemented as lower-star on `1 − normalized_map` so high-PEEK cells enter first) | `src/topoprun/topology.py:65` |
| Homology dims | 0 and 1 | `src/topoprun/topology.py:53` |
| Library | **GUDHI** `CubicalComplex` (not `cripser`) | `src/topoprun/topology.py:60,71` |
| Vectorization / summary | finite intervals + essential births; `finite_features`, `total/mean/max_persistence`; bottleneck + custom 2-Wasserstein between diagrams | `src/topoprun/topology.py:88,113` |
| Aggregation | image = independent unit; per module/dim median+IQR+mean + seed-42 10k-resample bootstrap CI; equal-weight mean across the 6 frozen modules | `src/topoprun/topology.py:187`, `docs/experiment-protocol.md:16` |
| Frozen modules | `[4, 6, 10, 16, 19, 22]` | `configs/pruning_yolo26.yaml:13` |
| **There is no per-channel `TopoScore`** | The plan's P0.1 asks to reimplement "the existing criterion" as a per-channel importance score. What exists is a **drift metric between two models' PEEK-map diagrams**, plus a *proposed* pruning method `topology` = "smallest increase in calibration-set topology drift" (`docs/pruning-protocol.md:20`) that is **not implemented**. | — |

### 5. How pruning is currently executed

**It is not.** `src/topoprun/pruning.py` contains only *record schemas and an acceptance/
scoring function* (`evaluate_candidate`, frozen gates, lower-is-better composite). The
actual structural removal is `dependency_engine: pending_checkpoint_integration`
(`configs/pruning_yolo26.yaml:18`); README to-do "Implement dependency-aware structural
removal" is **unchecked** (`README.md:313`). Planned unit: dependency-grouped output
channels with residual/concat/norm/attention safety rules (`docs/pruning-protocol.md:6`).
Four planned methods: `magnitude`, `random`, `peek_variance`, `topology`
(`src/topoprun/pruning.py:9`). No third-party pruning library is vendored.

### 6. Checkpoints and the baseline to prune

Baselines are **trained but not in Git** (see B2). From `docs/baseline-nano-results.csv`
via `docs/runs-summary.md`:

| Model | Init | Val mAP@.5:.95 | Test mAP@.5:.95 | GFLOPs | Weights MB |
|---|---|---:|---:|---:|---:|
| **YOLO26n** | **Pretrained** | **0.422** | **0.312** | 5.8 | 5.41 |
| YOLO26n | Scratch | 0.348 | 0.245 | 5.8 | 5.43 |
| YOLOv5n | Pretrained | 0.389 | 0.308 | 4.2 | 3.90 |
| YOLOv5n | Scratch | 0.334 | 0.262 | 4.2 | 3.90 |

Best tuned condition (val only, not test): YOLO26n pretrained, `cls=0.75, dfl=1.75` →
**0.4387** val mAP@.5:.95 (`docs/runs-summary.md:73`). The pruning-reference checkpoint is
**deliberately unresolved** pending validation-only selection (`configs/pruning_yolo26.yaml:7`).
Multi-seed (42–46) confirmation and any tuned-model test evaluation are **not done**
(`docs/runs-summary.md:110`).

### 7. Target hardware

**Found — no `[HUMAN]` needed.** **NVIDIA Jetson Orin Nano**, batch 1, 640 px, TensorRT
FP16 engine (`docs/deployment-protocol.md:5`). Desktop measurements (`profile_detector.py`:
params, serialized size, MACs/FLOPs, RSS, CUDA alloc, latency p50/p90/p95/p99) are
explicitly *proxy only* and never share a column with Jetson numbers. Jetson protocol:
JetPack/TRT/CUDA/cuDNN recorded, max power mode + `jetson_clocks`, 5-min thermal soak, 100
warm-up + 1000 measured end-to-end trials, `tegrastats` power. Not yet run
(`README.md:323`).

---

## Repo layout (what's already here)

```
src/topoprun/            # NOTE: name collides conceptually with the plan's topo_prune/
  topology.py            # cubical PH on PEEK maps, diagram compare, bootstrap aggregation
  peek_adapter.py        # activation C×H×W -> PEEK H×W via third_party/PEEK
  pruning.py             # record schemas + acceptance gates + composite score (NO removal)
  datasets.py            # WSD audit + Ultralytics YAML generation
  uncertainty.py         # across-seed Student-t intervals
  reproducibility.py     # seed_everything (torch/numpy/random + deterministic kernels)
  visualization.py       # deterministic PEEK heatmap + filtration frames
  cli.py                 # `topoprun` — PH on one map/pickle or a directory tree (joblib)
scripts/                 # 25 scripts: train/tune/evaluate baselines, WSD audits,
                         #   PEEK capture, topology compare, deployment profiling
configs/                 # frozen YAMLs: baseline_nano, tuning_*, pruning_yolo26, dataset
docs/                    # frozen protocols + machine-readable audits + run summaries
tests/                   # 6 test files (datasets, peek_adapter, pruning, topology×2, uncertainty)
third_party/             # ultralytics, yolov5, PEEK — pinned submodules (uninitialized here)
```

Existing conventions the plan's §4/§6 must slot into (not fight): frozen YAML configs,
`docs/*-protocol.md` freeze pattern, image-as-independent-unit + seed-42 10k bootstrap,
append-only machine-readable audit JSON, "exploratory until confirmation seeds + one-shot
test" labeling (`README.md:333`).

---

## Recommended decisions for Matas (`[HUMAN]`)

1. **Segmentation vs. detection (B1) — the fork.** Pick one:
   - **(a) Retarget the certificate to detection.** Define the topological output object as
     the per-scale class-logit heatmap from `Detect` (superlevel sets of that *are* a
     function on a grid). The plan's §3 math survives with `f_c` = class-`c` logit heatmap
     at a chosen scale; "predicted mask" = thresholded heatmap. Betti error becomes
     "connected components / holes of the class heatmap vs. ground-truth component count".
     Loses the crisp "one solar array split into three" framing slightly but keeps it
     mostly. Requires a new hook, not a new model.
   - **(b) Add a segmentation model.** Train a small seg head (or a UNet/DeepLab) on
     component **masks**. WSD ships boxes, not masks — needs mask labels (SPARK `[VERIFY]`,
     or SAM-assisted labeling, or synthetic renders per plan Route 8 / P1.5). Biggest
     scope increase; matches the plan verbatim.
   - **(c) Keep PEEK-map topology, drop the certificate.** Run P0 (hygiene / variance
     ablation) and P2 (two-sided intervention) on the *representation* topology as the
     repo already frames it, and cut P1/P3's certified Betti-preservation claim. Smallest
     change; also the weakest scientific claim (the plan's §1 argues against exactly this).
   - My read: **(a)** preserves the most of the plan for the least new infrastructure.
     Worth a short spike to confirm the `Detect` logit heatmap is accessible and non-degenerate.
2. **P0.1 reference (B3).** Which concrete artifact are "the SCITECH numbers"? If it's the
   unreproducible 0.808/0.570 draft row, P0.1's exit criterion needs restating against the
   reproducible v71 baseline table instead.
3. **Data + weights (B2).** Point me at the WSD v71 export and transfer the
   `yolo26n_pretrained_seed42` baseline checkpoint (+ the other three if the v5/v26 control
   matters for P2).
4. **Identity-disjoint split (Q3).** Does per-spacecraft-identity metadata exist for WSD?
   P3.4's distribution-shift claim depends on it.
5. **Package naming.** Plan wants a new `topo_prune/`; repo already has `src/topoprun/`.
   Extend the existing package, or add a parallel one and accept the near-homonym?

**Per the plan's own rules (`CLAUDE.md`: "`[HUMAN]` means stop and ask Matas"), P0.0 stops
here. No P0.1+ code until B1 is resolved.**
