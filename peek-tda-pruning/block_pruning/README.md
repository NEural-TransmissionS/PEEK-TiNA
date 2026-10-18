# block_pruning (topo-prune-shapes)

Initial pass at one question: **can you prune a YOLO26 detector by cutting the
latter-half blocks whose internal salience field carries the *least* topological
persistence, and lose little accuracy while cutting compute?**

Copy this directory together with `../src/topoprun` (the canonical PEEK adapter)
to a server, build the venv (it also needs the `peek` package, see
`../requirements.txt`), `sbatch`.

## The idea

For a scalar field (here the block's **PEEK map**, min-max normalised; the archived
`--basis activation` variant uses `|activation|_2` instead):

- **H0 persistence** = topographic prominence of the peaks. High = a few strong,
  isolated maxima. Low = a flat / uniformly textured field.
- **H1 persistence** = depth of an enclosed low pocket ringed by high values.
  High = sharp ring around a genuinely empty interior.

A block with **high** combined persistence has concentrated, spatially decisive
salience → keep it. A block with **low** persistence has diffuse salience → the
hypothesis is that it contributes little and can be removed.

We rank the four head `C3k2` blocks — **13** (P4 top-down), **16** (P3/8 out),
**19** (P4 out), **22** (P5 out) — by `0.5·norm(maxH0) + 0.5·norm(sumH1)`
averaged over 40 val images, then cut the `k` lowest for `k = 1, 2, 3`. With four
candidates and `k ≤ 3` the single highest-persistence block is always spared (on
these models that is almost always module 16).

**Single-feature criteria (exploratory).** `--criterion h0` / `--criterion h1`
order the cuts by the raw mean H0 (secondary-peak prominence) or H1 (pocket)
persistence alone instead of the 50/50 blend (`combined`, the default and the
original sweep). Same candidates, same cut mechanics, same training; results go
to `results/<dataset>_<size>_<criterion>_k<k>.json`, and `k = 0` (the full
model) is shared with the combined sweep rather than retrained:

```bash
CRITERION=h0 KS="1 2 3" sbatch slurm/sweep.slurm
CRITERION=h1 KS="1 2 3" sbatch slurm/sweep.slurm
```

**WSD (exploratory).** The same cut method on the real satellite-components data
(WSD v71), ranking with scratch YOLO26n/s baselines (`../scripts/train_wsd_baselines.slurm`)
and training every model from scratch under the WSD scratch protocol (1000 epochs,
patience 100, batch 16, deterministic, AMP) instead of the 60-epoch shapes schedule.
`k = 0` is retrained here too, so full and pruned share one pipeline. Validation split
only. Ranking reads 40 **train** images (val is left for evaluation), spread evenly
over the sorted list since WSD filenames are class-prefixed.

```bash
WSD_ROOT=/path/to/satellite_components-71 WSD_WEIGHTS_N=/path/to/n/best.pt WSD_WEIGHTS_S=/path/to/s/best.pt VENV=/path/to/venv bash slurm/submit_wsd.sh
```

"Cut" = replace that block's YAML entry with a **1×1 `Conv` of the same nominal
output width** (`topo_prune.build_pruned_yaml`). Every downstream `from` index and
channel width is unchanged, so the graph is always valid; what's removed is the
block's bottleneck stack. It's a "reduce the block to a minimal shape adapter"
prune, not total deletion — deleting a neck block outright inflates its consumer
(the next block widens to absorb the un-reduced channels) and can *raise* GFLOPs.
Params/GFLOPs are read from the rebuilt graph. Every model — full and pruned — is
trained **from scratch** on the shapes, same schedule, so the comparison is
architecture-vs-architecture.

## Layout

```
make_shapes.py         synthetic data generator (circles / triangles / squares / mixed)
yolo26.yaml            base architecture (ultralytics, pinned commit)
weights/               8 base checkpoints, yolo26{n,s} per dataset (40 epochs; see
                       ../results/base_models/) -- not committed: weights/README.md
xaikit/                vendored verbatim: YoloExplainer + H0/H1 persistence helpers
topo_prune.py          rank_layers() + build_pruned_yaml() + model_stats()
_train_one.py          train ONE model (full or pruned), own process, -> results/<name>.json
run_experiment.py      per (dataset,size): rank once, then spawn _train_one for k=0,1,2,3
summarize.py           results/*.json -> results.csv + summary.png
run_all.sh             local sequential run (no SLURM)
slurm/sweep.slurm      8-task array, one per (dataset,size)
slurm/submit.sh        prep -> sweep -> summarize, with dependencies
```

## Run on the server

Needs **Python ≥ 3.10** — the bare cluster `python3` is often older, so build a
venv against a loaded module (or point `VENV` at an existing env):

```bash
cd topo-prune-shapes
module load python/3.11              # whatever the cluster calls it
python3 -m venv .venv && . .venv/bin/activate
python -V                            # confirm >= 3.10
pip install -r requirements.txt
bash slurm/submit.sh
```

`submit.sh` / `sweep.slurm` env knobs:

| var | default | meaning |
|---|---|---|
| `VENV` | `.venv` | env to `source .../bin/activate` in each job |
| `PY_MODULE` | *(unset)* | `module load $PY_MODULE` in each job before activating the venv |
| `PYTHON` | `python3` | interpreter name (after venv activation) |
| `SBATCH_PARTITION` | `gpu2` (set in `sweep.slurm`) | GPU partition for the array; prep/summarize use the default CPU partition |
| `EPOCHS` | `60` | training length |
| `N_RANK` | `40` | val images used for the persistence ranking |

Both scripts abort up front if the resolved interpreter is < 3.10 rather than
failing deep in a job.

Local instead of SLURM:

```bash
. .venv/bin/activate
EPOCHS=60 DEVICE=0 bash run_all.sh
```

## Outputs (kept deliberately minimal)

- `ranking_peek/<dataset>_<size>.json` — per-block PEEK-map H0/H1 persistence + the
  cut order (`ranking/` for `--basis activation`).
- `results/<dataset>_<size>[_<criterion>]_k<k>_peek.json` — `mAP50-95`, `mAP50`,
  `gflops`, `params`, `layers_cut`, `basis`. A cut model depends only on which blocks
  are cut, so a run with the same cut set from another criterion is the same model;
  `../scripts/collect_results.py` maps each (ranking, k) to it.
- `results.csv` — the above flattened, plus `gflops_vs_full_%` and
  `mAP_delta_vs_full`.
- `summary.png` — mAP and GFLOPs vs blocks-cut, one line per (dataset, size).

## Scope

One seed (42), two metrics (mAP, GFLOPs), 4 datasets × 2 sizes × {full, −1, −2, −3}
= 32 from-scratch trainings. This is a signal check, not a full pruning study: no
per-class breakdown, no latency/wall-clock, no multi-seed CIs. If the low-persistence
blocks drop out cheaply, those come next.
