# Certified topological pruning — detector retarget (option A)

Workstream for [`../../TDA/TOPO_PRUNE_SAT_PLAN.md`](../../TDA/TOPO_PRUNE_SAT_PLAN.md).
Self-contained subtree, same pattern as `ood_monitor/`; it does not touch the
parent training code.

## The one thing to know

The plan is written for a **segmentation** model whose per-class logit map has
homology. This repo is a **YOLO detector**. Matas chose **option A**: retarget the
certificate to the detector's per-scale classification logit maps `f_c^(i)` from
the `Detect` head — their superlevel sets `{f_c^(i) >= tau}` carry genuine
homology, so every piece of the plan's §3 certificate transfers with only
interpretation changes. Full rationale and the open sign-off questions:
[`DEVIATIONS.md`](DEVIATIONS.md) D1 and [`docs/certificate-detection.md`](docs/certificate-detection.md).

## Status

**P1 ran locally** on a COCO-pretrained `yolo26n.pt` stand-in (no WSD checkpoint
yet — `../REPO_MAP.md` B2). Numbers: [`docs/p1-results.md`](docs/p1-results.md).
Headline: **0 audit violations under the sound certificate across all 10
prune arms** (E1) and **0 across 240 synthetic scenes with exact ground-truth
homology** (E2); certified coverage 28% → ~0 as the head-prune budget goes
10% → 50%.

| | State |
|---|---|
| Detector-free core — `filtrations`, `descriptors`, `certificate`, `metrics`, `synthetic` | **done, 53 tests pass** on Python 3.14 |
| `head_maps`, `prune`, `lipschitz` (empirical ε + single-layer certified ε) | **run** on `yolo26n.pt` CPU |
| `lipschitz.epsilon_certified` general mask→Δφ (spectral-norm product) | **`NotImplementedError`** — E1 uses the closed-form single-layer bound (`DEVIATIONS.md` D6) |
| E1 audit, E2 synthetic validation, `report.py` | **done** — `experiments/p1_certificate/` |
| Run on the real WSD checkpoint + identity-disjoint split; P2–P5 | pending the checkpoint (`../REPO_MAP.md` B2) |

## Reproduce the P1 experiments

```bash
# one-time: a venv with the topo core + CPU detector stack
python -m venv .venv && . .venv/Scripts/activate
pip install gudhi scipy numpy pyyaml joblib pytest
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -e third_party/ultralytics

python topo_prune/experiments/p1_certificate/run_p1.py         # E1: capture, prune, audit
python topo_prune/experiments/p1_certificate/synthetic_p15.py  # E2: synthetic validation
python topo_prune/src/topo_prune/report.py                     # regenerate docs/p1-results.md
```

## Layout

```
topo_prune/
  src/topo_prune/
    filtrations.py    superlevel cubical PH on a logit map; Euler-characteristic-curve control
    descriptors.py    Betti curves, persistence entropy, total persistence, ECC vector
    certificate.py    margin test (§3.3), certify, falsification audit (§3.4), coverage
    metrics.py        Betti-number error (primary metric), simplified Betti-matching error
    head_maps.py      hook Detect.one2one_cv3[i] -> f_c^(i)               [needs detector]
    prune.py          structured magnitude/random pruning of the class head [needs detector]
    lipschitz.py      epsilon_empirical / spectral norms / head bounds     [needs checkpoint]
    capture.py        run a detector over images -> class-logit maps       [needs detector]
    synthetic.py      logit-map scenes with exact ground-truth homology (P1.5)
    report.py         run.json manifests -> docs/p1-results.md
  configs/certificate_yolo26.yaml
  docs/certificate-detection.md   docs/p1-results.md   (generated)
  tests/               toy_fields.py + 6 test modules
  experiments/p1_certificate/    run_p1.py (E1), synthetic_p15.py (E2), runs/*.json
  DEVIATIONS.md  HYPOTHESES.md  RESULTS.md
```

## Run

```bash
make -C topo_prune test      # detector-free, runs anywhere with gudhi + scipy + numpy
```

On a GPU host with `make setup` done, the WSD export, and the tuned YOLO26n
checkpoint: wire `head_maps` + `lipschitz` into a capture script mirroring
`scripts/capture_peek_maps.py`, then run the P1.5 audit. Not scripted yet
(blocked, so not guessed).

## Rules inherited from `CLAUDE.md`

No PH on the inference path. No Rips. Betti error is the primary metric. Never
tune `tau` to make an audit violation disappear — a violation is a bug. Every
departure from the plan is in `DEVIATIONS.md` before the code.
