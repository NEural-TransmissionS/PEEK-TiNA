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

| | State |
|---|---|
| Detector-free core — `filtrations`, `descriptors`, `certificate`, `metrics` | **done, 28 tests pass** on Python 3.14 |
| `head_maps` (hook the one2one class-logit head), `lipschitz` (ε bounds) | written against pinned `third_party/ultralytics`, **not yet run** — no torch / checkpoint here |
| `lipschitz.epsilon_certified` (Δφ from a pruning mask) | **not implemented** — needs the checkpoint + a mask spec |
| P1.5 synthetic audit, coverage-vs-budget, all of P2–P5 | **blocked** on the WSD export + tuned checkpoint (`../REPO_MAP.md` B2) |

Same wall the `ood-monitor-phase1` branch hit: capture needs a GPU host with the
checkpoint and WSD. The detector-free science is built and tested; the rest is
ready to run there.

## Layout

```
topo_prune/
  src/topo_prune/
    filtrations.py    superlevel cubical PH on a logit map; Euler-characteristic-curve control
    descriptors.py    Betti curves, persistence entropy, total persistence, ECC vector
    certificate.py    margin test (§3.3), certify, falsification audit (§3.4), coverage
    metrics.py        Betti-number error (primary metric), simplified Betti-matching error
    head_maps.py      hook Detect.one2one_cv3[i] -> f_c^(i)               [needs detector]
    lipschitz.py      epsilon_empirical / epsilon_certified (plan §3.2)   [needs checkpoint]
  configs/certificate_yolo26.yaml
  docs/certificate-detection.md
  tests/               toy_fields.py + 4 test modules, all detector-free
  experiments/p1_certificate/
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
