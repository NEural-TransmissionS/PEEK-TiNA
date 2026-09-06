# RESULTS.md

Append-only (plan §6.6). Every completed run gets a row; failed and blocked runs
stay, marked. No hand-entered numbers in result tables — `report.py` builds them
from `run.json` manifests once runs exist.

---

## P0 — hygiene gate

| Run | Date | Status | Note |
|---|---|---|---|
| P0.0 discovery | 2026-09-06 | done | `../REPO_MAP.md`. Repo is a **detector**, not a segmentation model; existing criterion is topology of PEEK saliency maps (correlational quadrant). |
| P0.1 reproduce SCITECH number | — | **BLOCKED** | No checkpoint / WSD export on any reachable host (`REPO_MAP.md` B2). P0.1 target number also unresolved — draft's YOLOv5n 0.808/0.570 is known-unreproducible (`README.md:143`); needs restating against `docs/baseline-nano-results.csv` (`DEVIATIONS.md` open questions). |
| P0.2–P0.5 ablations | — | not started | Need `scores.TopoScore` reimplementation + a checkpoint. |

## P1 — certificate (detection retarget, `DEVIATIONS.md` D1)

| Component | Date | Status | Note |
|---|---|---|---|
| `filtrations.superlevel_diagram` + toy tests | 2026-09-06 | done | 3 hand-verified diagrams (disk / annulus / two-disks), matches GUDHI exactly. |
| `filtrations.euler_characteristic_curve` | 2026-09-06 | done | Cross-checks `beta_0 - beta_1` on all toy fields. |
| `certificate.certify` / `margin` / `audit` / `coverage` | 2026-09-06 | done | Audit catches a planted certified Betti change; ignores uncertified changes. |
| `metrics.betti_number_error` / `betti_matching_error` | 2026-09-06 | done | Betti-matching is a simplified 2-D approximation (`DEVIATIONS.md` D1.5). |
| `descriptors` (Betti curve, entropy, total persistence, ECC vector) | 2026-09-06 | done | |
| `head_maps.ClassLogitMapExtractor` | 2026-09-06 | written, **UNRUN** | Needs torch + checkpoint. Ready for the GPU host. |
| `lipschitz.epsilon_empirical` / `head_layer_lipschitz_bounds` | 2026-09-06 | written, **UNRUN** | Need checkpoint(s) (+ WSD for empirical). |
| `lipschitz.epsilon_certified` | 2026-09-06 | **NOT IMPLEMENTED** | `Δφ` from a mask needs the checkpoint + a mask spec (`DEVIATIONS.md` D1.4). |
| P1.5 synthetic 300-frame audit | — | not started | Needs a renderer with dial-able occlusion. |
| Coverage-vs-budget curve | — | **BLOCKED** | Needs pruned checkpoints. |

Test suite: `make -C topo_prune test` → **28 passed** (Python 3.14, gudhi 3.13,
detector-free).

## P2–P5

Not started. `HYPOTHESES.md` holds the P2 pre-registration skeleton.
