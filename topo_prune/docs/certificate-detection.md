# The certificate, retargeted to a detector (option A)

Design note for `DEVIATIONS.md` D1. Read the plan §3 first; this says only what
changes when the model emits boxes instead of a segmentation map, and why the
change is small.

## The object we certify

YOLO26's `Detect` head, for each of the `nl = 3` detection levels `i` and each
class `c`, computes a spatial classification logit map

```
f_c^(i) : grid_i -> R          shape [nc, H_i, W_i]   (nc = 4 for WSD)
```

before any sigmoid or box decode (`third_party/ultralytics/ultralytics/nn/modules/head.py:143`,
`.sigmoid()` applied later at `head.py:174`). For 640-px input the grids are
80×80, 40×40, 20×20 at strides 8/16/32. For an `end2end` model — YOLO26's default
(`head.py:129`) — the deployed predictions are the **one2one** head's, so we hook
`one2one_cv3[i]` (`topo_prune/src/topo_prune/head_maps.py`).

The predicted high-confidence region for `(c, i)` at operating point `p_conf` is
the superlevel set

```
S_c^(i) = { f_c^(i) >= tau },     tau = logit(p_conf) = ln(p_conf / (1 - p_conf))
```

`beta_0(S_c^(i))` counts the disjoint high-confidence blobs for class `c` ≈ the
number of predicted instances; `beta_1` counts enclosed holes (a truss or a
ring-shaped antenna is a real `beta_1 = 1`). The plan's operational failure —
"one solar array predicted as three components" — is `beta_0: 1 -> 3` at some
level. This is why the retarget works: the output *does* have measurable homology,
it is just per-level instead of on the full pixel grid.

## What transfers verbatim from plan §3

- **The stability bound.** Same grid, two functions:
  `d_B(D(f_c^(i)), D(f_c^(i),M)) <= ||f_c^(i) - f_c^(i),M||_inf`, Lipschitz 1.
- **The filtration.** Superlevel-set cubical PH, near-linear, no Rips
  (`filtrations.superlevel_diagram`, GUDHI, `DEVIATIONS.md` D2).
- **The certification check** (§3.3), the **audit / falsifiable prediction**
  (§3.4), **certified coverage**, and the **certified pruning objective** (§3.5).

## The two-condition check

Plan §3.3 certifies image `x` when `m(tau, D) > epsilon` **and**
`ell_min > 2*epsilon`. Our implementation keeps both, but note: for any bar that
straddles `tau` (`birth >= tau > death`),

```
lifetime = (birth - tau) + (tau - death) >= 2 * min(birth - tau, tau - death) >= 2 * m(tau, D)
```

so `m > epsilon` already implies `ell_min > 2*epsilon`. The second condition is
redundant given the first as we define `m` (min endpoint distance over *all*
bars, straddling included). We keep it because the plan lists it and it is free;
`test_certificate.py::test_straddling_condition_is_implied_by_the_margin_condition`
records the redundancy. If a reviewer wants `m` defined only over non-straddling
bars, the second condition stops being redundant — flag before changing.

## What needs Matas's sign-off (open in `DEVIATIONS.md` D1)

1. **Per-level vs. assigned-level certification.** Default: certify each `(c, i)`
   independently, report per-level coverage and an all-levels-certified fraction.
2. **Primary metric target.** `beta_0` error vs. GT component count per class, as
   the plan's Betti error — confirm vs. a box-count error.
3. **Synthetic P1.5 ground truth.** Renderer component counts, not mask homology.
4. **`ε_cert` scope.** Prune only the detection head + last neck block so the
   Lipschitz product is over ~3 conv layers (`lipschitz.head_layer_lipschitz_bounds`).
5. **Betti-matching metric.** Pure-Python 2-D approximation now
   (`metrics.betti_matching_error`), C++ `nstucki/Betti-matching` only if it is
   measurably too loose.

## Kill criterion (added, `DEVIATIONS.md` D1)

If the one2one class-logit maps are degenerate on the real checkpoint — every
level near-constant, or `beta_0` of `{f >= tau}` uncorrelated with GT component
counts across the calibration set — option A has no meaningful output topology.
Fall back to option B (segmentation head) or C (representation topology, no
certificate). Record it; do not tune `tau`.

## Build order (plan P1, adapted)

| Step | Needs | State |
|---|---|---|
| `superlevel_diagram` + toy tests | nothing | **done** |
| `euler_characteristic_curve` control | nothing | **done** |
| `certificate.certify` / `margin` / `audit` / `coverage` | nothing | **done** |
| `metrics.betti_number_error` / `betti_matching_error` | nothing | **done** |
| `head_maps.ClassLogitMapExtractor` | torch + checkpoint | written, unrun |
| `lipschitz.epsilon_empirical` | 2 checkpoints + WSD calib | written, unrun |
| `lipschitz.head_layer_lipschitz_bounds` | 1 checkpoint | written, unrun |
| `lipschitz.epsilon_certified` (Δφ from mask) | 1 checkpoint + mask spec | **`NotImplementedError`** |
| P1.5 synthetic 300-frame check | renderer | not started |
| coverage-vs-budget curve | pruned checkpoints | blocked |
