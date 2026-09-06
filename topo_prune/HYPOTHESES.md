# HYPOTHESES.md

Pre-registration for the `topo_prune` workstream (plan §6). **This file is
committed before any P2 run.** Post-hoc hypotheses are labelled exploratory in
the writeup.

Primary metric: **Betti-number error** of the predicted class-logit superlevel
sets (`DEVIATIONS.md` D1), not mAP. mAP@0.5:0.95 is reported alongside, never
alone.

---

## P1 — certificate soundness

**H0 (falsifiable prediction, plan §3.4).** On every certified `(image, class,
level)` triple, `beta_k(S_c^(i),M) = beta_k(S_c^(i))` for all `k`. Expected
audited violation count: **0**.

- Confirmed → a topology-preservation guarantee for a pruned detector.
- Violations > 0 → the bound / filtration convention / code is wrong. Debug; do
  not tune `tau`.

**H0b (coverage).** Certified coverage is a decreasing function of the pruning
budget, and is > 0 for at least the 10% and 20% budgets under `ε_emp`.

---

## P2 — two-sided intervention (arms per plan §4 `masks.py`)

Written out fully before P2 runs; placeholder until the checkpoint + WSD land.

> **H1.** At matched measured-FLOPs and matched L1 mass, pruning the channels the
> topological criterion calls **important** (`topo-high`) degrades Betti error
> significantly more than pruning the ones it calls **redundant** (`topo-low`),
> and the `topo-high` − `topo-low` gap exceeds the `random` − `topo-low` gap.
>
> **H2.** `cert-min` (minimise `ε_cert`, plan §3.5) achieves lower Betti error
> than `l1-low` and `taylor` at equal measured FLOPs, **before** retraining.
>
> **H3 (control).** If `ecc-low` matches `topo-low` within noise, homology adds
> no discriminative value over the Euler characteristic curve at ~100× the cost —
> and that is the headline finding, not a failure (`DEVIATIONS.md` D3, plan §2
> route 4).

Analysis: 5 seeds/arm, paired across shared seeds, Holm–Bonferroni across arms,
effect sizes reported. Selection and all parameter choices on the calibration
split only.

---

## Kill criteria carried into this workstream

- **D1 (option A viability).** One2one class-logit maps degenerate on the real
  checkpoint → option A has no output topology → fall back to B or C. Record it.
- **P2 (plan §P2).** `topo-high` and `topo-low` statistically indistinguishable
  at matched FLOPs → the score is not difference-making → publishable negative
  result about the method class. Write it up; do not keep tuning.
