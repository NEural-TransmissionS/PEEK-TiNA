# DEVIATIONS.md

Every departure from `../../TDA/TOPO_PRUNE_SAT_PLAN.md` (the plan), recorded **before** the
code that implements it, per `CLAUDE.md` ("Any departure from the plan goes in
`DEVIATIONS.md` **before** the code changes, with the reasoning").

Status legend: `PROPOSED` (awaiting Matas) · `ACCEPTED` (Matas signed off) · `IMPLEMENTED`.

---

## D0 — Package lives in a self-contained subtree, not at repo root · IMPLEMENTED

**Plan says:** §4 shows `topo_prune/*.py` plus `configs/`, `experiments/`, `RESULTS.md`,
`HYPOTHESES.md`, `DEVIATIONS.md` as repo-root siblings.

**What we did:** everything is under `topo_prune/` — `topo_prune/topo_prune/` (the import
package), `topo_prune/configs/`, `topo_prune/tests/`, `topo_prune/docs/`,
`topo_prune/experiments/`, and the four markdown ledgers.

**Why:** the repo already has a root-level `configs/` and its own protocol docs, and the
established precedent for adding a parallel workstream here is the `ood_monitor/` subtree
(branch `ood-monitor-phase1`, "direction C") — self-contained package + configs + docs +
Makefile + README under one directory. `CLAUDE.md` says "leave existing training code
alone". A subtree does that; scattering files across the root does not. Import name stays
`topo_prune` exactly as the plan and its Appendix B phase prompts write it.

---

## D1 — The certificate is retargeted from segmentation to detection · PROPOSED · `[HUMAN]`

**This is the fork from `REPO_MAP.md` §B1, resolved by Matas as "option A".** It is the
largest single deviation in the project and everything in P1/P3 depends on the specifics
below. Matas chose the direction; the specifics here still need his explicit sign-off
before P1 code is written against them.

**Plan says (§1, §3):** the model is a semantic segmentation network; the certified object
is a per-class logit map `f_c : Ω → ℝ` on the *full pixel grid*; the predicted mask is
`S_c = {f_c ≥ τ}`; the falsifiable claim is about `β_k(S_c)`.

**Reality (`REPO_MAP.md` Q1–Q2):** the model is a YOLO26n / YOLOv5n **detector**. No logit
map on the pixel grid exists. The `Detect` head output is `[1,300,6]` decoded boxes.

**What we retarget to.** The YOLO26 `Detect` head *does* expose a spatial per-class logit
map, before any decode:

- `ultralytics/nn/modules/head.py:143-144` — for each of the `nl = 3` detection levels
  `i`, `cls_head[i](x[i])` produces a tensor of shape `[B, nc, H_i, W_i]`. This is exactly
  a stack of `nc` scalar fields on a grid. For 640-px input the grids are 80×80, 40×40,
  20×20 (strides 8/16/32). `nc = 4` for WSD (`antenna`, `body`, `solar`, `thruster`).
- `_inference` (`head.py:174`) applies `.sigmoid()` to these to get probabilities. So the
  **logit map** is the pre-sigmoid `cls_head[i](x[i])`, and `τ` in logit space corresponds
  to a confidence operating point `p` via `τ = logit(p) = ln(p / (1−p))`.
- For YOLO26 `end2end` is `True` by default (`head.py:129`); the deployed predictions come
  from the **one2one** head, `one2one_cv3[i]`. The certificate hooks the one2one classification
  branch. (`one2many_cv3` is available too and is used only for the training-time loss.)

Define, per class `c` and level `i`:

```
f_c^{(i)} : grid_i → ℝ        the one2one class-c logit map at detection level i
S_c^{(i)} = { p : f_c^{(i)}(p) ≥ τ }
D(f_c^{(i)})                  superlevel-set cubical persistence diagram  (unchanged from plan §3)
```

Every piece of plan §3 then transfers **verbatim**: Cohen-Steiner–Edelsbrunner–Harer
stability gives `d_B(D(f_c^{(i)}), D(f_c^{(i),M})) ≤ ‖f_c^{(i)} − f_c^{(i),M}‖_∞` with
Lipschitz constant 1; the margin test (§3.3), the audit (§3.4), certified coverage (§3.4),
and the certified pruning objective (§3.5) are unchanged in form.

**What changes in interpretation (needs Matas sign-off):**

1. **Multi-scale.** There are three logit maps per class, not one. Proposed: **certify each
   `(c, i)` pair independently** and report per-level coverage plus an "all-levels-certified"
   fraction. An image counts as certified only if every `(c, i)` it contains certifies.
   Rationale: fusing scales (upsample + max) introduces an interpolation Lipschitz factor
   and destroys the exact-grid stability bound. Alternative if per-level is too weak:
   certify only the level that owns each GT object under YOLO's assignment. `[HUMAN]`

2. **What "Betti number of the prediction" means.** `β₀(S_c^{(i)})` = number of connected
   high-confidence blobs for class `c` at level `i` ≈ number of predicted instances;
   `β₁` = enclosed holes (rare, but a truss/antenna ring is a real β₁≈1 case). The plan's
   operational failure — "one solar array predicted as three components" — survives as
   `β₀: 1 → 3`. This is the primary metric (Betti error), same as the plan. `[HUMAN]` to
   confirm this is the right target vs. box-count error.

3. **Ground-truth topology for the P1.5 synthetic check.** The plan renders masks with
   known homology. Here the natural GT is *component count per class from the WSD boxes*
   (β₀), plus a hand-labelled β₁ for the few ring-shaped antennas. The synthetic testbed
   (plan Route 8 / P1.5) still renders spacecraft with dial-able occlusion, but the
   ground-truth signal is "how many disjoint components of class `c` are visible", read
   off the renderer, not a mask's homology. `[HUMAN]`

4. **`ε_cert` propagation path (plan §3.2a).** The plan already flags the analytic bound as
   "the single largest technical risk" and its mitigation 1 is "scope the certificate to
   the decoder head only". For a detector the natural short path is even shorter and
   cleaner: hook the input to `one2one_cv3[i]` (the neck feature `x[i]`, i.e. module
   outputs 16/19/22) and propagate the perturbation only through the **2–3 conv layers of
   the classification head** to its output. Pruning is then also scoped to those head
   layers plus, optionally, the neck blocks feeding them. This makes the spectral-norm
   product a product over ~3 layers, not over an encoder-decoder. `[HUMAN]` to confirm the
   certified claim may be scoped to "prune the detection-head + last neck block".

5. **Betti-matching metric.** Plan Appendix C leans on `nstucki/Betti-matching` (a C++
   extension, `[VERIFY]` in the plan). Proposed: implement plain **Betti number error**
   and **Betti-matching error** on 2-D grids directly against GUDHI (already a dependency,
   already builds on this environment) and only pull in the C++ package if the pure-Python
   2-D version is a measured bottleneck. `[HUMAN]` — low stakes.

**Kill criterion for D1 (added):** if the one2one class-logit maps are degenerate for the
WSD checkpoint — e.g. every level is near-constant, or `β₀` of the thresholded map bears no
relation to the GT component count across the calibration set — then option A does not have
a meaningful output topology and we fall back to option B (segmentation head, separate
branch) or option C (representation topology, no certificate). Record the negative result;
do not tune `τ` to manufacture structure.

**D1 status update (2026-09-06, local run on COCO-pretrained `yolo26n.pt`).** The kill
criterion did **not** fire on the stand-in checkpoint: the one2one class-logit maps are
non-degenerate (superlevel sets have varied `β₀`/`β₁`; pruning changes them on 40–127 of
555 audited triples depending on budget — `docs/p1-results.md` E1). Still must be re-checked
on the real WSD checkpoint against GT component counts. The **`ε_cert` propagation path
(D1.4)** was implemented as a single-layer closed form (final classification conv only),
not the full spectral-norm product through the neck — see D6.

---

## D2 — TDA backend: GUDHI, not `cripser` · PROPOSED

**Plan says:** §3.1 / Appendix C — "Compute with Cubical Ripser (`cripser`) or GUDHI";
"benchmark both once, pick one, record the choice".

**What we do:** use **GUDHI** and defer the `cripser` benchmark. The existing repo already
standardized on `gudhi.CubicalComplex` (`src/topoprun/topology.py:60`), it builds on the
Python 3.14 environment here (`cripser` wheels are less reliable), and re-using the same
backend keeps the new certificate diagrams directly comparable to the repo's existing PEEK
topology numbers. The `cripser` speed benchmark moves to a P1 "nice to have" and is only
done if cubical PH is a wall-clock problem at 80×80 grids (it will not be).

---

## D3 — Euler Characteristic Curve control uses a direct grid computation · PROPOSED

**Plan says:** §2 Route 4 and P0.5 — ECC as a linear-time control, `[VERIFY]` no library
named. Plan Appendix C cites Röell & Rieck DECT (an ICLR 2024 differentiable-ECT paper)
"as the basis of the Euler control".

**What we do:** compute the ECC directly as the alternating sum of `k`-cell counts of the
cubical complex across a filtration sweep (vectorised NumPy, genuinely `O(pixels · thresholds)`).
DECT / `torch-topological` are only needed if the ECC ever has to be *differentiable*
(it does not before P4). Recorded so the ECC number is reproducible from our code alone.

---

## D4 — Python 3.14 + fresh dependency pins for the topo_prune subtree · IMPLEMENTED

**Plan / repo says:** repo `pyproject.toml` pins `requires-python = ">=3.10"`, and the
`ood_monitor` precedent targets 3.10.

**Reality:** the only interpreter on this workstation is 3.14.5. The **topological core**
(numpy 2.5, scipy 1.18, gudhi 3.13, pytest 9) installs and runs cleanly on 3.14 in an
isolated venv.

**Update (2026-09-06).** The **detector stack also runs on 3.14 CPU**: `torch 2.14.0+cpu`,
`torchvision`, and `ultralytics 8.4.9` install from the pinned `third_party/ultralytics`
submodule, and `yolo26n.pt` (COCO) + COCO128 download fine. So E1 (capture → prune → audit)
ran here after all — on the COCO stand-in, not WSD. No CUDA GPU on this box; CPU inference
of `yolo26n` at 640 px is ~50 ms/image, fine for the 128-image experiments. `PEEK` is still
not installed (option A doesn't need it — it hooks the head directly, not PEEK maps).

**What we do:** `topo_prune/` declares its own minimal deps. `test_prune` uses
`pytest.importorskip` for torch/ultralytics; the rest of the suite is detector-free and
runs anywhere.

---

## D5 — Blocked-on-host items are stubbed with explicit `NotImplementedError`, not faked · IMPLEMENTED

Consistent with `ood_monitor`'s "No real detector output yet; capture is blocked on a GPU
host" stance and `CLAUDE.md` ("A recorded negative/blocked result is a deliverable. Do not
tune until the result looks better"). Functions that need the checkpoint or WSD raise with
a message naming exactly what is missing. `RESULTS.md` carries a BLOCKED row, not a fake
number.

---

## D6 — `ε_cert` is a single-layer, calibration-based bound (not the full §3.2a product) · IMPLEMENTED

**Plan §3.2a:** `ε_cert(M) = (∏_{k>ℓ} L_k) · ‖Δφ_ℓ‖_∞` — a product of per-layer Lipschitz
constants from the pruned layer to the output.

**What E1 actually computes:** we prune the **input channels of the final 1×1
classification conv** (`DEVIATIONS.md` D1.4), so the propagation path is that one linear
layer and the product collapses to a closed form:

```
ε_cert(c, level) = Σ_{k ∈ dropped} |W_final[c, level, k]| · M_k ,
   M_k = max over the calibration set and space of |penultimate_feature_k|
```

Two honesty points, both required by the plan (§3.2, §7 "Report `ε_cert` looseness
honestly"), stated in `docs/p1-results.md` and the run manifest:

1. **It is calibration-based, not worst-case.** `M_k` is a max over the calibration split.
   A true worst-case bound needs an input-domain bound on the penultimate activation
   (interval bound propagation from the image through the backbone+neck+head-prefix) — not
   done. So `ε_cert` here is the plan's §3.2 *mitigation 2* ("high-confidence bound"), not a
   certificate in the strict sense. Measured looseness vs `ε_emp`: ~2–5×.
2. **`lipschitz.epsilon_certified(model, mask, ...)`** (the general form that turns an
   arbitrary `mask` into `‖Δφ‖_∞` and propagates a spectral-norm product) still raises
   `NotImplementedError`. E1 bypasses it with the closed form above, which only covers
   final-conv-input pruning.

Widening the prune scope (into neck blocks) or claiming a strict certificate both require
finishing the general `epsilon_certified` with IBP. Tracked as the top P1 follow-up.

---

## Open `[HUMAN]` questions carried from REPO_MAP.md (not yet deviations, but gating)

- **P0.1 reference number.** What concrete artifact are "the SCITECH numbers" the plan's
  P0.1 must reproduce? The draft's YOLOv5n 0.808/0.570 row is known-unreproducible
  (`README.md:143`). Proposed: restate P0.1's exit criterion against the reproducible v71
  baseline table (`docs/baseline-nano-results.csv`), i.e. YOLO26n-pretrained 0.564 val /
  0.312 test mAP@.5:.95.
- **Identity-disjoint split** for P3.4 distribution-shift — does per-spacecraft-identity
  metadata exist for WSD v71?
- **Data + weights transfer** — WSD v71 Roboflow export + `yolo26n_pretrained_seed42`
  checkpoint (and ideally the tuned `cls0.75_dfl1.75` checkpoint) need to reach a host that
  can run them.
