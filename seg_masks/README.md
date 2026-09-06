# seg_masks — pseudo segmentation masks from WSD boxes (option B)

Branch `topo-prune/box-to-mask`. Parallel to branch `topo-prune/certified-detection`
(option A). See [`../REPO_MAP.md`](../REPO_MAP.md) B1 for the fork and
[`docs/feasibility.md`](docs/feasibility.md) for what this branch is testing.

**Goal.** WSD ships bounding boxes, not masks. Derive component masks from the
boxes so a segmentation head can be trained and the plan's §3 certificate applies
without the per-detection-level reinterpretation that option A needs. The
certificate code itself is reused from `../topo_prune/` unchanged.

**Open question (the reason it's a branch).** Do GrabCut masks carry topology the
boxes don't? `scripts/generate_masks.py` measures it per image
(`betti_drift_vs_boxfill`). Gate and decision rule in `docs/feasibility.md`.
Needs the WSD export — not on any reachable host yet (`../REPO_MAP.md` B2).

## Status

| Piece | State |
|---|---|
| `wsd_labels` — YOLO txt → boxes (WSD + COCO layouts) | done, tested |
| `box_to_mask` — GrabCut / box-fill / SAM(box-prompted) | GrabCut + box-fill done & tested; SAM path lazy (needs weights) |
| `topology_check` — β0/β1 of a binary mask (scipy) | done, tested vs disk / annulus / two-disks |
| `scripts/generate_masks.py` + `feasibility_report.py` | **run on COCO128** — [`docs/feasibility-results.md`](docs/feasibility-results.md) |
| segmentation head on the masks | not started — gated on the WSD feasibility number |

`make -C seg_masks test` → **13 passed** (Python 3.14, opencv 5.0, scipy 1.18).

**Pipeline check (COCO128 stand-in):** GrabCut masks differ topologically from
box-fill on **83%** of images (105/126), 719 β₁ holes total. Real signal; not the
WSD gate number (needs the WSD export — `../REPO_MAP.md` B2). Details +
fallback-rate caveat: [`docs/feasibility-results.md`](docs/feasibility-results.md).

## Layout

```
seg_masks/
  src/seg_masks/{wsd_labels,box_to_mask,topology_check}.py
  scripts/generate_masks.py
  configs/masks.yaml
  docs/feasibility.md
  tests/
```

## Deps beyond the parent

`opencv-python-headless` (GrabCut), `scipy` (already a parent dep). SAM path
optionally wants `ultralytics` SAM support + a checkpoint.
