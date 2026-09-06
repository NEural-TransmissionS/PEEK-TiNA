# Option B feasibility: masks over WSD boxes

`REPO_MAP.md` B1 gave three ways out of the segmentation/detection mismatch.
Matas picked **A** for the main line and **B on this branch** — "make masks over
boxes if possible". This note is the *if possible*.

## Why B is attractive

If WSD had component masks, the plan's §3 certificate would apply **verbatim**:
one segmentation logit map per class on the full pixel grid, one `tau`, no
per-detection-level interpretation (the wrinkle that branch
`topo-prune/certified-detection` has to argue through in `DEVIATIONS.md` D1.1).
The certificate core in `../topo_prune/` is reused unchanged — only the source of
`f_c` differs.

## The methods, and what each costs

| Method | Needs | Topology quality | Verdict |
|---|---|---|---|
| `boxfill_mask` | nothing | **none** — every component is a solid rectangle, `beta1 == 0` always, `beta0 == box count`. Identical information to the detector's box count. | baseline / fallback only |
| `grabcut_mask` | `opencv` (installed, works on 3.14) | real shapes when the box border is background. WSD spacecraft sit on dark space, so the border assumption usually holds. Recovers gaps between solar cells, truss holes. | **the candidate.** No training, no GPU, no downloads. |
| `sam_mask` | SAM / MobileSAM weights (~40–350 MB), ideally GPU | best boundaries | quality ceiling; use to spot-check GrabCut, or if GrabCut's Betti-drift rate is low |

## The feasibility gate

`scripts/generate_masks.py` records, per image, whether GrabCut's mask has
**different Betti numbers** from the box-fill mask (`betti_drift_vs_boxfill`).

- **GrabCut changes the topology on a meaningful fraction of images** (say > 20%,
  concentrated in `solar` and `antenna`) → the masks carry signal the boxes do
  not. Option B is worth training a segmentation head on. Proceed: masks →
  YOLO26n-seg (or a small UNet) → certificate on its logit map.
- **GrabCut rarely changes the topology, or its masks are unstable** (large
  `grabcut+boxfill` fallback rate) → the pseudo-masks are box-fill with noise.
  Option B adds machinery for nothing; **stay on branch A** and record this here.

This is a `[HUMAN]` decision once the number exists — it needs the WSD export,
which is not on any reachable host yet (`REPO_MAP.md` B2).

## What is built on this branch

| Piece | State |
|---|---|
| `wsd_labels.parse_label_file` | done, tested |
| `box_to_mask.{boxfill,grabcut,sam}_mask` | done; GrabCut + boxfill tested (SAM path lazy, needs weights) |
| `topology_check.betti_numbers` (scipy, no GUDHI needed for binary masks) | done, tested against disk/annulus/two-disks |
| `scripts/generate_masks.py` | written; unrun (no WSD) |
| segmentation head training on the masks | **not started** — gated on the feasibility number above |
| certificate on the seg logit map | reuses `../topo_prune/` unchanged once a seg checkpoint exists |

Test: `make -C seg_masks test` → 12 passed (Python 3.14, opencv 5.0, scipy 1.18).
The GrabCut test demonstrates the mechanism: on a synthetic bright ring it keeps
the central hole (`beta1 = 1`) that box-fill destroys (`beta1 = 0`).
