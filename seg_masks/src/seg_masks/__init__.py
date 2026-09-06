"""Pseudo segmentation masks from WSD bounding boxes (plan option B).

``REPO_MAP.md`` B1 / ``../topo_prune/DEVIATIONS.md`` D1: the plan's certificate
wants a segmentation model with a per-class logit map. WSD ships boxes, not
masks. This subtree is the "option B" experiment -- derive component masks from
the boxes so a real segmentation head can be trained and the plan's §3
certificate applies verbatim (no multi-scale interpretation needed, unlike the
detector retarget on branch ``topo-prune/certified-detection``).

Feasibility is the open question, hence the branch: see ``docs/feasibility.md``.

- ``wsd_labels``     read YOLO-format WSD labels -> per-class boxes
- ``box_to_mask``    box -> mask: GrabCut (no training), SAM (box-prompted,
                     optional weights), or box-fill (degenerate baseline)
- ``topology_check`` beta0 / beta1 of a generated mask, to sanity-check that the
                     masks carry real topology and not just filled rectangles
"""

from __future__ import annotations

__all__ = ["box_to_mask", "topology_check", "wsd_labels"]
__version__ = "0.0.1"
