"""Certified topological pruning for the WSD spacecraft-component detector.

Workstream for ``../../TDA/TOPO_PRUNE_SAT_PLAN.md``, retargeted from the plan's
segmentation model to this repo's YOLO26n detector (see ``DEVIATIONS.md`` D1):
the certified object is the per-scale, per-class **classification logit map**
``f_c^(i)`` from the ``Detect`` head, whose superlevel sets carry genuine
homology exactly as a segmentation logit map would.

Detector-free modules (developed and CI-tested on Python 3.14):

- ``filtrations``   superlevel-set cubical persistent homology on a 2-D logit map,
                    plus the linear-time Euler-characteristic-curve control
- ``descriptors``   barcode -> Betti curves, persistence entropy, ECC vector
- ``certificate``   margin test, certified-set logic, falsification audit, coverage
- ``metrics``       Betti-number error, simplified Betti-matching error

Detector-coupled modules (written here, validated later on a GPU host with the
WSD export + checkpoint -- see ``REPO_MAP.md`` B2):

- ``head_maps``     hook the one2one classification head, return ``f_c^(i)``
- ``lipschitz``     epsilon_empirical / epsilon_certified for a pruning mask

The cubical persistent-homology backend is GUDHI, matching the parent
``topoprun`` package; nothing here reimplements the PEEK equation.
"""

from __future__ import annotations

__all__ = [
    "certificate",
    "descriptors",
    "filtrations",
    "metrics",
]

__version__ = "0.0.1"
