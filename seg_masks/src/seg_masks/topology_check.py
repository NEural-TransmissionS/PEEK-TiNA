"""beta0 / beta1 of a binary mask -- to check generated masks carry real topology.

If GrabCut masks have the same Betti numbers as box-fill masks on every image
(beta1 == 0 everywhere, beta0 == box count), then option B buys nothing over the
detector retarget and the extra machinery is not worth it. This is the
feasibility gate in ``docs/feasibility.md``.

Uses ``scipy.ndimage`` (already a dependency) for connected components and the
Euler number; no GUDHI needed for a plain binary mask.
"""

from __future__ import annotations

import numpy as np


def betti_numbers(mask: np.ndarray, connectivity: int = 1) -> dict[str, int]:
    """``beta0`` (components) and ``beta1`` (holes) of a 2-D binary mask.

    ``beta0`` from connected-component labelling; ``beta1 = beta0 - chi`` where
    ``chi`` is the Euler number (``scipy.ndimage.label`` + the standard identity
    for a planar set).
    """
    try:
        from scipy import ndimage
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("topology_check needs scipy") from exc

    binary = np.asarray(mask) > 0
    if binary.ndim != 2:
        raise ValueError("mask must be 2-D")
    structure = ndimage.generate_binary_structure(2, connectivity)
    _, beta0 = ndimage.label(binary, structure=structure)
    if beta0 == 0:
        return {"beta0": 0, "beta1": 0}
    # Euler number with matching connectivity: label the complement with the
    # opposite connectivity (standard trick for 2-D digital topology).
    opp = 2 if connectivity == 1 else 1
    comp_structure = ndimage.generate_binary_structure(2, opp)
    padded = np.pad(binary, 1, constant_values=False)
    _, bg_components = ndimage.label(~padded, structure=comp_structure)
    holes = bg_components - 1  # one background component is the outside
    return {"beta0": int(beta0), "beta1": int(max(0, holes))}


def compare_topology(a: np.ndarray, b: np.ndarray, connectivity: int = 1) -> dict[str, int]:
    """Absolute Betti difference between two masks (e.g. grabcut vs boxfill)."""
    ba, bb = betti_numbers(a, connectivity), betti_numbers(b, connectivity)
    return {
        "d_beta0": abs(ba["beta0"] - bb["beta0"]),
        "d_beta1": abs(ba["beta1"] - bb["beta1"]),
    }
