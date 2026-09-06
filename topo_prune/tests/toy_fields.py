"""Hand-verifiable scalar fields with known homology (``CLAUDE.md``: an 8x8
disk, an annulus, two disjoint disks). Every diagram these produce is written
out explicitly in ``test_filtrations.py``.
"""

from __future__ import annotations

import numpy as np

O = 0.0


def solid_disk() -> np.ndarray:
    """8x8, one connected high plateau. beta0=1, beta1=0 for 0 < tau <= 3."""
    v = 3.0
    return np.array(
        [
            [O, O, O, O, O, O, O, O],
            [O, O, v, v, v, v, O, O],
            [O, v, v, v, v, v, v, O],
            [O, v, v, v, v, v, v, O],
            [O, v, v, v, v, v, v, O],
            [O, v, v, v, v, v, v, O],
            [O, O, v, v, v, v, O, O],
            [O, O, O, O, O, O, O, O],
        ],
        dtype=np.float64,
    )


def annulus() -> np.ndarray:
    """8x8 ring of 5s around a 2x2 hole of 0s.

    H0: one essential bar [5, -inf].
    H1: one bar [5, 0]  (hole fills only when the centre joins at tau=0).
    beta0=1, beta1=1 for 0 < tau <= 5.
    """
    v = 5.0
    return np.array(
        [
            [O, O, O, O, O, O, O, O],
            [O, v, v, v, v, v, v, O],
            [O, v, v, v, v, v, v, O],
            [O, v, v, O, O, v, v, O],
            [O, v, v, O, O, v, v, O],
            [O, v, v, v, v, v, v, O],
            [O, v, v, v, v, v, v, O],
            [O, O, O, O, O, O, O, O],
        ],
        dtype=np.float64,
    )


def two_disks() -> np.ndarray:
    """8x8, two disjoint plateaus at heights 2 (left) and 3 (right).

    H0: essential bar [3, -inf]; finite bar [2, 0] (blocks connect only through
        the background at tau=0).
    beta0=2 for 0 < tau <= 2, beta0=1 for 2 < tau <= 3.
    """
    a, b = 2.0, 3.0
    return np.array(
        [
            [O, O, O, O, O, O, O, O],
            [O, a, a, O, O, b, b, O],
            [O, a, a, O, O, b, b, O],
            [O, a, a, O, O, b, b, O],
            [O, O, O, O, O, O, O, O],
            [O, O, O, O, O, O, O, O],
            [O, O, O, O, O, O, O, O],
            [O, O, O, O, O, O, O, O],
        ],
        dtype=np.float64,
    )


def cone(size: int = 48, radius: float = 18.0, peak: float = 4.0) -> np.ndarray:
    """Smooth single bump: f = peak * max(0, 1 - r/radius). beta0=1, beta1=0."""
    yy, xx = np.mgrid[0:size, 0:size]
    c = (size - 1) / 2.0
    r = np.hypot(yy - c, xx - c)
    return (peak * np.clip(1.0 - r / radius, 0.0, None)).astype(np.float64)
