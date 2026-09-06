"""Barcode -> vector summaries (plan §4 ``descriptors.py``).

Kept deliberately small: the certificate itself only needs the raw diagram and
its endpoints. These summaries are for the P0 hygiene ablations (regressing the
score on cheap statistics) and for the ECC control.
"""

from __future__ import annotations

import numpy as np

from .filtrations import Diagram, betti_numbers_at, euler_characteristic_curve

__all__ = [
    "betti_curve",
    "ecc_vector",
    "persistence_entropy",
    "total_persistence",
]


def betti_curve(diagram: Diagram, thresholds: np.ndarray, dimension: int) -> np.ndarray:
    """``beta_dimension({f >= t})`` for each ``t`` in ``thresholds``."""
    thresholds = np.asarray(thresholds, dtype=np.float64)
    return np.array(
        [betti_numbers_at(diagram, float(t)).get(dimension, 0) for t in thresholds],
        dtype=np.int64,
    )


def _lifetimes(diagram: Diagram, dimension: int) -> np.ndarray:
    pairs = diagram.by_dim.get(dimension, np.empty((0, 2)))
    if pairs.size == 0:
        return np.empty(0)
    finite = pairs[np.isfinite(pairs[:, 1])]
    if finite.size == 0:
        return np.empty(0)
    return finite[:, 0] - finite[:, 1]


def total_persistence(diagram: Diagram, dimension: int, order: float = 1.0) -> float:
    """Sum of (finite) bar lifetimes raised to ``order``."""
    lifetimes = _lifetimes(diagram, dimension)
    if lifetimes.size == 0:
        return 0.0
    return float(np.sum(lifetimes**order))


def persistence_entropy(diagram: Diagram, dimension: int) -> float:
    """Shannon entropy of the normalised finite-bar lifetime distribution."""
    lifetimes = _lifetimes(diagram, dimension)
    if lifetimes.size == 0:
        return 0.0
    total = float(np.sum(lifetimes))
    if total <= 0.0:
        return 0.0
    p = lifetimes / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def ecc_vector(field: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Euler characteristic curve as a fixed-length vector (the cheap control)."""
    return euler_characteristic_curve(field, thresholds)
