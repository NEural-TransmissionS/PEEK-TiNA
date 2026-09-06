"""Topological accuracy metrics (plan §4 ``metrics.py``, primary metric §6).

Detector-free and implemented now:

- ``betti_number_error``   |beta_k(pred) - beta_k(gt)| at an operating point
- ``betti_matching_error`` simplified induced-matching count (``DEVIATIONS.md`` D1.5)

Detector-coupled and deferred to the GPU host (raise ``NotImplementedError``
with the reason, per ``DEVIATIONS.md`` D5):

- ``miou`` / ``per_class_iou`` / ``cl_dice``  -- need decoded predictions + masks
- ``flops`` / ``params`` / ``peak_activation_memory`` / ``int8_latency``
  -- ``scripts/profile_detector.py`` already covers these for the parent repo;
     this workstream will call into it rather than re-measure.
"""

from __future__ import annotations

import numpy as np

from .filtrations import Diagram, betti_numbers_at

__all__ = [
    "betti_matching_error",
    "betti_number_error",
]


def betti_number_error(
    prediction: Diagram,
    reference: Diagram,
    tau: float,
    dimensions: tuple[int, ...] = (0, 1),
) -> dict[str, int]:
    """Absolute Betti-number difference at ``tau``, per dimension and total.

    ``reference`` is the ground-truth topology (from GT component counts in the
    synthetic testbed, or from the unpruned model in a pruning comparison).
    """
    pred = betti_numbers_at(prediction, tau)
    ref = betti_numbers_at(reference, tau)
    out: dict[str, int] = {}
    total = 0
    for k in dimensions:
        diff = abs(int(pred.get(k, 0)) - int(ref.get(k, 0)))
        out[f"beta{k}"] = diff
        total += diff
    out["total"] = total
    return out


def _alive_bars(diagram: Diagram, tau: float, dimension: int) -> np.ndarray:
    pairs = diagram.by_dim.get(dimension, np.empty((0, 2)))
    if pairs.size == 0:
        return np.empty((0, 2))
    alive = (pairs[:, 0] >= tau) & (pairs[:, 1] < tau)
    return pairs[alive]


def betti_matching_error(
    prediction: Diagram,
    reference: Diagram,
    tau: float,
    dimensions: tuple[int, ...] = (0, 1),
    tolerance: float = 0.0,
) -> dict[str, int]:
    """Simplified Betti-matching error at ``tau`` (``DEVIATIONS.md`` D1.5).

    Full induced matching (Stucki et al., ICML 2023) needs the spatial location
    of each critical cell, which the raw diagram does not carry. Here we
    approximate it per dimension: optimally match the bars alive at ``tau`` in
    ``prediction`` and ``reference`` on L-inf ``(birth, death)`` distance, then
    count the bars on either side that end up unmatched (their nearest partner is
    further than ``tolerance``, or the counts differ). This equals
    ``|n_pred - n_ref|`` whenever the alive-bar counts differ and refines it with
    a location-in-barcode check when they agree.

    Replace with ``nstucki/Betti-matching`` if this proves too loose in P1.
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("Install scipy for Betti-matching error") from exc

    out: dict[str, int] = {}
    total = 0
    for k in dimensions:
        a = _alive_bars(prediction, tau, k)
        b = _alive_bars(reference, tau, k)
        n, m = len(a), len(b)
        if n == 0 or m == 0:
            out[f"beta{k}"] = max(n, m)
            total += max(n, m)
            continue
        # essential deaths (-inf) share one finite sentinel so they match each other.
        span = float(np.max(np.abs(np.concatenate([a[:, 0], b[:, 0]]) - tau))) + 1.0
        af = np.where(np.isfinite(a), a, tau - span)
        bf = np.where(np.isfinite(b), b, tau - span)
        cost = np.max(np.abs(af[:, None, :] - bf[None, :, :]), axis=2)
        rows, cols = linear_sum_assignment(cost)
        matched = int(np.count_nonzero(cost[rows, cols] <= tolerance))
        error = (n - matched) + (m - matched)
        out[f"beta{k}"] = error
        total += error
    out["total"] = total
    return out
