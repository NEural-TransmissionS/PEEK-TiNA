"""Superlevel-set cubical persistent homology on a 2-D logit map.

The certified object (``DEVIATIONS.md`` D1) is a per-class classification logit
map ``f_c : grid -> R`` from the detector head. The predicted high-confidence
region is the **superlevel set** ``{f_c >= tau}`` for an operating point ``tau``,
so the filtration that matters is the superlevel-set filtration of ``f_c``:
sweep ``tau`` from ``+inf`` downward and watch components appear and merge and
holes appear and fill.

GUDHI's ``CubicalComplex`` computes a *sublevel* (lower-star) filtration, so we
run it on ``-f`` and map every (birth, death) pair back into ``f``-value units.
In the returned diagrams the convention is therefore:

    birth >= death            (a feature is born high, dies lower)
    death == -inf             an essential class (never merges / never fills)

This module deliberately does **not** normalise the field: the certificate needs
distances measured in the model's own logit units (``DEVIATIONS.md`` D1, the
``epsilon`` bound is ``||f - f_M||_inf`` in those units).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_HOMOLOGY_DIMENSIONS: tuple[int, ...] = (0, 1)


@dataclass(frozen=True)
class Diagram:
    """Superlevel-set persistence diagram of one scalar field.

    ``by_dim[k]`` is an ``(n_k, 2)`` float array of ``[birth, death]`` pairs in
    field-value units, ``birth >= death``; an essential class has
    ``death == -inf``.
    """

    by_dim: dict[int, np.ndarray]

    def dimensions(self) -> tuple[int, ...]:
        return tuple(sorted(self.by_dim))

    def intervals(self, dimension: int) -> np.ndarray:
        return self.by_dim[dimension]

    def as_json(self) -> dict[str, list[list[float]]]:
        out: dict[str, list[list[float]]] = {}
        for dimension, pairs in self.by_dim.items():
            out[str(dimension)] = [
                [float(b), (float(d) if np.isfinite(d) else -np.inf)] for b, d in pairs
            ]
        return out


def _validate_field(field: np.ndarray) -> np.ndarray:
    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 2 or values.size == 0:
        raise ValueError(f"logit map must be a non-empty 2-D array, got shape {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("logit map contains non-finite values")
    return values


def superlevel_diagram(
    field: np.ndarray,
    homology_dimensions: tuple[int, ...] = DEFAULT_HOMOLOGY_DIMENSIONS,
) -> Diagram:
    """Return the superlevel-set cubical persistence diagram of ``field``."""
    try:
        import gudhi
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("Install gudhi to compute persistent homology") from exc

    values = _validate_field(field)
    # Superlevel filtration of f == sublevel filtration of -f.
    complex_ = gudhi.CubicalComplex(top_dimensional_cells=-values)
    complex_.compute_persistence(homology_coeff_field=2, min_persistence=0.0)

    by_dim: dict[int, np.ndarray] = {}
    for dimension in homology_dimensions:
        raw = complex_.persistence_intervals_in_dimension(dimension)
        if raw.size == 0:
            by_dim[dimension] = np.empty((0, 2), dtype=np.float64)
            continue
        # raw is in (-f) units with birth < death; map back to f units.
        birth = -raw[:, 0]
        death = np.where(np.isfinite(raw[:, 1]), -raw[:, 1], -np.inf)
        by_dim[dimension] = np.column_stack([birth, death]).astype(np.float64)
    return Diagram(by_dim=by_dim)


def betti_numbers_at(diagram: Diagram, tau: float) -> dict[int, int]:
    """Betti numbers of ``{f >= tau}`` read off a superlevel diagram.

    A dimension-``k`` feature is alive at ``tau`` when it has already been born
    (``birth >= tau``) and has not yet died (``death < tau``). Counting those is
    exactly ``beta_k({f >= tau})``.
    """
    out: dict[int, int] = {}
    for dimension, pairs in diagram.by_dim.items():
        if pairs.size == 0:
            out[dimension] = 0
            continue
        alive = (pairs[:, 0] >= tau) & (pairs[:, 1] < tau)
        out[dimension] = int(np.count_nonzero(alive))
    return out


def euler_characteristic_curve(field: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Euler characteristic of ``{f >= t}`` for each ``t`` in ``thresholds``.

    The linear-time control descriptor of the plan (§2 route 4, P0.5,
    ``DEVIATIONS.md`` D3). Computed directly from cubical cell counts, not from
    homology: with pixels as 2-cells, a lower-dimensional cell enters the
    superlevel set as soon as *any* incident pixel does, so its threshold is the
    max of the incident pixel values. Then

        chi(t) = #{vertices >= t} - #{edges >= t} + #{pixels >= t}

    which is genuinely ``O(pixels + len(thresholds))`` per row of the sweep.
    """
    values = _validate_field(field)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if thresholds.ndim != 1 or thresholds.size == 0:
        raise ValueError("thresholds must be a non-empty 1-D array")

    h, w = values.shape
    # Vertex value = max over up to four incident pixels.
    vert = np.full((h + 1, w + 1), -np.inf)
    for dy in (0, 1):
        for dx in (0, 1):
            sub = vert[dy : dy + h, dx : dx + w]
            np.maximum(sub, values, out=sub)
    # Horizontal edge (between vertically-adjacent... ) values: max over the up
    # to two pixels sharing that edge.
    h_edge = np.full((h + 1, w), -np.inf)  # edges running along x
    for dy in (0, 1):
        sub = h_edge[dy : dy + h, :]
        np.maximum(sub, values, out=sub)
    v_edge = np.full((h, w + 1), -np.inf)  # edges running along y
    for dx in (0, 1):
        sub = v_edge[:, dx : dx + w]
        np.maximum(sub, values, out=sub)

    vert_flat = vert.ravel()
    edge_flat = np.concatenate([h_edge.ravel(), v_edge.ravel()])
    face_flat = values.ravel()

    def _count_ge(sorted_asc: np.ndarray, t: float) -> int:
        # entries >= t in an ascending-sorted array
        return int(sorted_asc.size - np.searchsorted(sorted_asc, t, side="left"))

    vs = np.sort(vert_flat)
    es = np.sort(edge_flat)
    fs = np.sort(face_flat)
    chi = np.empty(thresholds.shape, dtype=np.int64)
    for idx, t in enumerate(thresholds):
        chi[idx] = _count_ge(vs, t) - _count_ge(es, t) + _count_ge(fs, t)
    return chi
