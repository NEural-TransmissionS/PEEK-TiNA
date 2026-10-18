"""Topological saliency maps from scalar activation fields.

The topology layer (``topology.py``) summarizes a PEEK map as persistence
diagrams and scalars. For an explainability comparison against CAM methods we
instead need a spatial attribution map on the same grid.

``peak_surface`` is the primary construction: run the 0-th persistence (merge
tree) of a superlevel filtration, then place a persistence-weighted Gaussian at
each local maximum and sum -- a persistence-weighted sum of localized features,
the usual shape of "topological saliency" in the literature. ``merge_tree_
saliency`` is the naive per-pixel alternative kept as an ablation: it assigns
every pixel the prominence of the basin it is absorbed into, which floods the
background with the essential class's full-range value.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, zoom


def normalize01(field: np.ndarray) -> np.ndarray:
    """Min-max a finite 2-D array to ``[0, 1]``; a constant field becomes zeros."""
    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 2 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("field must be a non-empty, finite 2-D array")
    low, high = float(values.min()), float(values.max())
    return np.zeros_like(values) if high == low else (values - low) / (high - low)


def resize_map(field: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Bilinearly resample a 2-D map to ``size`` (rows, cols)."""
    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("field must be a non-empty 2-D array")
    rows, cols = size
    factors = (rows / values.shape[0], cols / values.shape[1])
    out = zoom(values, factors, order=1, mode="nearest", grid_mode=True)
    out = out[:rows, :cols]
    if out.shape != (rows, cols):
        padded = np.zeros((rows, cols), dtype=out.dtype)
        padded[: out.shape[0], : out.shape[1]] = out
        out = padded
    return out


def _superlevel_merge_tree(
    values: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, int, float]]]:
    """0-th persistence of the superlevel filtration via a union-find sweep.

    Returns ``(born_persistence, peaks)`` where ``born_persistence`` stamps each
    cell with the persistence of the component it was born into, and ``peaks`` is
    ``(row, col, persistence)`` for every local maximum (the seed of a component),
    the global maximum included with ``birth - global_min``.
    """
    height, width = values.shape
    flat = values.ravel()
    order = np.argsort(flat, kind="stable")[::-1]

    parent = np.full(flat.size, -1, dtype=np.int64)
    added = np.zeros(flat.size, dtype=bool)
    birth: dict[int, float] = {}
    seed: dict[int, int] = {}
    members: dict[int, list[int]] = {}
    born_persistence = np.zeros(flat.size, dtype=np.float64)
    peaks: list[tuple[int, int, float]] = []

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = int(parent[root])
        while parent[node] != root:
            parent[node], node = root, int(parent[node])
        return root

    for raw in order:
        idx = int(raw)
        parent[idx] = idx
        added[idx] = True
        row, col = divmod(idx, width)

        neighbours: set[int] = set()
        if row > 0 and added[idx - width]:
            neighbours.add(find(idx - width))
        if row < height - 1 and added[idx + width]:
            neighbours.add(find(idx + width))
        if col > 0 and added[idx - 1]:
            neighbours.add(find(idx - 1))
        if col < width - 1 and added[idx + 1]:
            neighbours.add(find(idx + 1))

        level = float(flat[idx])
        if not neighbours:
            birth[idx] = level
            seed[idx] = idx
            members[idx] = [idx]
            continue

        elder = max(neighbours, key=lambda root: birth[root])
        for root in neighbours:
            if root == elder:
                continue
            persistence = birth[root] - level
            for cell in members.pop(root):
                born_persistence[cell] = persistence
            peaks.append((*divmod(seed[root], width), persistence))
            parent[root] = elder
        parent[idx] = elder
        members[elder].append(idx)

    global_min = float(flat.min())
    for root, cells in members.items():
        persistence = birth[root] - global_min
        for cell in cells:
            born_persistence[cell] = persistence
        peaks.append((*divmod(seed[root], width), persistence))

    return born_persistence.reshape(height, width), peaks


def _validated_field(field: np.ndarray, filtration: str) -> np.ndarray:
    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 2 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("field must be a non-empty, finite 2-D array")
    if filtration == "sublevel":
        return -values
    if filtration != "superlevel":
        raise ValueError("filtration must be 'superlevel' or 'sublevel'")
    return values


def local_max_persistence(
    field: np.ndarray, filtration: str = "superlevel"
) -> np.ndarray:
    """``(k, 3)`` array of ``[row, col, persistence]`` for every local maximum."""
    _, peaks = _superlevel_merge_tree(_validated_field(field, filtration))
    return np.array(peaks, dtype=np.float64).reshape(-1, 3)


def peak_surface(
    field: np.ndarray,
    filtration: str = "superlevel",
    *,
    sigma_frac: float = 0.06,
    weight: str = "persistence",
    normalize: bool = True,
) -> np.ndarray:
    """Sum of Gaussians, one per local maximum.

    Each local maximum contributes a Gaussian bump at its location; ``sigma`` is
    ``sigma_frac * max(H, W)``. With ``weight="persistence"`` the bump is scaled
    by the maximum's 0-th persistence (its prominence); with ``weight="uniform"``
    every maximum contributes equally -- the ablation that isolates whether the
    persistence weighting matters at all.
    """
    if weight not in ("persistence", "uniform"):
        raise ValueError("weight must be 'persistence' or 'uniform'")
    values = _validated_field(field, filtration)
    _, peaks = _superlevel_merge_tree(values)
    surface = np.zeros_like(values)
    for row, col, persistence in peaks:
        surface[int(row), int(col)] += persistence if weight == "persistence" else 1.0
    surface = gaussian_filter(surface, sigma=max(values.shape) * sigma_frac)
    return normalize01(surface) if normalize else surface


def merge_tree_saliency(
    field: np.ndarray,
    filtration: str = "superlevel",
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Naive per-pixel 0-th persistence (ablation).

    Sweeps the threshold from high to low; each cell is stamped with the
    prominence of the basin it is absorbed into. The one component containing the
    global maximum never dies and takes ``birth - global_min``, so most of the
    background inherits the maximum value -- which is why ``peak_surface`` is the
    primary construction.
    """
    values = _validated_field(field, filtration)
    born_persistence, _ = _superlevel_merge_tree(values)
    if normalize:
        peak = float(born_persistence.max())
        born_persistence = (
            born_persistence / peak if peak > 0 else np.zeros_like(born_persistence)
        )
    return born_persistence
