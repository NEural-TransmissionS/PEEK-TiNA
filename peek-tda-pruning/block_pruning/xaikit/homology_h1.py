"""H1 (loop) contribution to a topological saliency map -- the "filled hole" build.

An H1 class of the superlevel filtration is a ring of high field values enclosing
a pocket of lower values: the layer responds strongly *around* a region. This
paints that enclosed pocket, weighted by the loop's persistence (how strong the
ring is relative to the pocket). It uses more of the persistence diagram than the
H0 peak surface -- the 1-dimensional classes, with their spatial extent -- so it
is the closer approximation to "use the full persistent homology".
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .saliency import normalize01

_NEIGHBOURS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _validated(field: np.ndarray, filtration: str) -> np.ndarray:
    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 2 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("field must be a non-empty, finite 2-D array")
    if filtration == "sublevel":
        return -values
    if filtration != "superlevel":
        raise ValueError("filtration must be 'superlevel' or 'sublevel'")
    return values


def h1_pockets(
    field: np.ndarray, filtration: str = "superlevel", *, max_area_frac: float = 0.5
) -> list[tuple[np.ndarray, float]]:
    """``(mask, persistence)`` for every enclosed H1 pocket of the superlevel filtration.

    ``persistence`` = (field value where the ring closes) - (highest field value
    inside the pocket). Pockets that touch the grid border, or cover more than
    ``max_area_frac`` of it, are dropped as not-genuinely-enclosed.
    """
    import gudhi

    values = _validated(field, filtration)
    height, width = values.shape

    cubical = gudhi.CubicalComplex(top_dimensional_cells=-values)
    cubical.compute_persistence(homology_coeff_field=2)
    finite = cubical.cofaces_of_persistence_pairs()[0]
    if len(finite) < 2 or getattr(finite[1], "ndim", 0) != 2 or len(finite[1]) == 0:
        return []

    flat = values.ravel()
    pockets: list[tuple[np.ndarray, float]] = []
    for birth_idx, death_idx in finite[1]:
        t_birth = float(flat[birth_idx])
        persistence = t_birth - float(flat[death_idx])
        if persistence <= 0:
            continue
        below = values < t_birth
        seed = (int(death_idx) // width, int(death_idx) % width)
        mask = np.zeros((height, width), dtype=bool)
        mask[seed] = True
        stack = [seed]
        touches_border = False
        while stack:
            row, col = stack.pop()
            if row in (0, height - 1) or col in (0, width - 1):
                touches_border = True
            for d_row, d_col in _NEIGHBOURS:
                n_row, n_col = row + d_row, col + d_col
                if (
                    0 <= n_row < height
                    and 0 <= n_col < width
                    and below[n_row, n_col]
                    and not mask[n_row, n_col]
                ):
                    mask[n_row, n_col] = True
                    stack.append((n_row, n_col))
        if touches_border or mask.sum() > max_area_frac * mask.size:
            continue
        pockets.append((mask, persistence))
    return pockets


def h1_filled_surface(
    field: np.ndarray,
    filtration: str = "superlevel",
    *,
    sigma_frac: float = 0.06,
    normalize: bool = True,
) -> np.ndarray:
    """Persistence-weighted union of enclosed H1 pockets, Gaussian-smoothed."""
    values = _validated(field, filtration)
    surface = np.zeros_like(values)
    for mask, persistence in h1_pockets(values, "superlevel"):
        surface[mask] += persistence
    surface = gaussian_filter(surface, sigma=max(values.shape) * sigma_frac)
    return normalize01(surface) if normalize else surface
