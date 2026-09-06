"""Synthetic logit-map fields with *known* superlevel-set homology.

Plan P1.5 / route 8: a testbed where ground-truth topology is exact, to validate
``certificate.audit`` independent of any detector. A "scene" is a scalar field on
a grid standing in for a class-logit map ``f_c^(i)``; its superlevel set
``{f >= tau}`` at the scene's operating point has a component count and hole
count we set by construction.

- ``n_panels`` bright bars with a controllable ``gap`` -> beta_0 = n_panels
  (gap open) or 1 (gap closed)
- ``ring`` adds an annulus -> beta_1 += 1

Perturbations model what pruning does to the map:

- ``smooth_perturbation``   low-frequency random field, scaled to a target
  L-inf norm -- can lift a saddle over tau or push a peak under it
- ``bridge_perturbation``   a targeted bump that *deliberately* merges two
  components, for the "does the audit catch an intended change" test
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Scene:
    field: np.ndarray
    tau: float
    beta0: int
    beta1: int


def render_scene(
    *,
    n_panels: int = 2,
    gap: bool = True,
    ring: bool = False,
    size: int = 64,
    peak: float = 6.0,
    background: float = -4.0,
    tau: float = 0.0,
    seed: int = 0,
) -> Scene:
    """A logit-map scene with known ``beta0`` / ``beta1`` at ``tau``.

    Panels live in the lower band, an optional ring in the upper band -- disjoint,
    so Betti numbers add: ``beta0 = (n_panels if gap else 1) + ring``,
    ``beta1 = ring``.
    """
    rng = np.random.default_rng(seed)
    field = np.full((size, size), background, dtype=np.float64)
    m = max(2, size // 12)

    panel_top = size // 2 + m if ring else m
    panel_bot = size - m
    span = size - 2 * m
    if gap:
        gap_w = max(2, span // (2 * n_panels + 1))
        panel_w = (span - (n_panels - 1) * gap_w) // n_panels
    else:
        panel_w, gap_w = span // n_panels, 0
    for i in range(n_panels):
        x0 = m + i * (panel_w + gap_w)
        x1 = x0 + panel_w
        field[panel_top:panel_bot, x0:x1] = peak + rng.uniform(-0.3, 0.3)
    if not gap:
        field[panel_top:panel_bot, m : m + n_panels * panel_w] = peak + rng.uniform(-0.3, 0.3)

    beta0 = (1 if not gap else n_panels)
    beta1 = 0

    if ring:
        yy, xx = np.mgrid[0:size, 0:size]
        cy, cx = size // 4, size // 2
        r = np.hypot(yy - cy, xx - cx)
        r_out, r_in = size * 0.19, size * 0.09
        annulus = (r <= r_out) & (r >= r_in)
        field = np.where(annulus, peak + 0.5, field)
        beta0 += 1
        beta1 += 1

    field += rng.uniform(-0.4, 0.4, size=field.shape)  # sub-threshold texture
    return Scene(field=field, tau=tau, beta0=beta0, beta1=beta1)


def smooth_perturbation(shape: tuple[int, int], linf: float, seed: int, scale: int = 8) -> np.ndarray:
    """Low-frequency random field rescaled to exactly ``linf`` in sup-norm."""
    rng = np.random.default_rng(seed)
    h, w = shape
    low = rng.standard_normal((max(2, h // scale), max(2, w // scale)))
    # bilinear upsample without cv2
    ys = np.linspace(0, low.shape[0] - 1, h)
    xs = np.linspace(0, low.shape[1] - 1, w)
    y0 = np.floor(ys).astype(int).clip(0, low.shape[0] - 2)
    x0 = np.floor(xs).astype(int).clip(0, low.shape[1] - 2)
    fy = (ys - y0)[:, None]
    fx = (xs - x0)[None, :]
    top = low[y0][:, x0] * (1 - fx) + low[y0][:, x0 + 1] * fx
    bot = low[y0 + 1][:, x0] * (1 - fx) + low[y0 + 1][:, x0 + 1] * fx
    field = top * (1 - fy) + bot * fy
    peak = np.abs(field).max()
    return field * (linf / peak) if peak > 0 else field


def bridge_perturbation(shape: tuple[int, int], amplitude: float, band: int = 4) -> np.ndarray:
    """A positive horizontal band across the full width -- connects vertical panels
    into one component (a deliberate ``beta0`` change for the audit test)."""
    h, w = shape
    pert = np.zeros((h, w), dtype=np.float64)
    r0 = 3 * h // 4 - band // 2  # inside the panel band for every scene variant
    pert[r0 : r0 + max(1, band), :] = amplitude
    return pert
