"""The epsilon bound on ``||f_c^(i) - f_c^(i),M||_inf`` (plan §3.2).

Two routes, always both (plan §3.2):

- ``epsilon_empirical``  measured ``max`` over the calibration set. Not a
  certificate; the tight reference that tells us how loose the analytic bound is.
- ``epsilon_certified``  a Lipschitz-product upper bound propagated from the
  pruned layer to the head output.

``DEVIATIONS.md`` D1.4 scopes the certified route to the **detection head + last
neck block**, so the propagation path is 2-3 conv layers, not an encoder-decoder
-- the plan's §3.2 mitigation 1 ("scope the certificate to the decoder head
only"), specialised to a detector.

Both routes need the checkpoint (and the empirical one needs WSD too), which is
not on this workstation (``REPO_MAP.md`` B2). The layer-Lipschitz helper is
written so it can run the moment a checkpoint lands; the end-to-end functions
raise with the specific missing input until then.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EpsilonReport:
    """Per (class, level) epsilon values and the looseness factor (plan §3.2)."""

    epsilon_certified: dict[tuple[int, int], float]
    epsilon_empirical: dict[tuple[int, int], float]

    def looseness(self) -> dict[tuple[int, int], float]:
        out: dict[tuple[int, int], float] = {}
        for key, cert in self.epsilon_certified.items():
            emp = self.epsilon_empirical.get(key)
            out[key] = float("inf") if not emp else cert / emp
        return out


def conv_spectral_norm(weight: np.ndarray, n_iter: int = 50) -> float:
    """Upper bound on the Lipschitz constant of a conv layer via power iteration
    on the reshaped kernel (``out_ch x (in_ch*kh*kw)``). This is the cheap,
    standard over-estimate; a tighter Sedghi-style bound is a P1 refinement.
    """
    w = np.asarray(weight, dtype=np.float64)
    mat = w.reshape(w.shape[0], -1)
    if mat.size == 0:
        return 0.0
    v = np.random.default_rng(0).standard_normal(mat.shape[1])
    v /= np.linalg.norm(v) + 1e-12
    for _ in range(n_iter):
        u = mat @ v
        u /= np.linalg.norm(u) + 1e-12
        v = mat.T @ u
        v /= np.linalg.norm(v) + 1e-12
    return float(np.linalg.norm(mat @ v))


def head_layer_lipschitz_bounds(detect_module, level: int) -> list[float]:
    """Per-layer Lipschitz upper bounds for ``one2one_cv3[level]``.

    Conv -> spectral norm; BatchNorm -> ``max |gamma| / sqrt(var + eps)``;
    SiLU -> ``<= 1.1`` (its global Lipschitz constant); ReLU/identity -> 1.
    Runs on CPU from a loaded checkpoint, no data needed.
    """
    try:
        from torch import nn
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("head_layer_lipschitz_bounds needs torch") from exc

    branch = detect_module.one2one_cv3[level]
    bounds: list[float] = []
    for m in branch.modules():
        if isinstance(m, nn.Conv2d):
            bounds.append(conv_spectral_norm(m.weight.detach().cpu().numpy()))
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            gamma = m.weight.detach().cpu().numpy()
            var = (
                m.running_var.detach().cpu().numpy()
                if hasattr(m, "running_var")
                else np.ones_like(gamma)
            )
            bounds.append(float(np.max(np.abs(gamma) / np.sqrt(var + m.eps))))
        elif isinstance(m, nn.SiLU):
            bounds.append(1.1)
    return bounds


def epsilon_certified(model, mask, *, level: int, delta_phi_inf: float) -> float:
    """Lipschitz-product bound (plan §3.2a) for one detection level.

    ``ε_cert = (prod of per-layer L_k for k after the pruned layer) * ||Δφ||_inf``.
    ``delta_phi_inf`` is the sup-norm change at the head input caused by ``mask``;
    computing it exactly from ``mask`` and the neck weights is the remaining
    piece and needs the checkpoint.
    """
    raise NotImplementedError(
        "epsilon_certified needs the selected YOLO26n checkpoint to (a) read the "
        "head conv weights for the Lipschitz product and (b) turn `mask` into "
        "||Δφ||_inf at the head input. Blocked on REPO_MAP.md B2. "
        "head_layer_lipschitz_bounds() already works from a checkpoint alone."
    )


def epsilon_empirical(reference_maps, pruned_maps) -> dict[tuple[int, int], float]:
    """``max`` over the calibration set of ``||f_c^(i) - f_c^(i),M||_inf``.

    ``*_maps`` are iterables of ``head_maps.ClassLogitMaps`` for the same images
    in the same order, from the unpruned and pruned models.
    """
    worst: dict[tuple[int, int], float] = {}
    for ref, pruned in zip(reference_maps, pruned_maps, strict=True):
        for level, ref_arr in ref.maps.items():
            pruned_arr = pruned.maps[level]
            if ref_arr.shape != pruned_arr.shape:
                raise ValueError(
                    f"level {level}: shape mismatch {ref_arr.shape} vs {pruned_arr.shape}"
                )
            diff = np.abs(ref_arr - pruned_arr)
            for class_index in range(ref_arr.shape[0]):
                key = (class_index, level)
                worst[key] = max(worst.get(key, 0.0), float(diff[class_index].max()))
    if not worst:
        raise ValueError("no calibration maps supplied")
    return worst
