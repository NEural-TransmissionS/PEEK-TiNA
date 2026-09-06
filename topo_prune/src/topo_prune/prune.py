"""Structured pruning of the YOLO26 classification head, for the P1 certificate.

Scope (``DEVIATIONS.md`` D1.4): we prune the **input channels of the final
1x1 classification conv** ``one2one_cv3[level][-1]`` -- i.e. we drop channels of
the penultimate head feature. That conv is linear, so a dropped input channel's
contribution to the logit map is removed exactly (``W[:, k] * feat_k -> 0``),
which makes the perturbation clean to bound and keeps the certified propagation
path a single layer.

The producing conv + BN for each dropped channel are zeroed too so the FLOP
saving is real, but that does not change the logit map (the channel is already
gone at the final conv). Selection criteria:

- ``magnitude`` : smallest ``||W[:, k]||_2`` (L2 over output classes)
- ``random``    : seeded random subset
- ``topo``      : placeholder -> raises; the topological criterion is P2

Full dependency-aware removal across the neck is a P2 concern and is not done
here (matches the parent repo, whose ``dependency_engine`` is
``pending_checkpoint_integration``).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

PRUNE_METHODS = ("magnitude", "random")


@dataclass(frozen=True)
class HeadPruneRecord:
    level: int
    kept_channels: tuple[int, ...]
    dropped_channels: tuple[int, ...]
    method: str
    fraction: float

    @property
    def actual_fraction(self) -> float:
        total = len(self.kept_channels) + len(self.dropped_channels)
        return len(self.dropped_channels) / total if total else 0.0


def _resolve_detect(module):
    """``module`` is an nn.Module (DetectionModel) or its ``.model`` Sequential."""
    core = getattr(module, "model", module)
    return core[-1]


def _select_dropped(weight_cols_l2: np.ndarray, fraction: float, method: str, seed: int) -> np.ndarray:
    n = weight_cols_l2.shape[0]
    n_drop = round(fraction * n)
    n_drop = max(0, min(n - 1, n_drop))  # keep at least one channel
    if n_drop == 0:
        return np.empty(0, dtype=int)
    if method == "magnitude":
        return np.sort(np.argsort(weight_cols_l2)[:n_drop])
    if method == "random":
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(n, size=n_drop, replace=False))
    raise ValueError(f"unknown method {method!r}; topo importance is P2, not P1")


def prune_head(detection_module, *, fraction: float, method: str = "magnitude", seed: int = 42):
    """Return ``(pruned_module, [HeadPruneRecord per level])``.

    ``detection_module`` is the ultralytics ``DetectionModel`` nn.Module
    (``YOLO(...).model``), not the ``YOLO`` wrapper (which holds an unpicklable
    predictor lock). ``pruned_module`` is a deep copy; the input is untouched.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("prune_head needs torch + the detector stack") from exc
    if not 0.0 <= fraction < 1.0:
        raise ValueError(f"fraction must be in [0, 1), got {fraction}")

    pruned = copy.deepcopy(detection_module)
    detect = _resolve_detect(pruned)
    records: list[HeadPruneRecord] = []

    for level, branch in enumerate(detect.one2one_cv3):
        final_conv = branch[-1]                     # nn.Conv2d(c3, nc, 1)
        producer = branch[-2][-1]                   # Conv(c3, c3, 1): .conv + .bn
        w = final_conv.weight.detach().cpu().numpy()  # [nc, c3, 1, 1]
        col_l2 = np.sqrt((w[:, :, 0, 0] ** 2).sum(axis=0))  # [c3]
        dropped = _select_dropped(col_l2, fraction, method, seed + level)
        kept = np.array([c for c in range(w.shape[1]) if c not in set(dropped.tolist())])

        with torch.no_grad():
            for k in dropped.tolist():
                # exact: kill channel k's contribution to the logit map
                final_conv.weight[:, k, :, :] = 0.0
                # best-effort FLOP realism on the producing conv (fused or not)
                pconv = getattr(producer, "conv", producer)
                if k < pconv.weight.shape[0]:
                    pconv.weight[k].zero_()
                    if pconv.bias is not None:
                        pconv.bias[k].zero_()
                bn = getattr(producer, "bn", None)
                if bn is not None:
                    bn.weight[k].zero_()
                    bn.bias[k].zero_()
                    bn.running_mean[k].zero_()
                    bn.running_var[k].fill_(1.0)

        records.append(
            HeadPruneRecord(
                level=level,
                kept_channels=tuple(int(c) for c in kept),
                dropped_channels=tuple(int(c) for c in dropped),
                method=method,
                fraction=fraction,
            )
        )
    return pruned, records
