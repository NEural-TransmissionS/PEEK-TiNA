"""Thin adapters for the vendored PEEK package; PEEK math lives upstream."""

from __future__ import annotations

from typing import Any

import numpy as np


def to_peek_map(feature_map: Any) -> np.ndarray:
    """Convert one CHW activation into a map using canonical ``peek.PEEK``.

    ``feature_map`` is always a raw hook output (BCHW or CHW): every caller in
    this project draws from ``LatentExtractor.cache``, which forwards
    ``nn.Module`` outputs verbatim in PyTorch's channel-first convention.
    The channel axis is therefore always moved from position 0 to the end
    unconditionally, never inferred from the shape. A shape-based guess (e.g.
    "move it only if dim 0 is the largest") silently misidentifies the axes
    whenever the channel count is not larger than both spatial dimensions,
    which is the common case for early/wide layers.

    This project intentionally does not duplicate the PEEK equation. Install the
    pinned submodule with ``pip install -e third_party/PEEK``.
    """
    try:
        from peek.core import PEEK
    except ImportError as exc:
        raise RuntimeError("PEEK is not installed; run `pip install -e third_party/PEEK`") from exc

    if hasattr(feature_map, "detach"):
        feature_map = feature_map.detach().cpu().numpy()
    array = np.asarray(feature_map)
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError("Expected a single activation, not a batch")
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"Expected a CHW activation, got shape {array.shape}")

    array = np.moveaxis(array, 0, -1)
    return np.asarray(PEEK()(array), dtype=np.float64)
