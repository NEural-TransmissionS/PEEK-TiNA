"""Thin adapters for the vendored PEEK package; PEEK math lives upstream."""

from __future__ import annotations

from typing import Any

import numpy as np


def to_peek_map(feature_map: Any) -> np.ndarray:
    """Convert one CHW/HWC activation into a map using canonical ``peek.PEEK``.

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
        raise ValueError(f"Expected CHW or HWC activation, got shape {array.shape}")

    # Hook outputs are BCHW/CHW; accept HWC when the last axis is most plausibly channels.
    if array.shape[0] > array.shape[1] and array.shape[0] > array.shape[2]:
        array = np.moveaxis(array, 0, -1)
    return np.asarray(PEEK()(array), dtype=np.float64)
