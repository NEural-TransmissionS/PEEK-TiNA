"""Dependency-light, deterministic PEEK and filtration image export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def export_filtration(peek_map: np.ndarray, output: Path, filtration: str = "superlevel") -> None:
    values = np.asarray(peek_map, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("PEEK map must be a finite 2-D array")
    span = float(values.max() - values.min())
    normalized = np.zeros_like(values) if span == 0 else (values - values.min()) / span
    output.mkdir(parents=True, exist_ok=True)
    # Fixed blue-yellow gradient avoids backend- or matplotlib-version differences.
    red = np.clip(2 * normalized - 0.2, 0, 1)
    green = np.clip(1.6 * normalized, 0, 1)
    blue = np.clip(1.2 - 1.5 * normalized, 0, 1)
    heatmap = (np.stack([red, green, blue], axis=-1) * 255).astype(np.uint8)
    Image.fromarray(heatmap).save(output / "peek-map.png")
    for threshold in np.linspace(0, 1, 5):
        mask = normalized >= threshold if filtration == "superlevel" else normalized <= threshold
        Image.fromarray((mask * 255).astype(np.uint8)).save(
            output / f"filtration-{filtration}-{threshold:.2f}.png"
        )
