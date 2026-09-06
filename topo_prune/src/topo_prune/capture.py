"""Run a detector over images and collect the per-scale class-logit maps
(and, optionally, the penultimate head feature for the certified epsilon bound).

Needs torch + the detector stack. Deterministic: one image at a time, eval mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .head_maps import ClassLogitMapExtractor, _resolve_detect


@dataclass(frozen=True)
class Capture:
    image: str
    logit_maps: dict[int, np.ndarray]          # level -> [nc, H, W]
    penultimate: dict[int, np.ndarray] | None   # level -> [c3, H, W] (final-conv input)


def _penultimate_hooks(detect, store: dict[int, np.ndarray]):
    handles = []
    for level, branch in enumerate(detect.one2one_cv3):
        final_conv = branch[-1]

        def _pre_hook(_m, inputs, _level=level):
            store[_level] = inputs[0].detach().float().cpu().numpy()[0]

        handles.append(final_conv.register_forward_pre_hook(_pre_hook))
    return handles


def capture_images(
    model,
    image_paths: list[Path],
    *,
    imgsz: int = 640,
    device: str = "cpu",
    with_penultimate: bool = False,
) -> list[Capture]:
    detect = _resolve_detect(model)
    out: list[Capture] = []
    with ClassLogitMapExtractor(model) as ex:
        pen_store: dict[int, np.ndarray] = {}
        pen_handles = _penultimate_hooks(detect, pen_store) if with_penultimate else []
        try:
            for path in image_paths:
                ex.clear()
                pen_store.clear()
                model.predict(source=str(path), imgsz=imgsz, device=device, batch=1, verbose=False)
                maps = ex.take().maps
                out.append(
                    Capture(
                        image=Path(path).name,
                        logit_maps={k: v for k, v in maps.items()},
                        penultimate={k: v for k, v in pen_store.items()} if with_penultimate else None,
                    )
                )
        finally:
            for h in pen_handles:
                h.remove()
    return out


def linf_gap(a: dict[int, np.ndarray], b: dict[int, np.ndarray]) -> dict[tuple[int, int], float]:
    """Per (class, level) ``max |a - b|``."""
    out: dict[tuple[int, int], float] = {}
    for level, arr_a in a.items():
        arr_b = b[level]
        diff = np.abs(arr_a - arr_b)
        for c in range(arr_a.shape[0]):
            out[(c, level)] = float(diff[c].max())
    return out
