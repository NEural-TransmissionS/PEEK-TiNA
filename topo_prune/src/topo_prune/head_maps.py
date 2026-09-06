"""Hook the detector's per-scale, per-class classification logit maps.

``DEVIATIONS.md`` D1: the certified object ``f_c^(i)`` is the pre-sigmoid output
of the YOLO26 ``Detect`` head's classification branch at detection level ``i``.
For an ``end2end`` model (YOLO26 default) the deployed predictions come from the
``one2one`` head, so we hook ``one2one_cv3[i]``; ``ultralytics/nn/modules/head.py``
computes ``scores = cls_head[i](x[i])`` with shape ``[B, nc, H_i, W_i]`` and later
applies ``.sigmoid()`` in ``_inference`` -- so the hook output is exactly the
logit map, and the operating point in logit units is ``tau = logit(p_conf)``.

Nothing in this file runs on the current workstation (no torch / no checkpoint,
``REPO_MAP.md`` B2). It is written against the pinned ``third_party/ultralytics``
so it is ready for the GPU host. Import is lazy and guarded.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np


def logit(p: float) -> float:
    """Operating-point confidence -> threshold in logit units."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {p}")
    return math.log(p / (1.0 - p))


@dataclass(frozen=True)
class ClassLogitMaps:
    """One image's logit maps: ``maps[level]`` is ``[nc, H_level, W_level]``."""

    maps: dict[int, np.ndarray]
    strides: dict[int, int]
    class_names: tuple[str, ...]

    def field(self, class_index: int, level: int) -> np.ndarray:
        return self.maps[level][class_index]


def _resolve_detect(model):
    """Return the ``Detect`` module of a loaded ultralytics ``YOLO``."""
    core = getattr(model, "model", model)
    core = getattr(core, "model", core)
    return core[-1]


class ClassLogitMapExtractor:
    """Forward-hook the one2one classification head of a YOLO26 detector.

    Usage (on the GPU host)::

        from ultralytics import YOLO
        model = YOLO(weights)
        with ClassLogitMapExtractor(model) as ex:
            model.predict(source=img, imgsz=640, device=0, verbose=False)
            logit_maps = ex.take()
    """

    def __init__(self, model):
        try:
            import torch  # noqa: F401
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError("head_maps needs torch + the detector stack") from exc
        self._model = model
        self._detect = _resolve_detect(model)
        self._handles: list = []
        self._cache: dict[int, np.ndarray] = {}
        cls_head = getattr(self._detect, "one2one_cv3", None)
        if cls_head is None:
            raise RuntimeError(
                "Detect head has no one2one_cv3; this model is not end2end. "
                "Hook self._detect.cv3 instead and document it in DEVIATIONS.md."
            )
        self._cls_head = cls_head
        strides = list(getattr(self._detect, "stride", []))
        self._strides = {i: int(s) for i, s in enumerate(strides)} or {0: 8, 1: 16, 2: 32}
        names = getattr(model, "names", None) or getattr(self._detect, "names", None)
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names)]
        self._names = tuple(names) if names else tuple(
            f"class_{i}" for i in range(int(getattr(self._detect, "nc", 0)))
        )

    def _make_hook(self, level: int):
        def _hook(_module, _inputs, output):
            self._cache[level] = output.detach().float().cpu().numpy()[0]
        return _hook

    def __enter__(self) -> ClassLogitMapExtractor:  # noqa: PYI034
        for level, branch in enumerate(self._cls_head):
            self._handles.append(branch.register_forward_hook(self._make_hook(level)))
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def clear(self) -> None:
        self._cache.clear()

    def take(self) -> ClassLogitMaps:
        if set(self._cache) != set(range(len(self._cls_head))):
            raise RuntimeError(
                f"missing head activations: have {sorted(self._cache)}, "
                f"expected {list(range(len(self._cls_head)))}"
            )
        maps = {level: np.ascontiguousarray(arr) for level, arr in self._cache.items()}
        self.clear()
        return ClassLogitMaps(maps=maps, strides=dict(self._strides), class_names=self._names)


@contextmanager
def class_logit_maps(model) -> Iterator[ClassLogitMapExtractor]:
    with ClassLogitMapExtractor(model) as extractor:
        yield extractor
