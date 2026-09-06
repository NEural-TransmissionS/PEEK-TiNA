"""Box -> mask, three methods with increasing quality / cost.

- ``grabcut_mask``  : OpenCV GrabCut, one pass per box. No training, no GPU,
                      no downloads. Works well when the box border is mostly
                      background -- true for WSD spacecraft on dark space.
- ``sam_mask``      : SAM / SAM2, box-prompted. Best boundaries; needs weights.
                      Lazy import; raises with instructions if unavailable.
- ``boxfill_mask``  : filled rectangle. Degenerate (every component is a solid
                      rect, so beta1 == 0 always) -- kept only as a baseline and
                      as the fallback when GrabCut collapses to empty.

All return a ``uint8`` label image, ``0`` background and ``class_id + 1`` inside
components, same H×W as the input image. Overlaps resolve smaller-box-wins so a
thruster inside a body box keeps its own label.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .wsd_labels import Box


@dataclass(frozen=True)
class MaskResult:
    label_map: np.ndarray          # uint8 HxW, 0 = bg, class_id+1 = fg
    method: str
    per_box_pixels: list[int]      # foreground pixel count per input box, in order

    @property
    def coverage(self) -> float:
        return float(np.count_nonzero(self.label_map) / self.label_map.size)


def _paint_order(boxes: list[Box]) -> list[int]:
    """Largest box first so a smaller box painted later wins any overlap
    (a thruster inside a body box keeps its own label)."""
    areas = [b.width * b.height for b in boxes]
    return sorted(range(len(boxes)), key=lambda i: areas[i], reverse=True)


def boxfill_mask(image_shape: tuple[int, int], boxes: list[Box]) -> MaskResult:
    h, w = image_shape
    label_map = np.zeros((h, w), dtype=np.uint8)
    per_box = [0] * len(boxes)
    for i in _paint_order(boxes):
        b = boxes[i]
        label_map[b.y0 : b.y1, b.x0 : b.x1] = b.class_id + 1
        per_box[i] = b.width * b.height
    return MaskResult(label_map=label_map, method="boxfill", per_box_pixels=per_box)


def grabcut_mask(
    image: np.ndarray,
    boxes: list[Box],
    *,
    iterations: int = 5,
    margin: int = 8,
    min_foreground_fraction: float = 0.02,
) -> MaskResult:
    """GrabCut once per box on a padded crop, composited largest-box-first.

    A box whose GrabCut result covers less than ``min_foreground_fraction`` of
    the box falls back to a filled rectangle for that box (recorded in
    ``method`` as ``grabcut+boxfill``).
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("grabcut_mask needs opencv (opencv-python-headless)") from exc

    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    h, w = image.shape[:2]
    label_map = np.zeros((h, w), dtype=np.uint8)
    per_box = [0] * len(boxes)
    used_fallback = False

    for i in _paint_order(boxes):
        b = boxes[i]
        cx0, cy0 = max(0, b.x0 - margin), max(0, b.y0 - margin)
        cx1, cy1 = min(w, b.x1 + margin), min(h, b.y1 + margin)
        crop = np.ascontiguousarray(image[cy0:cy1, cx0:cx1])
        if crop.shape[0] < 3 or crop.shape[1] < 3:
            label_map[b.y0 : b.y1, b.x0 : b.x1] = b.class_id + 1
            per_box[i] = b.width * b.height
            used_fallback = True
            continue
        gc = np.zeros(crop.shape[:2], dtype=np.uint8)
        rect = (b.x0 - cx0, b.y0 - cy0, b.width, b.height)
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(crop, gc, rect, bgd, fgd, iterations, cv2.GC_INIT_WITH_RECT)
        except cv2.error:
            gc[:] = 0
        fg = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
        box_area = max(1, b.width * b.height)
        if fg.sum() < min_foreground_fraction * box_area:
            fg[:] = 0
            fg[b.y0 - cy0 : b.y1 - cy0, b.x0 - cx0 : b.x1 - cx0] = 1
            used_fallback = True
        region = label_map[cy0:cy1, cx0:cx1]
        region[fg.astype(bool)] = b.class_id + 1
        per_box[i] = int(fg.sum())

    method = "grabcut+boxfill" if used_fallback else "grabcut"
    return MaskResult(label_map=label_map, method=method, per_box_pixels=per_box)


def sam_mask(
    image: np.ndarray,
    boxes: list[Box],
    *,
    weights: str | None = None,
    device: str = "cpu",
) -> MaskResult:
    """Box-prompted SAM. Lazy import; ``weights`` defaults to ultralytics'
    auto-downloaded ``mobile_sam.pt`` if the ultralytics SAM API is present.
    """
    try:
        from ultralytics import SAM
    except ImportError as exc:  # pragma: no cover - optional path
        raise RuntimeError(
            "sam_mask needs `ultralytics` with SAM support. "
            "Install the detector stack (make setup) or pass weights= a SAM checkpoint."
        ) from exc

    model = SAM(weights or "mobile_sam.pt")
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    h, w = image.shape[:2]
    label_map = np.zeros((h, w), dtype=np.uint8)
    per_box = [0] * len(boxes)
    xyxy = [[b.x0, b.y0, b.x1, b.y1] for b in boxes]
    if not xyxy:
        return MaskResult(label_map=label_map, method="sam", per_box_pixels=per_box)
    results = model(image, bboxes=xyxy, device=device, verbose=False)
    masks = results[0].masks
    data = masks.data.cpu().numpy() if masks is not None else np.zeros((len(boxes), h, w))
    for i in _paint_order(boxes):
        m = data[i] > 0.5 if i < len(data) else np.zeros((h, w), bool)
        label_map[m] = boxes[i].class_id + 1
        per_box[i] = int(m.sum())
    return MaskResult(label_map=label_map, method="sam", per_box_pixels=per_box)


def to_binary(label_map: np.ndarray, class_id: int) -> np.ndarray:
    """Binary mask for one class from a label image."""
    return (label_map == class_id + 1).astype(np.uint8)
