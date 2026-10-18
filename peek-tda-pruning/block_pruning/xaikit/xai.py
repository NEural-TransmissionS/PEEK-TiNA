"""Saliency methods for the TDA-vs-CAM comparison on Ultralytics YOLO26.

One grad-enabled forward per image yields, at each hooked neck module:
  * the activation tensor and (optionally) its gradient w.r.t. a class-score
    target, for Grad-CAM / HiRes-CAM;
  * the raw class-score field, for the deletion/insertion faithfulness sweep.

Every method here returns a raw 2-D map at the module's own resolution. The
caller resamples to the common grid and min-max normalizes
(``saliency.resize_map`` / ``saliency.normalize01``).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from .saliency import merge_tree_saliency, peak_surface

NECK_MODULES = (16, 19, 22)


@dataclass(frozen=True)
class LetterboxMeta:
    ratio: float
    pad_w: float
    pad_h: float
    orig_hw: tuple[int, int]

    def map_box(self, x: float, y: float, w: float, h: float) -> tuple[float, float, float, float]:
        """COCO ``[x, y, w, h]`` (original px) -> ``[x1, y1, x2, y2]`` in the 640 frame."""
        x1 = x * self.ratio + self.pad_w
        y1 = y * self.ratio + self.pad_h
        return (x1, y1, (x + w) * self.ratio + self.pad_w, (y + h) * self.ratio + self.pad_h)


class YoloExplainer:
    """Wraps a YOLO26 checkpoint with neck-activation hooks and a class-score target."""

    def __init__(
        self,
        weights: str,
        device: str = "cuda",
        layers: tuple[int, ...] = NECK_MODULES,
        imgsz: int = 640,
    ) -> None:
        from ultralytics import YOLO

        self.device = torch.device(device if torch.cuda.is_available() or "cpu" in device else "cpu")
        self.yolo = YOLO(weights)
        self.model = self.yolo.model.to(self.device).eval()
        for param in self.model.parameters():
            param.requires_grad_(True)
        self.names: dict[int, str] = dict(self.yolo.names)
        self.layers = tuple(layers)
        self.imgsz = imgsz
        self._acts: dict[int, torch.Tensor] = {}
        self._handles = [
            self.model.model[i].register_forward_hook(self._capture(i)) for i in self.layers
        ]

    def _capture(self, index: int):
        def hook(_module, _inputs, output):
            self._acts[index] = output

        return hook

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    # -- preprocessing -------------------------------------------------------
    def preprocess(self, bgr: np.ndarray) -> tuple[torch.Tensor, LetterboxMeta]:
        h0, w0 = bgr.shape[:2]
        ratio = min(self.imgsz / h0, self.imgsz / w0, 1.0)
        new_w, new_h = round(w0 * ratio), round(h0 * ratio)
        pad_w, pad_h = (self.imgsz - new_w) / 2, (self.imgsz - new_h) / 2
        resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        top, left = round(pad_h - 0.1), round(pad_w - 0.1)
        canvas[top : top + new_h, left : left + new_w] = resized
        rgb = canvas[:, :, ::-1].transpose(2, 0, 1).copy()
        tensor = torch.from_numpy(rgb).float().div_(255.0).unsqueeze(0).to(self.device)
        return tensor, LetterboxMeta(ratio, left, top, (h0, w0))

    # -- forward -----------------------------------------------------------
    def class_scores(self, tensor: torch.Tensor) -> torch.Tensor:
        """``(80, 8400)`` raw class-score field from the one2many head; grad-capable."""
        self._acts.clear()
        with torch.enable_grad():  # override any ambient no_grad from ultralytics
            out = self.model(tensor)
        aux = out[1] if isinstance(out, (list, tuple)) and len(out) > 1 else out
        one2many = aux["one2many"] if isinstance(aux, dict) else aux[0]
        scores = one2many["scores"] if isinstance(one2many, dict) else one2many
        return scores[0]

    def activations(self) -> dict[int, torch.Tensor]:
        return {i: self._acts[i] for i in self.layers}

    def grad_maps(self, tensor: torch.Tensor, target_cls: int) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Return ``{module: (activation, gradient)}`` as CHW numpy for one target class."""
        scores = self.class_scores(tensor)
        acts = {i: self._acts[i] for i in self.layers}
        for act in acts.values():
            act.retain_grad()
        self.model.zero_grad(set_to_none=True)
        scores[target_cls].max().backward()
        return {
            i: (act.detach()[0].cpu().numpy(), act.grad[0].cpu().numpy())
            for i, act in acts.items()
        }

    @torch.no_grad()
    def target_score(self, tensor: torch.Tensor, target_cls: int) -> np.ndarray:
        """Max one2many class probability for ``target_cls`` per batch item.

        The head emits logits (unbounded); a sigmoid puts the deletion/insertion
        scalar back on ``[0, 1]`` so curve AUCs are comparable across images.
        Accepts a batch and returns one probability per row.
        """
        _preds, aux = self.model(tensor)
        logits = aux["one2many"]["scores"][:, target_cls].amax(dim=1)
        return torch.sigmoid(logits).cpu().numpy()

    def blurred_baseline(self, tensor: torch.Tensor, sigma: float) -> torch.Tensor:
        """Gaussian-blurred copy of a preprocessed tensor -- the deletion/insertion base."""
        chw = tensor.detach().cpu().numpy()[0].transpose(1, 2, 0)
        blur = cv2.GaussianBlur(chw, (0, 0), sigmaX=sigma, sigmaY=sigma)
        return torch.from_numpy(blur.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def score_batch(self, batch: torch.Tensor, target_cls: int) -> np.ndarray:
        return self.target_score(batch, target_cls)

    @torch.no_grad()
    def detect(self, tensor: torch.Tensor) -> np.ndarray:
        """``(n, 6)`` ``[x1, y1, x2, y2, conf, cls]`` end-to-end detections in the 640 frame."""
        preds, _aux = self.model(tensor)
        return preds[0].cpu().numpy()


# -- individual maps -------------------------------------------------------
def activation_norm(act: np.ndarray) -> np.ndarray:
    """Channel L2 norm -> ``(H, W)``. Also the scalar field the TDA maps run on."""
    return np.linalg.norm(np.asarray(act, dtype=np.float64), axis=0)


def tda_saliency(act: np.ndarray, filtration: str = "superlevel") -> np.ndarray:
    """Primary TDA map: persistence-weighted peak surface of the channel L2 field."""
    return peak_surface(activation_norm(act), filtration, normalize=False)


def tda_merge_saliency(act: np.ndarray, filtration: str = "superlevel") -> np.ndarray:
    """Ablation TDA map: naive per-pixel basin prominence."""
    return merge_tree_saliency(activation_norm(act), filtration, normalize=False)


def tda_h0_uniform_saliency(act: np.ndarray, filtration: str = "superlevel") -> np.ndarray:
    """Ablation: peak surface with every local maximum weighted equally (no persistence)."""
    return peak_surface(activation_norm(act), filtration, weight="uniform", normalize=False)


def tda_h1_saliency(act: np.ndarray, filtration: str = "superlevel") -> np.ndarray:
    """H1-only map: persistence-weighted enclosed loops (filled holes) of the L2 field."""
    from .homology_h1 import h1_filled_surface

    return h1_filled_surface(activation_norm(act), filtration, normalize=False)


def tda_h01_saliency(
    act: np.ndarray, filtration: str = "superlevel", *, alpha: float = 0.5
) -> np.ndarray:
    """Combined H0 + H1 map: ``alpha`` * (peak surface) + (1 - alpha) * (filled holes).

    Each part is normalized to [0, 1] before mixing so the weighting is meaningful.
    With no enclosed loops this reduces to the H0 peak surface.
    """
    from .homology_h1 import h1_filled_surface
    from .saliency import normalize01

    field = activation_norm(act)
    h0 = normalize01(peak_surface(field, filtration, normalize=False))
    h1 = normalize01(h1_filled_surface(field, filtration, normalize=False))
    return alpha * h0 + (1.0 - alpha) * h1


def eigen_cam(act: np.ndarray) -> np.ndarray:
    """Projection of activations onto their first principal component (Muhammad & Yeasin).

    No ReLU (EigenCAM keeps the full projection); the first-component sign is
    arbitrary, so it is oriented to correlate positively with the channel L2
    norm -- a deterministic, method-agnostic tie-break.
    """
    channels = np.asarray(act, dtype=np.float64)
    flat = channels.reshape(channels.shape[0], -1).T  # (HW, C)
    flat = flat - flat.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(flat, full_matrices=False)
    projection = flat @ vt[0]
    reference = np.linalg.norm(channels, axis=0).ravel()
    if np.dot(projection - projection.mean(), reference - reference.mean()) < 0:
        projection = -projection
    return projection.reshape(channels.shape[1:])


def grad_cam(act: np.ndarray, grad: np.ndarray) -> np.ndarray:
    weights = np.asarray(grad, dtype=np.float64).mean(axis=(1, 2), keepdims=True)
    return np.clip((weights * np.asarray(act, dtype=np.float64)).sum(axis=0), 0.0, None)


def hires_cam(act: np.ndarray, grad: np.ndarray) -> np.ndarray:
    product = np.asarray(grad, dtype=np.float64) * np.asarray(act, dtype=np.float64)
    return np.clip(product.sum(axis=0), 0.0, None)


def center_prior(shape: tuple[int, int]) -> np.ndarray:
    rows, cols = shape
    yy = np.linspace(-1.0, 1.0, rows)[:, None]
    xx = np.linspace(-1.0, 1.0, cols)[None, :]
    return np.exp(-(xx**2 + yy**2) / (2 * 0.5**2))


def random_saliency(shape: tuple[int, int], seed: int) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    return gaussian_filter(rng.standard_normal(shape), sigma=max(shape) / 16.0)
