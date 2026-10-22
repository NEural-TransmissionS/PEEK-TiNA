#!/usr/bin/env python3
"""Core logic for the low-persistence pruning test.

Ranks the four latter-half ``C3k2`` blocks of a YOLO26 detector by how much
H0 + H1 persistence each block's per-image PEEK map carries (``basis="peek"``,
the default; the archived ``basis="activation"`` reads the ``|activation|_2``
field instead). Measured on the min-max-normalised map, so blocks are compared on
topological *shape*, not raw magnitude. Then "cuts" the ``k`` lowest-ranked
blocks and reports the resulting GFLOPs.

Hypothesis under test: a block whose salience field is *diffuse* (low
persistence) carries little localised signal and can be cut with small accuracy
loss; a block whose field is *concentrated* (high persistence) is spatially
decisive and should be kept. With four candidates and ``k <= 3`` the single
highest-persistence block is always spared -- on these models that is almost
always module 16, the P3/8 small-object head.

"Cut" = replace the block's YAML entry with a 1x1 ``Conv`` that emits the same
channel count (so every downstream ``from`` index and channel width is
unchanged) but drops the block's bottleneck stack. This goes through the normal
``YOLO(yaml)`` path, so training behaves exactly as for the full model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from xaikit.homology_h1 import h1_pockets
from xaikit.saliency import local_max_persistence, normalize01
from xaikit.xai import YoloExplainer

BASE_YAML = HERE / "yolo26.yaml"

# The four head C3k2 blocks: 13 (P4 top-down), 16 (P3/8 out), 19 (P4 out),
# 22 (P5 out). The up-sample / concat nodes carry no parameters; the stride-2
# Convs change resolution -- neither is a useful cut target.
CANDIDATES: tuple[int, ...] = (13, 16, 19, 22)
# "all" adds the four backbone C3k2 blocks: 2 (P2/4), 4 (P3/8), 6 (P4/16), 8 (P5/32).
SCOPES: dict[str, tuple[int, ...]] = {"head": CANDIDATES, "all": (2, 4, 6, 8, *CANDIDATES)}


def ranking_path(dataset: str, size: str, basis: str = "peek", scope: str = "head", tag: str = "") -> Path:
    """ranking/ (activation) or ranking_<basis>/, with a _<scope> suffix beyond the head blocks
    and an optional _<tag> (e.g. _norm for the map-size-normalised rankings)."""
    rank_dir = "ranking" if basis == "activation" else f"ranking_{basis}"
    stem = f"{dataset}_{size}" + ("" if scope == "head" else f"_{scope}") + (f"_{tag}" if tag else "")
    return HERE / rank_dir / f"{stem}.json"


# --------------------------------------------------------------------------- #
# per-block persistence ranking                                              #
# --------------------------------------------------------------------------- #
def _l2_field(act) -> np.ndarray:
    a = act[0] if isinstance(act, (list, tuple)) else act
    a = a.detach().float().cpu().numpy()
    a = a[0] if a.ndim == 4 else a
    return np.linalg.norm(a, axis=0)


def _peek_field(act) -> np.ndarray:
    """Canonical PEEK map of the block output (PEEK-TiNA ``topoprun.peek_adapter``)."""
    sys.path.insert(0, str(HERE.parent / "src"))
    from topoprun.peek_adapter import to_peek_map

    return to_peek_map(act[0] if isinstance(act, (list, tuple)) else act)


FIELDS = {"activation": _l2_field, "peek": _peek_field}


def _metrics(field: np.ndarray) -> tuple[float, float]:
    """(secondary-peak H0 prominence, H1 pocket persistence) of a [0, 1] field."""
    peaks = local_max_persistence(field)
    pers = np.sort(peaks[:, 2])[::-1] if len(peaks) else np.zeros(1)
    return float(pers[1:].sum()), float(sum(p for _, p in h1_pockets(field, max_area_frac=0.6)))


def _pool(field: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    return F.adaptive_avg_pool2d(torch.from_numpy(np.ascontiguousarray(field, dtype=np.float32))[None, None],
                                 shape)[0, 0].numpy()


def _scores(h0: dict[int, float], h1: dict[int, float], candidates) -> dict:
    def _norm(d: dict[int, float]) -> dict[int, float]:
        lo, hi = min(d.values()), max(d.values())
        return {i: (v - lo) / (hi - lo) if hi > lo else 0.0 for i, v in d.items()}

    n0, n1 = _norm(h0), _norm(h1)
    combined = {i: round(0.5 * n0[i] + 0.5 * n1[i], 4) for i in candidates}
    return {
        "h0_secondary_sum": {str(i): round(v, 4) for i, v in h0.items()},
        "h1_sum": {str(i): round(v, 4) for i, v in h1.items()},
        "combined": {str(i): combined[i] for i in candidates},
        "cut_order_ascending": sorted(candidates, key=lambda i: combined[i]),  # lowest first
    }


def rank_layers(
    weights: str,
    val_img_dir: Path,
    n_images: int = 40,
    candidates: tuple[int, ...] = CANDIDATES,
    basis: str = "peek",
    norms: bool = False,
) -> dict:
    """Mean (max H0 persistence) and (total H1 persistence) per candidate block
    over ``n_images`` validation images, plus the ascending cut order. ``basis``
    picks the map the topology is read from: the ``|activation|_2`` field or the
    block's PEEK map.

    ``norms`` adds two map-size-normalised rankings from the same forward passes:
    ``grid`` average-pools every block's map to the coarsest candidate's grid before
    the topology, and ``null`` divides each mean by the same metric on a pixel-shuffled
    copy of the block's own map (same values and size, no spatial structure)."""
    if norms:
        return _rank_with_norms(weights, val_img_dir, n_images, candidates, basis)
    to_field = FIELDS[basis]
    explainer = YoloExplainer(weights, device="cuda", layers=tuple(candidates))
    # evenly spaced over the sorted list, so class-prefixed filenames (WSD) don't
    # put every ranking image in one class. (The shapes rankings from 2026-09-03
    # used the first ``n_images`` instead.)
    files = sorted(p for p in val_img_dir.iterdir() if p.suffix.lower() in {".png", ".jpg"})
    files = files[:: max(1, len(files) // n_images)][:n_images]
    if not files:
        raise SystemExit(f"no val images under {val_img_dir}")

    # H0: total prominence of the *secondary* peaks (the single global-max
    # component spans the whole normalised range, so it is dropped -- what
    # discriminates a concentrated field from a diffuse one is how much
    # prominence sits in the peaks below it). H1: total pocket persistence.
    h0_sec: dict[int, list] = {i: [] for i in candidates}
    h1_sum: dict[int, list] = {i: [] for i in candidates}
    shapes: dict[int, list] = {}
    for path in files:
        tensor, _ = explainer.preprocess(cv2.imread(str(path)))
        explainer.class_scores(tensor)
        for i in candidates:
            field = normalize01(to_field(explainer._acts[i]))
            shapes.setdefault(i, list(field.shape))
            peaks = local_max_persistence(field)
            pers = np.sort(peaks[:, 2])[::-1] if len(peaks) else np.zeros(1)
            h0_sec[i].append(float(pers[1:].sum()))
            h1_sum[i].append(float(sum(p for _, p in h1_pockets(field, max_area_frac=0.6))))
    explainer.close()

    def _mean(d: dict[int, list]) -> dict[int, float]:
        return {i: float(np.mean(v)) if v else 0.0 for i, v in d.items()}

    def _norm(d: dict[int, float]) -> dict[int, float]:
        lo, hi = min(d.values()), max(d.values())
        return {i: (v - lo) / (hi - lo) if hi > lo else 0.0 for i, v in d.items()}

    h0s, h1s = _mean(h0_sec), _mean(h1_sum)
    n0, n1 = _norm(h0s), _norm(h1s)
    combined = {i: round(0.5 * n0[i] + 0.5 * n1[i], 4) for i in candidates}
    order = sorted(candidates, key=lambda i: combined[i])  # lowest persistence first

    return {
        "weights": Path(weights).name,
        "basis": basis,
        "n_images": len(files),
        "candidates": list(candidates),
        "field_shape": {str(i): s for i, s in shapes.items()},  # map resolution per block
        "h0_secondary_sum": {str(i): round(v, 4) for i, v in h0s.items()},
        "h1_sum": {str(i): round(v, 4) for i, v in h1s.items()},
        "combined": {str(i): combined[i] for i in candidates},
        "cut_order_ascending": order,
    }


def _rank_with_norms(weights, val_img_dir, n_images, candidates, basis) -> dict:
    """rank_layers(norms=True): raw, common-grid and shuffled-null rankings in one pass."""
    to_field = FIELDS[basis]
    explainer = YoloExplainer(weights, device="cuda", layers=tuple(candidates))
    files = sorted(p for p in val_img_dir.iterdir() if p.suffix.lower() in {".png", ".jpg"})
    files = files[:: max(1, len(files) // n_images)][:n_images]
    if not files:
        raise SystemExit(f"no val images under {val_img_dir}")
    rng = np.random.default_rng(0)
    acc = {v: {i: ([], []) for i in candidates} for v in ("raw", "grid", "shuffled")}
    shapes: dict[int, list] = {}
    grid = None
    for path in files:
        tensor, _ = explainer.preprocess(cv2.imread(str(path)))
        explainer.class_scores(tensor)
        maps = {i: to_field(explainer._acts[i]) for i in candidates}
        grid = grid or min((m.shape for m in maps.values()), key=lambda s: s[0] * s[1])
        for i, m in maps.items():
            shapes.setdefault(i, list(m.shape))
            raw = normalize01(m)
            for v, field in (("raw", raw), ("grid", normalize01(_pool(m, grid))),
                             ("shuffled", rng.permutation(raw.ravel()).reshape(raw.shape))):
                h0, h1 = _metrics(field)
                acc[v][i][0].append(h0)
                acc[v][i][1].append(h1)
    explainer.close()

    mean = {v: ({i: float(np.mean(a[i][0])) for i in candidates},
                {i: float(np.mean(a[i][1])) for i in candidates}) for v, a in acc.items()}
    ratio = tuple({i: mean["raw"][k][i] / max(mean["shuffled"][k][i], 1e-12) for i in candidates}
                  for k in (0, 1))
    return {
        "weights": Path(weights).name,
        "basis": basis,
        "n_images": len(files),
        "candidates": list(candidates),
        "field_shape": {str(i): s for i, s in shapes.items()},
        **_scores(*mean["raw"], candidates),
        "grid": {"shape": list(grid), **_scores(*mean["grid"], candidates)},
        "null": {"shuffled_h0_secondary_sum": {str(i): round(v, 4) for i, v in mean["shuffled"][0].items()},
                 "shuffled_h1_sum": {str(i): round(v, 4) for i, v in mean["shuffled"][1].items()},
                 **_scores(*ratio, candidates)},
    }


# Which per-block score orders the cuts: the 50/50 blend, or H0 / H1 alone.
CRITERIA = {"combined": "combined", "h0": "h0_secondary_sum", "h1": "h1_sum"}


def cut_order(ranking: dict, criterion: str = "combined") -> tuple[list[int], dict[str, float]]:
    """Ascending cut order (lowest persistence first) + the scores it sorts by."""
    scores = ranking[CRITERIA[criterion]]
    return sorted(ranking["candidates"], key=lambda i: scores[str(i)]), scores


# --------------------------------------------------------------------------- #
# structural cut: replace block -> 1x1 Conv                                   #
# --------------------------------------------------------------------------- #
def build_pruned_yaml(size: str, cut_abs: list[int], out_dir: Path) -> Path:
    """Write a ``yolo26<size>`` YAML with each ``cut_abs`` block replaced by a
    1x1 ``Conv`` of the same nominal output width. ``cut_abs=[]`` -> full model."""
    spec = yaml.safe_load(BASE_YAML.read_text())
    spec["scale"] = size
    layers = [list(x) for x in spec["backbone"]] + [list(x) for x in spec["head"]]
    for i in cut_abs:
        frm, _repeats, _module, args = layers[i]
        layers[i] = [frm, 1, "Conv", [args[0], 1]]  # same nominal c2, 1x1, stride 1
    n_backbone = len(spec["backbone"])
    spec["backbone"], spec["head"] = layers[:n_backbone], layers[n_backbone:]

    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "full" if not cut_abs else "cut" + "-".join(str(i) for i in cut_abs)
    path = out_dir / f"yolo26{size}_{tag}.yaml"
    path.write_text(yaml.safe_dump(spec, sort_keys=False, default_flow_style=None))
    return path


def model_stats(model) -> dict:
    """``{params, gflops}`` for an instantiated ultralytics model. ``model.info()``
    only prints in this version, so read the numbers from the torch utils."""
    from ultralytics.utils.torch_utils import get_flops, get_num_params

    core = model.model if hasattr(model, "model") else model
    return {
        "params": int(get_num_params(core)),
        "gflops": round(float(get_flops(core, 640)), 3),
    }
