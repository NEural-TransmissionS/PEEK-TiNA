"""
Topological (persistent-homology) channel pruning for Ultralytics YOLO26.

Every channel of every prunable conv is scored by the persistence of a PEEK map
(``basis="peek"``, the default): the canonical PEEK map is computed from the
full-resolution activations, then a superlevel-set cubical filtration (high-PEEK
cells are born first) gives H0 (connected blobs) and H1 (loops/holes) diagrams.
Per-channel criteria read the PEEK map of each channel alone; a channel's score
is the average lifetime of its finite bars pooled over a calibration set:

    h0     mean finite H0 lifetime
    h1     mean finite H1 lifetime
    total  mean lifetime over H0 and H1 bars together

The field criteria (FIELD_CRITERIA) read the layer's PEEK map instead. The
archived ``basis="activation"`` variant reads raw activation maps / the
|activation|_2 field.

Within each layer the lowest-scoring channels are removed structurally with
Torch-Pruning (the conv is rebuilt with fewer filters; it is not a mask).
"""

from __future__ import annotations

import copy
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_TP_ROOT = Path(__file__).resolve().parent.parent / "third-party" / "Torch-Pruning"
if _TP_ROOT.exists() and str(_TP_ROOT) not in sys.path:
    sys.path.insert(0, str(_TP_ROOT))
# topoprun (canonical PEEK adapter + PEEK-map topology) lives in <repo>/src
_TOPOPRUN_SRC = Path(__file__).resolve().parents[2] / "src"
if (_TOPOPRUN_SRC / "topoprun").exists() and str(_TOPOPRUN_SRC) not in sys.path:
    sys.path.insert(0, str(_TOPOPRUN_SRC))

import torch_pruning as tp  # noqa: E402
from ultralytics.nn.modules import C2PSA, Conv, Detect  # noqa: E402
from ultralytics.nn.modules.block import C2f, PSABlock  # noqa: E402

TDA_CRITERIA = ("h0", "h1", "total")
# Layer-field criteria: a channel's score is how much the topology of its layer's
# PEEK map (or, on the activation basis, the |activation|_2 field) changes when
# that one channel is left out. Lowest change pruned first.
FIELD_CRITERIA = ("field_h0", "field_h1", "field_total")
# Top-level layers [0, PROTECT_EARLY) are never pruned: the stem through the
# stride-16 downsample (model.0-5 in YOLO26) forms the elementary edges/shapes,
# and pruning any one of them alone costs 60-99% of mAP (verify_yolo26_pruning.py).
PROTECT_EARLY = 6
BASELINE_CRITERIA = ("l2", "random")


# ---------------------------------------------------------------------------
# chunk/split -> two convs (Torch-Pruning cannot resolve YOLO's cv1 chunk)
# ---------------------------------------------------------------------------

def _half_conv(conv: Conv, start: int, stop: int) -> Conv:
    half = copy.deepcopy(conv)
    half.conv.weight = nn.Parameter(conv.conv.weight.data[start:stop].clone())
    half.conv.out_channels = stop - start
    bn = half.bn
    bn.weight = nn.Parameter(conv.bn.weight.data[start:stop].clone())
    bn.bias = nn.Parameter(conv.bn.bias.data[start:stop].clone())
    bn.running_mean = conv.bn.running_mean[start:stop].clone()
    bn.running_var = conv.bn.running_var[start:stop].clone()
    bn.num_features = stop - start
    return half


class _SplitBlock(nn.Module):
    """C2f/C3k2 and C2PSA with `cv1(x).chunk(2)` replaced by two convs (same math)."""

    def __init__(self, block: nn.Module):
        super().__init__()
        c = block.cv1.conv.out_channels // 2
        self.cv0 = _half_conv(block.cv1, 0, c)
        self.cv1 = _half_conv(block.cv1, c, 2 * c)
        self.cv2, self.m = block.cv2, block.m
        self.sequential = isinstance(block, C2PSA)
        for attr in ("f", "i", "type", "np"):  # graph bookkeeping DetectionModel reads
            if hasattr(block, attr):
                setattr(self, attr, getattr(block, attr))

    def forward(self, x):
        if self.sequential:  # C2PSA: a, m(b)
            return self.cv2(torch.cat((self.cv0(x), self.m(self.cv1(x))), 1))
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


def make_prunable(net: nn.Module) -> nn.Module:
    """Rewrite split blocks in place and enable grads; returns `net`."""
    layers = net.model
    for i, block in enumerate(layers):
        if isinstance(block, (C2f, C2PSA)):
            layers[i] = _SplitBlock(block)
    for p in net.parameters():
        p.requires_grad_(True)
    return net


def ignored_layers(net: nn.Module, protect: int = 0) -> list[nn.Module]:
    """Layers never pruned: the first `protect` top-level layers, the Detect head
    (fixed outputs), and any layer holding PSA attention, whose reshape
    hard-codes its channel count."""
    return [
        m for i, m in enumerate(net.model)
        if i < protect or isinstance(m, Detect) or any(isinstance(s, PSABlock) for s in m.modules())
    ]


def scored_convs(net: nn.Module) -> dict[str, Conv]:
    # protected early layers are still scored: the layer summary reports them, and
    # changing the protect cutoff needs no rescoring
    ignored = {id(s) for m in ignored_layers(net) for s in m.modules()}
    return {name: m for name, m in net.named_modules() if isinstance(m, Conv) and id(m) not in ignored}


# ---------------------------------------------------------------------------
# persistence scoring
# ---------------------------------------------------------------------------

def _channel_persistence(maps: np.ndarray) -> np.ndarray:
    """(C, H, W) activations -> (C, 4) [sum_h0, n_h0, sum_h1, n_h1] of finite bars."""
    import gudhi

    out = np.zeros((maps.shape[0], 4), dtype=np.float64)
    for c, fmap in enumerate(maps):
        cc = gudhi.CubicalComplex(top_dimensional_cells=-fmap)  # superlevel filtration
        cc.compute_persistence(homology_coeff_field=2, min_persistence=0.0)
        for k, dim in ((0, 0), (2, 1)):
            bars = cc.persistence_intervals_in_dimension(dim)
            if len(bars):
                life = bars[:, 1] - bars[:, 0]
                life = life[np.isfinite(life)]
                out[c, k], out[c, k + 1] = life.sum(), life.size
    return out


def _field_drift(maps: np.ndarray) -> np.ndarray:
    """(C, H, W) activations -> (C, 3) |change| in the layer field's summed finite
    [H0, H1, H0+H1] persistence when each channel is left out of the field."""
    sq = maps.astype(np.float64) ** 2
    full = sq.sum(0)
    fields = np.sqrt(np.maximum(np.concatenate([full[None], full[None] - sq]), 0))
    lo = fields.min(axis=(1, 2), keepdims=True)
    span = fields.max(axis=(1, 2), keepdims=True) - lo
    fields = np.divide(fields - lo, span, out=np.zeros_like(fields), where=span > 0)
    sums = _channel_persistence(fields)[:, [0, 2]]  # [sum_h0, sum_h1] per field
    d = np.abs(sums[1:] - sums[0])
    return np.column_stack([d, d.sum(1)])


def _peek_scores(maps: np.ndarray, max_side: int, field: bool) -> np.ndarray:
    """PEEK basis. Full-res (C, H, W) activations -> canonical PEEK maps (topoprun),
    pooled to ``max_side``, then topoprun's min-max-normalised superlevel persistence.
    Per channel: the PEEK map of that channel alone -> (C, 4) like _channel_persistence.
    field: the layer PEEK map with each channel left out -> (C, 3) like _field_drift."""
    from topoprun.peek_adapter import to_peek_map
    from topoprun.topology import persistence_diagrams

    torch.set_num_threads(1)
    idx = np.arange(len(maps))
    keep = [idx] + [np.delete(idx, j) for j in idx] if field else [idx[j : j + 1] for j in idx]
    peek = torch.from_numpy(np.stack([to_peek_map(maps[k]) for k in keep]))
    if max(peek.shape[-2:]) > max_side:
        peek = F.adaptive_avg_pool2d(peek, max_side)
    sums = np.zeros((len(keep), 4))
    for i, m in enumerate(peek.numpy()):
        for k, diagram in zip((0, 2), persistence_diagrams(m)):
            bars = np.asarray(diagram.intervals, dtype=np.float64).reshape(-1, 2)
            sums[i, k], sums[i, k + 1] = (bars[:, 1] - bars[:, 0]).sum(), len(bars)
    if not field:
        return sums
    d = np.abs(sums[1:, [0, 2]] - sums[0, [0, 2]])
    return np.column_stack([d, d.sum(1)])


@torch.no_grad()
def compute_persistence_scores(
    net: nn.Module,
    images: torch.Tensor,
    *,
    batch_size: int = 8,
    max_side: int = 64,
    workers: int = 1,
    field: bool = False,
    basis: str = "peek",
) -> dict[str, dict[str, torch.Tensor]]:
    """Per-channel h0/h1/total mean persistence for every scored conv, or with
    ``field=True`` the mean leave-one-out layer-field drift (FIELD_CRITERIA).
    ``basis="peek"`` reads the topology from PEEK maps instead of raw activations."""
    convs = scored_convs(net)
    maps: dict[str, torch.Tensor] = {}
    hooks = [m.register_forward_hook(lambda _m, _i, o, n=n: maps.__setitem__(n, o)) for n, m in convs.items()]
    device = next(net.parameters()).device
    totals = {n: np.zeros((m.conv.out_channels, 3 if field else 4)) for n, m in convs.items()}
    net.eval()
    try:
        # spawn: forking a CUDA-initialised parent is unsafe
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            for start in range(0, len(images), batch_size):
                net(images[start : start + batch_size].to(device))
                jobs = []
                for name, act in maps.items():
                    if basis == "peek":  # PEEK needs full-res activations; its map is pooled
                        for sample in act.float().cpu().numpy():
                            jobs.append((name, pool.submit(_peek_scores, sample, max_side, field)))
                        continue
                    if max(act.shape[-2:]) > max_side:
                        act = F.adaptive_avg_pool2d(act, max_side)
                    for sample in act.float().cpu().numpy():
                        fn = _field_drift if field else _channel_persistence
                        jobs.append((name, pool.submit(fn, sample)))
                for name, job in jobs:
                    totals[name] += job.result()
    finally:
        for h in hooks:
            h.remove()

    if field:  # drift columns are per-image sums; average over images
        return {
            f"{name}.conv": {k: torch.from_numpy(t[:, j] / len(images)).float() for j, k in enumerate(FIELD_CRITERIA)}
            for name, t in totals.items()
        }

    def _mean(s, n):
        return torch.from_numpy(np.divide(s, n, out=np.zeros_like(s), where=n > 0)).float()

    return {
        f"{name}.conv": {
            "h0": _mean(t[:, 0], t[:, 1]),
            "h1": _mean(t[:, 2], t[:, 3]),
            "total": _mean(t[:, 0] + t[:, 2], t[:, 1] + t[:, 3]),
        }
        for name, t in totals.items()
    }


class PersistenceImportance(tp.importance.GroupMagnitudeImportance):
    """Group importance = mean persistence of the coupled convs' output channels."""

    def __init__(self, net: nn.Module, scores: dict[str, dict[str, torch.Tensor]], criterion: str):
        super().__init__(group_reduction="mean", normalizer=None)
        modules = dict(net.named_modules())
        self.by_layer = {modules[name]: s[criterion] for name, s in scores.items() if name in modules}

    @torch.no_grad()
    def __call__(self, group):
        imps, idxs = [], []
        for i, (dep, dep_idxs) in enumerate(group):
            layer = dep.layer
            if dep.pruning_fn == tp.pruner.function.prune_conv_out_channels and layer in self.by_layer:
                imps.append(self.by_layer[layer][dep_idxs])
                idxs.append(group[i].root_idxs)
        return self._reduce(imps, idxs) if imps else None


def build_importance(criterion: str, net: nn.Module, scores=None):
    if criterion in TDA_CRITERIA or criterion in FIELD_CRITERIA:
        return PersistenceImportance(net, scores, criterion)
    if criterion == "l2":
        return tp.importance.GroupMagnitudeImportance(p=2)
    if criterion == "random":
        return tp.importance.RandomImportance()
    raise ValueError(f"Unknown criterion: {criterion}")


def prune(net: nn.Module, importance, ratio: float, example: torch.Tensor, protect: int = PROTECT_EARLY) -> nn.Module:
    """Per-layer structural pruning: each group outside the protected layers loses `ratio` of its channels."""
    net.train()  # Detect returns raw maps in train mode; the traced graph is simpler
    pruner = tp.pruner.MetaPruner(
        net,
        example,
        importance=importance,
        pruning_ratio=ratio,
        global_pruning=False,
        ignored_layers=ignored_layers(net, protect),
    )
    pruner.step()
    return net


@torch.no_grad()
def recalibrate_bn(net: nn.Module, images: torch.Tensor, batch_size: int = 16) -> nn.Module:
    """Re-estimate BatchNorm running stats. Removing a channel with a non-zero
    mean (post-SiLU activations are positive on average) shifts every downstream
    BN input away from its stored statistics; this, not the channel loss itself,
    causes most of the one-shot mAP drop."""
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None  # cumulative average over the calibration set
    device = next(net.parameters()).device
    net.train()
    for start in range(0, len(images), batch_size):
        net(images[start : start + batch_size].to(device))
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 0.03  # Ultralytics' default, for fine-tuning
    return net.eval()
