"""Structured head pruning (needs torch + ultralytics)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")

from topo_prune.prune import prune_head


@pytest.fixture(scope="module")
def detection_module():
    from ultralytics import YOLO

    return YOLO("yolo26n.yaml").model  # random init, no download needed


@pytest.mark.parametrize("fraction", [0.0, 0.25, 0.5])
def test_prune_head_zeroes_the_selected_final_conv_columns(detection_module, fraction):
    pruned, records = prune_head(detection_module, fraction=fraction, method="magnitude", seed=42)
    detect = pruned.model[-1]
    for rec in records:
        w = detect.one2one_cv3[rec.level][-1].weight.detach().cpu().numpy()
        for k in rec.dropped_channels:
            assert np.all(w[:, k] == 0.0)
        assert abs(rec.actual_fraction - fraction) < 1.0 / w.shape[1] + 1e-9


def test_prune_head_keeps_at_least_one_channel(detection_module):
    _, records = prune_head(detection_module, fraction=0.999, method="magnitude")
    assert all(len(r.kept_channels) >= 1 for r in records)


def test_prune_head_leaves_input_module_untouched(detection_module):
    before = detection_module.model[-1].one2one_cv3[0][-1].weight.clone()
    prune_head(detection_module, fraction=0.5, method="random", seed=1)
    assert torch.equal(before, detection_module.model[-1].one2one_cv3[0][-1].weight)


def test_prune_head_rejects_topo_method(detection_module):
    with pytest.raises(ValueError, match="P2"):
        prune_head(detection_module, fraction=0.2, method="topo")
