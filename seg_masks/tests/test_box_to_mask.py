from __future__ import annotations

import numpy as np
import pytest
from seg_masks.box_to_mask import boxfill_mask, grabcut_mask, to_binary
from seg_masks.topology_check import betti_numbers
from seg_masks.wsd_labels import Box

cv2 = pytest.importorskip("cv2")


def _ring_scene(size=120):
    """Bright ring on a dark 'space' background -- a real beta1=1 shape."""
    img = np.full((size, size, 3), 12, dtype=np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    c = (size - 1) / 2
    r = np.hypot(yy - c, xx - c)
    ring = (r <= 44) & (r >= 24)
    img[ring] = (210, 210, 210)
    return img, ring


def test_boxfill_is_a_solid_rectangle():
    boxes = [Box(2, 10, 10, 50, 40)]
    res = boxfill_mask((80, 80), boxes)
    assert res.method == "boxfill"
    assert betti_numbers(to_binary(res.label_map, 2)) == {"beta0": 1, "beta1": 0}


def test_grabcut_recovers_the_hole_that_boxfill_destroys():
    img, _ring = _ring_scene()
    box = [Box(2, 12, 12, 108, 108)]
    gc = grabcut_mask(img, box, iterations=8)
    fill = boxfill_mask(img.shape[:2], box)
    b_gc = betti_numbers(to_binary(gc.label_map, 2))
    b_fill = betti_numbers(to_binary(fill.label_map, 2))
    assert b_fill["beta1"] == 0
    # GrabCut on a high-contrast ring should keep the central hole
    assert b_gc["beta1"] == 1
    # and it should not fill the whole box
    assert gc.coverage < fill.coverage


def test_grabcut_falls_back_to_boxfill_when_it_collapses():
    # a flat gray scene gives GrabCut nothing to latch onto
    img = np.full((60, 60, 3), 128, dtype=np.uint8)
    res = grabcut_mask(img, [Box(1, 5, 5, 55, 55)], min_foreground_fraction=0.5)
    assert "boxfill" in res.method
    assert np.count_nonzero(res.label_map) > 0


def test_smaller_box_wins_on_overlap():
    boxes = [Box(1, 0, 0, 100, 100), Box(3, 40, 40, 60, 60)]  # body, thruster inside
    res = boxfill_mask((100, 100), boxes)
    assert res.label_map[50, 50] == 3 + 1
    assert res.label_map[5, 5] == 1 + 1
