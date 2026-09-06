from __future__ import annotations

import numpy as np
from seg_masks.topology_check import betti_numbers, compare_topology


def _annulus_mask(size=40, r_out=16, r_in=8):
    yy, xx = np.mgrid[0:size, 0:size]
    c = (size - 1) / 2
    r = np.hypot(yy - c, xx - c)
    return ((r <= r_out) & (r >= r_in)).astype(np.uint8)


def test_betti_of_solid_disk():
    yy, xx = np.mgrid[0:40, 0:40]
    disk = (np.hypot(yy - 20, xx - 20) <= 15).astype(np.uint8)
    assert betti_numbers(disk) == {"beta0": 1, "beta1": 0}


def test_betti_of_annulus_has_one_hole():
    assert betti_numbers(_annulus_mask()) == {"beta0": 1, "beta1": 1}


def test_betti_of_two_disks():
    m = np.zeros((40, 80), np.uint8)
    yy, xx = np.mgrid[0:40, 0:80]
    m[np.hypot(yy - 20, xx - 15) <= 10] = 1
    m[np.hypot(yy - 20, xx - 60) <= 10] = 1
    assert betti_numbers(m)["beta0"] == 2


def test_compare_topology_flags_a_filled_hole():
    annulus = _annulus_mask()
    filled = (annulus | _disk_fill(annulus)).astype(np.uint8)
    d = compare_topology(annulus, filled)
    assert d["d_beta1"] == 1


def _disk_fill(annulus):
    from scipy import ndimage

    return ndimage.binary_fill_holes(annulus)
