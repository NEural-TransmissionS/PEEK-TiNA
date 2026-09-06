"""Cubical PH and ECC against hand-computed toy diagrams (``CLAUDE.md``)."""

from __future__ import annotations

import numpy as np
import pytest
import toy_fields as tf
from topo_prune.filtrations import (
    betti_numbers_at,
    euler_characteristic_curve,
    superlevel_diagram,
)


def _sorted_rows(pairs: np.ndarray) -> list[list[float]]:
    return sorted([[float(b), float(d)] for b, d in pairs])


def test_solid_disk_one_essential_component():
    d = superlevel_diagram(tf.solid_disk())
    assert _sorted_rows(d.by_dim[0]) == [[3.0, -np.inf]]
    assert d.by_dim[1].size == 0
    for tau in (2.5, 1.5, 0.5):
        assert betti_numbers_at(d, tau) == {0: 1, 1: 0}


def test_annulus_has_one_hole_until_centre_fills():
    d = superlevel_diagram(tf.annulus())
    assert _sorted_rows(d.by_dim[0]) == [[5.0, -np.inf]]
    assert _sorted_rows(d.by_dim[1]) == [[5.0, 0.0]]
    assert betti_numbers_at(d, 2.5) == {0: 1, 1: 1}
    # at tau <= 0 the 2x2 centre joins the superlevel set and the hole closes
    assert betti_numbers_at(d, -0.5) == {0: 1, 1: 0}


def test_two_disks_merge_through_background():
    d = superlevel_diagram(tf.two_disks())
    assert _sorted_rows(d.by_dim[0]) == [[2.0, 0.0], [3.0, -np.inf]]
    assert d.by_dim[1].size == 0
    assert betti_numbers_at(d, 2.5) == {0: 1, 1: 0}  # only the taller block
    assert betti_numbers_at(d, 1.5) == {0: 2, 1: 0}  # both blocks, still disjoint
    assert betti_numbers_at(d, -0.5) == {0: 1, 1: 0}  # connected via background


def test_superlevel_diagram_rejects_non_finite():
    bad = tf.solid_disk()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        superlevel_diagram(bad)


def test_superlevel_diagram_rejects_non_2d():
    with pytest.raises(ValueError):
        superlevel_diagram(np.zeros((3, 3, 3)))


@pytest.mark.parametrize(
    "field, thresholds, expected",
    [
        (tf.solid_disk(), [2.5, 1.5, -0.5], [1, 1, 1]),
        (tf.annulus(), [2.5, 0.5, -0.5], [0, 0, 1]),
        (tf.two_disks(), [2.5, 1.5, -0.5], [1, 2, 1]),
    ],
)
def test_ecc_matches_betti_alternating_sum(field, thresholds, expected):
    thresholds = np.asarray(thresholds, dtype=float)
    chi = euler_characteristic_curve(field, thresholds)
    assert chi.tolist() == expected
    # cross-check: for a 2-D set chi = beta0 - beta1
    d = superlevel_diagram(field)
    for t, c in zip(thresholds, chi):
        b = betti_numbers_at(d, float(t))
        assert c == b[0] - b[1]


def test_ecc_full_grid_is_one():
    field = np.full((7, 9), 2.0)
    chi = euler_characteristic_curve(field, np.array([1.0]))
    assert chi.tolist() == [1]
