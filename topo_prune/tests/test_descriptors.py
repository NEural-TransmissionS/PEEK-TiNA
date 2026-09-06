"""Barcode vector summaries."""

from __future__ import annotations

import numpy as np
import toy_fields as tf
from topo_prune.descriptors import (
    betti_curve,
    ecc_vector,
    persistence_entropy,
    total_persistence,
)
from topo_prune.filtrations import superlevel_diagram


def test_betti_curve_tracks_component_count():
    d = superlevel_diagram(tf.two_disks())
    thresholds = np.array([2.5, 1.5, -0.5])
    assert betti_curve(d, thresholds, dimension=0).tolist() == [1, 2, 1]
    assert betti_curve(d, thresholds, dimension=1).tolist() == [0, 0, 0]


def test_total_persistence_and_entropy_are_zero_without_finite_bars():
    d = superlevel_diagram(tf.solid_disk())  # only an essential H0 bar
    assert total_persistence(d, dimension=0) == 0.0
    assert persistence_entropy(d, dimension=0) == 0.0


def test_total_persistence_counts_finite_bar_lifetime():
    d = superlevel_diagram(tf.two_disks())  # finite H0 bar [2, 0]
    assert total_persistence(d, dimension=0) == 2.0


def test_ecc_vector_is_the_euler_curve():
    field = tf.annulus()
    thresholds = np.array([2.5, -0.5])
    assert ecc_vector(field, thresholds).tolist() == [0, 1]
