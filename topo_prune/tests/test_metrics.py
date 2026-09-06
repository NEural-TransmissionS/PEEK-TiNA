"""Betti-number and simplified Betti-matching error."""

from __future__ import annotations

import numpy as np
import toy_fields as tf
from topo_prune.filtrations import Diagram, superlevel_diagram
from topo_prune.metrics import betti_matching_error, betti_number_error


def test_betti_number_error_zero_for_identical_diagrams():
    d = superlevel_diagram(tf.annulus())
    e = betti_number_error(d, d, tau=2.5)
    assert e == {"beta0": 0, "beta1": 0, "total": 0}


def test_betti_number_error_counts_a_lost_hole_and_split_component():
    ref = superlevel_diagram(tf.annulus())      # beta0=1 beta1=1 at 2.5
    pred = superlevel_diagram(tf.two_disks())   # beta0=1 beta1=0 at 2.5 (taller block)
    e = betti_number_error(pred, ref, tau=2.5)
    assert e["beta1"] == 1
    assert e["total"] == e["beta0"] + e["beta1"]


def test_betti_matching_error_matches_count_difference():
    ref = Diagram(by_dim={0: np.array([[5.0, -np.inf], [4.0, 1.0]]), 1: np.empty((0, 2))})
    pred = Diagram(by_dim={0: np.array([[5.0, -np.inf]]), 1: np.empty((0, 2))})
    e = betti_matching_error(pred, ref, tau=2.0)
    assert e["beta0"] == 1  # one reference component unmatched
    assert e["total"] == 1


def test_betti_matching_error_zero_when_bars_coincide():
    d = superlevel_diagram(tf.two_disks())
    e = betti_matching_error(d, d, tau=1.0, tolerance=1e-9)
    assert e["total"] == 0


def test_betti_matching_error_tolerance_allows_small_shift():
    ref = Diagram(by_dim={0: np.array([[5.0, 0.0]]), 1: np.empty((0, 2))})
    pred = Diagram(by_dim={0: np.array([[5.2, 0.1]]), 1: np.empty((0, 2))})
    assert betti_matching_error(pred, ref, tau=2.5, tolerance=0.5)["beta0"] == 0
    assert betti_matching_error(pred, ref, tau=2.5, tolerance=0.05)["beta0"] == 2
