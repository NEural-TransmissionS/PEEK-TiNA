"""Synthetic scenes have exactly the homology they claim (P1.5 testbed)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from topo_prune.filtrations import betti_numbers_at, superlevel_diagram
from topo_prune.metrics import betti_number_error
from topo_prune.synthetic import bridge_perturbation, render_scene, smooth_perturbation


@pytest.mark.parametrize(
    "n_panels,gap,ring",
    list(itertools.product([1, 2, 3, 4], [True, False], [False, True])),
)
def test_scene_betti_matches_construction(n_panels, gap, ring):
    for seed in range(8):
        s = render_scene(n_panels=n_panels, gap=gap, ring=ring, size=72, seed=seed)
        b = betti_numbers_at(superlevel_diagram(s.field), s.tau)
        assert (b[0], b[1]) == (s.beta0, s.beta1), f"seed {seed}: {b}"


def test_smooth_perturbation_hits_the_target_sup_norm():
    for linf in (0.5, 2.0, 7.5):
        p = smooth_perturbation((64, 80), linf=linf, seed=3)
        assert np.abs(p).max() == pytest.approx(linf, rel=1e-9)


def test_bridge_perturbation_merges_panels():
    s = render_scene(n_panels=3, gap=True, ring=False, size=72, seed=1)
    assert s.beta0 == 3
    bridged = s.field + bridge_perturbation(s.field.shape, amplitude=15.0)
    b = betti_numbers_at(superlevel_diagram(bridged), s.tau)
    assert b[0] < 3  # panels now connected


def test_small_perturbation_preserves_betti():
    s = render_scene(n_panels=2, gap=True, ring=True, size=72, seed=2)
    ref = superlevel_diagram(s.field)
    tiny = s.field + smooth_perturbation(s.field.shape, linf=0.2, seed=9)
    assert betti_number_error(superlevel_diagram(tiny), ref, s.tau)["total"] == 0
