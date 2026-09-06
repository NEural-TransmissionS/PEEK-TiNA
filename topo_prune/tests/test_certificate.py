"""Margin test, certification, and the falsification audit (plan §3.3-§3.4)."""

from __future__ import annotations

import numpy as np
import pytest
import toy_fields as tf
from topo_prune.certificate import (
    audit,
    certify,
    coverage,
    margin,
    straddling_min_lifetime,
)
from topo_prune.filtrations import superlevel_diagram


def test_margin_is_min_endpoint_distance_to_tau():
    d = superlevel_diagram(tf.annulus())  # H0 [5,-inf], H1 [5,0]
    # endpoints at 5 and 0; tau=2.5 is equidistant
    assert margin(d, 2.5) == pytest.approx(2.5)
    assert margin(d, 1.0) == pytest.approx(1.0)  # nearest endpoint is death=0


def test_straddling_min_lifetime_ignores_bars_off_tau():
    d = superlevel_diagram(tf.two_disks())  # H0 [2,0] and [3,-inf]
    # tau=1: both H0 bars straddle; lifetimes are 2 and +inf
    assert straddling_min_lifetime(d, 1.0) == pytest.approx(2.0)
    # tau=2.5: only the essential bar straddles -> +inf
    assert straddling_min_lifetime(d, 2.5) == np.inf


def test_certify_true_when_no_endpoint_can_cross_tau():
    d = superlevel_diagram(tf.annulus())
    r = certify(d, tau=2.5, epsilon=1.0)
    assert r.certified
    assert r.betti == {0: 1, 1: 1}


def test_certify_false_when_epsilon_exceeds_margin():
    d = superlevel_diagram(tf.annulus())
    assert not certify(d, tau=2.5, epsilon=3.0).certified


def test_straddling_condition_is_implied_by_the_margin_condition():
    # For any bar that straddles tau, lifetime = (birth - tau) + (tau - death)
    # >= 2 * min(birth - tau, tau - death) >= 2 * margin. So margin > epsilon
    # already forces straddling_min_lifetime > 2*epsilon. We keep both checks
    # because the plan (§3.3) lists both; this test records the redundancy.
    # See docs/certificate-detection.md "The two-condition check".
    d = superlevel_diagram(tf.two_disks())
    for tau in (0.5, 1.0, 1.5, 2.5):
        for epsilon in (0.05, 0.2, 0.4, 0.8):
            r = certify(d, tau=tau, epsilon=epsilon)
            if r.margin > epsilon:
                assert r.straddling_min_lifetime > 2 * epsilon
                assert r.certified
            else:
                assert not r.certified


def test_certify_rejects_bad_epsilon():
    d = superlevel_diagram(tf.solid_disk())
    for bad in (-1.0, np.inf, np.nan):
        with pytest.raises(ValueError):
            certify(d, tau=1.0, epsilon=bad)


def test_audit_no_violation_when_pruned_equals_reference():
    d = superlevel_diagram(tf.annulus())
    report = audit([("img0/solar/level1", d, d, 2.5, 1.0)])
    assert report.certified_count == 1
    assert report.violation_count == 0
    assert report.sound


def test_audit_flags_a_certified_betti_change():
    # reference says beta1=1 at tau=2.5 and certifies; pruned model in fact
    # produces beta1=0 -> the certificate is violated and the audit must catch it.
    ref = superlevel_diagram(tf.annulus())
    pruned = superlevel_diagram(tf.solid_disk())
    report = audit([("img7/solar/level1", ref, pruned, 2.5, 1.0)])
    assert report.certified_count == 1
    assert report.violation_count == 1
    assert not report.sound
    v = report.violations[0]
    assert v.betti_reference[1] == 1 and v.betti_pruned[1] == 0


def test_audit_does_not_flag_uncertified_changes():
    ref = superlevel_diagram(tf.annulus())
    pruned = superlevel_diagram(tf.solid_disk())
    # epsilon huge -> not certified -> a Betti change is allowed, not a violation
    report = audit([("img7/solar/level1", ref, pruned, 2.5, 100.0)])
    assert report.certified_count == 0
    assert report.uncertified_count == 1
    assert report.violation_count == 0


def test_coverage_fraction():
    d = superlevel_diagram(tf.annulus())
    results = [
        certify(d, 2.5, 1.0),   # certified
        certify(d, 2.5, 3.0),   # not
        certify(d, 2.5, 0.5),   # certified
    ]
    assert coverage(results) == pytest.approx(2 / 3)
    assert coverage([]) == 0.0
