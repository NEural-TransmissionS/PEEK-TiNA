import numpy as np
import pytest

from topoprun.topology import (
    compare_diagrams,
    compare_summaries,
    persistence_diagrams,
    summarize_map,
)

pytest.importorskip("gudhi")


def test_identical_map_has_zero_summary_drift():
    field = np.array([[0.0, 1.0, 0.0], [1.0, 0.2, 1.0], [0.0, 1.0, 0.0]])
    summary = summarize_map(field)
    assert compare_summaries(summary, summary) == 0.0


def test_rejects_non_finite_map():
    with pytest.raises(ValueError, match="finite 2-D"):
        summarize_map(np.array([[0.0, np.nan]]))


def test_identical_diagrams_have_zero_distance():
    field = np.array([[0.0, 1.0, 0.0], [1.0, 0.2, 1.0], [0.0, 1.0, 0.0]])
    diagrams = persistence_diagrams(field)
    distances = compare_diagrams(diagrams, diagrams)
    assert all(item["bottleneck"] == 0.0 for item in distances.values())
    assert all(item["wasserstein_2"] == 0.0 for item in distances.values())


def test_constant_map_is_numerically_finite():
    diagrams = persistence_diagrams(np.ones((4, 4), dtype=np.float16))
    assert all(np.isfinite(interval).all() for diagram in diagrams for interval in diagram.intervals)
