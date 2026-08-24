import pytest

from topoprun.uncertainty import summarize_seeds


def test_one_seed_is_not_reportable_uncertainty():
    result = summarize_seeds([0.4])
    assert result.mean == 0.4
    assert result.standard_deviation is None
    assert result.confidence_low is None
    assert not result.reportable


def test_repeated_seeds_have_t_interval():
    result = summarize_seeds([0.3, 0.4, 0.5])
    assert result.reportable
    assert result.confidence_low < result.mean < result.confidence_high
    assert result.standard_deviation == pytest.approx(0.1)
