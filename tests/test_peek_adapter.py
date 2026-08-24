import numpy as np

from topoprun.peek_adapter import to_peek_map


def test_constant_half_precision_activation_is_finite():
    activation = np.ones((8, 4, 4), dtype=np.float16)
    result = to_peek_map(activation)
    assert result.shape == (4, 4)
    assert np.isfinite(result).all()


def test_batched_activation_requires_one_item():
    try:
        to_peek_map(np.ones((2, 8, 4, 4), dtype=np.float32))
    except ValueError as error:
        assert "single activation" in str(error)
    else:
        raise AssertionError("Expected batched activation to be rejected")
