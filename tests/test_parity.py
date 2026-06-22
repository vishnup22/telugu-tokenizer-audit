import pytest

from telugu_audit.metrics.parity import parity_ratio


def test_parity_ratio():
    assert parity_ratio(2.0, 1.0) == 2.0


def test_parity_ratio_zero_english():
    with pytest.raises(ValueError):
        parity_ratio(1.0, 0.0)
