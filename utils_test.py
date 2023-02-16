"""
Unit tests for the utility module.

Run with `pytest utils_test.py`
"""
import torch as t
import pytest

import utils


def test_allclose():
    utils.allclose(t.tensor([1.0, 2.0, 3.0]), t.tensor([1.0, 2.0, 3.0]))
    utils.allclose(t.tensor([1.0, 2.0, 3.0]), t.tensor([1.0, 2.0, 3.0], dtype=t.float16))
    utils.allclose(t.tensor([1.0, 2.0, 3.0]), t.tensor([1.0, 2.0, 3.0], dtype=t.float64))

    with pytest.raises(AssertionError):
        utils.allclose(t.tensor([1.0, 2.0, 3.0]), t.tensor([1.001, 2.0, 3.0]))

    utils.allclose(t.tensor([1.000001, 1.999999, 3.0]), t.tensor([1.0, 2.0, 3.0]))


def test_allclose_zero_expected():
    """If the expected is zero, allclose fails for any nonzero difference - use atol instead."""
    with pytest.raises(AssertionError):
        utils.allclose(t.tensor([0.000000001, 2.0, 3.0], dtype=t.float64), t.tensor([0.0, 2.0, 3.0], dtype=t.float64))


def test_allclose_under_one_percent():
    """Old versions of allclose incorrectly passed when under 1% of entries were outside tolerance."""
    expected = t.ones(1000)
    actual = t.ones(1000)
    actual[0] += 1
    with pytest.raises(AssertionError):
        utils.allclose(actual, expected)


def test_allclose_atol_under_one_percent():
    """Old versions of allclose_atol incorrectly passed when under 1% of entries were outside tolerance."""
    expected = t.zeros(1000)
    actual = t.zeros(1000)
    actual[0] += 1
    with pytest.raises(AssertionError):
        utils.allclose_atol(actual, expected, 0.5)
