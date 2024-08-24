"""Module with test for the non-gaussian utilities."""

from typing import Literal

import pytest
from numpy.testing import assert_allclose

from sicore.utils.non_gaussian import _wasserstein_distance, generate_non_gaussian_rv


@pytest.mark.parametrize(
    "rv_name",
    ["t", "skewnorm", "gennormsteep", "gennormflat", "exponnorm"],
)
def test_generate_non_gaussian_rv(
    rv_name: Literal["t", "skewnorm", "gennormsteep", "gennormflat", "exponnorm"],
) -> None:
    """Test the generate non-gaussian random variable function."""
    rv = generate_non_gaussian_rv(rv_name, 0.03)
    assert_allclose(_wasserstein_distance(rv), 0.03)
    assert_allclose(rv.mean(), 0.0, rtol=1e-6, atol=1e-6)
    assert_allclose(rv.std(), 1.0, rtol=1e-6, atol=1e-6)
