"""Module with test for the non-gaussian utilities."""

from typing import Literal

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad  # type: ignore[import]
from scipy.stats import norm, rv_continuous  # type: ignore[import]

from sicore.utils.non_gaussian import generate_non_gaussian_rv


def wasserstein_distance(rv: rv_continuous) -> float:
    """Return the wasserstein distance of the random variable."""
    return quad(lambda z: np.abs(rv.cdf(z) - norm.cdf(z)), -np.inf, np.inf)[0]


@pytest.mark.parametrize(
    "rv_name",
    ["t", "skewnorm", "gennormsteep", "gennormflat", "exponnorm"],
)
def test_generate_non_gaussian_rv(
    rv_name: Literal["t", "skewnorm", "gennormsteep", "gennormflat", "exponnorm"],
) -> None:
    """Test the generate non-gaussian random variable function."""
    rv = generate_non_gaussian_rv(rv_name, 0.03)
    assert_allclose(wasserstein_distance(rv), 0.03)
    assert_allclose(rv.mean(), 0.0, rtol=1e-6, atol=1e-6)
    assert_allclose(rv.std(), 1.0, rtol=1e-6, atol=1e-6)
