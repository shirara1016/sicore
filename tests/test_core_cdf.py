"""Module with tests for the core cdf functions."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import chi, norm, rv_continuous  # type: ignore[import]

from sicore.core.cdf import truncated_cdf


@pytest.mark.parametrize(
    ("rv", "z", "intervals", "expected"),
    [
        (norm(), -np.inf, [[-np.inf, -2.0], [2.0, 7.0]], 0.0),
        (norm(), -1.7, [[-np.inf, -1.5], [-0.3, 0.5], [1.0, 12.0]], 0.08332542),
        (norm(), 0.8, [[-1.5, -1.2], [0.1, 2.3], [3.0, 12.0]], 0.5942752),
        (norm(), 7.3, [[-6.2, -6.1], [7.0, 12.0]], 0.99942286),
        (norm(), 2.0, [[1.0, 3.0], [4.0, 5.0]], 0.86378503),
        (norm(), -9.8, [[-np.inf, -19.0], [-10.0, -9.5]], 0.04671865),
        (norm(), 5.3, [[-10.0, -6.0], [4.0, 4.6], [5.0, 11.0]], 0.99806002),
        (norm(), -0.1, [[-7.0, 1.0], [2.0, 3.0]], 0.53338144),
        (chi(14), 2.5, [[1.6, 6.6]], 0.03985783),
        (chi(5), 6.7, [[6.4, 7.3], [18.9, 22.2], [24.7, 27.9]], 0.84278428),
        (chi(7), 2.3, [[0.0, 0.5], [1.0, 1.5], [2.0, np.inf]], 0.24739350),
        (chi(81), 8.2, [[2.1, 12.6], [18.9, 22.2], [24.7, 27.9]], 0.13653858),
        (chi(23), 2.6, [[2.5, 3.3], [5.9, 22.2]], 0.0028667),
        (chi(3), 5.2, [[2.5, 7.2], [15.0, 22.2]], 0.99994229),
        (chi(11), 3.9, [[2.5, 4.2], [8.0, 22.2]], 0.89193139),
        (chi(2), np.inf, [[0.0, 0.5], [1.0, 1.5], [2.0, np.inf]], 1.0),
    ],
)
def test_truncated_cdf(
    rv: rv_continuous,
    z: float,
    intervals: list[list[float]],
    *,
    expected: float,
) -> None:
    """Test the truncated cdf function."""
    value = truncated_cdf(rv, z, intervals)
    assert_allclose(value, expected, rtol=1e-6, atol=1e-6)
