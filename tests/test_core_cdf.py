"""Module with tests for the core cdf functions."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from sicore.core.cdf import truncated_chi_cdf, truncated_norm_cdf


@pytest.mark.parametrize(
    ("z", "intervals", "absolute", "expected"),
    [
        (-np.inf, [[-np.inf, -2.0], [2.0, 7.0]], False, 0.0),
        (-1.7, [[-np.inf, -1.5], [-0.3, 0.5], [1.0, 12.0]], False, 0.08332542),
        (0.8, [[-1.5, -1.2], [0.1, 2.3], [3.0, 12.0]], False, 0.5942752),
        (7.3, [[-6.2, -6.1], [7.0, 12.0]], False, 0.99942286),
        (-4.0, [[-5.0, -4.0], [-2.0, -1.0], [1.0, 3.0], [4.0, 5.0]], True, 0.99978597),
        (2.0, [[1.0, 3.0], [4.0, 5.0]], True, 0.86378503),
        (-9.8, [[-np.inf, -19.0], [-10.0, -9.5]], True, 0.95328134),
        (5.3, [[-10.0, -6.0], [4.0, 4.6], [5.0, 11.0]], True, 0.99802696),
        (-0.1, [[-7.0, 1.0], [2.0, 3.0]], True, 0.09232818),
        (-2.6, [[-3.0, -2.0]], True, 0.84526851),
        (1.9, [[-5.0, -2.0], [1.4, 2.0], [6.0, np.inf]], True, 0.64440857),
        (3.5, [[3.4, 3.7], [5.0, 5.6]], True, 0.45465432),
    ],
)
def test_truncated_norm_cdf(
    z: float,
    intervals: list[list[float]],
    *,
    absolute: bool,
    expected: float,
) -> None:
    """Test the truncated norm cdf function."""
    value = truncated_norm_cdf(z, intervals, absolute=absolute)
    assert_allclose(value, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("z", "df", "intervals", "expected"),
    [
        (2.5, 14, [[1.6, 6.6]], 0.03985783),
        (6.7, 5, [[6.4, 7.3], [18.9, 22.2], [24.7, 27.9]], 0.84278428),
        (2.3, 7, [[0.0, 0.5], [1.0, 1.5], [2.0, np.inf]], 0.24739350),
        (8.2, 81, [[2.1, 12.6], [18.9, 22.2], [24.7, 27.9]], 0.13653858),
        (2.6, 23, [[2.5, 3.3], [5.9, 22.2]], 0.0028667),
        (5.2, 3, [[2.5, 7.2], [15.0, 22.2]], 0.99994229),
        (3.9, 11, [[2.5, 4.2], [8.0, 22.2]], 0.89193139),
        (np.inf, 2, [[0.0, 0.5], [1.0, 1.5], [2.0, np.inf]], 1.0),
    ],
)
def test_truncated_chi_cdf(
    z: float,
    df: int,
    intervals: list[list[float]],
    expected: float,
) -> None:
    """Test the truncated chi cdf function."""
    value = truncated_chi_cdf(z, df, intervals)
    assert_allclose(value, expected, rtol=1e-6, atol=1e-6)
