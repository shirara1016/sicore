import pytest
import numpy as np
from numpy.testing import assert_allclose
from itertools import product
from sicore.utils.non_gaussian import generate_non_gaussian_rv, _wasserstein_distance


@pytest.mark.parametrize(
    "rv_name, distance",
    [
        (rv_name, distance)
        for rv_name, distance in product(
            ["t", "skewnorm", "gennormsteep", "gennormflat", "exponnorm"],
            [0.04, 0.08],
        )
    ],
)
def test_generate_non_gaussian_rv(rv_name, distance):
    rv = generate_non_gaussian_rv(rv_name, distance)
    value = _wasserstein_distance(rv)
    assert_allclose(value, distance, rtol=1e-6, atol=1e-6)
    assert_allclose(rv.mean(), 0.0, rtol=1e-6, atol=1e-6)
    assert_allclose(rv.std(), 1.0, rtol=1e-6, atol=1e-6)
