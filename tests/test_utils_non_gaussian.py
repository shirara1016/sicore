import pytest
from numpy.testing import assert_allclose
from sicore.utils.non_gaussian import generate_non_gaussian_rv, _wasserstein_distance


@pytest.mark.parametrize(
    "rv_name",
    ["t", "skewnorm", "gennormsteep", "gennormflat", "exponnorm"],
)
def test_generate_non_gaussian_rv(rv_name):
    rv = generate_non_gaussian_rv(rv_name, 0.03)
    assert_allclose(_wasserstein_distance(rv), 0.03)
    assert_allclose(rv.mean(), 0.0, rtol=1e-6, atol=1e-6)
    assert_allclose(rv.std(), 1.0, rtol=1e-6, atol=1e-6)
