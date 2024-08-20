import pytest
import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import norm, chi
from sicore.utils.evaluation import rejection_rate
from sicore.core.base import SelectiveInferenceResult
from sicore.core.real_subset import RealSubset


@pytest.mark.parametrize(
    "null_rv, alternative, expected",
    [
        (norm(), "two-sided", 0.384),
        (norm(), "less", 0.262),
        (norm(), "greater", 0.300),
        (chi(12), "less", 0.438),
    ],
)
def test_rejection_rate_naive(null_rv, alternative, expected):
    args = {
        "p_value": None,
        "inf_p": None,
        "sup_p": None,
        "searched_intervals": None,
        "truncated_intervals": None,
        "search_count": None,
        "detect_count": None,
        "_null_rv": null_rv,
        "_alternative": alternative,
    }
    stats = null_rv.rvs(random_state=0, size=1000)
    stats += np.sign(stats) * 1.1
    results = [SelectiveInferenceResult(**args, stat=stat) for stat in stats]

    value = rejection_rate(results, naive=True)
    assert value > 0.05 + 0.10
    assert_allclose(value, expected)


@pytest.mark.parametrize(
    "null_rv, alternative, expected",
    [
        (norm(), "two-sided", 0.0),
        (norm(), "less", 0.0),
        (norm(), "greater", 0.0),
        (chi(12), "less", 0.0),
    ],
)
def test_rejection_rate_bonferroni(null_rv, alternative, expected):
    args = {
        "p_value": None,
        "inf_p": None,
        "sup_p": None,
        "searched_intervals": None,
        "truncated_intervals": None,
        "search_count": None,
        "detect_count": None,
        "_null_rv": null_rv,
        "_alternative": alternative,
    }
    stats = null_rv.rvs(random_state=0, size=1000)
    stats += np.sign(stats) * 1.1
    results = [SelectiveInferenceResult(**args, stat=stat) for stat in stats]

    value = rejection_rate(
        results, bonferroni=True, log_num_comparisons=256.0 * np.log(2.0)
    )
    assert value < 0.02
    assert_allclose(value, expected)


@pytest.mark.parametrize(
    "seed, has_bias, expected",
    [(0, False, 0.045), (1, False, 0.047), (2, True, 0.220), (3, True, 0.226)],
)
def test_rejection_rate_p_values(seed, has_bias, expected):
    args = {
        "stat": None,
        "inf_p": None,
        "sup_p": None,
        "searched_intervals": None,
        "truncated_intervals": None,
        "search_count": None,
        "detect_count": None,
        "_null_rv": None,
        "_alternative": None,
    }
    rng = np.random.default_rng(seed)
    p_values = rng.uniform(size=1000)
    if has_bias:
        p_values = 4.0 * p_values**2.0 - 4.0 * p_values + 1.0
    results = [
        SelectiveInferenceResult(**args, p_value=p_value) for p_value in p_values
    ]

    value = rejection_rate(p_values)
    if has_bias:
        assert value > 0.05 + 0.10
    else:
        assert 0.05 - 0.01 < value < 0.05 + 0.01
    assert_allclose(value, expected)

    value = rejection_rate(results)
    if has_bias:
        assert value > 0.05 + 0.10
    else:
        assert 0.05 - 0.01 < value < 0.05 + 0.01
    assert_allclose(value, expected)
