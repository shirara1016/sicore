"""Module with tests for the evaluation utilities."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import chi, norm, rv_continuous  # type: ignore[import]

from sicore.core.base import SelectiveInferenceResult
from sicore.utils.evaluation import rejection_rate


@pytest.mark.parametrize(
    ("null_rv", "alternative", "expected"),
    [
        (norm(), "two-sided", 0.384),
        (norm(), "less", 0.262),
        (norm(), "greater", 0.300),
        (chi(12), "less", 0.438),
    ],
)
def test_rejection_rate_naive(
    null_rv: rv_continuous,
    alternative: str,
    expected: float,
) -> None:
    """Test the rejection rate function with naive option."""
    args = {
        "p_value": 0.5,
        "inf_p": 0.0,
        "sup_p": 1.0,
        "searched_intervals": [[-np.inf, np.inf]],
        "truncated_intervals": [[-np.inf, np.inf]],
        "search_count": 1,
        "detect_count": 1,
        "null_rv": null_rv,
        "alternative": alternative,
    }
    stats = null_rv.rvs(random_state=0, size=1000)
    stats += np.sign(stats) * 1.1
    results = [SelectiveInferenceResult(**args, stat=stat) for stat in stats]

    value = rejection_rate(results, naive=True)
    assert_allclose(value, expected)


@pytest.mark.parametrize(
    ("null_rv", "alternative", "expected"),
    [
        (norm(), "two-sided", 0.0),
        (norm(), "less", 0.0),
        (norm(), "greater", 0.0),
        (chi(12), "less", 0.0),
    ],
)
def test_rejection_rate_bonferroni(
    null_rv: rv_continuous,
    alternative: str,
    expected: float,
) -> None:
    """Test the rejection rate function with bonferroni option."""
    args = {
        "p_value": 0.5,
        "inf_p": 0.0,
        "sup_p": 1.0,
        "searched_intervals": [[-np.inf, np.inf]],
        "truncated_intervals": [[-np.inf, np.inf]],
        "search_count": 1,
        "detect_count": 1,
        "null_rv": null_rv,
        "alternative": alternative,
    }
    stats = null_rv.rvs(random_state=0, size=1000)
    stats += np.sign(stats) * 1.1
    results = [SelectiveInferenceResult(**args, stat=stat) for stat in stats]

    value = rejection_rate(
        results,
        bonferroni=True,
        log_num_comparisons=256.0 * np.log(2.0),
    )
    assert_allclose(value, expected)


@pytest.mark.parametrize(
    ("seed", "has_bias", "expected"),
    [(0, False, 0.045), (1, False, 0.047), (2, True, 0.220), (3, True, 0.226)],
)
def test_rejection_rate_p_values(seed: int, *, has_bias: bool, expected: float) -> None:
    """Test the rejection rate function for p-values."""
    args = {
        "stat": 0.0,
        "inf_p": 0.0,
        "sup_p": 1.0,
        "searched_intervals": [[-np.inf, np.inf]],
        "truncated_intervals": [[-np.inf, np.inf]],
        "search_count": 1,
        "detect_count": 1,
        "null_rv": norm(),
        "alternative": "two-sided",
    }
    rng = np.random.default_rng(seed)
    p_values = rng.uniform(size=1000)
    if has_bias:
        p_values = 4.0 * p_values**2.0 - 4.0 * p_values + 1.0
    results = [
        SelectiveInferenceResult(**args, p_value=p_value) for p_value in p_values
    ]

    value = rejection_rate(p_values)
    assert_allclose(value, expected)

    value = rejection_rate(results)
    assert_allclose(value, expected)
