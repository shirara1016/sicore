import pytest
import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import norm, chi
from sicore.utils.evaluation import _find_rejection_area, rejection_rate
from sicore.core.base import SelectiveInferenceResult
from sicore.core.real_subset import RealSubset


@pytest.mark.parametrize(
    "null_rv, alternative, alpha, log_num_comparisons, expected",
    [
        (
            norm(),
            "two-sided",
            0.05,
            0.0,
            RealSubset([[-np.inf, -1.95996398], [1.95996398, np.inf]]),
        ),
        (norm(), "less", 0.05, 0.0, RealSubset([1.64485363, np.inf])),
        (norm(), "greater", 0.05, 0.0, RealSubset([-np.inf, -1.64485363])),
        (chi(12), "less", 0.05, 0.0, RealSubset([4.58541926, np.inf])),
        (
            norm(),
            "two-sided",
            0.05,
            256.0 * np.log(2.0),
            RealSubset([[-np.inf, -18.829625], [18.829625, np.inf]]),
        ),
        (norm(), "less", 0.05, 256.0 * np.log(2.0), RealSubset([18.792881, np.inf])),
        (
            norm(),
            "greater",
            0.05,
            256.0 * np.log(2.0),
            RealSubset([-np.inf, -18.792881]),
        ),
        (chi(12), "less", 0.05, 256.0 * np.log(2.0), RealSubset([20.110980, np.inf])),
    ],
)
def test_find_rejection_area(
    null_rv, alternative, alpha, log_num_comparisons, expected
):
    rejection_area = _find_rejection_area(
        null_rv, alternative, alpha, log_num_comparisons
    )
    assert_allclose(rejection_area.intervals, expected.intervals)


@pytest.mark.skip()
@pytest.mark.parametrize(
    "p_values, naive, expected",
    [
        ([0.80, 0.80, 0.80, 0.80, 0.80], False, 0.0),
        ([0.01, 0.01, 0.80, 0.80, 0.80], False, 0.4),
        ([0.01, 0.01, 0.01, 0.80, 0.80], False, 0.6),
        ([0.01, 0.01, 0.01, 0.01, 0.01], False, 1.0),
        ([0.80, 0.80, 0.80, 0.80, 0.80], True, 0.0),
        ([0.01, 0.01, 0.80, 0.80, 0.80], True, 0.4),
        ([0.01, 0.01, 0.01, 0.80, 0.80], True, 0.6),
        ([0.01, 0.01, 0.01, 0.01, 0.01], True, 1.0),
    ],
)
def test_rejection_rate_results(p_values, naive, expected):
    args = {
        "stat": 2.0,
        "p_value": 0.0,
        "inf_p": 0.0,
        "sup_p": 1.0,
        "naive_p": 0.0,
        "searched_intervals": [[-np.inf, np.inf]],
        "truncated_intervals": [[-np.inf, np.inf]],
        "search_count": 1,
        "detect_count": 1,
        "_null_rv": None,
        "_alternative": None,
    }
    results = []
    for p_value in p_values:
        if naive:
            args["naive_p"] = p_value
        else:
            args["p_value"] = p_value
        result = SelectiveInferenceResult(**args)
        results.append(result)
    value = rejection_rate(results, naive=naive)
    assert_allclose(value, expected)


@pytest.mark.parametrize(
    "p_values, expected",
    [
        ([0.80, 0.80, 0.80, 0.80, 0.80], 0.0),
        ([0.01, 0.01, 0.80, 0.80, 0.80], 0.4),
        ([0.01, 0.01, 0.01, 0.80, 0.80], 0.6),
        ([0.01, 0.01, 0.01, 0.01, 0.01], 1.0),
    ],
)
def test_rejection_rate_pvalues(p_values, expected):
    value = rejection_rate(p_values)
    assert_allclose(value, expected)
