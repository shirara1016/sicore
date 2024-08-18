import pytest
import numpy as np
from numpy.testing import assert_allclose
from sicore.utils.evaluation import rejection_rate
from sicore.core.base import SelectiveInferenceResult


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
