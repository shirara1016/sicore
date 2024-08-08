from dataclasses import dataclass
import numpy as np
from scipy import sparse
from joblib import Parallel, delayed

from typing import Any, Callable

from .real_subset import RealSubset
from .cdf import tn_cdf, tc_cdf


@dataclass
class SelectiveInferenceResult:
    """A class containing the results of selective inference.

    Attributes:
        stat (float): Test statistic value.
        alpha (float): Significance level.
        p_value (float): Selective p-value.
        inf_p (float): Lower bound of selective p-value.
        sup_p (float): Upper bound of selective p-value.
        reject_or_not (bool): Whether to reject the null hypothesis.
        truncated_intervals (list[list[float]]): Intervals from which
            the selected_model is obtained.
        search_count (int): Number of times the truncated intervals were computed.
        detect_count (int): Number of times that the selected model was obtained.
    """

    stat: float
    alpha: float
    p_value: float
    inf_p: float
    sup_p: float
    reject_or_not: bool
    truncated_intervals: list[list[float]]
    search_count: int
    detect_count: int


class InfiniteLoopError(Exception):
    pass


def compute_pvalue(F: float, alternative: str) -> float:
    """Compute the p-value from the CDF value.

    Args:
        F (float): The CDF value.
        alternative (str): Must be one of 'two-sided', 'less', 'greater', or 'abs'.
            If 'two sided', the p-value is computed for the two-tailed test.
            If 'less', the p-value is computed for the right-tailed test.
            If 'greater', the p-value is computed for the left-tailed test.
            If 'abs', the p-value is computed for the two-tailed test with distribution
            of absolute values.

    Returns:
        float: The p-value.

    Raises:
        ValueError: If `alternative` is not one of 'two-sided', 'less', 'greater', or 'abs'.
    """
    match alternative:
        case "two-sided":
            return float(2 * np.min([F, 1.0 - F]))
        case "less" | "abs":
            return float(1.0 - F)
        case "greater":
            return float(F)
        case _:
            raise ValueError(
                "The alternative must be one of 'two-sided', 'less', 'greater', or 'abs'."
            )


def compute_pvalue_bounds(
    inf_F: float, sup_F: float, alternative: str
) -> tuple[float, float]:
    """Evaluate the lower and upper bounds of the p-value from the lower and upper bounds of the CDF values.

    Args:
        inf_F (float): The lower bound of the CDF value.
        sup_F (float): The upper bound of the CDF value.
        alternative (str): Must be one of 'two-sided', 'less', 'greater', or 'abs'.
            If 'two sided', the p-value is computed for the two-tailed test.
            If 'less', the p-value is computed for the right-tailed test.
            If 'greater', the p-value is computed for the left-tailed test.
            If 'abs', the p-value is computed for the two-tailed test with distribution
            of absolute values.

    Returns:
        tuple[float, float]: The lower and upper bounds of the p-value.

    Raises:
        ValueError: If the lower bound of the CDF value is greater than the upper bound.
        ValueError: If `n_jobs` is not a positive integer.
        ValueError: If `alternative` is not one of 'two-sided', 'less', 'greater', or 'abs'.
    """
    if inf_F > sup_F:
        raise ValueError(
            "The lower bound of the CDF value must be less than the upper bound."
        )

    p_value_from_inf = compute_pvalue(inf_F, alternative)
    p_value_from_sup = compute_pvalue(sup_F, alternative)
    inf_p, sup_p = np.sort([p_value_from_inf, p_value_from_sup])
    return inf_p, sup_p
