import numpy as np
from scipy.stats import rv_continuous  # type: ignore
from scipy.optimize import brentq  # type: ignore

from typing import Literal, cast

from ..core.base import SelectiveInferenceResult
from ..core.real_subset import RealSubset


def _find_rejection_area(
    null_rv: rv_continuous,
    alternative: Literal["two-sided", "less", "greater"],
    alpha: float = 0.05,
    log_num_tests: float = 0.0,
) -> RealSubset:
    """Find the rejection area of the unconditional test statistic.

    Args:
        null_rv (rv_continuous): Null distribution of the unconditional test statistic.
        alternative (Literal["two-sided", "less", "greater"]): Type of the alternative hypothesis.
        alpha (float, optional): Significance level. Defaults to 0.05.
        log_num_tests (float, optional): Logarithm of the number of tests
            for the Bonferroni correction. Defaults to 0.0, which means no correction.

    Returns:
        RealSubset: Rejection area of the unconditional test statistic.
    """
    support = RealSubset(np.array(null_rv.support()).reshape(-1, 2))
    match alternative:
        case "two-sided":
            f = (
                lambda z: null_rv.logcdf(-np.abs(z))
                - np.log(0.5 * alpha)
                + log_num_tests
            )
            edge = brentq(f, 0.0001, 1000.0)
            return RealSubset([[-np.inf, -edge], [edge, np.inf]]) & support
        case "less":
            f = lambda z: null_rv.logsf(z) - np.log(alpha) + log_num_tests
            edge = brentq(f, 0.0001, 1000.0)
            return RealSubset([[edge, np.inf]]) & support
        case "greater":
            f = lambda z: null_rv.logcdf(z) - np.log(alpha) + log_num_tests
            edge = brentq(f, -1000.0, -0.0001)
            return RealSubset([[-np.inf, edge]]) & support
        case _:
            raise ValueError("alternative must be 'two-sided', 'less', or 'greater'.")


def rejection_rate(
    results: list[SelectiveInferenceResult] | np.ndarray | list[float],
    alpha: float = 0.05,
    naive: bool = False,
    bonferroni: bool = False,
    log_num_tests: float = 0.0,
) -> float:
    """Compute rejection rate from the list of SelectiveInferenceResult objects or p-values.

    Args:
        results (list[SelectiveInferenceResult] | np.ndarray | list[float]):
            List of SelectiveInferenceResult objects or p-values.
        alpha (float, optional): Significance level. Defaults to 0.05.
        naive (bool, optional): Whether to compute rejection rate of naive inference.
            This option is available only when results are SelectiveInferenceResult objects.
            Defaults to False.
        bonferroni (bool, optional): Whether to compute rejection rate with Bonferroni correction.
            This option is available only when results are SelectiveInferenceResult objects.
            Defaults to False.
        log_num_tests (float, optional): Logarithm of the number of tests
            for the Bonferroni correction. This option is ignored when bonferroni is False.
            Defaults to 0.0, which means no correction.

    Returns:
        float: Rejection rate.
    """
    if isinstance(results[0], SelectiveInferenceResult):
        results = cast(list[SelectiveInferenceResult], results)
        if naive:
            p_values = np.array([result.naive_p for result in results])
        else:
            p_values = np.array([result.p_value for result in results])
    else:
        p_values = np.array(results)
    return np.count_nonzero(p_values <= alpha) / len(p_values)
