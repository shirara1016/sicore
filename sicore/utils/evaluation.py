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
    log_num_comparisons: float = 0.0,
) -> RealSubset:
    """Find the rejection area of the unconditional test statistic.

    Args:
        null_rv (rv_continuous): Null distribution of the unconditional test statistic.
        alternative (Literal["two-sided", "less", "greater"]): Type of the alternative hypothesis.
        alpha (float, optional): Significance level. Defaults to 0.05.
        log_num_comparisons (float, optional): Logarithm of the number of comparisons
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
                + log_num_comparisons
            )
            edge = brentq(f, 0.0001, 1000.0)
            return RealSubset([[-np.inf, -edge], [edge, np.inf]]) & support
        case "less":
            f = lambda z: null_rv.logsf(z) - np.log(alpha) + log_num_comparisons
            edge = brentq(f, 0.0001, 1000.0)
            return RealSubset([[edge, np.inf]]) & support
        case "greater":
            f = lambda z: null_rv.logcdf(z) - np.log(alpha) + log_num_comparisons
            edge = brentq(f, -1000.0, -0.0001)
            return RealSubset([[-np.inf, edge]]) & support
        case _:
            raise ValueError("alternative must be 'two-sided', 'less', or 'greater'.")


def rejection_rate(
    results: list[SelectiveInferenceResult] | np.ndarray | list[float],
    alpha: float = 0.05,
    naive: bool = False,
    bonferroni: bool = False,
    log_num_comparisons: float = 0.0,
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
        log_num_comparisons (float, optional): Logarithm of the number of comparisons
            for the Bonferroni correction. This option is ignored when bonferroni is False.
            Defaults to 0.0, which means no correction.

    Returns:
        float: Rejection rate.

    Raises:
        ValueError: naive and bonferroni cannot be True at the same time.
    """
    if naive and bonferroni:
        raise ValueError("naive and bonferroni cannot be True at the same time.")
    if naive:
        log_num_comparisons = 0.0

    match isinstance(results[0], SelectiveInferenceResult), naive or bonferroni:
        case True, True:
            results = cast(list[SelectiveInferenceResult], results)
            info_list = [
                result._null_rv.args + (result._alternative,) for result in results
            ]
            if len(set(info_list)) == 1:
                null_rv, alternative = results[0]._null_rv, results[0]._alternative
                intervals = _find_rejection_area(
                    null_rv, alternative, alpha, log_num_comparisons
                ).intervals
                stats = np.array([result.stat for result in results])
                rejects = np.any(
                    (intervals[:, 0] <= stats[:, np.newaxis])
                    & (stats[:, np.newaxis] <= intervals[:, 1]),
                    axis=1,
                )
            else:
                rejects = []
                for result in results:
                    rejection_area = _find_rejection_area(
                        result._null_rv, result._alternative, alpha, log_num_comparisons
                    )
                    rejects.append(result.stat in rejection_area)
        case True, False:
            results = cast(list[SelectiveInferenceResult], results)
            rejects = np.array([result.p_value for result in results]) <= alpha
        case False, _:
            results = cast(np.ndarray | list[float], results)
            rejects = np.array(results) <= alpha
    return np.count_nonzero(rejects) / len(rejects)
