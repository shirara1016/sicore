"""Module providing function for evaluating the results of selective inference."""

from typing import cast

import numpy as np

from sicore.core.base import SelectiveInferenceResult


def rejection_rate(
    results: list[SelectiveInferenceResult] | np.ndarray | list[float],
    alpha: float = 0.05,
    *,
    naive: bool = False,
    bonferroni: bool = False,
    log_num_comparisons: float = 0.0,
) -> float:
    """Compute rejection rate from list of SelectiveInferenceResult objects or p-values.

    Args:
        results (list[SelectiveInferenceResult] | np.ndarray | list[float]):
            List of SelectiveInferenceResult objects or p-values.
        alpha (float, optional): Significance level. Defaults to 0.05.
        naive (bool, optional):
            Whether to compute rejection rate of naive inference.
            This option is available only when results are
            SelectiveInferenceResult objects. Defaults to False.
        bonferroni (bool, optional):
            Whether to compute rejection rate with Bonferroni correction.
            This option is available only when results are
            SelectiveInferenceResult objects. Defaults to False.
        log_num_comparisons (float, optional):
            Logarithm of the number of comparisons for the Bonferroni correction.
            This option is ignored when bonferroni is False.
            Defaults to 0.0, which means no correction.

    Returns:
        float: Rejection rate.

    Raises:
        ValueError: naive and bonferroni cannot be True at the same time.
    """
    if naive and bonferroni:
        raise ValueError

    if isinstance(results[0], SelectiveInferenceResult):
        results = cast(list[SelectiveInferenceResult], results)
        if naive:
            p_values = np.array([result.naive_p_value() for result in results])
        elif bonferroni:
            p_values = np.array(
                [result.bonferroni_p_value(log_num_comparisons) for result in results],
            )
        else:
            p_values = np.array([result.p_value for result in results])
    else:
        p_values = np.array(results)

    return np.count_nonzero(p_values <= alpha) / len(p_values)
