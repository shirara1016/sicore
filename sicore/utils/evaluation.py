import numpy as np

from typing import cast

from ..core.base import SelectiveInferenceResult


def rejection_rate(
    results: list[SelectiveInferenceResult] | np.ndarray | list[float],
    alpha: float = 0.05,
    naive: bool = False,
):
    """Compute rejection rate from the list of SelectiveInferenceResult objects or p-values.

    Args:
        results (list[SelectiveInferenceResult] | np.ndarray | list[float]):
            List of SelectiveInferenceResult objects or p-values.
        alpha (float, optional): Significance level. Defaults to 0.05.
        naive (bool, optional): Whether to use naive p-values from
            SelectiveInferenceResult objects. Defaults to False.
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
