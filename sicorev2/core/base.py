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
        naive_p (float): Naive p-value.
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
    naive_p: float
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


class Inference:
    def __init__(self, data: np.ndarray, var: float | np.ndarray | sparse.csr_matrix):
        self.data = data
        self.var = var

        self.stat = None
        self.a = None
        self.b = None

        self._compute_pvalue = None
        self._evaluate_pvalue_bounds = None

        self.mode = None
        self.limits = None
        self.support = None
        self.null_rv = None
        self.cdf = None

    def _create_search_strategy(self):
        raise NotImplementedError()

    def _create_termination_criterion(self):
        raise NotImplementedError()

    def inference(
        self,
        algorithm: Callable[[np.ndarray, np.ndarray, float], Any],
        model_selector: Callable[[Any], bool],
        alternative: str = "abs",
        inference_mode: str = "parametric",  # parametric, exhaustive, or over_conditioning
        search_strategy: Callable[[RealSubset], list[float]] | str = "pi3",
        termination_criterion: (
            Callable[[RealSubset, RealSubset], bool] | str
        ) = "precision",
        max_iter: int = 1e6,
        n_jobs: int = 1,
        step: float = 1e-10,
        significance_level: float = 0.05,
        precision: float = 0.001,
    ) -> SelectiveInferenceResult:

        self.n_jobs = n_jobs
        self.step = step
        self.significance_level = significance_level
        self.precision = precision

        if isinstance(search_strategy, str):
            search_strategy = self._create_search_strategy(
                inference_mode, search_strategy
            )
        if isinstance(termination_criterion, str):
            termination_criterion = self._create_termination_criterion(
                inference_mode, termination_criterion
            )

        self.alternative = alternative

        searched_intervals = RealSubset()
        truncated_intervals = RealSubset()
        search_count, detect_count = 0, 0

        before_searched_intervals = RealSubset()
        while True:
            z_list = search_strategy(searched_intervals)

            if n_jobs == 1:
                results = []
                for z in z_list:
                    model, intervals = algorithm(self.a, self.b, z)
                    results.append((model, intervals))
            elif n_jobs > 1:
                with Parallel(n_jobs=n_jobs) as parallel:
                    results = parallel(
                        delayed(algorithm)(self.a, self.b, z) for z in z_list
                    )
            else:
                raise ValueError("The n_jobs must be positive integer.")

            for model, intervals in results:
                intervals = RealSubset(intervals)

                search_count += 1
                searched_intervals = searched_intervals | intervals

                if model_selector(model):
                    detect_count += 1
                    truncated_intervals = truncated_intervals | intervals

            if (
                search_count > max_iter
                or searched_intervals == before_searched_intervals
            ):
                raise InfiniteLoopError()
            before_searched_intervals = searched_intervals

            if termination_criterion(searched_intervals, truncated_intervals):
                break

        p_value = self._compute_pvalue(truncated_intervals)
        inf_p, sup_p = self._evaluate_pvalue_bounds(
            searched_intervals, truncated_intervals
        )
        naive_p = self._compute_pvalue(self.support)

        return SelectiveInferenceResult(
            self.stat,
            significance_level,
            p_value,
            inf_p,
            sup_p,
            naive_p,
            p_value <= significance_level,
            truncated_intervals.intervals,
            search_count,
            detect_count,
        )
