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

    def _compute_pvalue(self, truncated_intervals: RealSubset) -> float:
        """Compute p-value for the given truncated intervals.

        Args:
            truncated_intervals (RealSubset): The truncated intervals.

        Returns:
            float: The p-value from the truncated intervals.
        """
        absolute = self.alternative == "abs"
        F = tn_cdf(self.stat, truncated_intervals, absolute=absolute)
        return compute_pvalue(F, self.alternative)

    def _evaluate_pvalue_bounds(
        self,
        searched_intervals: RealSubset,
        truncated_intervals: RealSubset,
    ) -> tuple[float, float]:
        """Evaluate the lower and upper bounds of the p-value from the given truncated and searched intervals.

        Args:
            searched_intervals (RealSubset): The searched intervals.
            truncated_intervals (RealSubset): The truncated intervals.

        Returns:
            tuple[float, float]: The lower and upper bounds of the p-value.
        """
        absolute = self.alternative == "abs"
        if absolute:
            mask_intervals = RealSubset([[-np.abs(self.stat), np.abs(self.stat)]])
        else:
            mask_intervals = RealSubset([[-np.inf, self.stat]])

        unserched_intervals = ~searched_intervals

        inf_intervals = truncated_intervals | (unserched_intervals - mask_intervals)
        sup_intervals = truncated_intervals | (unserched_intervals & mask_intervals)

        # TODO: Restrict intervals

        inf_F = tn_cdf(self.stat, inf_intervals, absolute=absolute)
        sup_F = tn_cdf(self.stat, sup_intervals, absolute=absolute)

        inf_p, sup_p = compute_pvalue_bounds(inf_F, sup_F, self.alternative)
        return inf_p, sup_p

    def _create_search_strategy(
        self,
        inference_mode: str = "parametric",
        search_strategy_name: str = "pi3",
    ) -> Callable[[RealSubset], list[float]]:
        """Create a search strategy

        Args:
            inference_mode (str, optional): Must be one of 'parametric', 'exhaustive',
                or 'over_conditioning'. Defaults to 'parametric'.
            search_strategy_name (str, optional): Must be one of 'pi1', 'pi2', 'pi3', or 'parallel'.
                If 'pi1', the search strategy focuses on the truncated intervals.
                If 'pi2', the search strategy focuses on the searched intervals.
                If 'pi3', the search strategy focuses on the both of the truncated and searched intervals.
                If 'parallel', the search strategy focuses on the both of the
                truncated and searched intervals for the parallel computing.
                Defaults to 'pi3'.

        Returns:
            Callable[[RealSubset], list[float]]: The search strategy.
        """
        match inference_mode, search_strategy_name:
            case "exhaustive", _:
                return lambda searched_intervals: (
                    [searched_intervals.intervals[0][1] + self.step]
                    if searched_intervals != RealSubset()
                    else [self.limits.intervals[0][0]]
                )

            case "over_conditioning", _:
                return lambda searched_intervals: [self.stat]

            case "parametric", "pi1" | "pi2" | "pi3":
                match search_strategy_name:
                    case "pi1":
                        target_value = self.stat
                        metric = lambda z: np.abs(z - self.stat)
                    case "pi2":
                        target_value = self.mode
                        metric = lambda z: -self.null_rv.logpdf(z)
                    case "pi3":
                        target_value = self.stat
                        metric = lambda z: -self.null_rv.logpdf(z)

                def search_strategy(searched_intervals: RealSubset) -> list[float]:
                    if searched_intervals == RealSubset():
                        return [self.stat]
                    unsearched_intervals = self.support - searched_intervals
                    if target_value in unsearched_intervals:
                        return [target_value]

                    lower_mask = RealSubset([[-np.inf, target_value]])
                    upper_mask = RealSubset([[target_value, np.inf]])
                    unsearched_lower_intervals = unsearched_intervals & lower_mask
                    unsearched_upper_intervals = unsearched_intervals & upper_mask

                    candidates = []
                    if unsearched_lower_intervals != RealSubset():
                        edge = unsearched_lower_intervals.intervals[-1][1]
                        candidates.append(edge - self.step)
                    if unsearched_upper_intervals != RealSubset():
                        edge = unsearched_upper_intervals.intervals[0][0]
                        candidates.append(edge + self.step)
                    candidates = np.array(candidates)

                    return [candidates[np.argmin(metric(candidates))]]

                return search_strategy

            case _, "parallel":

                def search_strategy(intervals: RealSubset) -> list[float]:
                    rng = np.random.default_rng(0)
                    z_list = [self.stat] if intervals == RealSubset() else []
                    while len(z_list) < self.n_jobs * 5:
                        num = rng.binomial(n=4000, p=0.5)
                        samples_null = rng.normal(size=num) * 1.5
                        sample_stat = rng.normal(
                            loc=self.stat, scale=1.5, size=4000 - num
                        )
                        samples = np.concatenate([samples_null, sample_stat])
                        for z in samples:
                            if z not in intervals:
                                z_list.append(z)
                    return z_list[: self.n_jobs * 5]

                return search_strategy

            case _, _:
                raise ValueError("Invalid mode or name.")

    def _create_termination_criterion(
        self,
        inference_mode: str = "parametric",
        termination_criterion_name: str = "precision",
    ) -> Callable[[RealSubset, RealSubset], bool]:
        """Create a termination criterion

        Args:
            inference_mode (str, optional): Must be one of 'parametric', 'exhaustive',
                or 'over_conditioning'. Defaults to 'parametric'.
            termination_criterion_name (str, optional): Must be one of 'precision' or 'decision'.
                If 'precision', the termination criterion is based on
                the precision in the computation of the p-value.
                If 'decision', the termination criterion is based on
                the decision result by the p-value

        Returns:
            Callable[[RealSubset], list[float]]: The termination criterion.
        """
        match inference_mode, termination_criterion_name:
            case "exhaustive", _:

                def termination_criterion(
                    searched_intervals: RealSubset, truncated_intervals: RealSubset
                ) -> bool:
                    return (self.limits - searched_intervals) == RealSubset()

                return termination_criterion

            case "over_conditioning", _:
                return lambda searched_intervals, truncated_intervals: True

            case "parametric", "precision":

                def termination_criterion(
                    searched_intervals: RealSubset, truncated_intervals: RealSubset
                ) -> bool:
                    inf_p, sup_p = self._evaluate_pvalue_bounds(
                        searched_intervals, truncated_intervals
                    )
                    return np.abs(sup_p - inf_p) < self.precision

                return termination_criterion

            case "parametric", "decision":

                def termination_criterion(
                    searched_intervals: RealSubset, truncated_intervals: RealSubset
                ) -> bool:
                    inf_p, sup_p = self._evaluate_pvalue_bounds(
                        searched_intervals, truncated_intervals
                    )
                    return (
                        inf_p > self.significance_level
                        or sup_p <= self.significance_level
                    )

                return termination_criterion
