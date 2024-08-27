"""Module containing the base classes for selective inference."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Literal

import numpy as np
from joblib import Parallel, delayed  # type: ignore[import]
from scipy.stats import rv_continuous  # type: ignore[import]

from .real_subset import RealSubset


@dataclass
class SelectiveInferenceResult:
    """A class containing the results of selective inference.

    Attributes:
        stat (float): Test statistic value.
        p_value (float): Selective p-value.
        inf_p (float): Lower bound of selective p-value.
        sup_p (float): Upper bound of selective p-value.
        searched_intervals (list[list[float]]):
            Intervals where the search was performed.
        truncated_intervals (list[list[float]]):
            Intervals where the selected model is obtained.
        search_count (int): Number of times the search was performed.
        detect_count (int): Number of times the selected model was obtained.
        null_rv (rv_continuous): Null distribution of the unconditional test statistic.
        alternative (Literal["two-sided", "less", "greater"]): Type of the
            alternative hypothesis.
    """

    stat: float
    p_value: float
    inf_p: float
    sup_p: float
    searched_intervals: list[list[float]]
    truncated_intervals: list[list[float]]
    search_count: int
    detect_count: int
    null_rv: rv_continuous
    alternative: Literal["two-sided", "less", "greater"]

    def __post_init__(self) -> None:
        """Compute the logarithm of the naive p-value and store it in the cache."""
        match self.alternative:
            case "two-sided":
                self._log_naive_p_value = np.log(2.0) + self.null_rv.logcdf(
                    -np.abs(self.stat),
                )
            case "less":
                self._log_naive_p_value = self.null_rv.logsf(self.stat)
            case "greater":
                self._log_naive_p_value = self.null_rv.logcdf(self.stat)

    def naive_p_value(self) -> float:
        """Compute the naive p-value.

        Returns:
            float: The naive p-value.
        """
        return np.exp(self._log_naive_p_value)

    def bonferroni_p_value(self, log_num_comparisons: float) -> float:
        """Compute the Bonferroni-corrected p-value.

        Args:
            log_num_comparisons (float): Logarithm of the number of comparisons.

        Returns:
            float: The Bonferroni-corrected p-value.
        """
        log_bonferroni_p_value = np.clip(
            self._log_naive_p_value + log_num_comparisons,
            -np.inf,
            0.0,
        )
        return np.exp(log_bonferroni_p_value)

    def __str__(self) -> str:
        """Return a string representation of the SelectiveInferenceResult object.

        Returns:
            str: A string representation of the SelectiveInferenceResult object.
        """
        precision = 6

        def _convert(intervals: list[list[float]], precision: float) -> str:
            return (
                "["
                + ", ".join(
                    [f"[{l:.{precision}f}, {u:.{precision}f}]" for l, u in intervals],
                )
                + "]"
            )

        return "\n".join(
            [
                f"stat: {self.stat:.{precision}f}",
                f"p_value: {self.p_value:.{precision}f}",
                f"inf_p: {self.inf_p:.{precision}f}",
                f"sup_p: {self.sup_p:.{precision}f}",
                f"searched_intervals: {_convert(self.searched_intervals, precision)}",
                f"truncated_intervals: {_convert(self.truncated_intervals, precision)}",
                f"search_count: {self.search_count}",
                f"detect_count: {self.detect_count}",
                f"null_rv: {self.null_rv.dist.name}",
                f"alternative: {self.alternative}",
            ],
        )


class LoopType(Enum):
    """An enumeration class for loop types."""

    ITER = auto()
    SAME = auto()


class InfiniteLoopError(Exception):
    """Exception raised when infinite loop errors occur."""

    def __init__(self, loop_type: LoopType) -> None:
        """Initialize an InfiniteLoopError object."""
        if loop_type == LoopType.ITER:
            message = (
                "The search was performed a specified times and may have fallen"
                "into an infinite loop."
            )
        elif loop_type == LoopType.SAME:
            message = "The search did not proceed and fell into an infinite loop."
        super().__init__(message)


class InvalidAlternativeError(Exception):
    """Exception when raised invalid alternative is given."""

    def __init__(self, alternative: str) -> None:
        """Initialize an InvalidAlternativeError object."""
        super().__init__(
            f"'{alternative}' is not valid, must be 'two-sided', 'less', or 'greater'.",
        )


def _compute_pvalue(
    cdf_value: float,
    alternative: Literal["two-sided", "less", "greater"],
) -> float:
    """Compute the p-value from the CDF value.

    Args:
        cdf_value (float): The CDF value.
        alternative (Literal["two-sided", "less", "greater"]):
            Must be one of 'two-sided', 'less', or 'greater'.
            If 'two-sided', the p-value is computed for the two-tailed test.
            If 'less', the p-value is computed for the right-tailed test.
            If 'greater', the p-value is computed for the left-tailed test.

    Returns:
        float: The p-value.

    Raises:
        ValueError: If `alternative` is not one of 'two-sided', 'less', or 'greater'.
    """
    match alternative:
        case "two-sided" | "less":
            return float(1.0 - cdf_value)
        case "greater":
            return float(cdf_value)
        case _:
            raise InvalidAlternativeError(alternative)


def _evaluate_pvalue_bounds(
    inf_cdf: float,
    sup_cdf: float,
    alternative: Literal["two-sided", "less", "greater"],
) -> tuple[float, float]:
    """Evaluate the bounds of the p-value from the bounds of the CDF values.

    Args:
        inf_cdf (float): The lower bound of the CDF value.
        sup_cdf (float): The upper bound of the CDF value.
        alternative (Literal["two-sided", "less", "greater"]):
            Must be one of 'two-sided', 'less', or 'greater'.
            If 'two sided', the p-value is computed for the two-tailed test.
            If 'less', the p-value is computed for the right-tailed test.
            If 'greater', the p-value is computed for the left-tailed test.

    Returns:
        tuple[float, float]: The lower and upper bounds of the p-value.
    """
    p_value_from_inf = _compute_pvalue(inf_cdf, alternative)
    p_value_from_sup = _compute_pvalue(sup_cdf, alternative)
    inf_p, sup_p = np.sort([p_value_from_inf, p_value_from_sup])
    return inf_p, sup_p


class SelectiveInference:
    """An abstract class conducting selective inference.

    This class provides the basic structure for conducting selective inference.
    The user can inherit this class and implement the `__init__` method.
    """

    def __init__(self) -> None:
        """Initialize a SelectiveInference object."""
        self.stat: float

        self.a: np.ndarray
        self.b: np.ndarray

        self.support: RealSubset
        self.limits: RealSubset

        self.null_rv: rv_continuous
        self.mode: float
        self.alternative: Literal["two-sided", "less", "greater"]

        self.truncated_cdf: Callable[[float, RealSubset, bool], float]

    def inference(
        self,
        algorithm: Callable[
            [np.ndarray, np.ndarray, float],
            tuple[Any, list[list[float]] | RealSubset],
        ],
        model_selector: Callable[[Any], bool],
        alternative: Literal["two-sided", "less", "greater"] | None = None,
        inference_mode: Literal[
            "parametric",
            "exhaustive",
            "over_conditioning",
        ] = "parametric",
        search_strategy: (
            Callable[[RealSubset], list[float]]
            | Literal["pi1", "pi2", "pi3", "parallel"]
        ) = "pi3",
        termination_criterion: (
            Callable[[RealSubset, RealSubset], bool] | Literal["precision", "decision"]
        ) = "precision",
        max_iter: int = 100_000,
        n_jobs: int = 1,
        step: float = 1e-6,
        significance_level: float = 0.05,
        precision: float = 0.001,
    ) -> SelectiveInferenceResult:
        """Conduct selective inference.

        Args:
            algorithm (Callable[[np.ndarray, np.ndarray, float], tuple[Any, list[list[float]] | RealSubset]]):
                Callable function which takes two vectors a (np.ndarray) and
                b (np.ndarray), and a scalar z (float), and returns a model (Any) and
                intervals (list[list[float]] | RealSubset). For any point in
                the intervals, the same model must be selected.
            model_selector (Callable[[Any], bool]):
                Callable function which takes a model (Any) and returns a boolean value,
                indicating whether the model is the same as the selected model.
            alternative (Literal["two-sided", "less", "greater"] | None, optional):
                Must be one of 'two-sided', 'less', or 'greater' or None.
                If 'two-sided', we consider the two-tailed test.
                If 'less', we consider the right-tailed test.
                If 'greater', we consider the left-tailed test.
                If set to None, defaults to 'two-sided' for the normal distribution
                and 'less' for the chi distribution. Defaults to None.
            inference_mode (Literal["parametric", "exhaustive", "over_conditioning"], optional):
                Must be one of 'parametric', 'exhaustive',or 'over_conditioning'.
                Defaults to 'parametric'.
            search_strategy (Callable[[RealSubset], list[float]] | Literal["pi1", "pi2", "pi3", "parallel"], optional):
                Callable function which takes a searched_intervals (RealSubset) and
                returns next search points (list[float]).
                If not callable, it must be one of 'pi1', 'pi2', 'pi3', or 'parallel'.
                If 'pi1', the search strategy focuses on the truncated intervals.
                If 'pi2', the search strategy focuses on the searched intervals.
                If 'pi3', the search strategy focuses on the both of the truncated
                and searched intervals.
                If 'parallel', the search strategy focuses on the both of the
                truncated and searched intervals for the parallel computing.
                This option is ignored when the inference_mode is
                'exhaustive' or 'over_conditioning'. Defaults to 'pi3'.
            termination_criterion (Callable[[RealSubset, RealSubset], bool] | Literal["precision", "decision"], optional):
                Callable function which takes searched_intervals (RealSubset) and
                truncated_intervals (RealSubset) and returns a boolean value, indicating
                whether the search should be terminated.
                If not callable, it must be one of 'precision' or 'decision'.
                If 'precision', the termination criterion is based on
                the precision in the computation of the p-value.
                If 'decision', the termination criterion is based on
                the decision result by the p-value.
                This option is ignored when the inference_mode is 'exhaustive' or
                'over_conditioning'. Defaults to 'precision'.
            max_iter (int, optional):
                Maximum number of iterations. Defaults to 100_000.
            n_jobs (int, optional):
                Number of jobs to run in parallel. Defaults to 1.
            step (float, optional):
                Step size for the search strategy. Defaults to 1e-6.
            significance_level (float, optional):
                Significance level only for the termination criterion 'decision'.
                Defaults to 0.05.
            precision (float, optional):
                Precision only for the termination criterion 'precision'.
                Defaults to 0.001.

        Raises:
            InfiniteLoopError: If the search falls into an infinite loop.

        Returns:
            SelectiveInferenceResult: The result of the selective inference.
        """
        self.n_jobs = n_jobs
        self.step = step
        self.significance_level = significance_level
        self.precision = precision

        if not callable(search_strategy):
            search_strategy = self._create_search_strategy(
                inference_mode,
                search_strategy,
            )
        if not callable(termination_criterion):
            termination_criterion = self._create_termination_criterion(
                inference_mode,
                termination_criterion,
            )

        if alternative is not None:
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
            else:
                with Parallel(n_jobs=n_jobs) as parallel:
                    results = parallel(
                        delayed(algorithm)(self.a, self.b, z) for z in z_list
                    )

            for model, intervals_ in results:
                intervals = (
                    intervals_
                    if isinstance(intervals_, RealSubset)
                    else RealSubset(intervals_)
                )

                search_count += 1
                searched_intervals = searched_intervals | intervals

                if model_selector(model):
                    detect_count += 1
                    truncated_intervals = truncated_intervals | intervals

            if search_count > max_iter:
                raise InfiniteLoopError(LoopType.ITER)
            if searched_intervals == before_searched_intervals:
                raise InfiniteLoopError(LoopType.SAME)
            before_searched_intervals = searched_intervals

            if termination_criterion(searched_intervals, truncated_intervals):
                break

        finites = truncated_intervals.intervals[
            np.isfinite(truncated_intervals.intervals)
        ]
        min_finite, max_finite = np.min(finites), np.max(finites)
        if min_finite not in self.limits or max_finite not in self.limits:
            p_value = self._compute_pvalue(truncated_intervals & self.limits)
        else:
            p_value = self._compute_pvalue(truncated_intervals)

        inf_p, sup_p = self._evaluate_pvalue_bounds(
            searched_intervals,
            truncated_intervals,
        )

        return SelectiveInferenceResult(
            self.stat,
            p_value,
            inf_p,
            sup_p,
            searched_intervals.tolist(),
            truncated_intervals.tolist(),
            search_count,
            detect_count,
            self.null_rv,
            self.alternative,
        )

    def _compute_pvalue(self, truncated_intervals: RealSubset) -> float:
        """Compute p-value for the given truncated intervals.

        Args:
            truncated_intervals (RealSubset): The truncated intervals.

        Returns:
            float: The p-value from the truncated intervals.
        """
        absolute = self.alternative == "two-sided"
        cdf_value = self.truncated_cdf(self.stat, truncated_intervals, absolute)
        return _compute_pvalue(cdf_value, self.alternative)

    def _evaluate_pvalue_bounds(
        self,
        searched_intervals: RealSubset,
        truncated_intervals: RealSubset,
    ) -> tuple[float, float]:
        """Evaluate the bounds of the p-value from the truncated and searched intervals.

        Args:
            searched_intervals (RealSubset): The searched intervals.
            truncated_intervals (RealSubset): The truncated intervals.

        Returns:
            tuple[float, float]: The lower and upper bounds of the p-value.
        """
        absolute = self.alternative == "two-sided"
        if absolute:
            mask_intervals = RealSubset([[-np.abs(self.stat), np.abs(self.stat)]])
        else:
            mask_intervals = RealSubset([[-np.inf, self.stat]])

        unsearched_intervals = ~searched_intervals

        inf_intervals = truncated_intervals | (unsearched_intervals - mask_intervals)
        sup_intervals = truncated_intervals | (unsearched_intervals & mask_intervals)

        inf_intervals = inf_intervals & self.support
        sup_intervals = sup_intervals & self.support

        inf_finites = inf_intervals.intervals[np.isfinite(inf_intervals.intervals)]
        if len(inf_finites) != 0:
            inf_min_finite, inf_max_finite = np.min(inf_finites), np.max(inf_finites)
            if inf_min_finite not in self.limits or inf_max_finite not in self.limits:
                inf_intervals = inf_intervals & self.limits

        sup_finites = sup_intervals.intervals[np.isfinite(sup_intervals.intervals)]
        if len(sup_finites) != 0:
            sup_min_finite, sup_max_finite = np.min(sup_finites), np.max(sup_finites)
            if sup_min_finite not in self.limits or sup_max_finite not in self.limits:
                sup_intervals = sup_intervals & self.limits

        inf_f = self.truncated_cdf(self.stat, inf_intervals, absolute)
        sup_f = self.truncated_cdf(self.stat, sup_intervals, absolute)

        inf_p, sup_p = _evaluate_pvalue_bounds(inf_f, sup_f, self.alternative)
        return inf_p, sup_p

    def _create_search_strategy(
        self,
        inference_mode: Literal["parametric", "exhaustive", "over_conditioning"],
        search_strategy_name: Literal["pi1", "pi2", "pi3", "parallel"],
    ) -> Callable[[RealSubset], list[float]]:
        """Create a search strategy.

        Args:
            inference_mode (Literal["parametric", "exhaustive", "over_conditioning"]):
                Must be one of 'parametric', 'exhaustive', or 'over_conditioning'.
            search_strategy_name (Literal["pi1", "pi2", "pi3", "parallel"]):
                Must be one of 'pi1', 'pi2', 'pi3', or 'parallel'.
                If 'pi1', the search strategy focuses on the truncated intervals.
                If 'pi2', the search strategy focuses on the searched intervals.
                If 'pi3', the search strategy focuses on the both of
                the truncated and searched intervals.
                If 'parallel', the search strategy focuses on the both of the
                truncated and searched intervals for the parallel computing.

        Returns:
            Callable[[RealSubset], list[float]]: The search strategy.
        """
        match inference_mode, search_strategy_name:
            case "exhaustive", _:
                return lambda searched_intervals: (
                    [self.limits.intervals[0][0]]
                    if searched_intervals.is_empty()
                    else [searched_intervals.intervals[0][1] + self.step]
                )

            case "over_conditioning", _:
                return lambda _: [self.stat]

            case "parametric", "pi1" | "pi2" | "pi3":
                match search_strategy_name:
                    case "pi1":
                        target_value = self.stat

                        def metric(z: list[float]) -> list[float]:
                            return np.abs(np.array(z) - self.stat)
                    case "pi2":
                        target_value = self.mode

                        def metric(z: list[float]) -> list[float]:
                            return -self.null_rv.logpdf(np.array(z))
                    case "pi3":
                        target_value = self.stat

                        def metric(z: list[float]) -> list[float]:
                            return -self.null_rv.logpdf(np.array(z))

                def search_strategy(searched_intervals: RealSubset) -> list[float]:
                    min_step = 1e-11
                    if searched_intervals.is_empty():
                        return [self.stat]
                    unsearched_intervals = self.support - searched_intervals
                    if target_value in unsearched_intervals:
                        return [target_value]

                    candidates = []
                    l, u = searched_intervals.find_interval_containing(target_value)
                    for candidate, step_ in [(l, -self.step), (u, self.step)]:
                        step = step_
                        if candidate in unsearched_intervals and np.isfinite(candidate):
                            while np.abs(step) > min_step:
                                if candidate + step in unsearched_intervals:
                                    candidates.append(candidate + step)
                                    break
                                step /= 10

                    return [np.array(candidates)[np.argmin(metric(candidates))]]

                return search_strategy

            case "parametric", "parallel":

                def search_strategy(searched_intervals: RealSubset) -> list[float]:
                    num_points_per_core = 4
                    num_points = self.n_jobs * num_points_per_core
                    expand_width = 0.5

                    unsearched_intervals = self.support - searched_intervals
                    if self.stat in unsearched_intervals:
                        z_list = [self.stat]
                        loc = self.stat
                    else:
                        z_list = []
                        edges = searched_intervals.find_interval_containing(self.stat)
                        loc = edges[np.argmin(-self.null_rv.logpdf(edges))]

                    tail = 0.0
                    while len(z_list) < num_points:
                        inner, outer = tail, tail + expand_width
                        intervals = unsearched_intervals & RealSubset(
                            [[loc - outer, loc - inner], [loc + inner, loc + outer]],
                        )
                        for l, u in intervals.intervals:
                            if l + self.step < u:
                                z_list += np.arange(
                                    l + self.step,
                                    u,
                                    self.step,
                                ).tolist()
                            else:
                                z_list.append((l + u) / 2)
                        tail = outer
                    return z_list[:num_points]

                return search_strategy

        return search_strategy

    def _create_termination_criterion(
        self,
        inference_mode: Literal["parametric", "exhaustive", "over_conditioning"],
        termination_criterion_name: Literal["precision", "decision"],
    ) -> Callable[[RealSubset, RealSubset], bool]:
        """Create a termination criterion.

        Args:
            inference_mode (Literal["parametric", "exhaustive", "over_conditioning"]):
                Must be one of 'parametric', 'exhaustive', or 'over_conditioning'.
            termination_criterion_name (Literal["precision", "decision"]):
                Must be one of 'precision' or 'decision'.
                If 'precision', the termination criterion is based on
                the precision in the computation of the p-value.
                If 'decision', the termination criterion is based on
                the decision result by the p-value.

        Returns:
            Callable[[RealSubset, RealSubset], list[float]]: The termination criterion.
        """
        match inference_mode, termination_criterion_name:
            case "exhaustive", _:

                def termination_criterion(
                    searched_intervals: RealSubset,
                    truncated_intervals: RealSubset,
                ) -> bool:
                    _ = truncated_intervals
                    return self.limits <= searched_intervals

                return termination_criterion

            case "over_conditioning", _:

                def termination_criterion(
                    searched_intervals: RealSubset,
                    truncated_intervals: RealSubset,
                ) -> bool:
                    _ = searched_intervals, truncated_intervals
                    return True

            case "parametric", "precision":

                def termination_criterion(
                    searched_intervals: RealSubset,
                    truncated_intervals: RealSubset,
                ) -> bool:
                    inf_p, sup_p = self._evaluate_pvalue_bounds(
                        searched_intervals,
                        truncated_intervals,
                    )
                    return np.abs(sup_p - inf_p) < self.precision

                return termination_criterion

            case "parametric", "decision":

                def termination_criterion(
                    searched_intervals: RealSubset,
                    truncated_intervals: RealSubset,
                ) -> bool:
                    inf_p, sup_p = self._evaluate_pvalue_bounds(
                        searched_intervals,
                        truncated_intervals,
                    )
                    return (
                        inf_p > self.significance_level
                        or sup_p <= self.significance_level
                    )

                return termination_criterion

        return termination_criterion
