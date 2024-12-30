"""Module containing the base classes for selective inference."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Literal

import numpy as np
from joblib import Parallel, delayed  # type: ignore[import]
from scipy.stats import rv_continuous  # type: ignore[import]
from tqdm import tqdm  # type: ignore[import]

from .cdf import truncated_cdf
from .real_subset import RealSubset


@dataclass
class SelectiveInferenceResult:
    """A class containing the results of selective inference.

    Attributes
    ----------
    stat : float
        Test statistic value.
    p_value : float
        Selective p-value.
    inf_p : float
        Lower bound of selective p-value.
    sup_p : float
        Upper bound of selective p-value.
    searched_intervals : list[list[float]]
        Intervals where the search was performed.
    truncated_intervals : list[list[float]]
        Intervals where the selected model is obtained.
    search_count : int
        Number of times the search was performed.
    detect_count : int
        Number of times the selected model was obtained.
    null_rv : rv_continuous
        Null distribution of the unconditional test statistic.
    alternative : Literal["two-sided", "less", "greater"]
        Type of the alternative hypothesis.
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

        Returns
        -------
        float
            The naive p-value.
        """
        return np.exp(self._log_naive_p_value)

    def bonferroni_p_value(self, log_num_comparisons: float) -> float:
        """Compute the Bonferroni-corrected p-value.

        Parameters
        ----------
        log_num_comparisons : float
            Logarithm of the number of comparisons.

        Returns
        -------
        float
            The Bonferroni-corrected p-value.
        """
        log_bonferroni_p_value = np.clip(
            self._log_naive_p_value + log_num_comparisons,
            -np.inf,
            0.0,
        )
        return np.exp(log_bonferroni_p_value)

    def __str__(self) -> str:
        """Return a string representation of the SelectiveInferenceResult object.

        Returns
        -------
        str
            A string representation of the SelectiveInferenceResult object.
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
            Callable[[RealSubset], float] | Literal["pi1", "pi2", "pi3"]
        ) = "pi3",
        termination_criterion: (
            Callable[[RealSubset, RealSubset, tqdm | None], bool]
            | Literal["precision", "decision"]
        ) = "precision",
        max_iter: int = 100_000,
        n_jobs: int = 1,
        step: float = 1e-6,
        significance_level: float = 0.05,
        precision: float = 0.001,
        *,
        progress: bool = False,
    ) -> SelectiveInferenceResult:
        """Conduct selective inference.

        Parameters
        ----------
        algorithm : Callable[[np.ndarray, np.ndarray, float], tuple[Any, list[list[float]] | RealSubset]]
            Callable function which takes two vectors a (np.ndarray) and
            b (np.ndarray), and a scalar z (float), and returns a model (Any) and
            intervals (list[list[float]] | RealSubset). For any point in
            the intervals, the same model must be selected.
        model_selector : Callable[[Any], bool]
            Callable function which takes a model (Any) and returns a boolean value,
            indicating whether the model is the same as the selected model.
        alternative : Literal["two-sided", "less", "greater"] | None, optional
            Must be one of 'two-sided', 'less', or 'greater' or None.
            If 'two-sided', we consider the two-tailed test.
            If 'less', we consider the right-tailed test.
            If 'greater', we consider the left-tailed test.
            If set to None, defaults to 'two-sided' for the normal distribution
            and 'less' for the chi distribution. Defaults to None.
        inference_mode : Literal["parametric", "exhaustive", "over_conditioning"], optional
            Must be one of 'parametric', 'exhaustive',or 'over_conditioning'.
            Defaults to 'parametric'.
        search_strategy : Callable[[RealSubset], float] | Literal["pi1", "pi2", "pi3"], optional
            Callable function which takes a searched_intervals (RealSubset) and returns next search point (float).
            If not callable, it must be one of 'pi1', 'pi2', 'pi3', or 'parallel'.
            If 'pi1', the search strategy focuses on the truncated intervals.
            If 'pi2', the search strategy focuses on the searched intervals.
            If 'pi3', the search strategy focuses on the both of the truncated and searched intervals.
            This option is ignored when the `inference_mode` is 'exhaustive' or 'over_conditioning'.
            Defaults to 'pi3'.
        termination_criterion : Callable[[RealSubset, RealSubset, tqdm | None], bool] | Literal["precision", "decision"], optional
            Callable function which takes searched_intervals (RealSubset),
            truncated_intervals (RealSubset) and progress bar (tqdm, optional),
            updates the progress bar, and returns a boolean value, indicating
            whether the search should be terminated.
            If not callable, it must be one of 'precision' or 'decision'.
            If 'precision', the termination criterion is based on the precision in the computation of the p-value.
            If 'decision', the termination criterion is based on the decision result by the p-value.
            This option is ignored when the `inference_mode` is 'exhaustive' or 'over_conditioning'.
            Defaults to 'precision'.
        max_iter : int, optional
            Maximum number of iterations. Defaults to 100_000.
        n_jobs : int, optional
            Number of jobs to run in parallel. If set to other than 1, `inference_mode` is forced to
            'exhaustive' and then options `search_strategy` and `termination_criterion` are ignored.
            If set to -1, the all available cores are used. Defaults to 1.
        step : float, optional
            Step size for the search strategy. Defaults to 1e-6.
        significance_level : float, optional
            Significance level only for the termination criterion 'decision'. Defaults to 0.05.
        precision : float, optional
            Precision only for the termination criterion 'precision'. Defaults to 0.001.
        progress : bool, optional
            Whether to show the progress bar. Defaults to `False`.

        Raises
        ------
        InfiniteLoopError
            If the search falls into an infinite loop.

        Returns
        -------
        SelectiveInferenceResult
            The result of the selective inference.
        """
        self.max_iter = max_iter
        self.step = step
        self.significance_level = significance_level
        self.precision = precision
        self.progress = progress

        if alternative is not None:
            self.alternative = alternative

        if n_jobs > 1 or n_jobs == -1:
            return self._inference_parallel(
                algorithm,
                model_selector,
                n_jobs,
                progress=progress,
            )

        bar = None
        if progress:
            total = 100
            bar = tqdm(
                total=total,
                desc="Progress",
                unit="%",
                bar_format="{desc}: {percentage:3.2f}{unit}|{bar}| [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            )

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

        searched_intervals = RealSubset()
        truncated_intervals = RealSubset()
        search_count, detect_count = 0, 0

        before_searched_intervals = RealSubset()
        while True:
            z = search_strategy(searched_intervals)
            model, intervals_ = algorithm(self.a, self.b, z)
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

            if termination_criterion(searched_intervals, truncated_intervals, bar):
                break

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

        Parameters
        ----------
        truncated_intervals : RealSubset
            The truncated intervals.

        Returns
        -------
        float
            The p-value from the truncated intervals.
        """
        absolute = self.alternative == "two-sided"
        cdf_value = truncated_cdf(
            self.null_rv,
            self.stat,
            truncated_intervals,
            absolute=absolute,
        )
        return self._convert_cdf_value_to_pvalue(cdf_value)

    def _evaluate_pvalue_bounds(
        self,
        searched_intervals: RealSubset,
        truncated_intervals: RealSubset,
    ) -> tuple[float, float]:
        """Evaluate the bounds of the p-value from the truncated and searched intervals.

        Parameters
        ----------
        searched_intervals : RealSubset
            The searched intervals.
        truncated_intervals : RealSubset
            The truncated intervals.

        Returns
        -------
        tuple[float, float]
            The lower and upper bounds of the p-value.
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

        inf_cdf = truncated_cdf(
            self.null_rv,
            self.stat,
            inf_intervals,
            absolute=absolute,
        )
        sup_cdf = truncated_cdf(
            self.null_rv,
            self.stat,
            sup_intervals,
            absolute=absolute,
        )

        p_value_from_inf = self._convert_cdf_value_to_pvalue(inf_cdf)
        p_value_from_sup = self._convert_cdf_value_to_pvalue(sup_cdf)
        inf_p, sup_p = np.sort([p_value_from_inf, p_value_from_sup])
        return inf_p, sup_p

    def _convert_cdf_value_to_pvalue(
        self,
        cdf_value: float,
    ) -> float:
        """Convert the CDF value to the p-value.

        Parameters
        ----------
        cdf_value : float
            The CDF value.

        Returns
        -------
        float
            The p-value.
        """
        match self.alternative:
            case "two-sided" | "less":
                return float(1.0 - cdf_value)
            case "greater":
                return float(cdf_value)

    def _create_search_strategy(
        self,
        inference_mode: Literal["parametric", "exhaustive", "over_conditioning"],
        search_strategy_name: Literal["pi1", "pi2", "pi3"],
    ) -> Callable[[RealSubset], float]:
        """Create a search strategy.

        Parameters
        ----------
        inference_mode : Literal["parametric", "exhaustive", "over_conditioning"]
            Must be one of 'parametric', 'exhaustive', or 'over_conditioning'.
        search_strategy_name : Literal["pi1", "pi2", "pi3"]
            Must be one of 'pi1', 'pi2', 'pi3', or 'parallel'.
            If 'pi1', the search strategy focuses on the truncated intervals.
            If 'pi2', the search strategy focuses on the searched intervals.
            If 'pi3', the search strategy focuses on the both of the truncated and searched intervals.

        Returns
        -------
        Callable[[RealSubset], float]
            The search strategy.
        """
        match inference_mode:
            case "exhaustive", _:
                return lambda searched_intervals: (
                    self.limits.intervals[0][0]
                    if searched_intervals.is_empty()
                    else searched_intervals.intervals[0][1] + self.step
                )

            case "over_conditioning":
                return lambda _: self.stat

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

        def search_strategy(searched_intervals: RealSubset) -> float:
            if searched_intervals.is_empty():
                return self.stat
            unsearched_intervals = self.support - searched_intervals
            if target_value in unsearched_intervals:
                return target_value

            candidates, min_step = [], 1e-11
            l, u = searched_intervals.find_interval_containing(target_value)
            for candidate, step_ in [(l, -self.step), (u, self.step)]:
                step = step_
                if candidate in unsearched_intervals and np.isfinite(candidate):
                    while np.abs(step) > min_step:
                        if candidate + step in unsearched_intervals:
                            candidates.append(candidate + step)
                            break
                        step /= 10
            return np.array(candidates)[np.argmin(metric(candidates))]

        return search_strategy

    def _create_termination_criterion(
        self,
        inference_mode: Literal["parametric", "exhaustive", "over_conditioning"],
        termination_criterion_name: Literal["precision", "decision"],
    ) -> Callable[[RealSubset, RealSubset, tqdm | None], bool]:
        """Create a termination criterion.

        Parameters
        ----------
        inference_mode : Literal["parametric", "exhaustive", "over_conditioning"]
            Must be one of 'parametric', 'exhaustive', or 'over_conditioning'.
        termination_criterion_name : Literal["precision", "decision"]
            Must be one of 'precision' or 'decision'.
            If 'precision', the termination criterion is based on
            the precision in the computation of the p-value.
            If 'decision', the termination criterion is based on
            the decision result by the p-value.

        Returns
        -------
        Callable[[RealSubset, RealSubset], list[float]]
            The termination criterion.
        """
        match inference_mode, termination_criterion_name:
            case "exhaustive", _:

                def termination_criterion(
                    searched_intervals: RealSubset,
                    truncated_intervals: RealSubset,
                    bar: tqdm | None = None,
                ) -> bool:
                    _ = truncated_intervals
                    flag = self.limits <= searched_intervals
                    if bar is not None:
                        if flag:
                            bar.update(bar.total - bar.n)
                        else:
                            ratio = (
                                searched_intervals & self.limits
                            ).measure / self.limits.measure
                            bar.update(bar.total * ratio - bar.n)
                    return flag

                return termination_criterion

            case "over_conditioning", _:

                def termination_criterion(
                    searched_intervals: RealSubset,
                    truncated_intervals: RealSubset,
                    bar: tqdm | None = None,
                ) -> bool:
                    _ = searched_intervals, truncated_intervals
                    if bar is not None:
                        bar.update(bar.total)
                    return True

            case "parametric", "precision":

                def termination_criterion(
                    searched_intervals: RealSubset,
                    truncated_intervals: RealSubset,
                    bar: tqdm | None = None,
                ) -> bool:
                    inf_p, sup_p = self._evaluate_pvalue_bounds(
                        searched_intervals,
                        truncated_intervals,
                    )
                    value = np.abs(sup_p - inf_p)
                    if bar is not None:
                        shift = 0.001
                        start, end = 1.0, self.precision
                        scale = 1.0 / np.log((end + shift) / (start + shift))
                        bias = -scale * np.log(start + shift)
                        current = bar.total * (
                            scale * np.log(np.max([value, end]) + shift) + bias
                        )
                        bar.update(current - bar.n)
                    return value < self.precision

                return termination_criterion

            case "parametric", "decision":

                def termination_criterion(
                    searched_intervals: RealSubset,
                    truncated_intervals: RealSubset,
                    bar: tqdm | None = None,
                ) -> bool:
                    inf_p, sup_p = self._evaluate_pvalue_bounds(
                        searched_intervals,
                        truncated_intervals,
                    )
                    alpha = self.significance_level
                    value = np.min([alpha - inf_p, sup_p - alpha])
                    if bar is not None:
                        shift = 0.001
                        start, end = np.min([alpha, 1.0 - alpha]), 0.0
                        scale = 1.0 / np.log((end + shift) / (start + shift))
                        bias = -scale * np.log(start + shift)
                        current = bar.total * (
                            scale * np.log(np.max([value, end]) + shift) + bias
                        )
                        bar.update(current - bar.n)
                    return value < 0.0

                return termination_criterion

        return termination_criterion

    def _inference_parallel(
        self,
        algorithm: Callable[
            [np.ndarray, np.ndarray, float],
            tuple[Any, list[list[float]] | RealSubset],
        ],
        model_selector: Callable[[Any], bool],
        n_jobs: int,
        *,
        progress: bool = False,
    ) -> SelectiveInferenceResult:
        """Inference in parallel.

        Parameters
        ----------
        algorithm : Callable[[np.ndarray, np.ndarray, float], tuple[Any, list[list[float]] | RealSubset]])
            Callable function which takes two vectors a (np.ndarray) and
            b (np.ndarray), and a scalar z (float), and returns a model (Any) and
            intervals (list[list[float]] | RealSubset). For any point in
            the intervals, the same model must be selected.
        model_selector : Callable[[Any], bool]
            Callable function which takes a model (Any) and returns a boolean value,
            indicating whether the model is the same as the selected model.
        n_jobs : int
            Number of jobs to run in parallel. If set to -1, the all available cores are used.
        progress : bool, optional
            Whether to show the progress bar. Defaults to `False`.

        Returns
        -------
        SelectiveInferenceResult
            The result of the selective inference.
        """
        interval_list = []
        current_point = self.limits.intervals[0][0]
        each_length = self.limits.measure / n_jobs
        for _ in range(n_jobs):
            interval = RealSubset([[current_point, current_point + each_length]])
            interval_list.append(interval)
            current_point += each_length

        searched_intervals = RealSubset()
        truncated_intervals = RealSubset()
        search_count, detect_count = 0, 0

        with Parallel(n_jobs=n_jobs) as parallel:
            results = parallel(
                delayed(_search_interval)(
                    algorithm,
                    model_selector,
                    self.a,
                    self.b,
                    each_interval,
                    job_id=job_id,
                    max_iter=int(self.max_iter / n_jobs),
                    progress=progress,
                )
                for job_id, each_interval in enumerate(interval_list)
            )
        for result in results:
            (
                searched_intervals_,
                truncated_intervals_,
                search_count_,
                detect_count_,
            ) = result
            searched_intervals = searched_intervals | searched_intervals_
            truncated_intervals = truncated_intervals | truncated_intervals_
            search_count += search_count_
            detect_count += detect_count_

        return SelectiveInferenceResult(
            self.stat,
            self._compute_pvalue(truncated_intervals),
            *self._evaluate_pvalue_bounds(searched_intervals, truncated_intervals),
            searched_intervals.tolist(),
            truncated_intervals.tolist(),
            search_count,
            detect_count,
            self.null_rv,
            self.alternative,
        )


def _search_interval(
    algorithm: Callable[
        [np.ndarray, np.ndarray, float],
        tuple[Any, list[list[float]] | RealSubset],
    ],
    model_selector: Callable[[Any], bool],
    a: np.ndarray,
    b: np.ndarray,
    each_interval: RealSubset,
    job_id: int,
    max_iter: int,
    *,
    progress: bool = False,
) -> tuple[RealSubset, RealSubset, int, int]:
    """Search the interval for the parallel processing.

    Parameters
    ----------
    algorithm : Callable[[np.ndarray, np.ndarray, float], tuple[Any, list[list[float]] | RealSubset]])
        Callable function which takes two vectors a (np.ndarray) and
        b (np.ndarray), and a scalar z (float), and returns a model (Any) and
        intervals (list[list[float]] | RealSubset). For any point in
        the intervals, the same model must be selected.
    model_selector : Callable[[Any], bool]
        Callable function which takes a model (Any) and returns a boolean value,
        indicating whether the model is the same as the selected model.
    a : np.ndarray
        Search direction vector, whose shape is same to the data.
    b : np.ndarray
        Search direction vector, whose shape is same to the data.
    each_interval : RealSubset
        The interval for the search.
    job_id : int
        Job ID for the parallel processing.
    max_iter : int
        Maximum number of iterations.
    progress : bool, optional
        Whether to show the progress bar. Defaults to `False`.

    Returns
    -------
    tuple[RealSubset, RealSubset, int, int]
        The searched intervals, the truncated intervals,
        the number of times the search was performed,
        and the number of times the selected model was obtained.
    """
    searched_intervals = RealSubset()
    truncated_intervals = RealSubset()
    search_count, detect_count = 0, 0

    if progress:
        total = 100
        bar = tqdm(
            total=total,
            desc=f"Progress at job {job_id:02}",
            unit="%",
            bar_format="{desc}: {percentage:3.2f}{unit}|{bar}| [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        )

    z = (each_interval.intervals[0][1] - each_interval.intervals[0][0]) / 2

    before_searched_intervals = RealSubset()
    while True:
        model, intervals_ = algorithm(a, b, z)
        intervals = (
            intervals_ if isinstance(intervals_, RealSubset) else RealSubset(intervals_)
        )
        search_count += 1
        searched_intervals = searched_intervals | intervals
        if model_selector(model):
            detect_count += 1
            truncated_intervals = truncated_intervals | intervals

        unsearched_intervals = each_interval - searched_intervals
        if unsearched_intervals.is_empty():
            if progress:
                bar.update(total - bar.n)
            break

        if search_count > max_iter:
            raise InfiniteLoopError(LoopType.ITER)
        if searched_intervals == before_searched_intervals:
            raise InfiniteLoopError(LoopType.SAME)
        before_searched_intervals = searched_intervals

        if progress:
            current = (
                total
                * (searched_intervals & each_interval).measure
                / each_interval.measure
            )
            bar.update(current - bar.n)

        l, u = unsearched_intervals.intervals[0].tolist()
        z = (l + u) / 2
    return searched_intervals, truncated_intervals, search_count, detect_count
