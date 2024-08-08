import numpy as np
from scipy import sparse
from scipy.stats import norm

from typing import Callable

from .real_subset import RealSubset
from .cdf import tn_cdf
from .base import (
    SelectiveInferenceResult,
    InfiniteLoopError,
    Inference,
    compute_pvalue,
    compute_pvalue_bounds,
)


class SelectiveInferenceNorm(Inference):
    """A class conducting selective inference for the normal distribution.

    Args:
        Inference (_type_): _description_
    """

    def __init__(
        self,
        data: np.ndarray,
        var: float | np.ndarray | sparse.csr_array,
        eta: np.ndarray,
        use_sparse: bool = False,
        use_tf: bool = False,
        use_torch: bool = False,
    ):
        if np.sum([use_sparse, use_tf, use_torch]) > 1:
            raise ValueError(
                "Only one of use_sparse, use_tf, and use_torch can be True."
            )

        if use_tf:
            try:
                import tensorflow as tf
            except ModuleNotFoundError:
                raise ModuleNotFoundError("TensorFlow is not installed.")

            assert isinstance(data, tf.Tensor), "data must be a TensorFlow tensor."
            assert isinstance(eta, tf.Tensor), "eta must be a TensorFlow tensor."

            if isinstance(var, float):
                self.sigma_eta = var * eta
            elif len(var.shape) == 1:
                vars = tf.constant(var, dtype=data.dtype)
                self.sigma_eta = vars * eta
            elif len(var.shape) == 2:
                cov = tf.constant(var, dtype=data.dtype)
                self.sigma_eta = tf.tensordot(cov, eta, axes=1)
            self.eta_sigma_eta = tf.tensordot(eta, self.sigma_eta, axes=1)
            self.sqrt_eta_sigma_eta = tf.sqrt(self.eta_sigma_eta)
            self.stat = tf.tensordot(eta, data, axes=1) / self.sqrt_eta_sigma_eta

        elif use_torch:
            try:
                import torch
            except ModuleNotFoundError:
                raise Exception("Pytorch is not installed")

            assert isinstance(data, torch.Tensor), "data must be a PyTorch tensor."
            assert isinstance(eta, torch.Tensor), "eta must be a PyTorch tensor."

            if isinstance(var, float):
                self.sigma_eta = var * eta
            elif len(var.shape) == 1:
                vars = torch.tensor(var, dtype=data.dtype)
                self.sigma_eta = vars * eta
            elif len(var.shape) == 2:
                cov = torch.tensor(var, dtype=data.dtype)
                self.sigma_eta = torch.mv(cov, eta)
            self.eta_sigma_eta = torch.dot(eta, self.sigma_eta)
            self.sqrt_eta_sigma_eta = torch.sqrt(self.eta_sigma_eta)
            self.stat = torch.dot(eta, data) / self.sqrt_eta_sigma_eta

        else:
            data, eta = np.array(data), np.array(eta)
            if isinstance(var, float):
                self.sigma_eta = var * eta
            elif len(var.shape) == 1:
                vars = np.array(var)
                self.sigma_eta = vars * eta
            elif len(var.shape) == 2:
                cov = sparse.csr_array(var) if use_sparse else np.array(var)
                self.sigma_eta = cov @ eta
            self.eta_sigma_eta = eta @ self.sigma_eta
            self.sqrt_eta_sigma_eta = np.sqrt(self.eta_sigma_eta)
            self.stat = eta @ data / self.sqrt_eta_sigma_eta

        self.stat = float(self.stat)

        self.b = self.sigma_eta / self.sqrt_eta_sigma_eta
        self.a = data - self.stat * self.b

        self.support = RealSubset([[-np.inf, np.inf]])
        # lim = np.max([30.0, np.abs(self.stat) + 10.0])
        # self.limits = RealSubset([[-lim, lim]])
        self.limits = (
            RealSubset([[-10.0 - np.abs(self.stat), 10.0 + np.abs(self.stat)]])
            & self.support
        )
        self.mode = 0.0
        self.null_rv = norm()

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

            # case "parametric", "pi1":

            #     def search_strategy(intervals: RealSubset) -> list[float]:
            #         if intervals == RealSubset():
            #             return [self.stat]
            #         for lower, upper in intervals.intervals:
            #             if lower <= self.stat <= upper:
            #                 if np.abs(self.stat - lower) < np.abs(self.stat - upper):
            #                     return [lower - self.step]
            #                 else:
            #                     return [upper + self.step]

            #     return search_strategy

            # case "parametric", "pi2":

            #     def search_strategy(intervals: RealSubset) -> list[float]:
            #         if intervals == RealSubset():
            #             return [self.stat]
            #         if 0.0 not in intervals:
            #             return [0.0]
            #         for lower, upper in intervals.intervals:
            #             if lower <= self.mode <= upper:
            #                 if norm.logpdf(lower) > norm.logpdf(upper):
            #                     return [lower - self.step]
            #                 else:
            #                     return [upper + self.step]

            #     return search_strategy

            # case "parametric", "pi3":

            #     def search_strategy(intervals: RealSubset) -> list[float]:
            #         if intervals == RealSubset():
            #             return [self.stat]
            #         for lower, upper in intervals.intervals:
            #             if lower <= self.stat <= upper:
            #                 if norm.logpdf(lower) > norm.logpdf(upper):
            #                     return [lower - self.step]
            #                 else:
            #                     return [upper + self.step]

            #     return search_strategy

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
