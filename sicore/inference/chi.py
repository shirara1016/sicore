import numpy as np
from scipy import sparse
from scipy.stats import chi
from scipy.linalg import fractional_matrix_power

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from ..utils import is_int_or_float
from ..intervals import intersection, not_, union_all, _interval_to_intervals
from ..cdf_mpmath import tc_cdf_mpmath
from .base import (
    SelectiveInferenceResult,
    InfiniteLoopError,
    SearchChecker,
    calc_pvalue,
    calc_prange,
)


class InferenceChi(ABC):
    """Base inference class for a test statistic which follows chi squared distribution under null.

    Args:
        data (np.ndarray, tf.Tensor, torch.Tensor):
            Observation data in 1-D array. When given as a tensor,
            activate the corresponding option.
        var (float, np.ndarray, tf.Tensor, torch.Tensor, sparse.csr_array):
            Value of known variance covariance. Assuming that
            the shape of the input is a scalar or 1-D array or 2-D array,
            the variance-covariance matrix Cov is interpreted as follows.
            When the input is a scalar, Cov = input * Identity.
            When the input is a 1-D array, Cov = diag(input).
            When the input is a 2-D array, Cov = input.
            Also, activate the option, when given as a sparse array.
        P (np.ndarray, tf.Tensor, torch.Tensor, sparse.csr_array):
            Projection matrix in 2-D array. When given as
            a tensor or a sparse array, activate the corresponding option.
        degree (int):
            Dimension of the space projected by P matrix.
        use_sparse (boolean, optional):
            Whether to use sparse matrix or not. Defaults to False.
        use_tf (boolean, optional):
            Whether to use tensorflow or not. Defaults to False.
        use_torch (boolean, optional):
            Whether to use pytorch or not. Defaults to False.
    """

    def __init__(
        self,
        data: np.ndarray,
        var: float | np.ndarray | sparse.csr_array,
        P: np.ndarray | sparse.csr_array,
        use_sparse: bool = False,
        use_tf: bool = False,
        use_torch: bool = False,
    ):
        if use_sparse and (use_tf or use_torch):
            raise Exception(
                "Sparse matrix cannot be used with the deep learning framework at the same time."
            )
        if use_tf and use_torch:
            raise Exception("Only one of tensorflow and pytorch is available.")

        self.data = data  # unnecessary
        self.length = len(data)  # unnecessary
        self.P = P  # unnecessary
        self.degree = int(np.trace(P) + 1e-3)

        if use_tf:
            try:
                import tensorflow as tf
            except ModuleNotFoundError:
                raise Exception(
                    "The option use_tf is activated, but tensorflow was not found."
                )

            assert isinstance(data, tf.Tensor)
            assert isinstance(P, tf.Tensor)

            self.P_data = tf.tensordot(P, data, axes=1)
            if is_int_or_float(var):
                whitend_P_data = (var**-0.5) * self.P_data
            elif len(var.shape) == 1:
                vars = tf.constant(var, dtype=data.dtype)
                whitend_P_data = tf.pow(vars, -0.5) * self.P_data
            elif len(var.shape) == 2:
                cov = np.array(var)
                inv_sqrt_cov = fractional_matrix_power(cov, -0.5)
                inv_sqrt_cov = tf.constant(inv_sqrt_cov, dtype=data.dtype)
                whitend_P_data = tf.tensordot(inv_sqrt_cov, self.P_data, axes=1)
            self.stat = tf.norm(whitend_P_data, ord=2)

        elif use_torch:
            try:
                import torch
            except ModuleNotFoundError:
                raise Exception(
                    "The option use_torch is activated, but pytorch was not found."
                )

            assert isinstance(data, torch.Tensor)
            assert isinstance(P, torch.Tensor)

            self.P_data = torch.mv(P, data)
            if is_int_or_float(var):
                whitend_P_data = (var**-0.5) * self.P_data
            elif len(var.shape) == 1:
                vars = torch.tensor(var, dtype=data.dtype)
                whitend_P_data = torch.pow(vars, -0.5) * self.P_data
            elif len(var.shape) == 2:
                cov = np.array(var)
                inv_sqrt_cov = fractional_matrix_power(cov, -0.5)
                inv_sqrt_cov = torch.tensor(inv_sqrt_cov, dtype=data.dtype)
                whitend_P_data = torch.mv(inv_sqrt_cov, self.P_data)
            self.stat = torch.linalg.norm(whitend_P_data, ord=2)

        else:
            data = np.array(data)
            P = sparse.csr_array(P) if use_sparse else np.array(P)
            self.P_data = P @ data
            if is_int_or_float(var):
                whitend_P_data = (var**-0.5) * self.P_data
            elif len(var.shape) == 1:
                vars = np.array(var)
                whitend_P_data = np.power(vars, -0.5) * self.P_data
            elif len(var.shape) == 2:
                cov = np.array(var)
                inv_sqrt_cov = fractional_matrix_power(cov, -0.5)
                whitend_P_data = inv_sqrt_cov @ self.P_data
            self.stat = np.linalg.norm(whitend_P_data, ord=2)

    @abstractmethod
    def inference(self, *args, **kwargs):
        """Perform statistical inference."""
        pass


class NaiveInferenceChi(InferenceChi):
    """Naive inference for a test statistic which follows chi squared distribution under null.

    Args:
        data (np.ndarray, tf.Tensor, torch.Tensor):
            Observation data in 1-D array. When given as a tensor,
            activate the corresponding option.
        var (float, np.ndarray, tf.Tensor, torch.Tensor, sparse.csr_array):
            Value of known variance covariance. Assuming that
            the shape of the input is a scalar or 1-D array or 2-D array,
            the variance-covariance matrix Cov is interpreted as follows.
            When the input is a scalar, Cov = input * Identity.
            When the input is a 1-D array, Cov = diag(input).
            When the input is a 2-D array, Cov = input.
            Also, activate the option, when given as a sparse array.
        P (np.ndarray, tf.Tensor, torch.Tensor, sparse.csr_array):
            Projection matrix in 2-D array. When given as
            a tensor or a sparse array, activate the corresponding option.
        use_sparse (boolean, optional):
            Whether to use sparse matrix or not. Defaults to False.
        use_tf (boolean, optional):
            Whether to use tensorflow or not. Defaults to False.
        use_torch (boolean, optional):
            Whether to use pytorch or not. Defaults to False.
    """

    def inference(self, alternative: str = "less"):
        """Perform naive statistical inference.

        Args:
            alternative (str, optional):
                'two-sided' for two-tailed test,
                'less' for right-tailed test,
                'greater' for left-tailed test.
                Defaults to 'less'.

        Returns:
            float: p-value
        """
        stat = np.asarray(self.stat)
        F = chi.cdf(stat, self.degree)
        return calc_pvalue(F, alternative=alternative)


class SelectiveInferenceChi(InferenceChi):
    """Selective inference for a test statistic which follows chi squared distribution under null.

    Args:
        data (np.ndarray, tf.Tensor, torch.Tensor):
            Observation data in 1-D array. When given as a tensor,
            activate the corresponding option.
        var (float, np.ndarray, tf.Tensor, torch.Tensor, sparse.csr_array):
            Value of known variance covariance. Assuming that
            the shape of the input is a scalar or 1-D array or 2-D array,
            the variance-covariance matrix Cov is interpreted as follows.
            When the input is a scalar, Cov = input * Identity.
            When the input is a 1-D array, Cov = diag(input).
            When the input is a 2-D array, Cov = input.
            Also, activate the option, when given as a sparse array.
        P (np.ndarray, tf.Tensor, torch.Tensor, sparse.csr_array):
            Projection matrix in 2-D array. When given as
            a tensor or a sparse array, activate the corresponding option.
        use_sparse (boolean, optional):
            Whether to use sparse matrix or not. Defaults to False.
        use_tf (boolean, optional):
            Whether to use tensorflow or not. Defaults to False.
        use_torch (boolean, optional):
            Whether to use pytorch or not. Defaults to False.
    """

    def __init__(
        self,
        data: np.ndarray,
        var: float | np.ndarray | sparse.csr_array,
        P: np.ndarray | sparse.csr_array,
        use_sparse: bool = False,
        use_tf: bool = False,
        use_torch: bool = False,
    ):
        super().__init__(data, var, P, use_sparse, use_tf, use_torch)
        self.c = self.P_data / self.stat  # `b` vector in para si.
        self.z = data - self.P_data  # `a` vector in para si.

    def inference(
        self,
        algorithm: Callable[
            [np.ndarray, np.ndarray, float], tuple[list[list[float]], Any]
        ],
        model_selector: Callable[[Any], bool],
        significance_level: float = 0.05,
        precision: float = 0.001,
        termination_criterion: str = "precision",
        search_strategy: str = "pi3",
        step: float = 1e-10,
        max_iter: int = 1e6,
        over_conditioning: bool = False,
        exhaustive: bool = False,
        max_tail: float = 1e3,
        alternative: str = "abs",
        retain_observed_model: bool = False,
        retain_mappings: bool = False,
        tol: float = 1e-10,
        dps: int | str = "auto",
        max_dps: int = 5000,
        out_log: str = "test_log.log",
    ) -> SelectiveInferenceResult:
        """Perform Selective Inference.

        Args:
            algorithm (Callable[[np.ndarray, np.ndarray, float], Tuple[List[List[float]], Any]]):
                Callable function which takes two vectors (`a`, `b`) and a scalar `z`
                that can satisfy `data = a + b * z` as argument, and
                returns the obtained model (Any) and the intervals (np.ndarray) where it is obtained.
                A closure function might be helpful to implement this.
            model_selector (Callable[[Any], bool]):
                Callable function which takes a obtained model (any) as single argument, and
                returns True if the model is observed one, and False otherwise.
            significance_level (float, optional):
                Significance level for the test. Defaults to 0.05.
            precision (float, optional):
                Precision required to compute p-value. Defaults to 0.001.
            termination_criterion (str, optional):
                Specifies the termination criterion used to perform parametric selective inference.
                This option is ignored when the `over_conditioning` or `exhaustive` is activated.
                'precision' for computing p-value with `precision`.
                'decision' for making decision whether or not to reject at `significance_level`.
                Defaults to 'precision'.
            search_strategy (str, optional):
                Specifies the search strategy used to perform parametric selective inference.
                'pi1' focuses on the integral on
                'pi2', 'pi3' can be specified.
                Defaults to 'pi3'.
            step (float, optional):
                Step width for parametric search. Defaults to 1e-10.
            max_iter (int, optional):
                Maximum number of times to perform parametric search.
                If this value is exceeded, the loop is considered to be infinite.
                Defaults to 1e6.
            over_conditioning (bool, optional):
                Over conditioning inference. Defaults to False.
            exhaustive (bool, optional):
                Exhaustive search of the range specified by `max_tail`. Defaults to False.
            max_tail (float, optional):
                Specifies the range to be searched when exhaustive is activated. Defaults to 1e3.
            alternative (str, optional):
                'abs' for two-tailed test of the absolute value of test statistic.
                'two-sided' for two-tailed test.
                'less' for right-tailed test.
                'greater' for left-tailed test.
                Defaults to 'abs'.
            retain_observed_model (bool, optional):
                Whether retain observed model as returns or not. Defaults to False.
            retain_mappings (bool, optional):
                Whether retain mappings as returns or not. Defaults to False.
            tol (float, optional):
                Tolerance error parameter for intervals. Defaults to 1e-10.
            dps (int | str, optional):
                dps value for mpmath. Set 'auto' to select dps automatically. Defaults to 'auto'.
            max_dps (int, optional):
                Maximum dps value for mpmath. This option is valid when `dps` is set to 'auto'.
                Defaults to 5000.
            out_log (str, optional):
                Name for log file of mpmath. Defaults to 'test_log.log'.

        Raises:
            Exception:
                When `over_conditioning` and `exhaustive` are activated simultaneously

        Returns:
            SelectiveInferenceResult
        """

        lim = max(20 + np.sqrt(self.degree - 1), 10 + float(self.stat))
        self.restrict = [[-lim, lim]]

        if over_conditioning and exhaustive:
            raise Exception(
                "over_conditioning and exhaustive are activated simultaneously"
            )

        if over_conditioning:
            result = self._over_conditioned_inference(
                algorithm,
                significance_level,
                alternative,
                retain_observed_model,
                tol,
                dps,
                max_dps,
                out_log,
            )
            return result

        if exhaustive:
            result = self._exhaustive_parametric_inference(
                algorithm,
                model_selector,
                significance_level,
                step,
                max_iter,
                max_tail,
                alternative,
                retain_observed_model,
                retain_mappings,
                tol,
                dps,
                max_dps,
                out_log,
            )
            return result

        result = self._parametric_inference(
            algorithm,
            model_selector,
            significance_level,
            precision,
            termination_criterion,
            search_strategy,
            step,
            max_iter,
            alternative,
            retain_observed_model,
            retain_mappings,
            tol,
            dps,
            max_dps,
            out_log,
        )

        return result

    def _evaluate_pvalue(self, truncated_intervals, searched_intervals, alternative):
        unsearched_intervals = not_(searched_intervals)

        mask_intervals = [[-np.inf, float(self.stat)]]

        inf_intervals = union_all(
            truncated_intervals
            + intersection(unsearched_intervals, not_(mask_intervals)),
            tol=self.tol,
        )
        sup_intervals = union_all(
            truncated_intervals + intersection(unsearched_intervals, mask_intervals),
            tol=self.tol,
        )

        chi_inf_intervals = intersection(inf_intervals, [[0.0, np.inf]])
        chi_sup_intervals = intersection(sup_intervals, [[0.0, np.inf]])

        flatten = np.ravel(chi_inf_intervals)
        nonfinites = flatten[np.isfinite(flatten)]
        if len(nonfinites) != 0:
            if np.abs(nonfinites).max() > self.restrict[0][1]:
                chi_inf_intervals = intersection(chi_inf_intervals, self.restrict)
        flatten = np.ravel(chi_sup_intervals)
        nonfinites = flatten[np.isfinite(flatten)]
        if len(nonfinites) != 0:
            if np.abs(nonfinites).max() > self.restrict[0][1]:
                chi_sup_intervals = intersection(chi_sup_intervals, self.restrict)

        # chi_inf_intervals = intersection(chi_inf_intervals, self.restrict)
        # chi_sup_intervals = intersection(chi_sup_intervals, self.restrict)

        stat_chi = np.asarray(self.stat)

        inf_F = tc_cdf_mpmath(
            stat_chi,
            chi_inf_intervals,
            self.degree,
            dps=self.dps,
            max_dps=self.max_dps,
            out_log=self.out_log,
        )
        sup_F = tc_cdf_mpmath(
            stat_chi,
            chi_sup_intervals,
            self.degree,
            dps=self.dps,
            max_dps=self.max_dps,
            out_log=self.out_log,
        )

        inf_p, sup_p = calc_prange(inf_F, sup_F, alternative)

        return inf_p, sup_p

    def _determine_next_search_data(self, search_strategy, searched_intervals):
        unsearched_intervals = intersection(not_(searched_intervals), [[0.0, np.inf]])
        candidates = list()
        mode = np.sqrt(self.degree - 1)

        if search_strategy == "pi2":
            for interval in unsearched_intervals:
                l = interval[0]
                if np.isinf(interval[1]):
                    u = max(mode + 2, interval[0] + 2)
                else:
                    u = interval[1]
                if u - l > 2 * self.step:
                    candidates += list(
                        np.linspace(
                            l + self.step,
                            u - self.step,
                            int(1000 / len(unsearched_intervals)),
                        )
                    )

        if search_strategy == "pi1" or search_strategy == "pi3":
            unsearched_lower_stat = intersection(
                unsearched_intervals, [[-np.inf, float(self.stat)]]
            )
            unsearched_upper_stat = intersection(
                unsearched_intervals, [[float(self.stat), np.inf]]
            )
            if len(unsearched_lower_stat) != 0:
                candidates.append(unsearched_lower_stat[-1][-1] - self.step)
            if len(unsearched_upper_stat) != 0:
                candidates.append(unsearched_upper_stat[0][0] + self.step)

        if search_strategy == "pi1":

            def method(z):
                return -np.abs(z - float(self.stat))

        if search_strategy == "pi3" or search_strategy == "pi2":

            def method(z):
                return chi.logpdf(z, self.degree)

        candidates = np.array(candidates)
        return candidates[np.argmax(method(candidates))]

    def _next_search_data(self):
        intervals = not_(self.searched_intervals)
        if len(intervals) == 0:
            return None
        return intervals[0][0] + self.step

    def _execute_callback(self, callback, progress):
        self.search_history.append(callback(progress))

    def _parametric_inference(
        self,
        algorithm,
        model_selector,
        significance_level,
        precision,
        termination_criterion,
        search_strategy,
        step,
        max_iter,
        alternative,
        retain_observed_model,
        retain_mappings,
        tol,
        dps,
        max_dps,
        out_log,
    ):
        self.tol = tol
        self.step = step
        self.dps = dps
        self.max_dps = max_dps
        self.out_log = out_log
        self.searched_intervals = list()

        self.search_history = list()
        self.search_checker = SearchChecker(max_iter)

        mappings = dict() if retain_mappings else None
        truncated_intervals = list()

        search_count = 0
        detect_count = 0

        z = self.stat
        while True:
            search_count += 1

            model, interval = algorithm(self.z, self.c, z)
            interval = np.asarray(interval)
            intervals = _interval_to_intervals(interval)

            if retain_mappings:
                for interval in intervals:
                    interval = tuple(interval)
                    if interval in mappings:
                        raise Exception(
                            "An interval appeared a second time. Usually, numerical error "
                            "causes this exception. Consider increasing the tol parameter "
                            "or decreasing max_tail parameter to avoid it."
                        )
                    mappings[interval] = model

            if model_selector(model):
                selected_model = model if retain_observed_model else None
                truncated_intervals += intervals
                detect_count += 1

            self.searched_intervals = union_all(
                self.searched_intervals + intervals, tol=self.tol
            )
            self.search_checker.verify_progress(self.searched_intervals)

            inf_p, sup_p = self._evaluate_pvalue(
                truncated_intervals, self.searched_intervals, alternative
            )

            if termination_criterion == "precision":
                if np.abs(sup_p - inf_p) < precision:
                    break
            if termination_criterion == "decision":
                if sup_p <= significance_level:
                    reject_or_not = True
                    break
                if inf_p > significance_level:
                    reject_or_not = False
                    break

            z = self._determine_next_search_data(
                search_strategy, self.searched_intervals
            )

        stat_chi = float(self.stat)
        truncated_intervals = union_all(truncated_intervals, tol=self.tol)
        chi_intervals = intersection(truncated_intervals, [[0.0, np.inf]])
        chi_intervals = intersection(chi_intervals, self.restrict)
        F = tc_cdf_mpmath(
            stat_chi,
            chi_intervals,
            self.degree,
            absolute=False,
            dps=self.dps,
            max_dps=self.max_dps,
            out_log=self.out_log,
        )
        p_value = calc_pvalue(F, alternative)
        if termination_criterion == "precision":
            reject_or_not = p_value <= significance_level

        return SelectiveInferenceResult(
            stat_chi,
            significance_level,
            p_value,
            inf_p,
            sup_p,
            reject_or_not,
            chi_intervals,
            search_count,
            detect_count,
            selected_model,
            mappings,
        )

    def _exhaustive_parametric_inference(
        self,
        algorithm,
        model_selector,
        significance_level,
        step,
        max_iter,
        max_tail,
        alternative,
        retain_observed_model,
        retain_mappings,
        tol,
        dps,
        max_dps,
        out_log,
    ):
        self.tol = tol
        self.step = step
        self.dps = dps
        self.max_dps = max_dps
        self.out_log = out_log
        self.searched_intervals = union_all(
            [[-np.inf, 0.0], [float(max_tail), np.inf]], tol=self.tol
        )

        mappings = dict() if retain_mappings else None
        result_intervals = list()

        search_count = 0
        detect_count = 0

        z = self._next_search_data()
        while True:
            search_count += 1
            if search_count > max_iter:
                raise Exception(
                    f"The number of searches exceeds {int(max_iter)} times, suggesting an infinite loop."
                )

            model, interval = algorithm(self.z, self.c, z)
            interval = np.asarray(interval)
            intervals = _interval_to_intervals(interval)

            if retain_mappings:
                for interval in intervals:
                    interval = tuple(interval)
                    if interval in mappings:
                        raise Exception(
                            "An interval appeared a second time. Usually, numerical error "
                            "causes this exception. Consider increasing the tol parameter "
                            "or decreasing max_tail parameter to avoid it."
                        )
                    mappings[interval] = model

            if model_selector(model):
                selected_model = model if retain_observed_model else None
                result_intervals += intervals
                detect_count += 1

            self.searched_intervals = union_all(
                self.searched_intervals + intervals, tol=self.tol
            )

            prev_z = z
            z = self._next_search_data()

            if z is None:
                break

            if np.abs(prev_z - z) < self.step * 0.5:
                raise InfiniteLoopError

        stat_chi = float(self.stat)
        truncated_intervals = union_all(result_intervals, tol=self.tol)
        chi_intervals = intersection(truncated_intervals, [[0.0, np.inf]])
        chi_intervals = intersection(chi_intervals, self.restrict)
        F = tc_cdf_mpmath(
            stat_chi,
            chi_intervals,
            self.degree,
            absolute=False,
            dps=self.dps,
            max_dps=self.max_dps,
            out_log=self.out_log,
        )
        p_value = calc_pvalue(F, alternative)

        inf_p, sup_p = self._evaluate_pvalue(
            truncated_intervals, [[1e-5, float(max_tail)]], alternative
        )

        return SelectiveInferenceResult(
            stat_chi,
            significance_level,
            p_value,
            inf_p,
            sup_p,
            (p_value <= significance_level),
            chi_intervals,
            search_count,
            detect_count,
            selected_model,
            mappings,
        )

    def _over_conditioned_inference(
        self,
        algorithm,
        significance_level,
        alternative,
        retain_observed_model,
        tol,
        dps,
        max_dps,
        out_log,
    ):
        self.tol = tol
        self.dps = dps
        self.max_dps = max_dps
        self.out_log = out_log

        model, interval = algorithm(self.z, self.c, self.stat)
        interval = np.asarray(interval)
        intervals = _interval_to_intervals(interval)

        stat_chi = float(self.stat)
        chi_intervals = intersection(intervals, [[0.0, np.inf]])
        chi_intervals = intersection(chi_intervals, self.restrict)
        F = tc_cdf_mpmath(
            stat_chi,
            chi_intervals,
            self.degree,
            absolute=False,
            dps=self.dps,
            max_dps=self.max_dps,
            out_log=self.out_log,
        )
        p_value = calc_pvalue(F, alternative)

        inf_p, sup_p = self._evaluate_pvalue(intervals, intervals, alternative)

        return SelectiveInferenceResult(
            stat_chi,
            significance_level,
            p_value,
            inf_p,
            sup_p,
            (p_value <= significance_level),
            chi_intervals,
            1,
            1,
            None,
            model if retain_observed_model else None,
        )
