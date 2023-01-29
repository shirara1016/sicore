from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import fractional_matrix_power
from ..utils import is_int_or_float
from ..intervals import intersection, not_, poly_lt_zero, union_all, _interval_to_intervals
from ..cdf_mpmath import tc2_cdf_mpmath
from .base import *

from scipy import sparse

from scipy.stats import chi2
from typing import Callable, List, Tuple, Type


class InferenceChiSquared(ABC):
    """Base inference class for a test statistic which follows chi squared distribution under null.

    Args:
        data (np.ndarray, tf.Tensor, torch.Tensor):
            Observation data in 1-D array. When given as a tensor,
            activate the corresponding option.
        var (float, np.ndarray, tf.Tensor, torch.Tensor, sparse.spmatrix):
            Value of known variance covariance. Assuming that
            the shape of the input is a scalar or 1-D array or 2-D array,
            the variance-covariance matrix Cov is interpreted as follows.
            When the input is a scalar, Cov = input * Identity.
            When the input is a 1-D array, Cov = diag(input).
            When the input is a 2-D array, Cov = input.
            Also, activate the option, when given as a sparse matrix.
        P (np.ndarray, tf.Tensor, torch.Tensor, sparse.spmatrix):
            Projection matrix in 2-D array. When given as
            a tensor or a sparse matrix, activate the corresponding option.
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
        var: float | np.ndarray | sparse.spmatrix,
        P: np.ndarray | sparse.spmatrix,
        degree: int,
        use_sparse: bool = False,
        use_tf: bool = False,
        use_torch: bool = False
    ):

        if use_sparse and (use_tf or use_torch):
            raise Exception(
                'Sparse matrix cannot be used with the deep learning framework at the same time.')
        if use_tf and use_torch:
            raise Exception(
                'Only one of tensorflow and pytorch is available.')

        self.data = data  # unnecessary
        self.length = len(data)  # unnecessary
        self.P = P  # unnecessary
        self.degree = degree

        if use_tf:
            try:
                import tensorflow as tf
            except ModuleNotFoundError:
                raise Exception(
                    'The option use_tf is activated, but tensorflow was not found.')

            assert isinstance(data, tf.Tensor)
            assert isinstance(P, tf.Tensor)

            self.P_data = tf.tensordot(P, data, axes=1)
            if is_int_or_float(var):
                whitend_P_data = (var ** -0.5) * self.P_data
            elif len(var.shape) == 1:
                vars = tf.constant(var, dtype=data.dtype)
                whitend_P_data = tf.pow(vars, -0.5) * self.P_data
            elif len(var.shape) == 2:
                cov = np.array(var)
                inv_sqrt_cov = fractional_matrix_power(cov, -0.5)
                inv_sqrt_cov = tf.constant(inv_sqrt_cov, dtype=data.dtype)
                whitend_P_data = tf.tensordot(
                    inv_sqrt_cov, self.P_data, axes=1)
            self.stat = tf.norm(whitend_P_data, ord=2)

        elif use_torch:
            try:
                import torch
            except ModuleNotFoundError:
                raise Exception(
                    'The option use_torch is activated, but pytorch was not found.')

            assert isinstance(data, torch.Tensor)
            assert isinstance(P, torch.Tensor)

            self.P_data = torch.mv(P, data)
            if is_int_or_float(var):
                whitend_P_data = (var ** -0.5) * self.P_data
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
            P = sparse.csr_matrix(P) if use_sparse else np.array(P)
            self.P_data = P @ data
            if is_int_or_float(var):
                whitend_P_data = (var ** -0.5) * self.P_data
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


class NaiveInferenceChiSquared(InferenceChiSquared):
    """Naive inference for a test statistic which follows chi squared distribution under null.

    Args:
        data (np.ndarray, tf.Tensor, torch.Tensor):
            Observation data in 1-D array. When given as a tensor,
            activate the corresponding option.
        var (float, np.ndarray, tf.Tensor, torch.Tensor, sparse.spmatrix):
            Value of known variance covariance. Assuming that
            the shape of the input is a scalar or 1-D array or 2-D array,
            the variance-covariance matrix Cov is interpreted as follows.
            When the input is a scalar, Cov = input * Identity.
            When the input is a 1-D array, Cov = diag(input).
            When the input is a 2-D array, Cov = input.
            Also, activate the option, when given as a sparse matrix.
        P (np.ndarray, tf.Tensor, torch.Tensor, sparse.spmatrix):
            Projection matrix in 2-D array. When given as
            a tensor or a sparse matrix, activate the corresponding option.
        degree (int):
            Dimension of the space projected by P matrix.
        use_sparse (boolean, optional):
            Whether to use sparse matrix or not. Defaults to False.
        use_tf (boolean, optional):
            Whether to use tensorflow or not. Defaults to False.
        use_torch (boolean, optional):
            Whether to use pytorch or not. Defaults to False.
    """

    def inference(self, tail: str = "right"):
        """Perform naive statistical inference.

        Args:
            tail (str, optional):
            'double' for double-tailed test,
            'right' for right-tailed test, and
            'left' for left-tailed test. Defaults to 'right'.

        Returns:
            float: p-value
        """
        stat = np.asarray(self.stat) ** 2
        F = chi2.cdf(stat, self.degree)
        return calc_pvalue(F, tail=tail)


class SelectiveInferenceChiSquared(InferenceChiSquared):
    """Selective inference for a test statistic which follows chi squared distribution under null.

    Args:
        data (np.ndarray, tf.Tensor, torch.Tensor):
            Observation data in 1-D array. When given as a tensor,
            activate the corresponding option.
        var (float, np.ndarray, tf.Tensor, torch.Tensor, sparse.spmatrix):
            Value of known variance covariance. Assuming that
            the shape of the input is a scalar or 1-D array or 2-D array,
            the variance-covariance matrix Cov is interpreted as follows.
            When the input is a scalar, Cov = input * Identity.
            When the input is a 1-D array, Cov = diag(input).
            When the input is a 2-D array, Cov = input.
            Also, activate the option, when given as a sparse matrix.
        P (np.ndarray, tf.Tensor, torch.Tensor, sparse.spmatrix):
            Projection matrix in 2-D array. When given as
            a tensor or a sparse matrix, activate the corresponding option.
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
        var: float | np.ndarray | sparse.spmatrix,
        P: np.ndarray | sparse.spmatrix,
        degree: int,
        use_sparse: bool = False,
        use_tf: bool = False,
        use_torch: bool = False
    ):
        super().__init__(data, var, P, degree, use_sparse, use_tf, use_torch)
        self.c = self.P_data / self.stat  # `b` vector in para si.
        self.z = data - self.P_data  # `a` vector in para si.

    def inference(
        self,
        algorithm: Callable[[np.ndarray, np.ndarray, float], Tuple[List[List[float]], Any]],
        model_selector: Callable[[Any], bool],
        significance_level: float = 0.05,
        parametric_mode: str = 'p_value',
        over_conditioning: bool = False,
        tail: str = 'right',
        threshold: float = 1e-3,
        line_search: bool = True,
        max_tail: float = 1e3,
        choose_method: str = 'near_stat',
        retain_selected_model: bool = False,
        retain_mappings: bool = False,
        tol: float = 1e-10,
        step: float = 1e-10,
        dps: int | str = 'auto',
        max_dps: int = 5000,
        out_log: str = 'test_log.log'
    ) -> Type[SelectiveInferenceResult]:
        """Perform Selective Inference.

        Args:
            algorithm (Callable[[np.ndarray, np.ndarray, float], Tuple[List[List[float]], Any]]):
                Callable function which takes two vectors (`a`, `b`)
                and a scalar `z` that can satisfy `data = a + b * z`
                as arguments, and returns the selected model (any) and
                the truncation intervals (array-like). A closure function might be
                helpful to implement this.
            model_selector (Callable[[Any], bool]):
                Callable function which takes a selected model (any) as single argument, and
                returns True if the model is used for the testing, and False otherwise.
            significance_level (float, optional):
                Significance level for the testing. Defaults to 0.05.
            parametric_mode (str, optional):
                Specifies the method used to perform parametric selective inference. This option is
                ignored when the over_conditioning option is activated.
                'p_value' for calculation of p-value with guaranteed accuracy specified by the threshold option.
                'reject_or_not' for only determining whether the null hypothesis is rejected.
                'all_search' for all searches of the interval specified by the max_tail option.
            over_conditioning (bool, optional):
                Over conditioning Inference. Defaults to False.
            tail (str, optional):
                'double' for double-tailed test, 'right' for right-tailed test, and
                'left' for left-tailed test. Defaults to 'right'.
            threshold (float, optional):
                Guaranteed accuracy when calculating p-value. Defaults to 1e-3.
            line_search (bool, optional):
                Wheter to perform a line search or a random search. Defaults to True.
            max_tail (float, optional):
                Maximum tail value to be parametrically searched when
                the parametric_mode option is set to all_search. Defaults to 1e3.
            choose_method (str, optional):
                How to determine the search direction when parametric_mode
                is other than all_search. 'near_stat', 'high_pdf', and 'random'
                can be specified. Defaults to 'near_stat'.
            retain_selected_model (bool, optional):
                Whether retain selected model as returns or not. Defaults to False.
            retain_mappings (bool, optional):
                Whether retain mappings as returns or not. Defaults to False.
            tol (float, optional):
                Tolerance error parameter for intervals. Defaults to 1e-10.
            step (float, optional):
                Step width for parametric search. Defaults to 1e-10.
            dps (int | str, optional):
                dps value for mpmath. Set 'auto' to select dps
                automatically. Defaults to 'auto'.
            max_dps (int, optional):
                Maximum dps value for mpmath. This option is valid
                when `dps` is set to 'auto'. Defaults to 5000.
            out_log (str, optional):
                Name for log file of mpmath. Defaults to 'test_log.log'.

        Raises:
            Exception:
                   The parametric_mode option is not p_value, reject_or_not, or all_search,
                   and the over_conditioning option is set False.

        Returns:
            Type[SelectiveInferenceResult]
        """

        if over_conditioning:
            result = self._over_conditioned_inference(
                algorithm, significance_level, tail, retain_selected_model,
                tol, dps, max_dps, out_log)
            return result

        elif parametric_mode == 'p_value' or parametric_mode == 'reject_or_not':
            result = self._parametric_inference(
                algorithm, model_selector, significance_level, parametric_mode,
                tail, threshold, choose_method, retain_selected_model, retain_mappings,
                tol, step, dps, max_dps, out_log)

        elif parametric_mode == 'all_search':
            result = self._all_search_parametric_inference(
                algorithm, model_selector, significance_level, tail,
                line_search, max_tail, retain_selected_model, retain_mappings,
                tol, step, dps, max_dps, out_log)

        else:
            raise Exception(
                'Please activate either parametric_mode or over_conditioning option.')

        return result

    def _calc_range_of_cdf_value(self, truncated_intervals, searched_intervals):

        unsearched_intervals = not_(searched_intervals)
        s = intersection(unsearched_intervals, [
            NINF, float(self.stat)])[-1][1]
        e = intersection(unsearched_intervals, [
            float(self.stat), INF])[0][0]

        self.left_end = s
        self.right_end = e

        sup_intervals = union_all(
            truncated_intervals + [[NINF, s]], tol=self.tol)
        inf_intervals = union_all(
            truncated_intervals + [[e, INF]], tol=self.tol)

        chi_sup_intervals = intersection(
            sup_intervals, [[1e-5, INF]])
        chi_inf_intervals = intersection(
            inf_intervals, [[1e-5, INF]])

        chisq_sup_intervals = np.power(chi_sup_intervals, 2)
        chisq_inf_intervals = np.power(chi_inf_intervals, 2)

        stat_chisq = np.asarray(self.stat) ** 2

        sup_F = tc2_cdf_mpmath(stat_chisq, chisq_sup_intervals, self.degree,
                               dps=self.dps, max_dps=self.max_dps, out_log=self.out_log)
        inf_F = tc2_cdf_mpmath(stat_chisq, chisq_inf_intervals, self.degree,
                               dps=self.dps, max_dps=self.max_dps, out_log=self.out_log)

        return inf_F, sup_F

    def _determine_next_search_data(self, choose_method, *args):
        if choose_method == 'near_stat':
            def method(z): return -np.abs(z - float(self.stat))
        if choose_method == 'high_pdf':
            def method(z): return chi2.pdf(z ** 2, self.degree)
        if choose_method == 'random':
            return random.choice(args)
        return max(args, key=method)

    def _next_search_data(self, line_search):
        intervals = not_(self.searched_intervals)
        if len(intervals) == 0:
            return None
        if line_search:
            param = intervals[0][0] + self.step
        else:
            s, e = random.choice(intervals)
            param = (e + s) / 2
        return param

    def _parametric_inference(
            self, algorithm, model_selector, significance_level, parametric_mode,
            tail, threshold, choose_method, retain_selected_model, retain_mappings,
            tol, step, dps, max_dps, out_log):

        self.tol = tol
        self.step = step
        self.dps = dps
        self.max_dps = max_dps
        self.out_log = out_log
        self.searched_intervals = list()

        mappings = dict() if retain_mappings else None
        truncated_intervals = list()

        search_count = 0
        detect_count = 0

        z = self.stat
        while True:
            search_count += 1
            if search_count > 1e6:
                raise Exception(
                    'The number of searches exceeds 100,000 times, suggesting an infinite loop.')

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
                selected_model = model if retain_selected_model else None
                truncated_intervals += intervals
                detect_count += 1

            self.searched_intervals = union_all(
                self.searched_intervals + intervals, tol=self.tol)

            inf_F, sup_F = self._calc_range_of_cdf_value(
                truncated_intervals, self.searched_intervals)
            inf_p, sup_p = calc_p_range(inf_F, sup_F, tail=tail)

            if parametric_mode == 'p_value':
                if np.abs(sup_p - inf_p) < threshold:
                    break
            if parametric_mode == 'reject_or_not':
                if sup_p <= significance_level:
                    reject_or_not = True
                    break
                if inf_p > significance_level:
                    reject_or_not = False
                    break

            z_l = self.left_end - self.step
            z_r = self.right_end + self.step
            z = self._determine_next_search_data(choose_method, z_l, z_r)
            if z <= 1e-5:
                z = z_r

        stat_chisq = np.asarray(self.stat) ** 2
        truncated_intervals = union_all(truncated_intervals, tol=self.tol)
        chi_intervals = intersection(truncated_intervals, [[1e-5, INF]])
        chisq_intervals = np.power(chi_intervals, 2)
        F = tc2_cdf_mpmath(stat_chisq, chisq_intervals, self.degree,
                           dps=self.dps, max_dps=self.max_dps, out_log=self.out_log)
        p_value = calc_pvalue(F, tail=tail)
        if parametric_mode == 'p_value':
            reject_or_not = (p_value <= significance_level)

        return SelectiveInferenceResult(
            stat_chisq, significance_level, p_value, inf_p, sup_p,
            reject_or_not, chisq_intervals,
            search_count, detect_count, selected_model, mappings)

    def _all_search_parametric_inference(
            self, algorithm, model_selector, significance_level, tail,
            line_search, max_tail, retain_selected_model, retain_mappings,
            tol, step, dps, max_dps, out_log):

        self.tol = tol
        self.step = step
        self.dps = dps
        self.max_dps = max_dps
        self.out_log = out_log
        self.searched_intervals = union_all(
            [[NINF, 1e-5], [float(max_tail), INF]], tol=self.tol)

        mappings = dict() if retain_mappings else None
        result_intervals = list()

        search_count = 0
        detect_count = 0

        z = self._next_search_data(line_search)
        while True:

            if z is None:
                break

            search_count += 1
            if search_count > 1e6:
                raise Exception(
                    'The number of searches exceeds 100,000 times, suggesting an infinite loop.')

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
                            "or decreasing max_tail parameter to avoid it.")
                    mappings[interval] = model

            if model_selector(model):
                selected_model = model if retain_selected_model else None
                result_intervals += intervals
                detect_count += 1

            self.searched_intervals = union_all(
                self.searched_intervals + intervals, tol=self.tol)

            z = self._next_search_data(line_search)

        stat_chisq = np.asarray(self.stat) ** 2
        truncated_intervals = union_all(result_intervals, tol=self.tol)
        chi_intervals = intersection(truncated_intervals, [[1e-5, INF]])
        chisq_intervals = np.power(chi_intervals, 2)
        F = tc2_cdf_mpmath(stat_chisq, chisq_intervals, self.degree,
                           dps=self.dps, max_dps=self.max_dps, out_log=self.out_log)
        p_value = calc_pvalue(F, tail=tail)

        inf_F, sup_F = self._calc_range_of_cdf_value(
            truncated_intervals, [[1e-5, float(max_tail)]])
        inf_p, sup_p = calc_p_range(inf_F, sup_F, tail=tail)

        return SelectiveInferenceResult(
            stat_chisq, significance_level, p_value, inf_p, sup_p,
            (p_value <= significance_level), chisq_intervals,
            search_count, detect_count, selected_model, mappings)

    def _over_conditioned_inference(
            self, algorithm, significance_level, tail,
            retain_selected_model, tol, dps, max_dps, out_log):

        self.tol = tol
        self.dps = dps
        self.max_dps = max_dps
        self.out_log = out_log

        model, interval = algorithm(self.z, self.c, self.stat)
        interval = np.asarray(interval)
        intervals = _interval_to_intervals(interval)

        stat_chisq = np.asarray(self.stat) ** 2
        chi_intervals = intersection(intervals, [[1e-5, INF]])
        chisq_intervals = np.power(chi_intervals, 2)
        F = tc2_cdf_mpmath(stat_chisq, chisq_intervals, self.degree,
                           dps=self.dps, max_dps=self.max_dps, out_log=self.out_log)
        p_value = calc_pvalue(F, tail=tail)

        inf_F, sup_F = self._calc_range_of_cdf_value(intervals, intervals)
        inf_p, sup_p = calc_p_range(inf_F, sup_F, tail=tail)

        return SelectiveInferenceResult(
            stat_chisq, significance_level, p_value, inf_p, sup_p,
            (p_value <= significance_level), chisq_intervals, 1, 1,
            None, model if retain_selected_model else None)
