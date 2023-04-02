import random
import numpy as np
from scipy.stats import norm, ttest_ind, ttest_1samp

from ..utils import is_int_or_float
from ..intervals import intersection

from typing import List, Any, Dict
from dataclasses import dataclass


@dataclass
class SelectiveInferenceResult():
    """
    This class contains the results of selective inference.

    Attributes:
        stat (float): Observed value of test statistic.
        alpha (float): Significance level.
        p_value (float): p-value from test performed.
        inf_p (float): Lower bound of p-value.
        sup_p (float): Upper bound of p-value.
        reject_or_not (bool): Whether or not to reject the null hypothesis.
        truncated_intervals (List[List[float]]): Intervals from which the selected model is obtained.
        search_count (int): Number of times the truncated intervals were searched.
        detect_count (int): Number of times that the same model as the observed model was obtained.
        selected_model (Any | None): The model selected for the observed test statistic.
        mappings (Dict[tuple[float], Any] | None): A dictionary that holds the model obtained at any point.
    """
    stat: float
    alpha: float
    p_value: float
    inf_p: float
    sup_p: float
    reject_or_not: bool
    truncated_intervals: List[List[float]]
    search_count: int
    detect_count: int
    selected_model: Any | None
    mappings: Dict[tuple[float], Any] | None


@dataclass
class SearchProgress():
    """
    This class contains the intermediate results of each search process.

    Attributes:
    """
    stat: float
    alpha: float
    p_value: float
    inf_p: float
    sup_p: float
    truncated_intervals: List[List[float]]
    searched_intervals: List[List[float]]
    search_point: float
    search_count: int
    detect_count: int
    choose_method: str
    null_distribution: str


class InfiniteLoopError(Exception):
    pass


class SearchChecker:
    def __init__(self, max_iter: int = 1e6):
        self.last_width = -np.inf
        self.last_length = 0
        self.num_search = 0
        self.min = np.inf
        self.max = -np.inf
        self.max_iter = max_iter

    def _compute_logarithm_width(self, intervals):
        width = 0
        if len(intervals) == 0:
            return -np.inf

        self.min = np.min([self.min, intervals[0][-1]])
        self.max = np.max([self.max, intervals[-1][0]])

        if np.isneginf(intervals[0][0]):
            self.min = self.min - 2
        else:
            self.min = intervals[0][0]
        if np.isposinf(intervals[-1][-1]):
            self.max = self.max + 2
        else:
            self.max = intervals[-1][-1]

        intervals = intersection(intervals, [[self.min, self.max]])
        for interval in intervals:
            width += np.log(interval[1] - interval[0])
        return width

    def verify_progress(self, intervals: List[List[float]]):
        length = len(intervals)
        width = self._compute_logarithm_width(intervals)
        if length == self.last_length and width <= self.last_width:
            raise InfiniteLoopError()
        if self.num_search > self.max_iter:
            raise InfiniteLoopError()
        self.last_length = length
        self.last_width = width
        self.num_search += 1


INF = float('inf')
NINF = -INF

random.seed(0)


def calc_pvalue(F, alternative):
    """
    Calculate p-value.

    Args:
        F (float): CDF value at the observed test statistic.
        alternative (str):
            'two-sided' for two-tailed test,
            'less' for right-tailed test,
            'greater' for left-tailed test,
            'abs' for two-tailed test for
            the distribution of absolute values
            following the null distribution.

    Returns:
        float: p-value
    """
    if alternative == 'two-sided':
        return float(2 * min(F, 1 - F))
    elif alternative == 'less' or alternative == 'abs':
        return float(1 - F)
    elif alternative == 'greater':
        return float(F)


def calc_prange(inf_F, sup_F, alternative):
    """
    Calculate possible range of p-value.

    Args:
        inf_F (float): Infimum of CDF value at the observed test statistic.
        sup_F (float): Supremum of CDF value at the observed test statistic.
        alternative (str, optional):
            'two-sided' for two-tailed test,
            'less' for right-tailed test,
            'greater' for left-tailed test,
            'abs' for two-tailed test for
            the distribution of absolute values
            following the null distribution.

    Returns:
        (float, float): (Infimum of p-value, Supremum of p-value)
    """
    if alternative == 'two-sided':
        sup_p = float(2 * min(sup_F, 1 - inf_F))
        inf_p = float(2 * min(inf_F, 1 - sup_F))
    elif alternative == 'less' or alternative == 'abs':
        sup_p = float(1 - inf_F)
        inf_p = float(1 - sup_F)
    elif alternative == 'greater':
        sup_p = float(sup_F)
        inf_p = float(inf_F)
    return (max(inf_p, 0), min(sup_p, 1))


def standardize(x, mean=0, var=1):
    """
    Standardize a random variable.

    Args:
        x (float, list): The value of random variable.
        mean (float, optional): Mean. Defaults to 0.
        var (float, optional): Variance. Defaults to 1.
    """
    sd = np.sqrt(var)
    return (np.asarray(x) - mean) / sd


def one_sample_test(data, popmean, var=None, tail='double'):
    """
    One sample hypothesis testing for population mean.

    var=float: Z-test
    var=None: T-test

    Args:
        data (array-like): Dataset.
        popmean (float): Population mean of `data` under null hypothesis is
            true.
        var (float, optional): Known population variance of the dataset. If
            None, the population variance is unknown. Defaults to None.
        tail (str, optional): 'double' for double-tailed test, 'right' for
            right-tailed test, and 'left' for left-tailed test. Defaults to
            'double'.

    Returns:
        float: p-value
    """
    if var is None:
        pvalue, stat = ttest_1samp(data, popmean)
        F = pvalue / 2 if stat < 0 else 1 - pvalue / 2
        return calc_pvalue(F, tail=tail)
    else:
        estimator = np.mean(data)
        var = var / len(data)
        stat = standardize(estimator, popmean, var)
        F = norm.cdf(stat)
        return calc_pvalue(F, tail=tail)


def two_sample_test(data1, data2, var=None, equal_var=True, tail='double'):
    """
    Two sample hypothesis testing for the difference between population means.

    var=float, list: Z-test
    var=None & equal_var=True: T-test
    var=None & equal_var=False: Welch's T-test

    Args:
        data1 (array-like): Dataset1.
        data2 (array-like): Dataset2.
        var (float, list, optional): Known population variance of each dataset
            in the form of single value or tuple `(var1, var2)`. If None, the
            population variance is unknown. Defaults to None.
        equal_var (bool, optional): If True, two population variances are
            assumed to be the same. Defaults to True.
        tail (str, optional): 'double' for double-tailed test, 'right' for
            right-tailed test, and 'left' for left-tailed test. Defaults to
            'double'.

    Returns:
        float: p-value
    """
    if var is None:
        pvalue, stat = ttest_ind(data1, data2, equal_var=equal_var)
        F = pvalue / 2 if stat < 0 else 1 - pvalue / 2
        return calc_pvalue(F, tail=tail)
    else:
        if is_int_or_float(var):
            var1, var2 = var, var
        else:
            var1, var2 = var

        estimator = np.mean(data1) - np.mean(data2)
        var = var1 / len(data1) + var2 / len(data2)
        stat = standardize(estimator, var=var)
        F = norm.cdf(stat)
        return calc_pvalue(F, tail=tail)
