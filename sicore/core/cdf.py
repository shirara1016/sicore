"""Module providing the cumulative distribution functions."""

import numpy as np
from scipy.special import logsumexp  # type: ignore[import]
from scipy.stats import chi, norm, rv_continuous  # type: ignore[import]

from .real_subset import NotBelongToSubsetError, RealSubset


def truncated_cdf(
    rv: rv_continuous,
    z: float,
    intervals: np.ndarray | list[list[float]] | RealSubset,
    *,
    absolute: bool = False,
) -> float:
    """Compute the cdf value of the truncated distribution.

    Parameters
    ----------
    rv : rv_continuous
        The rv_continuous instance to be truncated.
    z : float
        The value at which to compute the cdf of the truncated distribution.
    intervals : np.ndarray | list[list[float]] | RealSubset
        The truncated intervals [[l1, u1], [l2, u2], ...].
    absolute : bool, optional
        Whether to compute the cdf for the distribution of the
        absolute value of the random variable. Defaults to False.

    Returns
    -------
    float
        The cdf value at z.

    Raises
    ------
    ValueError
        If the value z is not belong to the truncated intervals.
    """
    if not isinstance(intervals, RealSubset):
        intervals = RealSubset(intervals)

    if z not in intervals:
        raise NotBelongToSubsetError(z, intervals)

    mask_intervals = (
        RealSubset([[-np.abs(z), np.abs(z)]])
        if absolute
        else RealSubset([[-np.inf, z]])
    )

    inner_intervals = intervals & mask_intervals
    outer_intervals = intervals - mask_intervals

    inner_log_area = _compute_log_area(rv, inner_intervals)
    outer_log_area = _compute_log_area(rv, outer_intervals)
    # equal to 1.0 / (1.0 + np.exp(outer_log_area - inner_log_area))
    return np.exp(-np.log1p(np.exp(outer_log_area - inner_log_area)))


def _compute_log_area(rv: rv_continuous, intervals: RealSubset) -> float:
    """Compute the logarithm of the integral of the pdf over the each interval.

    Parameters
    ----------
    rv : rv_continuous
        The rv_continuous instance to be integrated.
    intervals : RealSubset
        The intervals on which to compute the integral.

    Returns
    -------
    float
        The logarithm of the integral.
    """
    left_ends, right_ends = intervals.intervals.T
    log_each_area = np.empty(len(intervals))
    mask = left_ends < rv.median()

    left_log_cdf, right_log_cdf = (
        rv.logcdf(left_ends[mask]),
        rv.logcdf(right_ends[mask]),
    )
    log_each_area[mask] = right_log_cdf + _log1mexp(left_log_cdf - right_log_cdf)

    left_log_sf, right_log_sf = rv.logsf(left_ends[~mask]), rv.logsf(right_ends[~mask])
    log_each_area[~mask] = left_log_sf + _log1mexp(right_log_sf - left_log_sf)

    return logsumexp(log_each_area)


def _log1mexp(z: np.ndarray) -> np.ndarray:
    """Compute the logarithm of one minus the exponential of the input array, element-wise.

    Parameters
    ----------
    z : np.ndarray
        Input values.

    Returns
    -------
    np.ndarray
        Logarithm of one minus the exponential of the input array.
    """
    z = np.asarray(z)
    values = np.empty_like(z)
    halflog = -0.693147  # equal to log(0.5)
    mask = z < halflog
    values[mask] = np.log1p(-np.exp(z[mask]))
    values[~mask] = np.log(-np.expm1(z[~mask]))
    return values


def truncated_norm_cdf(
    z: float,
    intervals: np.ndarray | list[list[float]] | RealSubset,
    *,
    absolute: bool = False,
) -> float:
    """Compute the cdf value of the truncated normal distribution.

    Parameters
    ----------
    z : float
        The value at which to compute the cdf of the truncated distribution.
    intervals : np.ndarray | list[list[float]] | RealSubset
        The truncated intervals [[l1, u1], [l2, u2], ...].
    absolute : bool, optional
        Whether to compute the cdf for the distribution of the
        absolute value of the random variable. Defaults to False.

    Returns
    -------
    float
        The cdf value at z.

    Raises
    ------
    ValueError
        If the value z is not belong to the truncated intervals.
    """
    return truncated_cdf(norm(), z, intervals, absolute=absolute)


def truncated_chi_cdf(
    z: float,
    df: int,
    intervals: np.ndarray | list[list[float]] | RealSubset,
    *,
    absolute: bool = False,
) -> float:
    """Compute the cdf value of the truncated normal distribution.

    Parameters
    ----------
    z : float
        The value at which to compute the cdf of the truncated distribution.
    df : int
        The degrees of freedom.
    intervals : np.ndarray | list[list[float]] | RealSubset
        The truncated intervals [[l1, u1], [l2, u2], ...].
    absolute : bool, optional
        Whether to compute the cdf for the distribution of the
        absolute value of the random variable. Defaults to False.

    Returns
    -------
    float
        The cdf value at z.

    Raises
    ------
    ValueError
        If the value z is not belong to the truncated intervals.
    """
    return truncated_cdf(chi(df), z, intervals, absolute=absolute)
