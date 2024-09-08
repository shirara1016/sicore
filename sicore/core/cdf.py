"""Module providing the cumulative distribution functions."""

from collections.abc import Callable

import numpy as np
from mpmath import mp  # type: ignore[import]
from scipy.special import logsumexp  # type: ignore[import]
from scipy.stats import rv_continuous  # type: ignore[import]

from .real_subset import NotBelongToSubsetError, RealSubset

mp.dps = 1000


def chi_cdf(z: float, df: int) -> float:
    """Compute the CDF of a chi distribution.

    Args:
        z (float): The value at which to compute the CDF.
        df (int): The degree of freedom.

    Returns:
        float: The CDF value at `z`.
    """
    return mp.gammainc(df / 2, a=0, b=z**2 / 2, regularized=True)


def _truncated_cdf_from_cdf(
    cdf: Callable[[float], float],
    z: float,
    intervals: np.ndarray | list[list[float]] | RealSubset,
    *,
    absolute: bool = False,
    dps: int | str = "auto",
    init_dps: int = 30,
    max_dps: int = 5000,
    scale_dps: int = 2,
) -> float:
    """Compute the CDF of a truncated distribution from a CDF function.

    Args:
        cdf (callable[[float], float]):
            The CDF function of the distribution to be truncated.
        z (float): The value at which to compute the CDF.
        intervals (np.ndarray | list[list[float]] | RealSubset): The truncated intervals
            [[l1, u1], [l2, u2], ...].
        absolute (bool, optional): Whether to compute the CDF of the distribution of the
            absolute value of the random variable. Defaults to False.
        dps (int | str, optional): The number of decimal precision in mpmath. If "auto",
            it will be calculated automatically, However, it will not work well when the
            interval is extremely narrow and the CDF values are almost the same.
            To faster computation, set an integer value since the auto selection
            can incur overhead. Defaults to "auto".
        init_dps (int, optional):
            Initial dps value for mpmath when dps is set to "auto".
            Defaults to 30.
        max_dps (int, optional):
            Maximum dps value for mpmath when dps is set to "auto".
            Defaults to 5000.
        scale_dps (int, optional): Multiplier for the dps value for mpmath when `dps` is
            set to "auto". Defaults to 2.

    Returns:
        float: The CDF value at `z`.

    Raises:
        ValueError: If the value `z` is not belong to the truncated intervals.
        ZeroDivisionError: If the denominator is zero. Consider increasing the dps.
    """
    if not isinstance(intervals, RealSubset):
        intervals = RealSubset(intervals)

    if z not in intervals:
        raise NotBelongToSubsetError(z, intervals)

    if dps == "auto":
        mp.dps = init_dps

        edge_points = np.ravel(intervals.intervals)
        finite_edge_points = edge_points[np.isfinite(edge_points)]

        if len(finite_edge_points) != 0:
            extreme = np.max(np.abs(finite_edge_points))
            literal = mp.nstr(
                cdf(extreme),
                init_dps - 15,
                min_fixed=-np.inf,
                max_fixed=np.inf,
            )
            if literal == "1.0":
                next_init_dps = int(init_dps * scale_dps)
                if next_init_dps <= max_dps:
                    return _truncated_cdf_from_cdf(
                        cdf,
                        z,
                        intervals,
                        absolute=absolute,
                        dps=dps,
                        init_dps=next_init_dps,
                        max_dps=max_dps,
                        scale_dps=scale_dps,
                    )
    else:
        mp.dps = dps

    nominator = 0.0
    denominator = 0.0

    mask_intervals = (
        RealSubset([[-np.abs(z), np.abs(z)]])
        if absolute
        else RealSubset([[-np.inf, z]])
    )

    inner_intervals = intervals & mask_intervals
    outer_intervals = intervals - mask_intervals

    for lower, upper in inner_intervals.intervals:
        diff = cdf(upper) - cdf(lower)
        nominator += diff
        denominator += diff

    for lower, upper in outer_intervals.intervals:
        diff = cdf(upper) - cdf(lower)
        denominator += diff

    return nominator / denominator


def truncated_norm_cdf(
    z: float,
    intervals: np.ndarray | list[list[float]] | RealSubset,
    *,
    absolute: bool = False,
) -> float:
    """Compute the CDF of a truncated normal distribution.

    Args:
        z (float): The value at which to compute the CDF.
        intervals (np.ndarray | list[list[float]] | RealSubset): The truncated intervals
            [[l1, u1], [l2, u2], ...].
        absolute (bool, optional): Whether to compute the CDF of the distribution of the
            absolute value of the random variable. Defaults to False.

    Returns:
        float: The CDF value of the truncated normal distribution at `z`.

    Raises:
        ValueError: If the value `z` is not belong to the truncated intervals.
        ZeroDivisionError: If the denominator is zero. Consider increasing the dps.
    """
    cdf = mp.ncdf
    return float(_truncated_cdf_from_cdf(cdf, z, intervals, absolute=absolute))


def truncated_chi_cdf(
    z: float,
    df: int,
    intervals: np.ndarray | list[list[float]] | RealSubset,
    *,
    absolute: bool = False,
) -> float:
    """Compute the CDF of a truncated chi distribution.

    Args:
        z (float): The value at which to compute the CDF.
        df (int): The degree of freedom.
        intervals (np.ndarray | list[list[float]] | RealSubset): The truncated intervals
            [[l1, u1], [l2, u2], ...].
        absolute (bool, optional): Whether to compute the CDF of the distribution of the
            absolute value of the random variable. Defaults to False.

    Returns:
        float: The CDF value of the truncated chi distribution at `z`.

    Raises:
        ValueError: If the value `z` is not belong to the truncated intervals.
        ZeroDivisionError: If the denominator is zero. Consider increasing the dps.

    """
    return float(
        _truncated_cdf_from_cdf(
            lambda z: chi_cdf(z, df),
            z,
            intervals,
            absolute=absolute,
        ),
    )


def truncated_cdf(
    rv: rv_continuous,
    z: float,
    intervals: np.ndarray | list[list[float]] | RealSubset,
    *,
    absolute: bool = False,
) -> float:
    """Compute the cdf value of the truncated distribution.

    Args:
        rv (rv_continuous): The rv_continuous instance to be truncated.
        z (float): The value at which to compute the cdf of the truncated distribution.
        intervals (np.ndarray | list[list[float]] | RealSubset): The truncated intervals
            [[l1, u1], [l2, u2], ...].
        absolute (bool, optional): Whether to compute the cdf for the distribution of the
            absolute value of the random variable. Defaults to False.

    Returns:
        float: The cdf value at z.

    Raises:
        ValueError: If the value z is not belong to the truncated intervals.
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


def _log1mexp(z: np.ndarray) -> np.ndarray:
    """Compute the logarithm of one minus the exponential of the input array, element-wise.

    Args:
        z (np.ndarray): Input values.

    Returns:
        np.ndarray: Logarithm of one minus the exponential of the input array.
    """
    z = np.asarray(z)
    values = np.empty_like(z)
    halflog = -0.693147  # equal to log(0.5)
    mask = z < halflog
    values[mask] = np.log1p(-np.exp(z[mask]))
    values[~mask] = np.log(-np.expm1(z[~mask]))
    return values


def _compute_log_area(rv: rv_continuous, intervals: RealSubset) -> float:
    """Compute the logarithm of the integral of the pdf over the each interval.

    Args:
        rv (rv_continuous): The rv_continuous instance to be integrated.
        intervals (RealSubset): The intervals on which to compute the integral.

    Returns:
        float: The logarithm of the integral.
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
