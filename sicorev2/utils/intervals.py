import numpy as np
from ..core.real_subset import simplify, complement, union, intersection


def difference(intervals1: np.ndarray, intervals2: np.ndarray) -> np.ndarray:
    """Take the difference of first intervals with second intervals.

    Args:
        intervals1 (np.ndarray): Intervals [[l1, u1], [l2, u2], ...].
        intervals2 (np.ndarray): Intervals [[l1, u1], [l2, u2], ...].

    Returns:
        np.ndarray: Difference of the firt input intervals with the second input intervals [[l1', u1'], [l2', u2'], ...].
    """
    return intersection(intervals1, complement(intervals2))


def symmetric_difference(intervals1: np.ndarray, intervals2: np.ndarray) -> np.ndarray:
    """Take the symmetric difference of two intervals.

    Args:
        intervals1 (np.ndarray): Intervals [[l1, u1], [l2, u2], ...].
        intervals2 (np.ndarray): Intervals [[l1, u1], [l2, u2], ...].

    Returns:
        np.ndarray: Symmetric difference of the two input intervals [[l1', u1'], [l2', u2'], ...].
    """
    return union(difference(intervals1, intervals2), difference(intervals2, intervals1))
