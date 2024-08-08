from __future__ import annotations
import numpy as np


def _simplify(intervals: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Simplify (merge overlapping) intervals.

    Args:
        intervals (np.ndarray): Intervals [[l1, u1], [l2, u2], ...].
        tol (float, optional): Tolerance error parameter. When `tol`>=0.1,
            the intervals [[0, 1], [1.1, 2]] will be simplified to [[0, 2]].
            Defaults to 1e-10.

    Returns:
        np.ndarray: Simplified intervals [[l1', u1'], [l2', u2'], ...].
    """
    if len(intervals) == 0:
        return np.array([]).reshape(0, 2)

    # TODO: Delete try-except block
    try:
        intervals = intervals[np.argsort(intervals[:, 0])]
    except:
        print(intervals)
        print(np.argsort(intervals[:, 0]))
    simplified = [intervals[0]]
    for interval in intervals[1:]:
        if interval[0] <= simplified[-1][1] + tol:
            simplified[-1][1] = np.max([simplified[-1][1], interval[1]])
        else:
            simplified.append(interval)
    return np.array(simplified)


def complement(intervals: np.ndarray) -> np.ndarray:
    """Take complement of intervals.

    Args:
        intervals (np.ndarray): Intervals [[l1, u1], [l2, u2], ...].

    Returns:
        np.ndarray: Complement of the input intervals [[l1', u1'], [l2', u2'], ...].
    """
    if len(intervals) == 0:
        return np.array([[-np.inf, np.inf]])

    result = []
    if intervals[0][0] > -np.inf:
        result.append([-np.inf, intervals[0][0]])

    for i in range(len(intervals) - 1):
        result.append([intervals[i][1], intervals[i + 1][0]])

    if intervals[-1][1] < np.inf:
        result.append([intervals[-1][1], np.inf])

    return _simplify(np.array(result))


def intersection(intervals1: np.ndarray, intervals2: np.ndarray) -> np.ndarray:
    """Take intersection of two intervals.

    Args:
        intervals1 (np.ndarray): Intervals [[l1, u1], [l2, u2], ...].
        intervals2 (np.ndarray): Intervals [[l1, u1], [l2, u2], ...].

    Returns:
        np.ndarray: Intersection of the two input intervals [[l1', u1'], [l2', u2'], ...].
    """
    result = []
    i, j = 0, 0
    while i < len(intervals1) and j < len(intervals2):
        low = np.max([intervals1[i][0], intervals2[j][0]])
        high = np.min([intervals1[i][1], intervals2[j][1]])
        if low < high:
            result.append([low, high])
        if intervals1[i][1] < intervals2[j][1]:
            i += 1
        else:
            j += 1
    return _simplify(np.array(result))


def union(intervals1: np.ndarray, intervals2: np.ndarray) -> np.ndarray:
    """Take union of two intervals.

    Args:
        intervals1 (np.ndarray): Intervals [[l1, u1], [l2, u2], ...].
        intervals2 (np.ndarray): Intervals [[l1, u1], [l2, u2], ...].

    Returns:
        np.ndarray: Union of the two input intervals [[l1', u1'], [l2', u2'], ...].
    """
    intervals = np.vstack([intervals1, intervals2])
    return _simplify(intervals)


class RealSubset:
    """A class representing a subset of real numbers as a collection of intervals.

    This class allows for various set operations on subsets of real numbers, including
    complement, intersection, and union. It uses numpy arrays for efficient storage
    and computation.

    Attributes:
        intervals (np.ndarray): An array of shape (n, 2) representing n intervals.
                                    Each row [a, b] represents the interval [a, b].

    Args:
        intervals (np.array, list[list[int]], optional): Initial intervals to
            create the subset. Defaults to [], which creates an empty set.
        simplify (bool, optional): Whether to simplify (merge overlapping) intervals
            upon creation. Defaults to True.

    Note:
        - Infinite intervals can be represented using np.inf.
        - An empty array represents the empty set.

    Examples:
        >>> A = RealSubset([[1.0, 3.0], [5.0, 7.0]])
        >>> B = RealSubset([[2.0, 4.0], [6.0, 8.0]])
        >>> print(A & B)  # Intersection
        [[2.0, 3.0], [6.0, 7.0]]
        >>> print(A | B)  # Union
        [[1.0, 4.0], [5.0, 8.0]]
        >>> print(~A)     # Complement
        [[-inf, 1.0], [3.0, 5.0], [7.0, inf]]
    """

    def __init__(
        self, intervals: np.ndarray | list[list[float]] = [], simplify: bool = True
    ) -> None:
        """Initialize a RealSubset object.

        Args:
            intervals (np.array, list[list[int]], optional): Initial intervals to
                create the subset. Defaults to [], which creates an empty set.
            simplify (bool, optional): Whether to simplify (merge overlapping) intervals
                upon creation. Defaults to True.
        """
        if len(intervals) == 0:
            self.intervals = np.array([]).reshape(0, 2)
        else:
            self.intervals = np.array(intervals).reshape(-1, 2)
            if simplify:
                self._simplify()

    def _simplify(self) -> None:
        """Simplify the intervals of the subset"""
        self.intervals = _simplify(self.intervals)

    def complement(self) -> RealSubset:
        """Take the complement of the subset.

        Returns:
            RealSubset: Complement of the subset.
        """
        return RealSubset(complement(self.intervals), simplify=False)

    def intersection(self, other: RealSubset) -> RealSubset:
        """Take the intersection of the subset with another subset.

        Args:
            other (RealSubset): Another subset to take intersection with.

        Returns:
            RealSubset: Intersection of the two subsets.
        """
        return RealSubset(intersection(self.intervals, other.intervals), simplify=False)

    def union(self, other: RealSubset) -> RealSubset:
        """Take the union of the subset with another subset.

        Args:
            other (RealSubset): Another subset to take union with.

        Returns:
            RealSubset: Union of the two subsets.
        """
        return RealSubset(union(self.intervals, other.intervals), simplify=False)

    def __str__(self) -> str:
        """Return a string representation of the subset.

        Returns:
            str: String representation of the subset.
        """
        if len(self.intervals) == 0:
            return "[]"
        return "[" + ", ".join([f"[{a}, {b}]" for a, b in self.intervals]) + "]"

    def __repr__(self) -> str:
        """Return a string representation of the RealSubset that can be used to recreate the object.

        Returns:
            str: String representation of the RealSubset in the format
                "RealSubset([[a, b], [c, d], ...])".
        """
        return f"RealSubset({self.intervals.tolist()})"

    def __invert__(self) -> RealSubset:
        """Take the complement of the subset

        Returns:
            RealSubset: Complement of the subset.
        """
        return self.complement()

    def __and__(self, other: RealSubset) -> RealSubset:
        """Take the intersection of the subset with another subset.

        Args:
            other (RealSubset): Another subset to take intersection with.

        Returns:
            RealSubset: Intersection of the two subsets.
        """
        return self.intersection(other)

    def __or__(self, other: RealSubset) -> RealSubset:
        """Take the union of the subset with another subset.

        Args:
            other (RealSubset): Another subset to take union with.

        Returns:
            RealSubset: Union of the two subsets.
        """
        return self.union(other)

    def __sub__(self, other: RealSubset) -> RealSubset:
        """Take the difference of the subset with another subset.

        Args:
            other (RealSubset): Another subset to take difference with.

        Returns:
            RealSubset: Difference of the subset with another subset.
        """
        return self.intersection(~other)

    def __eq__(self, other: RealSubset) -> bool:
        """Check if two RealSubset objects are equal.

        Args:
            other (RealSubset): Another RealSubset object to compare with.

        Returns:
            bool: True if the two RealSubset objects are equal, False otherwise.
        """
        if self.intervals.shape != other.intervals.shape:
            return False
        return np.allclose(self.intervals, other.intervals)  # TODO: Tune atol and rtol

    def __contains__(self, z: float) -> bool:
        """Check if a real number is in the subset.

        Args:
            z (float): Real number to check for membership.

        Returns:
            bool: True if z is in the subset, False otherwise.
        """
        return np.any((self.intervals[:, 0] <= z) & (z <= self.intervals[:, 1]))
