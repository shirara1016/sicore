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


def intersection(intervals1: np.ndarray, intervals2: np.ndarray) -> np.ndarray:
    """Take intersection of two intervals.

    Args:
        intervals1 (np.ndarray): Intervals [[l1, u1], [l2, u2], ...].
        intervals2 (np.ndarray): Intervals [[l1, u1], [l2, u2], ...].

    Returns:
        np.ndarray: Intersection of the two input intervals [[l1', u1'], [l2', u2'], ...].
    """
    return complement(union(complement(intervals1), complement(intervals2)))


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
        >>> print(~A)     # Complement
        [[-inf, 1.0], [3.0, 5.0], [7.0, inf]]
        >>> print(A | B)  # Union
        [[1.0, 4.0], [5.0, 8.0]]
        >>> print(A & B)  # Intersection
        [[2.0, 3.0], [6.0, 7.0]]
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

    def __invert__(self) -> RealSubset:
        """Take the complement of the subset

        Returns:
            RealSubset: Complement of the subset.
        """
        return self.complement()

    def union(self, other: RealSubset) -> RealSubset:
        """Take the union of the subset with another subset.

        Args:
            other (RealSubset): Another subset to take union with.

        Returns:
            RealSubset: Union of the two subsets.
        """
        return RealSubset(union(self.intervals, other.intervals), simplify=False)

    def __or__(self, other: RealSubset) -> RealSubset:
        """Take the union of the subset with another subset.

        Args:
            other (RealSubset): Another subset to take union with.

        Returns:
            RealSubset: Union of the two subsets.
        """
        return self.union(other)

    def union_update(self, other: RealSubset) -> None:
        """Update the subset with the union of itself and another subset.

        Args:
            other (RealSubset): Another subset to take union with.
        """
        self.intervals = (self | other).intervals

    def __ior__(self, other: RealSubset) -> RealSubset:
        """Update the subset with the union of itself and another subset.

        Args:
            other (RealSubset): Another subset to take union with.

        Returns:
            RealSubset: Updated subset with the union of itself and another subset.
        """
        self.union_update(other)
        return self

    def intersection(self, other: RealSubset) -> RealSubset:
        """Take the intersection of the subset with another subset.

        Args:
            other (RealSubset): Another subset to take intersection with.

        Returns:
            RealSubset: Intersection of the two subsets.
        """
        return ~((~self) | (~other))

    def __and__(self, other: RealSubset) -> RealSubset:
        """Take the intersection of the subset with another subset.

        Args:
            other (RealSubset): Another subset to take intersection with.

        Returns:
            RealSubset: Intersection of the two subsets.
        """
        return self.intersection(other)

    def intersection_update(self, other: RealSubset) -> None:
        """Update the subset with the intersection of itself and another subset.

        Args:
            other (RealSubset): Another subset to take intersection with.
        """
        self.intervals = (self & other).intervals

    def __iand__(self, other: RealSubset) -> RealSubset:
        """Update the subset with the intersection of itself and another subset.

        Args:
            other (RealSubset): Another subset to take intersection with.

        Returns:
            RealSubset: Updated subset with the intersection of itself and another subset.
        """
        self.intersection_update(other)
        return self

    def difference(self, other: RealSubset) -> RealSubset:
        """Take the difference of the subset with another subset.

        Args:
            other (RealSubset): Another subset to take difference with.

        Returns:
            RealSubset: Difference of the subset with another subset.
        """
        return self & ~other

    def __sub__(self, other: RealSubset) -> RealSubset:
        """Take the difference of the subset with another subset.

        Args:
            other (RealSubset): Another subset to take difference with.

        Returns:
            RealSubset: Difference of the subset with another subset.
        """
        return self.difference(other)

    def difference_update(self, other: RealSubset) -> None:
        """Update the subset with the difference of itself with another subset.

        Args:
            other (RealSubset): Another subset to take difference with.
        """
        self.intervals = (self - other).intervals

    def __isub__(self, other: RealSubset) -> RealSubset:
        """Update the subset with the difference of itself with another subset.

        Args:
            other (RealSubset): Another subset to take difference with.

        Returns:
            RealSubset: Updated subset with the difference of itself with another subset.
        """
        self.difference_update(other)
        return self

    def symmetric_difference(self, other: RealSubset) -> RealSubset:
        """Take the symmetric difference of the subset with another subset.

        Args:
            other (RealSubset): Another subset to take symmetric difference with.

        Returns:
            RealSubset: Symmetric difference of the subset with another subset.
        """
        return (self - other) | (other - self)

    def __xor__(self, other: RealSubset) -> RealSubset:
        """Take the symmetric difference of the subset with another subset.

        Args:
            other (RealSubset): Another subset to take symmetric difference with.

        Returns:
            RealSubset: Symmetric difference of the subset with another subset.
        """
        return self.symmetric_difference(other)

    def symmetric_difference_update(self, other: RealSubset) -> None:
        """Update the subset with the symmetric difference of itself with another subset.

        Args:
            other (RealSubset): Another subset to take symmetric difference with.
        """
        self.intervals = (self ^ other).intervals

    def __ixor__(self, other: RealSubset) -> RealSubset:
        """Update the subset with the symmetric difference of itself with another subset.

        Args:
            other (RealSubset): Another subset to take symmetric difference with.

        Returns:
            RealSubset: Updated subset with the symmetric difference of itself with another subset.
        """
        self.symmetric_difference_update(other)
        return self

    def is_empty(self) -> bool:
        """Check if the subset is empty.

        Returns:
            bool: True if the subset is empty, False otherwise.
        """
        return len(self.intervals) == 0

    def __eq__(self, other: RealSubset) -> bool:
        """Check if two subsets are equal.

        Args:
            other (RealSubset): Another subset to compare with.

        Returns:
            bool: True if the two subsets are equal, False otherwise.
        """
        if self.intervals.shape != other.intervals.shape:
            return False
        return np.allclose(
            self.intervals, other.intervals, rtol=1e-10, atol=1e-10
        )  # TODO: Tune atol and rtol

    def issubset(self, other: RealSubset) -> bool:
        """Check if the subset is a subset of another subset.

        Args:
            other (RealSubset): Another subset to check for subset.

        Returns:
            bool: True if the subset is a subset of another subset, False otherwise.
        """
        return (self - other).is_empty()

    def __le__(self, other: RealSubset) -> bool:
        """Check if the subset is a subset of another subset.

        Args:
            other (RealSubset): Another subset to check for subset.

        Returns:
            bool: True if the subset is a subset of another subset, False otherwise.
        """
        return self.issubset(other)

    def __lt__(self, other: RealSubset) -> bool:
        """Check if the subset is a proper subset of another subset.

        Args:
            other (RealSubset): Another subset to check for proper subset.

        Returns:
            bool: True if the subset is a proper subset of another subset, False otherwise.
        """
        return self.issubset(other) and self != other

    def issuperset(self, other: RealSubset) -> bool:
        """Check if the subset is a superset of another subset.

        Args:
            other (RealSubset): Another subset to check for superset.

        Returns:
            bool: True if the subset is a superset of another subset, False otherwise.
        """
        return other.issubset(self)

    def __ge__(self, other: RealSubset) -> bool:
        """Check if the subset is a superset of another subset.

        Args:
            other (RealSubset): Another subset to check for superset.

        Returns:
            bool: True if the subset is a superset of another subset, False otherwise.
        """
        return self.issuperset(other)

    def __gt__(self, other: RealSubset) -> bool:
        """Check if the subset is a proper superset of another subset.

        Args:
            other (RealSubset): Another subset to check for proper superset.

        Returns:
            bool: True if the subset is a proper superset of another subset, False otherwise.
        """
        return self.issuperset(other) and self != other

    def isdisjoint(self, other: RealSubset) -> bool:
        """Check if the subset is disjoint with another subset.

        Args:
            other (RealSubset): Another subset to check for disjoint.

        Returns:
            bool: True if the subset is disjoint with another subset, False otherwise.
        """
        return (self & other).is_empty()

    def __contains__(self, z: float) -> bool:
        """Check if a real number is in the subset.

        Args:
            z (float): Real number to check for membership.

        Returns:
            bool: True if z is in the subset, False otherwise.
        """
        if len(self.intervals) == 0:
            return False
        return np.any((self.intervals[:, 0] <= z) & (z <= self.intervals[:, 1]))

    def find_interval_containing(self, z: float) -> tuple[float, float] | None:
        """Find the interval containing a real number.

        Args:
            z (float): Real number to find the interval containing it.

        Returns:
            tuple[float, float] | None: Interval containing z if found, None otherwise.
        """
        if len(self.intervals) == 0:
            return None
        mask = (self.intervals[:, 0] <= z) & (z <= self.intervals[:, 1])
        if np.sum(mask) == 0:
            return None
        assert np.sum(mask) == 1
        return self.intervals[mask][0]

    def __len__(self) -> int:
        """Return the number of intervals in the subset.

        Returns:
            int: Number of intervals in the subset.
        """
        return len(self.intervals)

    def __str__(self) -> str:
        """Return a string representation of the subset.

        Returns:
            str: String representation of the subset.
        """
        if len(self.intervals) == 0:
            return "[]"
        return "[" + ", ".join([f"[{l:.6f}, {u:.6f}]" for l, u in self.intervals]) + "]"

    def __repr__(self) -> str:
        """Return a string representation of the RealSubset that can be used to recreate the object.

        Returns:
            str: String representation of the RealSubset in the format
                "RealSubset([[a, b], [c, d], ...])".
        """
        return f"RealSubset({self.intervals.tolist()})"
