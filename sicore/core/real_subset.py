"""Module defining the RealSubset class for representing subsets of real numbers."""

from __future__ import annotations

import numpy as np
from typing_extensions import Self


def simplify(intervals: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Simplify (merge overlapping) the intervals.

    Parameters
    ----------
    intervals : np.ndarray
        Intervals [[l1, u1], [l2, u2], ...].
    tol : float, optional
        Tolerance error parameter. When `tol`>=0.1, the intervals [[0, 1], [1.1, 2]]
        will be simplified to [[0, 2]]. Defaults to 1e-10.

    Returns
    -------
    np.ndarray
        Simplified intervals [[l1', u1'], [l2', u2'], ...].
    """
    if len(intervals) == 0:
        return np.array([]).reshape(0, 2)

    intervals = intervals[np.argsort(intervals[:, 0])]
    simplified = [intervals[0]]
    for interval in intervals[1:]:
        if interval[0] <= simplified[-1][1] + tol:
            simplified[-1][1] = np.max([simplified[-1][1], interval[1]])
        else:
            simplified.append(interval)
    return np.array(simplified)


def complement(intervals: np.ndarray) -> np.ndarray:
    """Take the complement of intervals.

    Parameters
    ----------
    intervals : np.ndarray
        Intervals [[l1, u1], [l2, u2], ...].

    Returns
    -------
    np.ndarray
        Complement of the input intervals [[l1', u1'], [l2', u2'], ...].
    """
    if len(intervals) == 0 or len(intervals[0]) == 0:
        return np.array([[-np.inf, np.inf]])

    result = []
    if intervals[0][0] > -np.inf:
        result.append([-np.inf, intervals[0][0]])

    result += [
        [intervals[i][1], intervals[i + 1][0]] for i in range(len(intervals) - 1)
    ]

    if intervals[-1][1] < np.inf:
        result.append([intervals[-1][1], np.inf])

    return simplify(np.array(result))


def union(intervals1: np.ndarray, intervals2: np.ndarray) -> np.ndarray:
    """Take the union of two intervals.

    Parameters
    ----------
    intervals1 : np.ndarray
        Intervals [[l1, u1], [l2, u2], ...].
    intervals2 : np.ndarray
        Intervals [[l1, u1], [l2, u2], ...].

    Returns
    -------
    np.ndarray
        Union of the two input intervals [[l1', u1'], [l2', u2'], ...].
    """
    intervals = np.vstack([intervals1, intervals2])
    return simplify(intervals)


def intersection(intervals1: np.ndarray, intervals2: np.ndarray) -> np.ndarray:
    """Take the intersection of two intervals.

    Parameters
    ----------
    intervals1 : np.ndarray
        Intervals [[l1, u1], [l2, u2], ...].
    intervals2 : np.ndarray
        Intervals [[l1, u1], [l2, u2], ...].

    Returns
    -------
    np.ndarray
        Intersection of the two input intervals [[l1', u1'], [l2', u2'], ...].
    """
    return complement(union(complement(intervals1), complement(intervals2)))


class RealSubset:
    """A class representing a subset of real numbers as a collection of intervals.

    This class allows for various set operations on subsets of real numbers, including
    complement, intersection, and union. It uses numpy arrays for efficient storage
    and computation.

    Parameters
    ----------
    intervals : np.array | list[list[int]] | None, optional
        Initial intervals to create the subset. Defaults to None, which creates an empty set.
    is_simplify : bool, optional
        Whether to simplify (merge overlapping) intervals upon creation. Defaults to True.

    Attributes
    ----------
    intervals : np.ndarray
        An array of shape (n, 2) representing n intervals. Each row [a, b] represents the interval [a, b].

    Note
    ----
    - Infinite intervals can be represented using np.inf.
    - An empty array represents the empty set.

    Examples
    --------
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
        self,
        intervals: np.ndarray | list[list[float]] | None = None,
        *,
        is_simplify: bool = True,
    ) -> None:
        """Initialize a RealSubset object.

        Parameters
        ----------
        intervals : np.array | list[list[int]] | None, optional
            Initial intervals to create the subset. Defaults to None, which creates an empty set.
        is_simplify : bool, optional
            Whether to simplify (merge overlapping) intervals upon creation. Defaults to True.
        """
        if intervals is None:
            self.intervals = np.array([]).reshape(0, 2)
        else:
            self.intervals = np.array(intervals).reshape(-1, 2)
            if is_simplify:
                self.simplify()

    def simplify(self) -> None:
        """Simplify the intervals of the subset."""
        self.intervals = simplify(self.intervals)

    def complement(self) -> RealSubset:
        """Take the complement.

        Returns
        -------
        RealSubset
            Complement of the subset.
        """
        return RealSubset(complement(self.intervals), is_simplify=False)

    def __invert__(self) -> RealSubset:
        """Take the complement.

        Returns
        -------
        RealSubset
            Complement of the subset.
        """
        return self.complement()

    def union(self, other: RealSubset) -> RealSubset:
        """Take the union and can be used by the | operator.

        Parameters
        ----------
        other : RealSubset
            Another subset to take union with.

        Returns
        -------
        RealSubset
            Union of the two subsets.
        """
        return RealSubset(union(self.intervals, other.intervals), is_simplify=False)

    def __or__(self, other: RealSubset) -> RealSubset:
        """Take the union.

        Parameters
        ----------
        other : RealSubset
            Another subset to take union with.

        Returns
        -------
        RealSubset
            Union of the two subsets.
        """
        return self.union(other)

    def union_update(self, other: RealSubset) -> None:
        """Update the subset by taking the union.

        Parameters
        ----------
        other : RealSubset
            Another subset to take union with.
        """
        self.intervals = (self | other).intervals

    def __ior__(self, other: Self) -> Self:
        """Update the subset by taking the union.

        Parameters
        ----------
        other : Self
            Another subset to take union with.

        Returns
        -------
        Self
            Updated subset with the union of itself and another subset.
        """
        self.union_update(other)
        return self

    def intersection(self, other: RealSubset) -> RealSubset:
        """Take the intersection and can be used by the & operator.

        Parameters
        ----------
        other : RealSubset
            Another subset to take intersection with.

        Returns
        -------
        RealSubset
            Intersection of the two subsets.
        """
        return ~((~self) | (~other))

    def __and__(self, other: RealSubset) -> RealSubset:
        """Take the intersection.

        Parameters
        ----------
        other : RealSubset
            Another subset to take intersection with.

        Returns
        -------
        RealSubset
            Intersection of the two subsets.
        """
        return self.intersection(other)

    def intersection_update(self, other: RealSubset) -> None:
        """Update the subset by taking the intersection.

        Parameters
        ----------
        other : RealSubset
            Another subset to take intersection with.
        """
        self.intervals = (self & other).intervals

    def __iand__(self, other: Self) -> Self:
        """Update the subset by taking the intersection.

        Parameters
        ----------
        other : Self
            Another subset to take intersection with.

        Returns
        -------
        Self
            Updated subset with the intersection of itself and another subset.
        """
        self.intersection_update(other)
        return self

    def difference(self, other: RealSubset) -> RealSubset:
        """Take the difference and can be used by the - operator.

        Parameters
        ----------
        other : RealSubset
            Another subset to take difference with.

        Returns
        -------
        RealSubset
            Difference of the subset with another subset.
        """
        return self & ~other

    def __sub__(self, other: RealSubset) -> RealSubset:
        """Take the difference.

        Parameters
        ----------
        other : RealSubset
            Another subset to take difference with.

        Returns
        -------
        RealSubset
            Difference of the subset with another subset.
        """
        return self.difference(other)

    def difference_update(self, other: RealSubset) -> None:
        """Update the subset by taking the difference.

        Parameters
        ----------
        other : RealSubset
            Another subset to take difference with.
        """
        self.intervals = (self - other).intervals

    def __isub__(self, other: Self) -> Self:
        """Update the subset by taking the difference.

        Parameters
        ----------
        other : Self
            Another subset to take difference with.

        Returns
        -------
        Self
            Updated subset with the difference of itself with another subset.
        """
        self.difference_update(other)
        return self

    def symmetric_difference(self, other: RealSubset) -> RealSubset:
        """Take the symmetric difference and can be used by the ^ operator.

        Parameters
        ----------
        other : RealSubset
            Another subset to take symmetric difference with.

        Returns
        -------
        RealSubset
            Symmetric difference of the subset with another subset.
        """
        return (self - other) | (other - self)

    def __xor__(self, other: RealSubset) -> RealSubset:
        """Take the symmetric difference.

        Parameters
        ----------
        other : RealSubset
            Another subset to take symmetric difference with.

        Returns
        -------
        RealSubset
            Symmetric difference of the subset with another subset.
        """
        return self.symmetric_difference(other)

    def symmetric_difference_update(self, other: RealSubset) -> None:
        """Update the subset by taking the symmetric difference.

        Parameters
        ----------
        other : RealSubset
            Another subset to take symmetric difference with.
        """
        self.intervals = (self ^ other).intervals

    def __ixor__(self, other: Self) -> Self:
        """Update the subset by taking the symmetric difference.

        Parameters
        ----------
        other : Self
            Another subset to take symmetric difference with.

        Returns
        -------
        Self
            Updated subset with the symmetric difference of itself with another subset.
        """
        self.symmetric_difference_update(other)
        return self

    def is_empty(self) -> bool:
        """Check if the subset is empty.

        Returns
        -------
        bool
            True if the subset is empty, False otherwise.
        """
        return len(self.intervals) == 0

    def __eq__(self, other: object) -> bool:
        """Check if two objects are equal.

        Parameters
        ----------
        other : Any
            Another object to compare with.

        Returns
        -------
        bool
            True if another object is a RealSubset and two subsets are equal, False otherwise.
        """
        if not isinstance(other, RealSubset):
            return False
        if self.intervals.shape != other.intervals.shape:
            return False
        return np.allclose(self.intervals, other.intervals, rtol=1e-12, atol=1e-12)

    def issubset(self, other: RealSubset) -> bool:
        """Check if the subset is a subset of another subset.

        Parameters
        ----------
        other : RealSubset
            Another subset to check for subset.

        Returns
        -------
        bool
            True if the subset is a subset of another subset, False otherwise.
        """
        return (self - other).is_empty()

    def __le__(self, other: RealSubset) -> bool:
        """Check if the subset is a subset of another subset.

        Parameters
        ----------
        other : RealSubset
            Another subset to check for subset.

        Returns
        -------
        bool
            True if the subset is a subset of another subset, False otherwise.
        """
        return self.issubset(other)

    def __lt__(self, other: RealSubset) -> bool:
        """Check if the subset is a proper subset of another subset.

        Parameters
        ----------
        other : RealSubset
            Another subset to check for proper subset.

        Returns
        -------
        bool
            True if the subset is a proper subset of another subset, False otherwise.
        """
        return self.issubset(other) and self != other

    def issuperset(self, other: RealSubset) -> bool:
        """Check if the subset is a superset of another subset.

        Parameters
        ----------
        other : RealSubset
            Another subset to check for superset.

        Returns
        -------
        bool
            True if the subset is a superset of another subset, False otherwise.
        """
        return other.issubset(self)

    def __ge__(self, other: RealSubset) -> bool:
        """Check if the subset is a superset of another subset.

        Parameters
        ----------
        other : RealSubset
            Another subset to check for superset.

        Returns
        -------
        bool
            True if the subset is a superset of another subset, False otherwise.
        """
        return self.issuperset(other)

    def __gt__(self, other: RealSubset) -> bool:
        """Check if the subset is a proper superset of another subset.

        Parameters
        ----------
        other : RealSubset
            Another subset to check for proper superset.

        Returns
        -------
        bool
            True if the subset is a proper superset of another subset, False otherwise.
        """
        return self.issuperset(other) and self != other

    def isdisjoint(self, other: RealSubset) -> bool:
        """Check if the subset is disjoint with another subset.

        Parameters
        ----------
        other : RealSubset
            Another subset to check for disjoint.

        Returns
        -------
        bool
            True if the subset is disjoint with another subset, False otherwise.
        """
        return (self & other).is_empty()

    def __contains__(self, z: float) -> bool:
        """Check if a real number is in the subset.

        Parameters
        ----------
        z : float
            Real number to check for membership.

        Returns
        -------
        bool
            True if z is in the subset, False otherwise.
        """
        if len(self.intervals) == 0:
            return False
        return np.any((self.intervals[:, 0] <= z) & (z <= self.intervals[:, 1])).item()

    def find_interval_containing(self, z: float) -> list[float]:
        """Find the interval containing a real number.

        Parameters
        ----------
        z : float
            Real number to find the interval containing it.

        Returns
        -------
        list[float]
            Interval containing z

        Raises
        ------
        ValueError
            If the subset is empty or no interval contains z.
        """
        if len(self.intervals) == 0:
            raise NotBelongToSubsetError(z, self)
        mask = (self.intervals[:, 0] <= z) & (z <= self.intervals[:, 1])
        if np.sum(mask) == 0:
            raise NotBelongToSubsetError(z, self)
        return self.intervals[mask][0].tolist()

    def tolist(self) -> list[list[float]]:
        """Return the intervals as a list of lists.

        Returns
        -------
        list[list[float]]
            Intervals as a list of lists.
        """
        return self.intervals.tolist()

    @property
    def measure(self) -> float:
        """Return the measure of the subset.

        Returns
        -------
        float
            Measure of the subset.
        """
        return np.sum(self.intervals[:, 1] - self.intervals[:, 0]).item()

    def __len__(self) -> int:
        """Return the number of intervals in the subset.

        Returns
        -------
        int
            Number of intervals in the subset.
        """
        return len(self.intervals)

    def __str__(self) -> str:
        """Return a string representation of the subset.

        Returns
        -------
        str
            String representation of the subset.
        """
        if len(self.intervals) == 0:
            return "[]"
        return "[" + ", ".join([f"[{l:.6f}, {u:.6f}]" for l, u in self.intervals]) + "]"

    def __repr__(self) -> str:
        """Return a string representation that can be used to recreate the object.

        Returns
        -------
        str
            String representation of the RealSubset in the format "RealSubset([[a, b], [c, d], ...])".
        """
        return f"RealSubset({self.tolist()})"


class NotBelongToSubsetError(Exception):
    """Error raised when a value does not belong to a subset."""

    def __init__(self, value: float, subset: list[list[float]] | RealSubset) -> None:
        """Initialize a NotBelongToSubsetError.

        Parameters
        ----------
        value : float
            Value that does not belong to the subset.
        subset : list[list[float]] | RealSubset
            Subset that the value does not belong to.
        """
        subset = subset if isinstance(subset, RealSubset) else RealSubset(subset)
        super().__init__(
            f"The value {value:.6f} does not belong to the subset {subset}.",
        )
