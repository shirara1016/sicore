"""Module with tests for the RealSubset class."""

import numpy as np
import pytest

from sicore.core.real_subset import NotBelongToSubsetError, RealSubset


@pytest.mark.parametrize(
    ("intervals", "expected"),
    [
        ([[2.0, 4.0], [7.0, 9.0]], [[2.0, 4.0], [7.0, 9.0]]),  # same intervals
        ([[7.0, 9.0], [2.0, 4.0]], [[2.0, 4.0], [7.0, 9.0]]),  # automatically sorted
        ([[1.0, 3.0], [3.0, 5.0]], [[1.0, 5.0]]),  # merge intervals
        ([[-3.0, -1.0], [-1.0 + 1e-12, 2.0]], [[-3.0, 2.0]]),  # ignore small difference
    ],
)
def test_simplify(intervals: list[list[float]], expected: list[list[float]]) -> None:
    """Test the simplify method."""
    assert RealSubset(intervals) == RealSubset(expected)


@pytest.mark.parametrize(
    ("intervals", "expected"),
    [
        ([[]], [[-np.inf, np.inf]]),
        ([[-np.inf, np.inf]], [[]]),
        ([[-5.0, np.inf]], [[-np.inf, -5.0]]),
        ([[-4.0, 0.0], [1.0, 3.0]], [[-np.inf, -4.0], [0.0, 1.0], [3.0, np.inf]]),
    ],
)
def test_complement(intervals: list[list[float]], expected: list[list[float]]) -> None:
    """Test the complement method."""
    assert ~RealSubset(intervals) == RealSubset(expected)


@pytest.mark.parametrize(
    ("intervals1", "intervals2", "expected"),
    [
        ([[1.0, 3.0], [5.0, 7.0]], [[2.0, 4.0], [6.0, 8.0]], [[1.0, 4.0], [5.0, 8.0]]),
        ([[1.0, 3.0], [5.0, 7.0]], [[1.0, 7.0]], [[1.0, 7.0]]),
        ([[2.0, 4.0]], [[0.0, 1.0], [5.0, 7.0]], [[0.0, 1.0], [2.0, 4.0], [5.0, 7.0]]),
        ([[-np.inf, 2.0]], [[1.0, 4.0], [5.0, 7.0]], [[-np.inf, 4.0], [5.0, 7.0]]),
    ],
)
def test_union(
    intervals1: list[list[float]],
    intervals2: list[list[float]],
    expected: list[list[float]],
) -> None:
    """Test the union method."""
    assert RealSubset(intervals1) | RealSubset(intervals2) == RealSubset(expected)


@pytest.mark.parametrize(
    ("intervals1", "intervals2", "expected"),
    [
        ([[1.0, 3.0], [5.0, 7.0]], [[2.0, 4.0], [6.0, 8.0]], [[2.0, 3.0], [6.0, 7.0]]),
        ([[1.0, 3.0], [5.0, 7.0]], [[1.0, 7.0]], [[1.0, 3.0], [5.0, 7.0]]),
        ([[2.0, 4.0]], [[0.0, 1.0], [5.0, 7.0]], [[]]),
        ([[-np.inf, 2.0]], [[1.0, 4.0], [5.0, 7.0]], [[1.0, 2.0]]),
    ],
)
def test_intersection(
    intervals1: list[list[float]],
    intervals2: list[list[float]],
    expected: list[list[float]],
) -> None:
    """Test the intersection method."""
    assert RealSubset(intervals1) & RealSubset(intervals2) == RealSubset(expected)


@pytest.mark.parametrize(
    ("intervals1", "intervals2", "expected"),
    [
        ([[1.0, 3.0], [5.0, 7.0]], [[2.0, 4.0], [6.0, 8.0]], [[1.0, 2.0], [5.0, 6.0]]),
        ([[1.0, 3.0], [5.0, 7.0]], [[1.0, 7.0]], [[]]),
        ([[2.0, 4.0]], [[0.0, 1.0], [5.0, 7.0]], [[2.0, 4.0]]),
        ([[-np.inf, 2.0]], [[1.0, 4.0], [5.0, 7.0]], [[-np.inf, 1.0]]),
    ],
)
def test_difference(
    intervals1: list[list[float]],
    intervals2: list[list[float]],
    expected: list[list[float]],
) -> None:
    """Test the difference method."""
    assert RealSubset(intervals1) - RealSubset(intervals2) == RealSubset(expected)


@pytest.mark.parametrize(
    ("intervals1", "intervals2", "expected"),
    [
        (
            [[1.0, 3.0], [5.0, 7.0]],
            [[2.0, 4.0], [6.0, 8.0]],
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
        ),
        ([[1.0, 3.0], [5.0, 7.0]], [[1.0, 7.0]], [[3.0, 5.0]]),
        ([[2.0, 4.0]], [[0.0, 1.0], [5.0, 7.0]], [[0.0, 1.0], [2.0, 4.0], [5.0, 7.0]]),
        (
            [[-np.inf, 2.0]],
            [[1.0, 4.0], [5.0, 7.0]],
            [[-np.inf, 1.0], [2.0, 4.0], [5.0, 7.0]],
        ),
    ],
)
def test_symmetric_difference(
    intervals1: list[list[float]],
    intervals2: list[list[float]],
    expected: list[list[float]],
) -> None:
    """Test the symmetric difference method."""
    assert RealSubset(intervals1) ^ RealSubset(intervals2) == RealSubset(expected)


@pytest.mark.parametrize(
    ("intervals1", "intervals2", "expected"),
    [
        ([[1.0, 3.0], [5.0, 7.0]], [[2.0, 4.0], [6.0, 8.0]], False),
        ([[1.0, 3.0], [5.0, 7.0]], [[1.0, 7.0]], True),
        ([[1.0, 7.0]], [[1.0, 3.0], [5.0, 7.0]], False),
        ([[-np.inf, 2.0]], [[-np.inf, 2.0]], True),
    ],
)
def test_less_equal(
    intervals1: list[list[float]],
    intervals2: list[list[float]],
    *,
    expected: bool,
) -> None:
    """Test the less equal method."""
    assert (RealSubset(intervals1) <= RealSubset(intervals2)) == expected


@pytest.mark.parametrize(
    ("intervals1", "intervals2", "expected"),
    [
        ([[1.0, 3.0], [5.0, 7.0]], [[2.0, 4.0], [6.0, 8.0]], False),
        ([[1.0, 3.0], [5.0, 7.0]], [[1.0, 7.0]], True),
        ([[1.0, 7.0]], [[1.0, 3.0], [5.0, 7.0]], False),
        ([[-np.inf, 2.0]], [[-np.inf, 2.0]], False),
    ],
)
def test_less_than(
    intervals1: list[list[float]],
    intervals2: list[list[float]],
    expected: bool,
) -> None:
    """Test the less than method."""
    assert (RealSubset(intervals1) < RealSubset(intervals2)) == expected


@pytest.mark.parametrize(
    ("intervals1", "intervals2", "expected"),
    [
        ([[1.0, 3.0], [5.0, 7.0]], [[2.0, 4.0], [6.0, 8.0]], False),
        ([[1.0, 3.0], [5.0, 7.0]], [[1.0, 7.0]], False),
        ([[1.0, 7.0]], [[1.0, 3.0], [5.0, 7.0]], True),
        ([[-np.inf, 2.0]], [[-np.inf, 2.0]], True),
    ],
)
def test_greater_equal(
    intervals1: list[list[float]],
    intervals2: list[list[float]],
    *,
    expected: bool,
) -> None:
    """Test the greater equal method."""
    assert (RealSubset(intervals1) >= RealSubset(intervals2)) == expected


@pytest.mark.parametrize(
    ("intervals1", "intervals2", "expected"),
    [
        ([[1.0, 3.0], [5.0, 7.0]], [[2.0, 4.0], [6.0, 8.0]], False),
        ([[1.0, 3.0], [5.0, 7.0]], [[1.0, 7.0]], False),
        ([[1.0, 7.0]], [[1.0, 3.0], [5.0, 7.0]], True),
        ([[-np.inf, 2.0]], [[-np.inf, 2.0]], False),
    ],
)
def test_greater_than(
    intervals1: list[list[float]],
    intervals2: list[list[float]],
    *,
    expected: bool,
) -> None:
    """Test the greater than method."""
    assert (RealSubset(intervals1) > RealSubset(intervals2)) == expected


@pytest.mark.parametrize(
    ("intervals1", "intervals2", "expected"),
    [
        ([[1.0, 3.0], [5.0, 7.0]], [[3.0, 6.0]], False),
        ([[1.0, 3.0], [5.0, 7.0]], [[8.0, 10.0]], True),
    ],
)
def test_isdisjoint(
    intervals1: list[list[float]],
    intervals2: list[list[float]],
    expected: bool,
) -> None:
    """Test the isdisjoint method."""
    assert RealSubset(intervals1).isdisjoint(RealSubset(intervals2)) == expected


@pytest.mark.parametrize(
    ("intervals", "value", "expected"),
    [
        ([[1.0, 3.0], [5.0, 7.0]], 4.0, False),
        ([[1.0, 3.0], [5.0, 7.0]], 6.0, True),
    ],
)
def test_contains(
    intervals: list[list[float]],
    value: float,
    expected: bool,
) -> None:
    """Test the contains method."""
    assert (value in RealSubset(intervals)) == expected


@pytest.mark.parametrize(
    ("intervals", "value", "expected_result"),
    [
        ([[1.0, 3.0], [5.0, 7.0]], 6.0, [5.0, 7.0]),
        ([[1.0, 3.0], [5.0, 7.0]], 2.0, [1.0, 3.0]),
        ([[-np.inf, 3.0]], 2.0, [-np.inf, 3.0]),
        ([[1.0, 3.0], [5.0, 7.0]], 4.0, None),
    ],
)
def test_find_interval_containing(
    intervals: list[list[float]],
    value: float,
    expected_result: list[float] | None,
) -> None:
    """Test the find_interval_containing method."""
    if expected_result is None:
        with pytest.raises(NotBelongToSubsetError):
            RealSubset(intervals).find_interval_containing(value)
    else:
        assert RealSubset(intervals).find_interval_containing(value) == expected_result


@pytest.mark.parametrize(
    "intervals",
    [
        [[1.0, 3.0], [5.0, 7.0]],
        [[-np.inf, 2.0], [3.0, 4.0], [5.0, 7.0]],
    ],
)
def test_tolist(intervals: list[list[float]]) -> None:
    """Test the tolist method."""
    assert RealSubset(intervals).tolist() == intervals


@pytest.mark.parametrize(
    ("intervals", "measure"),
    [
        ([[1.0, 3.0], [5.0, 7.0]], 4.0),
        ([[-np.inf, 2.0], [3.0, 4.0], [5.0, 7.0]], np.inf),
    ],
)
def test_measure(intervals: list[list[float]], measure: float) -> None:
    """Test the tolist method."""
    assert RealSubset(intervals).measure == measure


@pytest.mark.parametrize(
    ("intervals", "expected"),
    [
        ([[1.0, 3.0], [5.0, 7.0]], 2),
        ([[]], 0),
    ],
)
def test_len(intervals: list[list[float]], expected: int) -> None:
    """Test the len method."""
    assert len(RealSubset(intervals)) == expected
