import pytest
import numpy as np
from sicore.core.real_subset import RealSubset


@pytest.mark.parametrize(
    "intervals, expected",
    [
        ([[2.0, 4.0], [7.0, 9.0]], [[2.0, 4.0], [7.0, 9.0]]),  # same intervals
        ([[7.0, 9.0], [2.0, 4.0]], [[2.0, 4.0], [7.0, 9.0]]),  # automatically sorted
        ([[1.0, 3.0], [3.0, 5.0]], [[1.0, 5.0]]),  # merge intervals
        ([[-3.0, -1.0], [-1.0 + 1e-12, 2.0]], [[-3.0, 2.0]]),  # ignore small difference
    ],
)
def test_simplify(intervals, expected):
    assert RealSubset(intervals) == RealSubset(expected)


@pytest.mark.parametrize(
    "intervals, expected",
    [
        ([[]], [[-np.inf, np.inf]]),
        ([[-np.inf, np.inf]], [[]]),
        ([[-5.0, np.inf]], [[-np.inf, -5.0]]),
        ([[-4.0, 0.0], [1.0, 3.0]], [[-np.inf, -4.0], [0.0, 1.0], [3.0, np.inf]]),
    ],
)
def test_complement(intervals, expected):
    assert ~RealSubset(intervals) == RealSubset(expected)


@pytest.mark.parametrize(
    "intervals1, intervals2, expected",
    [
        ([[1.0, 3.0], [5.0, 7.0]], [[2.0, 4.0], [6.0, 8.0]], [[1.0, 4.0], [5.0, 8.0]]),
        ([[1.0, 3.0], [5.0, 7.0]], [[1.0, 7.0]], [[1.0, 7.0]]),
        ([[2.0, 4.0]], [[0.0, 1.0], [5.0, 7.0]], [[0.0, 1.0], [2.0, 4.0], [5.0, 7.0]]),
        ([[-np.inf, 2.0]], [[1.0, 4.0], [5.0, 7.0]], [[-np.inf, 4.0], [5.0, 7.0]]),
    ],
)
def test_union(intervals1, intervals2, expected):
    assert RealSubset(intervals1) | RealSubset(intervals2) == RealSubset(expected)


@pytest.mark.parametrize(
    "intervals1, intervals2, expected",
    [
        ([[1.0, 3.0], [5.0, 7.0]], [[2.0, 4.0], [6.0, 8.0]], [[2.0, 3.0], [6.0, 7.0]]),
        ([[1.0, 3.0], [5.0, 7.0]], [[1.0, 7.0]], [[1.0, 3.0], [5.0, 7.0]]),
        ([[2.0, 4.0]], [[0.0, 1.0], [5.0, 7.0]], [[]]),
        ([[-np.inf, 2.0]], [[1.0, 4.0], [5.0, 7.0]], [[1.0, 2.0]]),
    ],
)
def test_intersection(intervals1, intervals2, expected):
    assert RealSubset(intervals1) & RealSubset(intervals2) == RealSubset(expected)


@pytest.mark.parametrize(
    "intervals1, intervals2, expected",
    [
        ([[1.0, 3.0], [5.0, 7.0]], [[2.0, 4.0], [6.0, 8.0]], [[1.0, 2.0], [5.0, 6.0]]),
        ([[1.0, 3.0], [5.0, 7.0]], [[1.0, 7.0]], [[]]),
        ([[2.0, 4.0]], [[0.0, 1.0], [5.0, 7.0]], [[2.0, 4.0]]),
        ([[-np.inf, 2.0]], [[1.0, 4.0], [5.0, 7.0]], [[-np.inf, 1.0]]),
    ],
)
def test_difference(intervals1, intervals2, expected):
    assert RealSubset(intervals1) - RealSubset(intervals2) == RealSubset(expected)


@pytest.mark.parametrize(
    "intervals1, intervals2, expected",
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
def test_symmetric_difference(intervals1, intervals2, expected):
    assert RealSubset(intervals1) ^ RealSubset(intervals2) == RealSubset(expected)


@pytest.mark.parametrize(
    "intervals1, intervals2, expected",
    [
        ([[1.0, 3.0], [5.0, 7.0]], [[2.0, 4.0], [6.0, 8.0]], False),
        ([[1.0, 3.0], [5.0, 7.0]], [[1.0, 7.0]], True),
        ([[1.0, 7.0]], [[1.0, 3.0], [5.0, 7.0]], False),
        ([[-np.inf, 2.0]], [[-np.inf, 2.0]], True),
    ],
)
def test_less_equal(intervals1, intervals2, expected):
    assert (RealSubset(intervals1) <= RealSubset(intervals2)) == expected


@pytest.mark.parametrize(
    "intervals1, intervals2, expected",
    [
        ([[1.0, 3.0], [5.0, 7.0]], [[2.0, 4.0], [6.0, 8.0]], False),
        ([[1.0, 3.0], [5.0, 7.0]], [[1.0, 7.0]], True),
        ([[1.0, 7.0]], [[1.0, 3.0], [5.0, 7.0]], False),
        ([[-np.inf, 2.0]], [[-np.inf, 2.0]], False),
    ],
)
def test_less_than(intervals1, intervals2, expected):
    assert (RealSubset(intervals1) < RealSubset(intervals2)) == expected


@pytest.mark.parametrize(
    "intervals1, intervals2, expected",
    [
        ([[1.0, 3.0], [5.0, 7.0]], [[2.0, 4.0], [6.0, 8.0]], False),
        ([[1.0, 3.0], [5.0, 7.0]], [[1.0, 7.0]], False),
        ([[1.0, 7.0]], [[1.0, 3.0], [5.0, 7.0]], True),
        ([[-np.inf, 2.0]], [[-np.inf, 2.0]], True),
    ],
)
def test_greater_equal(intervals1, intervals2, expected):
    assert (RealSubset(intervals1) >= RealSubset(intervals2)) == expected


@pytest.mark.parametrize(
    "intervals1, intervals2, expected",
    [
        ([[1.0, 3.0], [5.0, 7.0]], [[2.0, 4.0], [6.0, 8.0]], False),
        ([[1.0, 3.0], [5.0, 7.0]], [[1.0, 7.0]], False),
        ([[1.0, 7.0]], [[1.0, 3.0], [5.0, 7.0]], True),
        ([[-np.inf, 2.0]], [[-np.inf, 2.0]], False),
    ],
)
def test_greater_than(intervals1, intervals2, expected):
    assert (RealSubset(intervals1) > RealSubset(intervals2)) == expected


@pytest.mark.parametrize(
    "intervals1, intervals2, expected",
    [
        ([[1.0, 3.0], [5.0, 7.0]], [[3.0, 6.0]], False),
        ([[1.0, 3.0], [5.0, 7.0]], [[8.0, 10.0]], True),
    ],
)
def test_isdisjoint(intervals1, intervals2, expected):
    assert RealSubset(intervals1).isdisjoint(RealSubset(intervals2)) == expected


@pytest.mark.parametrize(
    "intervals, value, expected",
    [
        ([[1.0, 3.0], [5.0, 7.0]], 4.0, False),
        ([[1.0, 3.0], [5.0, 7.0]], 6.0, True),
    ],
)
def test_contains(intervals, value, expected):
    assert (value in RealSubset(intervals)) == expected


@pytest.mark.parametrize(
    "intervals, value, expected_result, expected_exception",
    [
        ([[1.0, 3.0], [5.0, 7.0]], 4.0, None, ValueError),
        ([[1.0, 3.0], [5.0, 7.0]], 6.0, [5.0, 7.0], None),
    ],
)
def test_find_interval_containing(
    intervals, value, expected_result, expected_exception
):
    if expected_exception is not None:
        with pytest.raises(expected_exception):
            RealSubset(intervals).find_interval_containing(value)
    else:
        assert RealSubset(intervals).find_interval_containing(value) == expected_result


@pytest.mark.parametrize(
    "intervals",
    [
        ([[1.0, 3.0], [5.0, 7.0]]),
        ([[-np.inf, 2.0], [3.0, 4.0], [5.0, 7.0]]),
    ],
)
def test_tolist(intervals):
    assert RealSubset(intervals).tolist() == intervals


@pytest.mark.parametrize(
    "intervals, expected",
    [
        ([[1.0, 3.0], [5.0, 7.0]], 2),
        ([[]], 0),
    ],
)
def test_len(intervals, expected):
    assert len(RealSubset(intervals)) == expected
