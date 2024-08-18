import pytest
import numpy as np
from numpy.testing import assert_allclose
from sicore.utils.intervals import (
    complement,
    union,
    intersection,
    difference,
    symmetric_difference,
    polynomial_below_zero,
    polytope_below_zero,
    degree_one_polynomials_below_zero,
)


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
    assert_allclose(complement(intervals), np.array(expected).reshape(-1, 2))


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
    assert_allclose(union(intervals1, intervals2), np.array(expected).reshape(-1, 2))


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
    assert_allclose(
        intersection(intervals1, intervals2), np.array(expected).reshape(-1, 2)
    )


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
    assert_allclose(
        difference(intervals1, intervals2), np.array(expected).reshape(-1, 2)
    )


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
    assert_allclose(
        symmetric_difference(intervals1, intervals2), np.array(expected).reshape(-1, 2)
    )


# def test_poly_lt_zero():
#     testcase = [
#         ([1], []),
#         ([0], [[NINF, INF]]),
#         ([-1], [[NINF, INF]]),
#         ([1, 1], [[NINF, -1]]),
#         ([-1, 1], [[1, INF]]),
#         ([1, -2, 2], []),
#         ([1, -2, 1], []),
#         ([1, 0, -1], [[-1, 1]]),
#         ([-1, 0, 1], [[NINF, -1], [1, INF]]),
#         ([-1, 2, -1], [[NINF, INF]]),
#         ([-1, 2, -2], [[NINF, INF]]),
#         ([1, -6, 11, -6], [[NINF, 1], [2, 3]]),
#         ([1, -3, 4, -2], [[NINF, 1]]),
#         ([1, 0, 0, 0], [[NINF, 0]]),
#     ]

#     for coef, expected in testcase:
#         assert_allclose(poly_lt_zero(coef), expected)

#     for coef, expected in testcase:
#         poly = np.poly1d(coef)
#         assert_allclose(poly_lt_zero(poly), expected)
