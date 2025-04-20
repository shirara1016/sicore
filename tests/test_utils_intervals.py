"""Module with tests for the intervals utilities."""

import numpy as np
import pytest
from numpy.polynomial import Polynomial
from numpy.testing import assert_allclose

from sicore.utils.intervals import (
    complement,
    difference,
    intersection,
    linear_polynomials_below_zero,
    polynomial_below_zero,
    polynomial_iso_sign_interval,
    polytope_below_zero,
    symmetric_difference,
    union,
)


@pytest.mark.parametrize(
    ("intervals", "expected"),
    [
        ([[]], [[-np.inf, np.inf]]),
        ([[-np.inf, np.inf]], [[]]),
        ([[-5.0, np.inf]], [[-np.inf, -5.0]]),
        ([[-4.0, 0.0], [1.0, 3.0]], [[-np.inf, -4.0], [0.0, 1.0], [3.0, np.inf]]),
    ],
)
def test_complement(
    intervals: list[list[float]] | list,
    expected: list[list[float]],
) -> None:
    """Test the complement function."""
    assert_allclose(complement(np.array(intervals)), np.array(expected).reshape(-1, 2))


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
    """Test the union function."""
    assert_allclose(
        union(np.array(intervals1), np.array(intervals2)),
        np.array(expected).reshape(-1, 2),
    )


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
    """Test the intersection function."""
    assert_allclose(
        intersection(np.array(intervals1), np.array(intervals2)),
        np.array(expected).reshape(-1, 2),
    )


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
    """Test the difference function."""
    assert_allclose(
        difference(np.array(intervals1), np.array(intervals2)),
        np.array(expected).reshape(-1, 2),
    )


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
    """Test the symmetric difference function."""
    assert_allclose(
        symmetric_difference(np.array(intervals1), np.array(intervals2)),
        np.array(expected).reshape(-1, 2),
    )


@pytest.mark.parametrize(
    ("coef", "z", "expected"),
    [
        ([1.0], 0.0, [[-np.inf, np.inf]]),
        ([-1.0, 0.0, 1.0], 0.5, [[-1.0, 1.0]]),
        ([-1.0, 0.0, 1.0], 1.5, [[1.0, np.inf]]),
        ([-2.0, 2.0, -1.0], 1.5, [[1.0, np.inf]]),
        ([-6.0, 11.0, -6.0, 1.0], 2.5, [[2.0, 3.0]]),
    ],
)
def test_polynomial_iso_sign_interval(
    coef: list[float],
    z: float,
    expected: list[list[float]],
) -> None:
    """Test the polynomial iso-sign interval function."""
    assert_allclose(polynomial_iso_sign_interval(coef, z), expected)

    poly = Polynomial(coef)
    assert_allclose(polynomial_iso_sign_interval(poly, z), expected)


@pytest.mark.parametrize(
    ("coef", "expected"),
    [
        ([1.0], []),
        ([0.0], [[-np.inf, np.inf]]),
        ([-1.0], [[-np.inf, np.inf]]),
        ([1.0, 1.0], [[-np.inf, -1.0]]),
        ([1.0, -1.0], [[1.0, np.inf]]),
        ([2.0, -2.0, 1.0], []),
        ([1.0, -2.0, 1.0], []),
        ([-1.0, 0.0, 1.0], [[-1.0, 1.0]]),
        ([1.0, 0.0, -1.0], [[-np.inf, -1.0], [1.0, np.inf]]),
        ([-1.0, 2.0, -1.0], [[-np.inf, np.inf]]),
        ([-2.0, 2.0, -1.0], [[-np.inf, np.inf]]),
        ([-6.0, 11.0, -6.0, 1.0], [[-np.inf, 1.0], [2.0, 3.0]]),
        ([-2.0, 4.0, -3.0, 1.0], [[-np.inf, 1.0]]),
        ([0.0, 0.0, 0.0, 1.0], [[-np.inf, 0.0]]),
    ],
)
def test_polynomial_below_zero(coef: list[float], expected: list[list[float]]) -> None:
    """Test the polynomial below zero function."""
    assert_allclose(polynomial_below_zero(coef), expected)

    poly = Polynomial(coef)
    assert_allclose(polynomial_below_zero(poly), expected)


@pytest.mark.parametrize(
    ("a_vector", "b_vector", "a", "b", "c", "expected"),
    [
        ([0.0, -2.0], [1.0, 1.0], None, [1.0, 1.0], 4.0, [[-np.inf, -1.0]]),
        ([0.0, -2.0], [-1.0, -1.0], None, [1.0, 1.0], 4.0, [[1.0, np.inf]]),
        ([-1.0, -1.0], [1.0, 1.0], np.eye(2), None, -2.0, [[0.0, 2.0]]),
        (
            [-1.0, -1.0],
            [1.0, 1.0],
            -np.eye(2),
            None,
            2.0,
            [[-np.inf, 0.0], [2.0, np.inf]],
        ),
    ],
)
def test_polytope_below_zero(
    a_vector: list[float],
    b_vector: list[float],
    a: np.ndarray | None,
    b: np.ndarray | None,
    c: float,
    expected: list[list[float]],
) -> None:
    """Test the polytope below zero function."""
    a_vector_, b_vector_ = np.array(a_vector), np.array(b_vector)
    assert_allclose(polytope_below_zero(a_vector_, b_vector_, a, b, c), expected)


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ([2.0, -3.0], [-2.0, 1.0], [[1.0, 3.0]]),
        ([-3.0, -2.0], [-3.0, -1.0], [[-1.0, np.inf]]),
        ([-6.0, -6.0], [1.0, 2.0], [[-np.inf, 3.0]]),
        ([5.0, 2.0], [-1.0, 0.0], []),
    ],
)
def test_linear_polynomials_below_zero(
    a: list[float],
    b: list[float],
    expected: list[list[float]],
) -> None:
    """Test the degree one polynomials below zero function."""
    a_, b_ = np.array(a), np.array(b)
    assert_allclose(linear_polynomials_below_zero(a_, b_), expected)
