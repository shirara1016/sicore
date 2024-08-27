"""Module providing functions for computing intervals and solving inequalities."""

from itertools import pairwise

import numpy as np
from numpy.polynomial import Polynomial
from scipy import sparse  # type: ignore[import]

from sicore.core.real_subset import complement, intersection, simplify, union


def difference(intervals1: np.ndarray, intervals2: np.ndarray) -> np.ndarray:
    """Take the difference of first intervals with second intervals.

    Args:
        intervals1 (np.ndarray): Intervals [[l1, u1], [l2, u2], ...].
        intervals2 (np.ndarray): Intervals [[l1, u1], [l2, u2], ...].

    Returns:
        np.ndarray:
            Difference of the first input intervals with the second input intervals
            [[l1', u1'], [l2', u2'], ...].
    """
    return intersection(intervals1, complement(intervals2))


def symmetric_difference(intervals1: np.ndarray, intervals2: np.ndarray) -> np.ndarray:
    """Take the symmetric difference of two intervals.

    Args:
        intervals1 (np.ndarray): Intervals [[l1, u1], [l2, u2], ...].
        intervals2 (np.ndarray): Intervals [[l1, u1], [l2, u2], ...].

    Returns:
        np.ndarray:
            Symmetric difference of the two input intervals
            [[l1', u1'], [l2', u2'], ...].
    """
    return union(difference(intervals1, intervals2), difference(intervals2, intervals1))


def polynomial_below_zero(
    poly_or_coef: Polynomial | np.ndarray | list[float],
    tol: float = 1e-10,
) -> list[list[float]]:
    """Compute intervals where a given polynomial is below zero.

    Args:
        poly_or_coef (Polynomial | np.ndarray | list[float]):
            Polynomial or its coefficients e.g. [a, b, c] for a + b * z + c * z ** 2.
        tol (float, optional): Tolerance error parameter. It is recommended to set a
            large value (about 1e-5) for high order polynomial (>= 3) or a polynomial
            with multiple root. Defaults to 1e-10.

    Returns:
        list[list[float]]: Intervals where the polynomial is below zero.
    """
    if isinstance(poly_or_coef, Polynomial):
        coef = poly_or_coef.coef.tolist()
    else:
        coef = np.array(poly_or_coef).tolist()

    coef = [0 if -tol < c_ < tol else c_ for c_ in coef]
    poly = Polynomial(coef).trim()

    if poly.degree() == 0:
        if poly.coef[0] <= 0:
            return [[-np.inf, np.inf]]
        return []

    roots_: np.ndarray = poly.roots()
    if np.issubdtype(roots_.dtype, np.complexfloating):
        roots = [root.real for root in roots_ if -tol < root.imag < tol]
        if len(roots) == 0:
            if poly(0.0) <= 0:
                return [[-np.inf, np.inf]]
            return []
    else:
        roots = roots_.tolist()

    roots = np.unique(roots)
    intervals = []

    if poly(roots[0] - 1) <= 0:
        intervals.append([-np.inf, roots[0]])
    for start, end in pairwise(roots):
        if end - start < tol:
            continue
        mid = (start + end) / 2
        if poly(mid) <= 0:
            intervals.append([start, end])
    if poly(roots[-1] + 1) <= 0:
        intervals.append([roots[-1], np.inf])

    return simplify(np.array(intervals)).tolist()


def polytope_below_zero(
    a_vector: np.ndarray,
    b_vector: np.ndarray,
    a: np.ndarray | sparse.csr_matrix | None = None,
    b: np.ndarray | None = None,
    c: float | None = None,
    *,
    tol: float = 1e-10,
    use_sparse: bool = False,
) -> list[list[float]]:
    """Compute intervals where a given polytope is below zero.

    The polytope is defined as the set of z such that
    (a_vec+b_vec*z)^T A (a_vec+b_vec*z) + b^T (a_vec+b_vec*z) + c < 0.0.

    Args:
        a_vector (np.ndarray): Vector a_vec in the polytope.
        b_vector (np.ndarray): Vector b_vec in the polytope.
        a (np.ndarray | sparse.csr_matrix, optional):
            Matrix A in the polytope. Defaults to None.
        b (np.ndarray, optional):
            Vector b in the polytope. Defaults to None.
        c (float, optional):
            Scalar c in the polytope. Defaults to None.
        tol (float, optional): Tolerance error parameter. Defaults to 1e-10.
        use_sparse (bool, optional):
            Whether to use sparse matrix for computation of A matrix. Defaults to False.

    Returns:
        list[list[float]]: Intervals where the polytope is below zero.
    """
    alpha, beta, gamma = 0.0, 0.0, 0.0

    if a is not None:
        a_mat = np.array(a) if not use_sparse else sparse.csr_matrix(a)
        a_a_mat = a_vector @ a_mat
        b_a_mat = b_vector @ a_mat
        alpha += b_a_mat @ b_vector
        beta += a_a_mat @ b_vector + b_a_mat @ a_vector
        gamma += a_a_mat @ a_vector

    if b is not None:
        beta += b @ b_vector
        gamma += b @ a_vector

    if c is not None:
        gamma += c

    return polynomial_below_zero([gamma, beta, alpha], tol=tol)


def linear_polynomials_below_zero(
    a: np.ndarray,
    b: np.ndarray,
) -> list[list[float]]:
    """Compute intervals where given linear polynomials are all below zero.

    The linear polynomials are defined as the set of z such that
    a_i + b_i * z < 0.0 for all i in [len(a)].

    Args:
        a (np.ndarray): Constant terms of the linear polynomials.
        b (np.ndarray): Coefficients of the linear polynomials.

    Returns:
        list[list[float]]: Intervals where the linear polynomials are all below zero.
    """
    l, u = -np.inf, np.inf
    if np.any(a[b == 0.0] > 0.0):
        return []
    l_candidates = -a[b < 0] / b[b < 0]
    u_candidates = -a[b > 0] / b[b > 0]
    if len(l_candidates) > 0:
        l = np.max(l_candidates)
    if len(u_candidates) > 0:
        u = np.min(u_candidates)
    return np.array([[l, u]]).tolist()
