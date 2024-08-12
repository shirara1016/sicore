import numpy as np
from scipy import sparse
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


def polynomial_below_zero(
    poly_or_coef: np.poly1d | np.ndarray | list[float], tol: float = 1e-10
) -> list[list[float]]:
    """Compute intervals where a given polynomial is below zero.

    Args:
        poly_or_coef (np.poly1d | np.ndarray | list[float]): Polynomial or its
            coefficients e.g. [a, b, c] for a*z^2 + b*z + c.
        tol (float, optional): Tolerance error parameter. It is recommended to set a
            large value (about 1e-5) for high order polynomial (>= 3) or a polynomial
            with multiple root. Defaults to 1e-10.

    Returns:
        list[list[float]]: Intervals where the polynomial is below zero.
    """
    if isinstance(poly_or_coef, np.poly1d):
        coef = poly_or_coef.coef
    else:
        coef = poly_or_coef

    coef = [0 if -tol < c < tol else c for c in coef]
    poly = np.poly1d(coef)

    if poly.order == 0:
        if poly.coef[0] <= 0:
            return [[-np.inf, np.inf]]
        else:
            return []

    roots = []
    if np.issubdtype(poly.coef.dtype, complex):
        for root in poly.roots:
            if -tol < root.imag < tol:
                roots.append(root.real)

        if len(roots) == 0:
            if poly(0) <= 0:
                return [[-np.inf, np.inf]]
            else:
                return []
    else:
        roots = poly.roots

    roots = np.unique(roots)
    intervals = []

    if poly(roots[0] - 1) <= 0:
        intervals.append([-np.inf, roots[0]])
    for start, end in zip(roots[:-1], roots[1:]):
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
    A: np.ndarray | sparse.csr_matrix | None = None,
    b: np.ndarray | None = None,
    c: np.ndarray | None = None,
    tol: float = 1e-10,
    use_sparse: bool = False,
) -> list[list[float]]:
    """Compute intervals where a given polytope (a_vec+b_vec*z)^T A (a_vec+b_vec*z) + b^T (a_vec+b_vec*z) + c is below zero.

    Args:
        a_vector (np.ndarray): Vector a_vec in the polytope.
        b_vector (np.ndarray): Vector b_vec in the polytope.
        A (np.ndarray | sparse.csr_matrix, optional): Matrix A in the polytope. Defaults to None.
        b (np.ndarray, optional): Vector b in the polytope. Defaults to None.
        c (np.ndarray, optional): Vector c in the polytope. Defaults to None.
        tol (float, optional): Tolerance error parameter. Defaults to 1e-10.
        use_sparse (bool, optional): Whether to use sparse matrix for computation of A matrix.
            Defaults to False.

    Returns:
        list[list[float]]: Intervals where the polytope is below zero.
    """
    alpha, beta, gamma = 0.0, 0.0, 0.0

    if A is not None:
        if use_sparse:
            A = sparse.csr_matrix(A)
        else:
            A = np.array(A)
        aA = a_vector @ A
        bA = b_vector @ A
        alpha += bA @ b_vector
        beta += aA @ b_vector + bA @ a_vector
        gamma += aA @ a_vector

    if b is not None:
        beta += b @ b_vector
        gamma += b @ a_vector

    if c is not None:
        gamma += c

    return polynomial_below_zero([alpha, beta, gamma], tol=tol)


def degree_one_polynomials_below_zero(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute intervals where given degree one polynomials a_i + b_i * z are all below zero.

    Args:
        a (np.ndarray): Constant terms of the degree one polynomials.
        b (np.ndarray): Coefficinets of the degree one polynomials.

    Returns:
        list[list[float]]: Intervals where the degree one polynomials are below zero.
    """
    l, u = -np.inf, np.inf
    l_candidates = -a[b < 0] / b[b < 0]
    u_candidates = -a[b > 0] / b[b > 0]
    if len(l_candidates) > 0:
        l = np.max(l_candidates)
    if len(u_candidates) > 0:
        u = np.min(u_candidates)
    return [[l, u]]