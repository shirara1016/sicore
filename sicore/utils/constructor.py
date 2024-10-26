"""Module providing functions for constructing various objects."""

import numpy as np


class OneVector:
    """Vector whose elements from position `i` to `j` are set to 1, and 0 otherwise.

    Attributes
    ----------
    int
        Dimension of the vector.
    """

    def __init__(self, length: int) -> None:
        """Initialize the vector constructor.

        Parameters
        ----------
        length : int
            Dimension of the vector.
        """
        self.length = length

    def get(self, i: int, j: int | None = None) -> np.ndarray:
        """Get the vector.

        Parameters
        ----------
        i : int
            Start index of 1 (1<=i<=`length`).
        j : int, optional
            End index of 1 (1<=j<=`length`).
            If None, it returns a vector whose `i`-th element is set to 1, and 0 otherwise.

        Returns
        -------
        np.ndarray
            One-zero vector
        """
        vector = np.zeros(self.length)

        if j is None:
            vector[i - 1] = 1
        else:
            vector[i - 1 : j] = 1

        return vector


def construct_projection_matrix(
    basis: np.ndarray | list[list[float]],
    *,
    verify: bool = False,
) -> np.ndarray:
    """Construct projection matrix from basis.

    Parameters
    ----------
    basis : np.ndarray | list[list[float]]
        The basis of the k-dimensional subspace to be projected.
        The shape of the basis should be (k, n), where n is the dimension of the data space.
    verify : bool, optional
        Whether to verify the constructed projection matrix. Defaults to False.

    Raises
    ------
    ValueError:
        The constructed projection matrix is not consistent with the definition.

    Returns
    -------
    np.ndarray
        The constructed projection matrix
    """
    basis = np.array(basis)
    u, _, _ = np.linalg.svd(basis.T, full_matrices=False)
    p = u @ u.T
    tol = 1e-5
    if verify:
        if np.sum(np.abs(p.T - p)) > tol:
            raise ValueError
        if np.sum(np.abs(p @ p - p)) > tol:
            raise ValueError
    return p
