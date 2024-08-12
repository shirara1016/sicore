import numpy as np


class OneVec:
    """Vector whose elements from position `i` to `j` are set to 1, and 0 otherwise.

    Args:
        length (int): Dimension of the vector.
    """

    def __init__(self, length: int) -> None:
        """Initialize the vector constructor.

        Args:
            length (int): Dimension of the vector.
        """
        self.length = length

    def get(self, i, j=None):
        """
        Get the vector.

        Args:
            i (int): Start index of 1 (1<=i<=`length`).
            j (int, optional): End index of 1 (1<=j<=`length`). If None, it returns a
                vector whose `i`-th element is set to 1, and 0 otherwise. Defaults to None.

        Returns:
            np.ndarray: One-zero vector
        """
        vec = np.zeros(self.length)

        if j is None:
            vec[i - 1] = 1
        else:
            vec[i - 1 : j] = 1

        return vec


def construct_projection_matrix(basis, verify=False):
    """Construct projection matrix from basis.

    Args:
        basis (np.ndarray): The basis of the k-dimensional subspace to be projected.
            The shape of the basis should be (k, n), where n is the dimension of the data space.
        verify (bool, optional): Whether to verify the constructed projection matrix. Defaults to False.

    Raises:
        Exception: The constructed projection matrix is not consistent with the definition.

    Returns:
        np.ndarray: The constructed projection matrix
    """
    basis = np.array(basis)
    U, _, _ = np.linalg.svd(basis.T, full_matrices=False)
    P = U @ U.T
    if verify:
        if np.sum(np.abs(P.T - P)) > 1e-5:
            raise Exception("The projection matrix is not constructed correctly")
        else:
            if np.sum(np.abs(P @ P - P)) > 1e-5:
                raise Exception("The projection matrix is not constructed correctly")
    return P
