import numpy as np


def is_int_or_float(value):
    """Check if it is a scalar value.

    Check if a value (including a numpy object) has integer or floating data type. Note
    that infinity values have floating data type.
    """
    type_ = type(value)
    return np.issubdtype(type_, np.integer) or np.issubdtype(type_, np.floating)


class OneVec():
    """Vector whose elements from position `i` to `j` are set to 1, and 0 otherwise.

    Args:
        length (int): Dimension of the vector.
    """

    def __init__(self, length):
        self.length = length

    def get(self, i, j=None):
        """
        Get the vector.

        Args:
            i (int): Start index of 1 (1<=i<=`length`).
            j (int, optional): End index of 1 (1<=j<=`length`). If None, it returns a
                vector whose `i`th element is set to 1, and 0 otherwise. Defaults to
                None.

        Returns:
            np.ndarray
        """
        vec = np.zeros(self.length)

        if j is None:
            vec[i - 1] = 1
        else:
            vec[i - 1: j] = 1

        return vec


def construct_projection_matrix(basis, verbose=False):
    """Construct projection matrix from basis.

    Args:
        basis (np.ndarray, List[float]): List of basis of the space to be projected

    Raises:
        Exception: When the constructed matrix does not satisfy
            the definition of a projection matrix.

    Returns:
        np.ndarray: projection matrix
    """

    basis = np.array(basis)
    V = np.linalg.qr(basis.T)[0]
    P = np.dot(V, V.T)
    if verbose:
        if np.sum(np.abs(P.T - P)) > 1e-5:
            raise Exception(
                'The projection matrix is not constructed correctly')
        else:
            if np.sum(np.abs(np.dot(P, P) - P)) > 1e-5:
                raise Exception(
                    'The projection matrix is not constructed correctly')
    return P
