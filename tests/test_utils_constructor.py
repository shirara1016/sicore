"""Module with tests for the constructor utilities."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from sicore.utils.constructor import OneVector, construct_projection_matrix


@pytest.mark.parametrize(
    ("indexes", "expected"),
    [
        ((1,), [1, 0, 0, 0, 0]),
        ((4,), [0, 0, 0, 1, 0]),
        ((2, 2), [0, 1, 0, 0, 0]),
        ((2, 4), [0, 1, 1, 1, 0]),
    ],
)
def test_one_vector(indexes: tuple[int], expected: list[float]) -> None:
    """Test the OneVector class."""
    one = OneVector(5)
    assert_allclose(one.get(*indexes), expected)


@pytest.mark.parametrize(
    ("basis", "expected"),
    [
        (
            [[0.0, 1.0]],
            np.array([[0.0, 0.0], [0.0, 1.0]]),
        ),
        (
            [[1.0, 0.0, 0.0]],
            np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        ),
        (
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        ),
        (
            [[1.0, 0.0, 1.0]],
            np.array([[0.5, 0.0, 0.5], [0.0, 0.0, 0.0], [0.5, 0.0, 0.5]]),
        ),
    ],
)
def test_construct_projection_matrix(
    basis: list[list[float]],
    expected: np.ndarray,
) -> None:
    """Test the construct projection matrix function."""
    assert_allclose(construct_projection_matrix(basis, verify=True), expected)
