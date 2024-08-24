import pytest
import numpy as np
from numpy.testing import assert_allclose
from sicore.utils.constructor import OneVector, construct_projection_matrix


@pytest.mark.parametrize(
    "indexes, expected",
    [
        ((1,), [1, 0, 0, 0, 0]),
        ((4,), [0, 0, 0, 1, 0]),
        ((2, 2), [0, 1, 0, 0, 0]),
        ((2, 4), [0, 1, 1, 1, 0]),
    ],
)
def test_OneVec(indexes, expected):
    one = OneVector(5)
    assert_allclose(one.get(*indexes), expected)


@pytest.mark.parametrize(
    "basis, expected",
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
def test_construct_projection_matrix(basis, expected):
    assert_allclose(construct_projection_matrix(basis, verify=True), expected)
