"""Module with tests for the main inference module."""

from typing import Literal

import numpy as np
import pytest
from numpy.testing import assert_allclose

from sicore.core.base import SelectiveInferenceResult
from sicore.core.real_subset import NotBelongToSubsetError, RealSubset
from sicore.main.inference import SelectiveInferenceChi, SelectiveInferenceNorm
from sicore.utils.constructor import construct_projection_matrix
from sicore.utils.intervals import degree_one_polynomials_below_zero


class MarginalScreening:
    """A class for marginal screening."""

    def __init__(self, X: np.ndarray, y: np.ndarray, sigma: float, k: int) -> None:
        """Initialize a MarginalScreening object.

        Args:
            X (np.ndarray): Feature matrix, whose shape is (n_samples, n_features).
            y (np.ndarray): Response vector, whose shape is (n_samples,).
            sigma (float): Standard deviation of the response vector.
            k (int): Number of features to select.
        """
        self.X, self.y, self.sigma, self.k = X, y, sigma, k
        self.M = self._feature_selection(X, y, k)

    def _feature_selection(self, X: np.ndarray, y: np.ndarray, k: int) -> list[int]:
        """Select the top k features based on the correlation with the response vector.

        Args:
            X (np.ndarray): Feature matrix, whose shape is (n_samples, n_features).
            y (np.ndarray): Response vector, whose shape is (n_samples,).
            k (int): Number of features to select.

        Returns:
            list[int]: Indices of the selected features.
        """
        return np.argsort(np.abs(X.T @ y))[::-1][:k].tolist()

    def algorithm(
        self,
        a: np.ndarray,
        b: np.ndarray,
        z: float,
    ) -> tuple[list[int], RealSubset]:
        """A function to conduct selective inference.

        It takes a, b, and z as input and apply marginal screening to the dataset
        (self.X, a + b * z) to select the top self.k features M. It returns
        the selected features and the intervals where the same features are selected,
        i.e., from (self.X, a + b * r) we can select the features M for any r in the intervals.

        Args:
            a (np.ndarray): _description_
            b (np.ndarray): _description_
            z (float): _description_

        Returns:
            tuple[list[int], RealSubset]: _description_
        """
        a, b = self.X.T @ a, self.X.T @ b

        signs = np.sign(a + b * z)
        intervals_ = degree_one_polynomials_below_zero(-signs * a, -signs * b)
        intervals = RealSubset(intervals_)
        a, b = signs * a, signs * b

        collerations = a + b * z
        indexes = np.argsort(collerations)[::-1]

        active_set = indexes[: self.k]
        inactive_set = indexes[self.k :]

        for active in active_set:
            temp_intervals = degree_one_polynomials_below_zero(
                a[inactive_set] - a[active],
                b[inactive_set] - b[active],
            )
            intervals = intervals & RealSubset(temp_intervals)

        if z not in intervals:
            raise NotBelongToSubsetError(z, intervals)
        return indexes[: self.k].tolist(), intervals

    def model_selector(self, M: list[int]) -> bool:
        """A function to conduct selective inference.

        It takes a list of indices M as input and returns True if the selected features
        are the same as the features selected from the obsearved dataset (self.X, self.y).

        Args:
            M (list[int]): A list of indices of the selected features.

        Returns:
            bool: True if the selected features are the same as the features selected
                from the obsearved dataset (self.X, self.y).
        """
        return set(self.M) == set(M)


class MarginalScreeningNorm(MarginalScreening):
    """A class for marginal screening for the normal distribution."""

    def construct_eta(self, index: int) -> np.ndarray:
        """Construct the eta vector for the selective inference.

        Args:
            index (int): Target index for the selective inference.

        Returns:
            np.ndarray: Constructed eta vector.
        """
        return (
            self.X[:, self.M]
            @ np.linalg.inv(self.X[:, self.M].T @ self.X[:, self.M])[:, index]
        )

    def inference(
        self,
        index: int,
        search_strategy: Literal["pi1", "pi2", "pi3"],
        termination_criterion: Literal["precision", "decision"],
    ) -> SelectiveInferenceResult:
        """Conduct selective inference for the normal distribution.

        Args:
            index (int): Target index for the selective inference.
            search_strategy (Literal["pi1", "pi2", "pi3"]):
                Search strategy for the test.
            termination_criterion (Literal["precision", "decision"]):
                Termination criterion for the test.

        Returns:
            SelectiveInferenceResult: The result of the selective inference.
        """
        eta = self.construct_eta(index)
        si = SelectiveInferenceNorm(self.y, self.sigma, eta)
        return si.inference(
            self.algorithm,
            self.model_selector,
            search_strategy=search_strategy,
            termination_criterion=termination_criterion,
        )


class MarginalScreeningChi(MarginalScreening):
    """A class for marginal screening for the chi distribution."""

    def construct_projection(self, indexes: list[int]) -> np.ndarray:
        """Construct the P matrix for the selective inference.

        Args:
            indexes (int): Target indexes for the selective inference.

        Returns:
            np.ndarray: Constructed P vector.
        """
        return construct_projection_matrix(self.X[:, np.array(self.M)[indexes]].T)

    def inference(
        self,
        indexes: list[int],
        search_strategy: Literal["pi1", "pi2", "pi3"],
        termination_criterion: Literal["precision", "decision"],
    ) -> SelectiveInferenceResult:
        """Conduct selective inference for the chi distribution.

        Args:
            indexes (list[int]): Target indexes for the selective inference.
            search_strategy (Literal["pi1", "pi2", "pi3"]):
                Search strategy for the test.
            termination_criterion (Literal["precision", "decision"]):
                Termination criterion for the test.

        Returns:
            SelectiveInferenceResult: The result of the selective inference.
        """
        projection = self.construct_projection(indexes)
        si = SelectiveInferenceChi(self.y, self.sigma, projection)
        return si.inference(
            self.algorithm,
            self.model_selector,
            search_strategy=search_strategy,
            termination_criterion=termination_criterion,
        )


@pytest.mark.parametrize(
    ("seed", "expected_stat", "expected_p_value"),
    [
        (0, 1.997181, 0.067986),
        (1, -0.715869, 0.802903),
        (2, -0.959323, 0.551424),
        (3, 0.966223, 0.598692),
    ],
)
def test_marginal_screening_norm(
    seed: int,
    expected_stat: float,
    expected_p_value: float,
) -> None:
    """Test the SelectiveInferenceNorm class."""
    precision, significance_level = 0.001, 0.05
    rng = np.random.default_rng(seed)
    n, p, k, sigma = 100, 10, 5, 1.0

    X = rng.normal(size=(n, p))
    y = rng.normal(size=n)
    index = rng.choice(k)

    ms = MarginalScreeningNorm(X, y, sigma, k)

    strategies: list[Literal["pi1", "pi2", "pi3"]] = ["pi1", "pi2", "pi3"]
    for search_strategy in strategies:
        result = ms.inference(
            index,
            termination_criterion="precision",
            search_strategy=search_strategy,
        )
        assert np.abs(result.p_value - expected_p_value) < precision
        assert_allclose(result.stat, expected_stat, rtol=1e-4, atol=1e-4)

        result = ms.inference(
            index,
            termination_criterion="decision",
            search_strategy=search_strategy,
        )
        assert (result.p_value <= significance_level) == (
            expected_p_value <= significance_level
        )
        assert_allclose(result.stat, expected_stat, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    ("seed", "expected_stat", "expected_p_value"),
    [
        (0, 1.503175, 0.568306),
        (1, 3.020218, 0.130227),
        (2, 2.121942, 0.292879),
        (3, 2.627566, 0.385533),
    ],
)
def test_marginal_screening_chi(
    seed: int,
    expected_stat: float,
    expected_p_value: float,
) -> None:
    """Test the SelectiveInferenceChi class."""
    precision, significance_level = 0.001, 0.05
    rng = np.random.default_rng(seed)
    n, p, k, sigma = 100, 10, 5, 1.0

    X = rng.normal(size=(n, p))
    y = rng.normal(size=n)
    indexes = rng.choice(k, rng.choice(k - 1) + 2, replace=False).tolist()

    ms = MarginalScreeningChi(X, y, sigma, k)

    strategies: list[Literal["pi1", "pi2", "pi3"]] = ["pi1", "pi2", "pi3"]
    for search_strategy in strategies:
        result = ms.inference(
            indexes,
            termination_criterion="precision",
            search_strategy=search_strategy,
        )
        assert np.abs(result.p_value - expected_p_value) < precision
        assert_allclose(result.stat, expected_stat, rtol=1e-4, atol=1e-4)

        result = ms.inference(
            indexes,
            termination_criterion="decision",
            search_strategy=search_strategy,
        )
        assert (result.p_value <= significance_level) == (
            expected_p_value <= significance_level
        )
        assert_allclose(result.stat, expected_stat, rtol=1e-4, atol=1e-4)
