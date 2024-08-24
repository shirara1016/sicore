"""Module containing classes for selective inference for the specific distributions."""

import numpy as np
from scipy import sparse  # type: ignore[import]
from scipy.stats import chi, norm  # type: ignore[import]

from sicore.core.base import SelectiveInference
from sicore.core.cdf import truncated_chi_cdf, truncated_norm_cdf
from sicore.core.real_subset import RealSubset


class ManyOptionsError(Exception):
    """Exception raised when multiple options are activated."""

    def __init__(self) -> None:
        """Initialize an ManyOptionsError object."""
        super().__init__(
            "Only one of use_sparse, use_tf, and use_torch can be True.",
        )


class NotTensorFlowTensorError(Exception):
    """Exception raised when the input is not a TensorFlow tensor."""

    def __init__(self) -> None:
        """Initialize an NotTensorFlowTensorError object."""
        super().__init__(
            "Input must be a TensorFlow tensor when use_tf is True.",
        )


class NotPyTorchTensorError(Exception):
    """Exception raised when the input is not a PyTorch tensor."""

    def __init__(self) -> None:
        """Initialize an NotPyTorchTensorError object."""
        super().__init__(
            "Input must be a PyTorch tensor when use_torch is True.",
        )


class SelectiveInferenceNorm(SelectiveInference):
    """A class conducting selective inference for the normal distribution.

    Args:
        data (np.ndarray): Observed data in 1D array.
        var (float | np.ndarray | sparse.csr_matrix): Known covariance matrix.
            If float, covariance matrix equals to the scalar times identity matrix.
            If 1D array, covariance matrix equals to the diagonal matrix
            with the given array.
            If 2D array, covariance matrix equals to the given array.
        eta (np.ndarray): The direction of the test statistic in 1D array.
        use_sparse (bool, optional): Whether to use sparse matrix.
            If True, the `var` must be given as a sparse matrix. Defaults to False.
        use_tf (bool, optional): Whether to use TensorFlow.
            If True, the `data`, `eta`, and `var` must be given as TensorFlow tensors.
            Defaults to False.
        use_torch (bool, optional): Whether to use PyTorch.
            If True, the `data`, `eta`, and `var` must be given as PyTorch tensors.
            Defaults to False.
    """

    def __init__(
        self,
        data: np.ndarray,
        var: float | np.ndarray | sparse.csr_array,
        eta: np.ndarray,
        *,
        use_sparse: bool = False,
        use_tf: bool = False,
        use_torch: bool = False,
    ) -> None:
        """Initialize a SelectiveInferenceNorm object.

        Args:
            data (np.ndarray): Observed data in 1D array.
            var (float | np.ndarray | sparse.csr_matrix): Known covariance matrix.
                If float, covariance matrix equals to the scalar times identity matrix.
                If 1D array, covariance matrix equals to the diagonal matrix
                with the given array.
                If 2D array, covariance matrix equals to the given array.
            eta (np.ndarray): The direction of the test statistic in 1D array.
            use_sparse (bool, optional): Whether to use sparse matrix.
                If True, the `var` must be given as a sparse matrix. Defaults to False.
            use_tf (bool, optional): Whether to use TensorFlow.
                If True, the `data`, `eta`, and `var` must be given as
                TensorFlow tensors. Defaults to False.
            use_torch (bool, optional): Whether to use PyTorch.
                If True, the `data`, `eta`, and `var` must be given as PyTorch tensors.
                Defaults to False.
        """
        if np.sum([use_sparse, use_tf, use_torch]) > 1:
            raise ManyOptionsError
        if use_tf:
            import tensorflow as tf  # type: ignore[import]

            if not isinstance(data, tf.Tensor) or not isinstance(eta, tf.Tensor):
                raise NotTensorFlowTensorError

            if isinstance(var, float):
                sigma_eta = var * eta
            elif len(var.shape) == 1:
                diag_cov = tf.constant(var, dtype=data.dtype)
                sigma_eta = diag_cov * eta
            else:
                cov = tf.constant(var, dtype=data.dtype)
                sigma_eta = tf.tensordot(cov, eta, axes=1)
            eta_sigma_eta = tf.tensordot(eta, sigma_eta, axes=1)
            sqrt_eta_sigma_eta = tf.sqrt(eta_sigma_eta)
            self.stat = tf.tensordot(eta, data, axes=1) / sqrt_eta_sigma_eta

        elif use_torch:
            import torch

            if not isinstance(data, torch.Tensor) or not isinstance(eta, torch.Tensor):
                raise NotPyTorchTensorError

            if isinstance(var, float):
                sigma_eta = var * eta
            elif len(var.shape) == 1:
                diag_cov = torch.tensor(var, dtype=data.dtype)
                sigma_eta = diag_cov * eta
            else:
                cov = torch.tensor(var, dtype=data.dtype)
                sigma_eta = torch.mv(cov, eta)
            eta_sigma_eta = torch.dot(eta, sigma_eta)
            sqrt_eta_sigma_eta = torch.sqrt(eta_sigma_eta)
            self.stat = torch.dot(eta, data) / sqrt_eta_sigma_eta

        else:
            data, eta = np.array(data), np.array(eta)
            if isinstance(var, float):
                sigma_eta = var * eta
            elif len(var.shape) == 1:
                diag_cov = np.array(var)
                sigma_eta = diag_cov * eta
            else:
                cov = sparse.csr_array(var) if use_sparse else np.array(var)
                sigma_eta = cov @ eta
            eta_sigma_eta = eta @ sigma_eta
            sqrt_eta_sigma_eta = np.sqrt(eta_sigma_eta)
            self.stat = eta @ data / sqrt_eta_sigma_eta

        self.stat = float(self.stat)

        self.b = sigma_eta / sqrt_eta_sigma_eta
        self.a = data - self.stat * self.b

        self.mode = 0.0
        self.support = RealSubset([[-np.inf, np.inf]])
        self.limits = (
            RealSubset([[-10.0 - np.abs(self.stat), 10.0 + np.abs(self.stat)]])
            & self.support
        )
        self.null_rv = norm()
        self.alternative = "two-sided"

        self.truncated_cdf = lambda z, intervals, absolute: truncated_norm_cdf(
            z,
            intervals,
            absolute=absolute,
        )


class SelectiveInferenceChi(SelectiveInference):
    """A class conducting selective inference for the chi distribution.

    Args:
        data (np.ndarray): Observed data in 1D array.
        var (float): Known covariance matrix, which equals to
            the scalar times identity matrix.
        projection (np.ndarray): The space of the test statistic in 2D array.
        use_sparse (bool, optional): Whether to use sparse matrix.
            If True, the `P` must be given as a sparse matrix. Defaults to False.
        use_tf (bool, optional): Whether to use TensorFlow.
            If True, the `data` and `P` must be given as TensorFlow tensors.
            Defaults to False.
        use_torch (bool, optional): Whether to use PyTorch.
            If True, the `data` and `P` must be given as PyTorch tensors.
            Defaults to False.
    """

    def __init__(
        self,
        data: np.ndarray,
        var: float,
        projection: np.ndarray,
        *,
        use_sparse: bool = False,
        use_tf: bool = False,
        use_torch: bool = False,
    ) -> None:
        """Initialize a SelectiveInferenceChi object.

        Args:
            data (np.ndarray): Observed data in 1D array.
            var (float): Known covariance matrix, which equals to
                the scalar times identity matrix.
            projection (np.ndarray): The space of the test statistic in 2D array.
            use_sparse (bool, optional): Whether to use sparse matrix.
                If True, the `P` must be given as a sparse matrix. Defaults to False.
            use_tf (bool, optional): Whether to use TensorFlow.
                If True, the `data` and `P` must be given as TensorFlow tensors.
                Defaults to False.
            use_torch (bool, optional): Whether to use PyTorch.
                If True, the `data` and `P` must be given as PyTorch tensors.
                Defaults to False.
        """
        if np.sum([use_sparse, use_tf, use_torch]) > 1:
            raise ManyOptionsError

        if use_tf:
            import tensorflow as tf  # type: ignore[import]

            if not isinstance(data, tf.Tensor) or not isinstance(projection, tf.Tensor):
                raise NotTensorFlowTensorError

            degree = int(tf.linalg.trace(projection) + 1e-3)
            projected_data = tf.tensordot(projection, data, axes=1)
            self.stat = tf.norm((var**-0.5) * projected_data, ord=2)

        elif use_torch:
            import torch

            if not isinstance(data, torch.Tensor) or not isinstance(
                projection,
                torch.Tensor,
            ):
                raise NotPyTorchTensorError

            # trace of P
            degree = int(torch.trace(projection) + 1e-3)
            projected_data = torch.mv(projection, data)
            self.stat = torch.linalg.norm((var**-0.5) * projected_data, ord=2)

        else:
            data = np.array(data)
            projection = (
                sparse.csr_array(projection) if use_sparse else np.array(projection)
            )
            degree = int(projection.trace() + 1e-3)
            projected_data = projection @ data
            self.stat = np.linalg.norm((var**-0.5) * projected_data, ord=2).item()

        self.stat = float(self.stat)

        self.b = projected_data / self.stat
        self.a = data - self.stat * self.b

        self.mode = np.sqrt(degree - 1)
        self.support = RealSubset([[0.0, np.inf]])
        self.limits = (
            RealSubset([[self.mode - 20.0, np.max([self.stat, self.mode]) + 10.0]])
            & self.support
        )
        self.null_rv = chi(df=degree)
        self.alternative = "less"

        self.truncated_cdf = lambda z, intervals, absolute: truncated_chi_cdf(
            z,
            degree,
            intervals,
            absolute=absolute,
        )
