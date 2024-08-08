import numpy as np
from scipy import sparse
from scipy.stats import norm

from .real_subset import RealSubset
from .cdf import tn_cdf
from .base import Inference


class SelectiveInferenceNorm(Inference):
    """A class conducting selective inference for the normal distribution."""

    def __init__(
        self,
        data: np.ndarray,
        var: float | np.ndarray | sparse.csr_array,
        eta: np.ndarray,
        use_sparse: bool = False,
        use_tf: bool = False,
        use_torch: bool = False,
    ):
        """Initialize a SelectiveInferenceNorm object.

        Args:
            data (np.ndarray): Observed data in 1D array.
            var (float | np.ndarray | sparse.csr_matrix): Known covariance matrix.
                If float, covariance matrix equals to the scalar times identity matrix.
                If 1D array, covariance matrix equals to the diagonal matrix with the given array.
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
        if np.sum([use_sparse, use_tf, use_torch]) > 1:
            raise ValueError(
                "Only one of use_sparse, use_tf, and use_torch can be True."
            )

        if use_tf:
            try:
                import tensorflow as tf
            except ModuleNotFoundError:
                raise ModuleNotFoundError("TensorFlow is not installed.")

            assert isinstance(data, tf.Tensor), "data must be a TensorFlow tensor."
            assert isinstance(eta, tf.Tensor), "eta must be a TensorFlow tensor."

            if isinstance(var, float):
                sigma_eta = var * eta
            elif len(var.shape) == 1:
                vars = tf.constant(var, dtype=data.dtype)
                sigma_eta = vars * eta
            elif len(var.shape) == 2:
                cov = tf.constant(var, dtype=data.dtype)
                sigma_eta = tf.tensordot(cov, eta, axes=1)
            eta_sigma_eta = tf.tensordot(eta, sigma_eta, axes=1)
            sqrt_eta_sigma_eta = tf.sqrt(eta_sigma_eta)
            self.stat = tf.tensordot(eta, data, axes=1) / sqrt_eta_sigma_eta

        elif use_torch:
            try:
                import torch
            except ModuleNotFoundError:
                raise Exception("Pytorch is not installed")

            assert isinstance(data, torch.Tensor), "data must be a PyTorch tensor."
            assert isinstance(eta, torch.Tensor), "eta must be a PyTorch tensor."

            if isinstance(var, float):
                sigma_eta = var * eta
            elif len(var.shape) == 1:
                vars = torch.tensor(var, dtype=data.dtype)
                sigma_eta = vars * eta
            elif len(var.shape) == 2:
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
                vars = np.array(var)
                sigma_eta = vars * eta
            elif len(var.shape) == 2:
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

        self.truncated_cdf = lambda z, intervals, absolute: tn_cdf(
            z, intervals, absolute=absolute
        )
