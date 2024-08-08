import numpy as np
from scipy import sparse
from scipy.stats import norm

from typing import Callable

from .real_subset import RealSubset
from .cdf import tn_cdf
from .base import (
    SelectiveInferenceResult,
    InfiniteLoopError,
    Inference,
    compute_pvalue,
    compute_pvalue_bounds,
)


class SelectiveInferenceNorm(Inference):
    """A class conducting selective inference for the normal distribution.

    Args:
        Inference (_type_): _description_
    """

    def __init__(
        self,
        data: np.ndarray,
        var: float | np.ndarray | sparse.csr_array,
        eta: np.ndarray,
        use_sparse: bool = False,
        use_tf: bool = False,
        use_torch: bool = False,
    ):
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
                self.sigma_eta = var * eta
            elif len(var.shape) == 1:
                vars = tf.constant(var, dtype=data.dtype)
                self.sigma_eta = vars * eta
            elif len(var.shape) == 2:
                cov = tf.constant(var, dtype=data.dtype)
                self.sigma_eta = tf.tensordot(cov, eta, axes=1)
            self.eta_sigma_eta = tf.tensordot(eta, self.sigma_eta, axes=1)
            self.sqrt_eta_sigma_eta = tf.sqrt(self.eta_sigma_eta)
            self.stat = tf.tensordot(eta, data, axes=1) / self.sqrt_eta_sigma_eta

        elif use_torch:
            try:
                import torch
            except ModuleNotFoundError:
                raise Exception("Pytorch is not installed")

            assert isinstance(data, torch.Tensor), "data must be a PyTorch tensor."
            assert isinstance(eta, torch.Tensor), "eta must be a PyTorch tensor."

            if isinstance(var, float):
                self.sigma_eta = var * eta
            elif len(var.shape) == 1:
                vars = torch.tensor(var, dtype=data.dtype)
                self.sigma_eta = vars * eta
            elif len(var.shape) == 2:
                cov = torch.tensor(var, dtype=data.dtype)
                self.sigma_eta = torch.mv(cov, eta)
            self.eta_sigma_eta = torch.dot(eta, self.sigma_eta)
            self.sqrt_eta_sigma_eta = torch.sqrt(self.eta_sigma_eta)
            self.stat = torch.dot(eta, data) / self.sqrt_eta_sigma_eta

        else:
            data, eta = np.array(data), np.array(eta)
            if isinstance(var, float):
                self.sigma_eta = var * eta
            elif len(var.shape) == 1:
                vars = np.array(var)
                self.sigma_eta = vars * eta
            elif len(var.shape) == 2:
                cov = sparse.csr_array(var) if use_sparse else np.array(var)
                self.sigma_eta = cov @ eta
            self.eta_sigma_eta = eta @ self.sigma_eta
            self.sqrt_eta_sigma_eta = np.sqrt(self.eta_sigma_eta)
            self.stat = eta @ data / self.sqrt_eta_sigma_eta

        self.stat = float(self.stat)

        self.b = self.sigma_eta / self.sqrt_eta_sigma_eta
        self.a = data - self.stat * self.b

        self.support = RealSubset([[-np.inf, np.inf]])
        self.limits = (
            RealSubset([[-10.0 - np.abs(self.stat), 10.0 + np.abs(self.stat)]])
            & self.support
        )
        self.null_rv = norm()
        self.mode = 0.0

        self.truncated_cdf = lambda z, intervals, absolute: tn_cdf(
            z, intervals, absolute=absolute
        )
