import numpy as np
from scipy import sparse
from scipy.stats import chi

from .real_subset import RealSubset
from .cdf import tc_cdf
from .base import SelectiveInference


class SelectiveInferenceChi(SelectiveInference):
    """A class conducting selective inference for the normal distribution."""

    def __init__(
        self,
        data: np.ndarray,
        var: float,
        P: np.ndarray,
        use_sparse: bool = False,
        use_tf: bool = False,
        use_torch: bool = False,
    ):
        """Initialize a SelectiveInferenceNorm object.

        Args:
            data (np.ndarray): Observed data in 1D array.
            var (float): Known covariance matrix, which equals to the scalar times identity matrix.
            P (np.ndarray): The space of the test statistic in 2D array.
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
            raise ValueError(
                "Only one of use_sparse, use_tf, and use_torch can be True."
            )

        if use_tf:
            try:
                import tensorflow as tf
            except ModuleNotFoundError:
                raise ModuleNotFoundError("TensorFlow is not installed.")

            assert isinstance(data, tf.Tensor), "data must be a TensorFlow tensor."
            assert isinstance(P, tf.Tensor), "P must be a TensorFlow tensor."

            degree = int(tf.linalg.trace(P) + 1e-3)
            P_data = tf.tensordot(P, data, axes=1)
            self.stat = tf.norm((var**-0.5) * P_data, ord=2)

        elif use_torch:
            try:
                import torch
            except ModuleNotFoundError:
                raise Exception("Pytorch is not installed")

            assert isinstance(data, torch.Tensor), "data must be a PyTorch tensor."
            assert isinstance(P, torch.Tensor), "P must be a PyTorch tensor."

            # trace of P
            degree = int(torch.trace(P) + 1e-3)
            P_data = torch.mv(P, data)
            self.stat = torch.linalg.norm((var**-0.5) * P_data, ord=2)

        else:
            data = np.array(data)
            P = sparse.csr_array(P) if use_sparse else np.array(P)
            degree = int(P.trace() + 1e-3)
            P_data = P @ data
            self.stat = np.linalg.norm((var**-0.5) * P_data, ord=2)

        self.stat = float(self.stat)

        self.b = P_data / self.stat
        self.a = data - self.stat * self.b

        self.mode = np.sqrt(degree - 1)
        self.support = RealSubset([[0.0, np.inf]])
        self.limits = (
            RealSubset([[self.mode - 20.0, np.max([self.stat, self.mode]) + 10.0]])
            & self.support
        )
        self.null_rv = chi(df=degree)

        self.truncated_cdf = lambda z, intervals, absolute: tc_cdf(
            z, degree, intervals, absolute=absolute
        )
