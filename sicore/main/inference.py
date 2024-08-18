import numpy as np
from scipy import sparse  # type: ignore
from scipy.stats import norm, chi  # type: ignore

from ..core.real_subset import RealSubset
from ..core.base import SelectiveInference
from ..core.cdf import truncated_norm_cdf, truncated_chi_cdf


class SelectiveInferenceNorm(SelectiveInference):
    """A class conducting selective inference for the normal distribution.

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
                import tensorflow as tf  # type: ignore
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

        self.truncated_cdf = lambda z, intervals, absolute: truncated_norm_cdf(
            z, intervals, absolute=absolute
        )


class SelectiveInferenceChi(SelectiveInference):
    """A class conducting selective inference for the chi distribution.

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

    def __init__(
        self,
        data: np.ndarray,
        var: float,
        P: np.ndarray,
        use_sparse: bool = False,
        use_tf: bool = False,
        use_torch: bool = False,
    ):
        """Initialize a SelectiveInferenceChi object.

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
                import tensorflow as tf  # type: ignore
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

        self.truncated_cdf = lambda z, intervals, absolute: truncated_chi_cdf(
            z, degree, intervals, absolute=absolute
        )
