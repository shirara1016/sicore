"""Module containing classes for selective inference for the specific distributions."""

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
from scipy import sparse  # type: ignore[import]
from scipy.integrate import quad  # type: ignore[import]
from scipy.optimize import (  # type: ignore[import]
    OptimizeResult,
    RootResults,
    minimize_scalar,
    root_scalar,
)
from scipy.stats import chi, norm  # type: ignore[import]

from sicore.core.base import (
    RandomizedInferenceResult,
    SelectiveInference,
    SelectiveInferenceResult,
)
from sicore.core.dist import compute_log_area, truncated_cdf, truncated_pdf
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

    Parameters
    ----------
    data : np.ndarray
        Observed data in 1D array.
    var : float | np.ndarray | sparse.csr_matrix
        Known covariance matrix.
        If float, covariance matrix equals to the scalar times identity matrix.
        If 1D array, covariance matrix equals to the diagonal matrix with the given array.
        If 2D array, covariance matrix equals to the given array.
    eta : np.ndarray
        The direction of the test statistic in 1D array.
    alternative : Literal["two-sided", "less", "greater"], optional
        Type of the alternative hypothesis. Defaults to "two-sided".
    null_value : float, optional
        The null value of the hypothesis test. Defaults to 0.0.
    use_sparse : bool, optional
        Whether to use sparse matrix.
        If True, the `var` must be given as a sparse matrix. Defaults to False.
    use_tf : bool, optional
        Whether to use TensorFlow.
        If True, the `data`, `eta`, and `var` must be given as TensorFlow tensors. Defaults to False.
    use_torch : bool, optional
        Whether to use PyTorch.
        If True, the `data`, `eta`, and `var` must be given as PyTorch tensors. Defaults to False.
    """

    def __init__(
        self,
        data: np.ndarray,
        var: float | np.ndarray | sparse.csr_array,
        eta: np.ndarray,
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
        null_value: float = 0.0,
        *,
        use_sparse: bool = False,
        use_tf: bool = False,
        use_torch: bool = False,
    ) -> None:
        """Initialize a SelectiveInferenceNorm object.

        Parameters
        ----------
        data : np.ndarray
            Observed data in 1D array.
        var : float | np.ndarray | sparse.csr_matrix
            Known covariance matrix.
            If float, covariance matrix equals to the scalar times identity matrix.
            If 1D array, covariance matrix equals to the diagonal matrix with the given array.
            If 2D array, covariance matrix equals to the given array.
        eta : np.ndarray
            The direction of the test statistic in 1D array.
        alternative : Literal["two-sided", "less", "greater"], optional
            Type of the alternative hypothesis. Defaults to "two-sided".
        null_value : float, optional
            The null value of the hypothesis test. Defaults to 0.0.
        use_sparse : bool, optional
            Whether to use sparse matrix.
            If True, the `var` must be given as a sparse matrix. Defaults to False.
        use_tf : bool, optional
            Whether to use TensorFlow.
            If True, the `data`, `eta`, and `var` must be given as TensorFlow tensors. Defaults to False.
        use_torch : bool, optional
            Whether to use PyTorch.
            If True, the `data`, `eta`, and `var` must be given as PyTorch tensors. Defaults to False.
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
        self.stat_scale = float(sqrt_eta_sigma_eta)

        self.b = sigma_eta / sqrt_eta_sigma_eta
        self.a = data - self.stat * self.b

        self.null_rv = norm(loc=null_value)
        self.mode = null_value
        self.support = RealSubset([[-np.inf, np.inf]])
        self.alternative = alternative

        self.limits = (
            RealSubset([[-10.0 - np.abs(self.stat), 10.0 + np.abs(self.stat)]])
            & self.support
        )

    def _inverse_cdf(self, p: float, result: SelectiveInferenceResult) -> float:
        scale = result.null_rv.kwds.get("scale", 1.0)
        a, b = -1.0, 1.0
        while True:
            val_a = (
                truncated_cdf(
                    norm(loc=a, scale=scale),
                    result.stat,
                    result.truncated_intervals,
                )
                - p
            )
            val_b = (
                truncated_cdf(
                    norm(loc=b, scale=scale),
                    result.stat,
                    result.truncated_intervals,
                )
                - p
            )
            if val_a * val_b < 0:
                break
            a *= 2
            b *= 2

        value: RootResults = root_scalar(
            lambda mu: truncated_cdf(
                norm(loc=mu, scale=scale),
                result.stat,
                result.truncated_intervals,
            )
            - p,
            bracket=(a, b),
        )
        return float(value.root)

    def interval_estimate(
        self,
        result: SelectiveInferenceResult,
        confidence_level: float,
    ) -> tuple[float, float]:
        """Compute the selective confidence interval.

        Note that it has been pointed out that the interval length can theoretically become infinite.

        Parameters
        ----------
        result : SelectiveInferenceResult
            The result of inference method. It should be computed with 'inference_mode="exhaustive"'.
        confidence_level : float
            Confidence level. The selective confidence interval has the coverage of 'confidence_level'.

        Returns
        -------
        tuple[float, float]
            Lower and upper bounds of the selective confidence interval.
        """
        return (
            self._inverse_cdf((1.0 + confidence_level) / 2.0, result),
            self._inverse_cdf((1.0 - confidence_level) / 2.0, result),
        )

    def point_estimate(self, result: SelectiveInferenceResult) -> float:
        """Compute the selective maximum likelihood estimate.

        Parameters
        ----------
        result : SelectiveInferenceResult
            The result of inference method. It should be computed with 'inference_mode="exhaustive"'.

        Returns
        -------
        float
            The selective maximum likelihood estimate.
        """
        scale = result.null_rv.kwds.get("scale", 1.0)
        point_estimate: OptimizeResult = minimize_scalar(
            lambda mu: -truncated_pdf(
                norm(loc=mu, scale=scale),
                result.stat,
                result.truncated_intervals,
            ),
        )
        return float(point_estimate.x)


class SelectiveInferenceChi(SelectiveInference):
    """A class conducting selective inference for the chi distribution.

    Parameters
    ----------
    data : np.ndarray
        Observed data in 1D array.
    var : float
        Known covariance matrix, which equals to the scalar times identity matrix.
    projection : np.ndarray
        The space of the test statistic in 2D array.
    use_sparse : bool, optional
        Whether to use sparse matrix.
        If True, the `P` must be given as a sparse matrix. Defaults to False.
    use_tf : bool, optional
        Whether to use TensorFlow.
        If True, the `data` and `P` must be given as TensorFlow tensors. Defaults to False.
    use_torch : bool, optional
        Whether to use PyTorch.
        If True, the `data` and `P` must be given as PyTorch tensors. Defaults to False.
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

        Parameters
        ----------
        data : np.ndarray
            Observed data in 1D array.
        var : float
            Known covariance matrix, which equals to the scalar times identity matrix.
        projection : np.ndarray
            The space of the test statistic in 2D array.
        use_sparse : bool, optional
            Whether to use sparse matrix.
            If True, the `P` must be given as a sparse matrix. Defaults to False.
        use_tf : bool, optional
            Whether to use TensorFlow.
            If True, the `data` and `P` must be given as TensorFlow tensors. Defaults to False.
        use_torch : bool, optional
            Whether to use PyTorch.
            If True, the `data` and `P` must be given as PyTorch tensors. Defaults to False.
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
        self.stat_scale = float(var**0.5)

        self.b = projected_data / self.stat
        self.a = data - self.stat * self.b

        self.null_rv = chi(df=degree)
        self.mode = np.sqrt(degree - 1)
        self.support = RealSubset([[0.0, np.inf]])
        self.alternative = "less"

        left, right = np.sort([self.stat, self.mode])
        self.limits = RealSubset([[left - 10.0, right + 10.0]]) & self.support


class RandomizedSelectiveInference:
    """A class conducting randomized selective inference for the normal distribution.

    Parameters
    ----------
    data : np.ndarray
        Observed data in 1D array.
    var : float | np.ndarray | sparse.csr_matrix
        Known covariance matrix.
        If float, covariance matrix equals to the scalar times identity matrix.
        If 1D array, covariance matrix equals to the diagonal matrix with the given array.
        If 2D array, covariance matrix equals to the given array.
    randomizer: np.ndarray
        Added randomization in 1D array.
    randomized_var: float
        Known variance of the added randomization.
    eta : np.ndarray
        The direction of the test statistic in 1D array.
    alternative : Literal["two-sided", "less", "greater"], optional
        Type of the alternative hypothesis. Defaults to "two-sided".
    null_value : float, optional
        The null value of the hypothesis test. Defaults to 0.0.
    use_sparse : bool, optional
        Whether to use sparse matrix.
        If True, the `var` must be given as a sparse matrix. Defaults to False.
    use_tf : bool, optional
        Whether to use TensorFlow.
        If True, the `data`, `eta`, and `var` must be given as TensorFlow tensors. Defaults to False.
    use_torch : bool, optional
        Whether to use PyTorch.
        If True, the `data`, `eta`, and `var` must be given as PyTorch tensors. Defaults to False.
    """

    def __init__(
        self,
        data: np.ndarray,
        var: float | np.ndarray | sparse.csr_array,
        randomizer: np.ndarray,
        randomized_var: float,
        eta: np.ndarray,
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
        null_value: float = 0.0,
        *,
        use_sparse: bool = False,
        use_tf: bool = False,
        use_torch: bool = False,
    ) -> None:
        """Initialize a SelectiveInferenceNorm object.

        Parameters
        ----------
        data : np.ndarray
            Observed data in 1D array.
        var : float | np.ndarray | sparse.csr_matrix
            Known covariance matrix.
            If float, covariance matrix equals to the scalar times identity matrix.
            If 1D array, covariance matrix equals to the diagonal matrix with the given array.
            If 2D array, covariance matrix equals to the given array.
        randomizer : np.ndarray
            Added randomization in 1D array.
        randomized_var : float
            Known variance of the added randomization.
        eta : np.ndarray
            The direction of the test statistic in 1D array.
        alternative : Literal["two-sided", "less", "greater"], optional
            Type of the alternative hypothesis. Defaults to "two-sided".
        null_value : float, optional
            The null value of the hypothesis test. Defaults to 0.0.
        use_sparse : bool, optional
            Whether to use sparse matrix.
            If True, the `var` must be given as a sparse matrix. Defaults to False.
        use_tf : bool, optional
            Whether to use TensorFlow.
            If True, the `data`, `eta`, and `var` must be given as TensorFlow tensors. Defaults to False.
        use_torch : bool, optional
            Whether to use PyTorch.
            If True, the `data`, `eta`, and `var` must be given as PyTorch tensors. Defaults to False.
        """
        if np.sum([use_sparse, use_tf, use_torch]) > 1:
            raise ManyOptionsError
        if use_tf:
            import tensorflow as tf  # type: ignore[import]

            if not isinstance(data, tf.Tensor) or not isinstance(eta, tf.Tensor):
                raise NotTensorFlowTensorError

            eta_norm = tf.norm(eta, ord=2)
            if isinstance(var, float):
                sigma = eta_norm * (var**0.5)
                var = var + randomized_var
            elif len(var.shape) == 1:
                var = tf.constant(var, dtype=data.dtype)
                sigma = tf.reduce_sum(eta**2 * var) ** 0.5
                var = var + randomized_var
            else:
                var = tf.constant(var, dtype=data.dtype)
                sigma = tf.sqrt(
                    tf.tensordot(eta, tf.tensordot(var, eta, axes=1), axes=1),
                )
                var[np.diag_indices_from(var)] += randomized_var
            stat = tf.tensordot(eta, data, axes=1)

        elif use_torch:
            import torch

            if not isinstance(data, torch.Tensor) or not isinstance(eta, torch.Tensor):
                raise NotPyTorchTensorError

            eta_norm = torch.norm(eta, p=2)
            if isinstance(var, float):
                sigma = eta_norm * (var**0.5)
                var = var + randomized_var
            elif len(var.shape) == 1:
                var = torch.tensor(var, dtype=data.dtype)
                sigma = torch.sqrt(torch.dot(eta**2, var))
                var = var + randomized_var
            else:
                var = torch.tensor(var, dtype=data.dtype)
                sigma = torch.sqrt(torch.dot(eta, torch.mv(var, eta)))
                var[np.diag_indices_from(var)] += randomized_var
            stat = torch.dot(eta, data)

        else:
            eta_norm = np.linalg.norm(eta, ord=2)
            data, eta = np.array(data), np.array(eta)
            if isinstance(var, float):
                sigma = eta_norm * (var**0.5)
                var = var + randomized_var
            elif len(var.shape) == 1:
                var = np.array(var)
                sigma = np.sqrt(np.sum(eta**2 * var))
                var = var + randomized_var
            else:
                var = sparse.csr_array(var) if use_sparse else np.array(var)
                sigma = np.sqrt(eta @ (var @ eta))
                var[np.diag_indices_from(var)] += randomized_var
            stat = eta @ data

        self.si = SelectiveInferenceNorm(
            data=data + randomizer,
            var=var,
            eta=eta,
            alternative=alternative,
            null_value=null_value,
            use_sparse=use_sparse,
            use_tf=use_tf,
            use_torch=use_torch,
        )
        self.sigma = float(sigma)
        self.tau = float(eta_norm) * float(np.sqrt(randomized_var))
        self.stat = float(stat)
        self.alternative = alternative
        self.null_value = null_value

    def inference(
        self,
        algorithm: Callable[
            [np.ndarray, np.ndarray, float],
            tuple[Any, list[list[float]] | RealSubset],
        ],
        model_selector: Callable[[Any], bool],
        inference_mode: Literal[
            "parametric",
            "over_conditioning",
        ] = "parametric",
        confidence_level: float = 0.95,
        max_iter: int = 100_000,
        n_jobs: int = 1,
        step: float = 1e-6,
        *,
        progress: bool = False,
    ) -> RandomizedInferenceResult:
        """Conduct inference.

        Parameters
        ----------
        algorithm : Callable[[np.ndarray, np.ndarray, float], tuple[Any, list[list[float]] | RealSubset]]
            Callable function which takes two vectors a (np.ndarray) and
            b (np.ndarray), and a scalar z (float), and returns a model (Any) and
            intervals (list[list[float]] | RealSubset). For any point in
            the intervals, the same model must be selected.
        model_selector : Callable[[Any], bool]
            Callable function which takes a model (Any) and returns a boolean value,
            indicating whether the model is the same as the selected model.
        inference_mode : Literal["parametric", "over_conditioning"], optional
            Must be 'parametric' or 'over_conditioning'.
            Defaults to 'parametric'.
        confidence_level : float, optional
            Confidence level for the selective confidence interval.
            The selective confidence interval has the coverage of `confidence_level`.
            Defaults to 0.95.
        max_iter : int, optional
            Maximum number of iterations. Defaults to 100_000.
        n_jobs : int, optional
            Number of jobs to run in parallel. If set to other than 1, `inference_mode` is forced to
            'exhaustive' and then options `search_strategy` and `termination_criterion` are ignored.
            If set to -1, the all available cores are used. Defaults to 1.
        step : float, optional
            Step size for the search strategy. Defaults to 1e-6.
        progress : bool, optional
            Whether to show the progress bar. Defaults to `False`.

        Raises
        ------
        InfiniteLoopError
            If the search falls into an infinite loop.

        Returns
        -------
        RandomizedInferenceResult
            The result of the selective inference.
        """
        result = self.si.inference(
            algorithm,
            model_selector,
            inference_mode="exhaustive"
            if inference_mode == "parametric"
            else "over_conditioning",
            max_iter=max_iter,
            n_jobs=n_jobs,
            step=step,
            progress=progress,
        )

        truncated_region = RealSubset(result.truncated_intervals)
        pivot = _randomized_cdf(
            stat=self.stat,
            sigma=self.sigma,
            tau=self.tau,
            truncated_region=truncated_region,
            mu=self.null_value,
        )
        match self.alternative:
            case "two-sided":
                p_value = 2.0 * min(pivot, 1.0 - pivot)
            case "less":
                p_value = 1.0 - pivot
            case "greater":
                p_value = pivot

        ci_lower = _randomized_ppf(
            stat=self.stat,
            sigma=self.sigma,
            tau=self.tau,
            truncated_region=truncated_region,
            q=(1 + confidence_level) / 2,
        )
        ci_upper = _randomized_ppf(
            stat=self.stat,
            sigma=self.sigma,
            tau=self.tau,
            truncated_region=truncated_region,
            q=(1 - confidence_level) / 2,
        )

        mle: OptimizeResult = minimize_scalar(
            lambda mu: -_randomized_pdf(
                stat=self.stat,
                sigma=self.sigma,
                tau=self.tau,
                truncated_region=truncated_region,
                mu=mu,
            ),
        )

        return RandomizedInferenceResult(
            stat=self.stat,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            point_estimate=mle.x,
            searched_intervals=result.searched_intervals,
            truncated_intervals=result.truncated_intervals,
            search_count=result.search_count,
            detect_count=result.detect_count,
            null_rv=norm(loc=self.null_value * self.sigma, scale=self.sigma),
            alternative=self.alternative,
        )


def _randomized_cdf(
    stat: float,
    sigma: float,
    tau: float,
    truncated_region: RealSubset,
    mu: float,
) -> float:
    rho_sq = tau**2 / (sigma**2 + tau**2)
    log_denominator = compute_log_area(
        norm(loc=mu, scale=np.sqrt(sigma**2 + tau**2)),
        truncated_region,
    )

    def integrand(v: float) -> float:
        log_numerator = norm.logcdf(
            stat,
            loc=rho_sq * mu + (1 - rho_sq) * v,
            scale=sigma * np.sqrt(rho_sq),
        )
        return np.exp(
            norm.logpdf(v, loc=mu, scale=np.sqrt(sigma**2 + tau**2))
            + log_numerator
            - log_denominator,
        ).item()

    value = 0.0
    for interval in truncated_region.tolist():
        a, b = interval
        value += quad(integrand, float(a), float(b))[0]
    return value


def _randomized_ppf(
    stat: float,
    sigma: float,
    tau: float,
    truncated_region: RealSubset,
    q: float,
) -> float:
    a, b = -1.0, 1.0
    while True:
        val_a = (
            _randomized_cdf(
                stat=stat,
                sigma=sigma,
                tau=tau,
                truncated_region=truncated_region,
                mu=a,
            )
            - q
        )
        val_b = (
            _randomized_cdf(
                stat=stat,
                sigma=sigma,
                tau=tau,
                truncated_region=truncated_region,
                mu=b,
            )
            - q
        )
        if val_a * val_b < 0:
            break
        a *= 2
        b *= 2
    res: RootResults = root_scalar(
        lambda mu: _randomized_cdf(
            stat=stat,
            sigma=sigma,
            tau=tau,
            truncated_region=truncated_region,
            mu=mu,
        )
        - q,
        bracket=(a, b),
    )
    return float(res.root)


def _randomized_pdf(
    stat: float,
    sigma: float,
    tau: float,
    truncated_region: RealSubset,
    mu: float,
) -> float:
    rho_sq = tau**2 / (sigma**2 + tau**2)
    log_denominator = compute_log_area(
        norm(loc=mu, scale=np.sqrt(sigma**2 + tau**2)),
        truncated_region,
    )

    def integrand(v: float) -> float:
        log_numerator = norm.logpdf(
            stat,
            loc=rho_sq * mu + (1 - rho_sq) * v,
            scale=sigma * np.sqrt(rho_sq),
        )
        return np.exp(
            norm.logpdf(v, loc=mu, scale=np.sqrt(sigma**2 + tau**2))
            + log_numerator
            - log_denominator,
        ).item()

    value = 0.0
    for interval in truncated_region.tolist():
        a, b = interval
        value += quad(integrand, float(a), float(b))[0]
    return value
