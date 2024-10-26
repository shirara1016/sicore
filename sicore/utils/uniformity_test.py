"""Module providing tests for uniformity of given samples."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from collections.abc import Callable

import numpy as np
from numpy.polynomial import Polynomial
from scipy.stats import chi2, ecdf, kstwo, norm  # type: ignore[import]
from scipy.stats._hypotests import _cdf_cvm  # type: ignore[import]


@dataclass
class UniformityTestResult:
    """A class containing the result of uniformity tests.

    Attributes
    ----------
    is_rejected:
        A dictionary containing the test name and a boolean value indicating
        whether the null hypothesis is rejected
    """

    is_rejected: dict[str, bool]

    def __post_init__(self) -> None:
        """Separate the tests that are rejected and not rejected."""
        self._rejected_tests = [
            test_name
            for test_name, is_rejected in self.is_rejected.items()
            if is_rejected
        ]
        self._not_rejected_tests = [
            test_name
            for test_name, is_rejected in self.is_rejected.items()
            if not is_rejected
        ]

    def __str__(self) -> str:
        """Return a string representation of the UniformityTestResult object.

        Returns
        -------
        str
            String representation of the UniformityTestResult object.
        """
        return "\n".join(
            [
                f"Rejected tests ({len(self._rejected_tests)}/25):",
                *self._rejected_tests,
                "",
                f"Not rejected tests ({len(self._not_rejected_tests)}/25):",
                *self._not_rejected_tests,
            ],
        )

    def __repr__(self) -> str:
        """Return a string representation that can be used to recreate the object.

        Returns
        -------
        str
            String representation of the UniformityTestResult object.
        """
        return "\n".join(
            [
                f"{test_name}: {'rejected' if is_rejected else 'not rejected'}"
                for test_name, is_rejected in self.is_rejected.items()
            ],
        )


class UniformityTest:
    """Class for testing uniformity of given samples."""

    def __init__(self) -> None:
        """Initialize UniformityTest."""
        self.alternative: Literal["two-sided", "less"]
        self._asymptotic_sample_size: int | None = None
        self._monte_carlo_sf: Callable[[np.ndarray], np.ndarray] | None = None
        self._sample_size: int
        self.name: str

    @staticmethod
    def validate_samples(samples: np.ndarray | list[float]) -> np.ndarray:
        """Validate given samples."""
        return np.atleast_2d(samples)

    @staticmethod
    def sort_samples(samples: np.ndarray | list[float]) -> np.ndarray:
        """Sort given samples."""
        return np.sort(UniformityTest.validate_samples(samples), axis=1)

    @staticmethod
    def diff_samples(samples: np.ndarray | list[float]) -> np.ndarray:
        """Compute differences between samples."""
        samples = UniformityTest.sort_samples(samples)
        samples = np.hstack(
            [np.zeros((samples.shape[0], 1)), samples, np.ones((samples.shape[0], 1))],
        )
        return np.diff(samples, axis=1)

    def test(
        self,
        samples: np.ndarray | list[float],
        alpha: float = 0.05,
    ) -> np.ndarray | bool:
        """Test uniformity of given samples."""
        samples = UniformityTest.validate_samples(samples)
        rejection_area = self._load_rejection_area(samples.shape[1], alpha)
        if rejection_area is not None:
            stats = self._statistic(samples)
            rejects = np.any(
                (rejection_area[:, 0] < stats[:, None])
                & (stats[:, None] < rejection_area[:, 1]),
                axis=1,
            )
        else:
            rejects = self.compute_pvalues(samples) < alpha

        if len(rejects) == 1:
            return rejects[0].item()
        return rejects

    def compute_pvalues(self, samples: np.ndarray | list[float]) -> np.ndarray:
        """Compute p-values for given samples."""
        samples = UniformityTest.validate_samples(samples)
        self._sample_size = samples.shape[1]
        stats = self._statistic(samples)
        sf_values = self._sf(stats)

        match self.alternative:
            case "two-sided":
                p_values = 2 * np.minimum(sf_values, 1.0 - sf_values)
            case "less":
                p_values = sf_values
        return np.clip(p_values, 0.0, 1.0)

    def _load_rejection_area(self, sample_size: int, alpha: float) -> np.ndarray | None:
        """Load rejection area for given sample size and alpha."""
        if alpha not in (0.01, 0.05, 0.1):
            return None
        quantiles = QUANTILES_TABLE.get(self.name, {}).get(sample_size)
        if quantiles is None:
            return None

        quantiles_as_dict = dict(
            zip(
                [0.005, 0.01, 0.025, 0.05, 0.1, 0.9, 0.95, 0.975, 0.99, 0.995],
                quantiles,
                strict=True,
            ),
        )

        match self.alternative:
            case "two-sided":
                return np.array(
                    [
                        [-np.inf, quantiles_as_dict[alpha / 2]],
                        [quantiles_as_dict[1 - alpha / 2], np.inf],
                    ],
                )
            case "less":
                return np.array([[quantiles_as_dict[1 - alpha], np.inf]])

    def _statistic(self, samples: np.ndarray) -> np.ndarray:
        """Compute test statistics."""
        raise NotImplementedError

    def _sf(self, stats: np.ndarray) -> np.ndarray:
        """Compute values of survival function."""
        if self._monte_carlo_sf is None:
            self._construct_sf()
            assert self._monte_carlo_sf is not None  # noqa: S101
        return self._monte_carlo_sf(stats)

    def _construct_sf(self) -> None:
        """Construct survival function by monte carlo simulation and cache it."""
        rng = np.random.default_rng(12345)
        num_simulations = 10_000
        simulation_sample_size = (
            self._sample_size
            if self._asymptotic_sample_size is None
            else self._asymptotic_sample_size
        )

        monte_carlo_samples = self._statistic(
            rng.uniform(size=(num_simulations, simulation_sample_size)),
        )
        self._monte_carlo_sf = ecdf(monte_carlo_samples).sf.evaluate


class KolmogorovSmirnovTest(UniformityTest):
    """Class for Kolmogorov-Smirnov test of uniformity."""

    def __init__(self) -> None:
        """Initialize KolmogorovSmirnovTest."""
        super().__init__()
        self.alternative = "less"
        self.name = "kolmogorov_smirnov"

    def _statistic(self, samples: np.ndarray) -> np.ndarray:
        n = samples.shape[1]
        sort_samples = UniformityTest.sort_samples(samples)
        d_plus = np.max(np.arange(1.0, n + 1.0) / n - sort_samples, axis=1)
        d_minus = np.max(sort_samples - np.arange(0.0, n) / n, axis=1)
        return np.maximum(d_plus, d_minus)

    def _sf(self, stats: np.ndarray) -> np.ndarray:
        return kstwo.sf(stats, self._sample_size)


class KuiperTest(UniformityTest):
    """Class for Kuiper test of uniformity."""

    def __init__(self) -> None:
        """Initialize KuiperTest."""
        super().__init__()
        self.alternative = "less"
        self.name = "kuiper"

    def _statistic(self, samples: np.ndarray) -> np.ndarray:
        n = samples.shape[1]
        sort_samples = UniformityTest.sort_samples(samples)
        d_plus = np.max(np.arange(1.0, n + 1.0) / n - sort_samples, axis=1)
        d_minus = np.max(sort_samples - np.arange(0.0, n) / n, axis=1)
        return d_plus + d_minus

    def _sf(self, stats: np.ndarray) -> np.ndarray:
        n = self._sample_size
        stats = stats * np.sqrt(n)

        precision = 1e-10
        tot, cond = np.zeros_like(stats), np.ones_like(stats, dtype=bool)
        k = 1
        while np.any(cond):
            term_ = k * k * stats[cond] ** 2.0
            exp_term_ = np.exp(-2.0 * term_)
            term1 = ((-1.0) ** (k - 1)) * (4.0 * term_ - 1.0) * exp_term_
            term2 = k * k * (4.0 * term_ - 3.0) * exp_term_
            term = 2.0 * term1 - 8.0 * stats[cond] * term2 / (3.0 * np.sqrt(n))
            tot[cond] += term
            cond[cond] = np.abs(term) >= precision
            k += 1
        return tot


class CramerVonMisesTest(UniformityTest):
    """Class for Cramer-von Mises test of uniformity."""

    def __init__(self) -> None:
        """Initialize CramerVonMisesTest."""
        super().__init__()
        self.alternative = "less"
        self.name = "cramer_von_mises"

    def _statistic(self, samples: np.ndarray) -> np.ndarray:
        n = samples.shape[1]
        sort_samples = UniformityTest.sort_samples(samples)
        indices = np.arange(1.0, n + 1.0)
        term = np.sum((sort_samples - (2.0 * indices - 1.0) / (2.0 * n)) ** 2.0, axis=1)
        return term + (1.0 / (12.0 * n))

    def _sf(self, stats: np.ndarray) -> np.ndarray:
        return 1.0 - _cdf_cvm(stats, self._sample_size)


class AndersonDarlingTest(UniformityTest):
    """Class for Anderson-Darling test of uniformity."""

    def __init__(self) -> None:
        """Initialize AndersonDarlingTest."""
        super().__init__()
        self.alternative = "less"
        self._asymptotic_sample_size = 100
        self.name = "anderson_darling"

    def _statistic(self, samples: np.ndarray) -> np.ndarray:
        n = samples.shape[1]
        sort_samples = UniformityTest.sort_samples(samples)
        indices = np.arange(1.0, n + 1.0)
        term = np.sum(
            (2.0 * indices - 1.0) * np.log(sort_samples)
            + (2.0 * (n - indices) + 1.0) * np.log(1.0 - sort_samples),
            axis=1,
        )
        return -n - (term / n)


class WatsonTest(UniformityTest):
    """Class for Watson test of uniformity."""

    def __init__(self) -> None:
        """Initialize WatsonTest."""
        super().__init__()
        self.alternative = "less"
        self.name = "watson"

    def _statistic(self, samples: np.ndarray) -> np.ndarray:
        n = samples.shape[1]
        sort_samples = UniformityTest.sort_samples(samples)
        indices = np.arange(1.0, n + 1.0)
        term = np.sum((sort_samples - (2.0 * indices - 1.0) / (2.0 * n)) ** 2.0, axis=1)
        return (
            term + (1.0 / (12.0 * n)) - n * (np.mean(sort_samples, axis=1) - 0.5) ** 2.0
        )

    def _sf(self, stats: np.ndarray) -> np.ndarray:
        precision = 1e-10
        tot, cond = np.zeros_like(stats), np.ones_like(stats, dtype=bool)
        k = 1
        while np.any(cond):
            term = ((-1.0) ** (k - 1)) * np.exp(
                -2.0 * k * k * np.pi * np.pi * stats[cond],
            )
            tot[cond] += term
            cond[cond] = np.abs(term) >= precision
            k += 1
        return 2.0 * tot


class ZhangKolmogorovSmirnovTest(UniformityTest):
    """Class for Zhang's Kolmogorov-Smirnov test of uniformity."""

    def __init__(self) -> None:
        """Initialize ZhangKolmogorovSmirnovTest."""
        super().__init__()
        self.alternative = "less"
        self.name = "zhang_kolmogorov_smirnov"

    def _statistic(self, samples: np.ndarray) -> np.ndarray:
        n = samples.shape[1]
        sort_samples = UniformityTest.sort_samples(samples)
        indices = np.arange(1.0, n + 1.0)
        term1 = (indices - 0.5) * np.log((indices - 0.5) / (n * sort_samples))
        term2 = (n - indices + 0.5) * np.log(
            (n - indices + 0.5) / (n * (1.0 - sort_samples)),
        )
        return np.max(term1 + term2, axis=1)


class ZhangCramerVonMisesTest(UniformityTest):
    """Class for Zhang's Cramer-von Mises test of uniformity."""

    def __init__(self) -> None:
        """Initialize ZhangCramerVonMisesTest."""
        super().__init__()
        self.alternative = "less"
        self.name = "zhang_cramer_von_mises"

    def _statistic(self, samples: np.ndarray) -> np.ndarray:
        n = samples.shape[1]
        sort_samples = UniformityTest.sort_samples(samples)
        indices = np.arange(1.0, n + 1.0)
        term = np.log((1.0 / sort_samples - 1.0) / ((n - 0.5) / (indices - 0.75) - 1.0))
        return np.sum(term**2.0, axis=1)


class ZhangAndersonDarlingTest(UniformityTest):
    """Class for Zhang's Anderson-Darling test of uniformity."""

    def __init__(self) -> None:
        """Initialize ZhangAndersonDarlingTest."""
        super().__init__()
        self.alternative = "less"
        self.name = "zhang_anderson_darling"

    def _statistic(self, samples: np.ndarray) -> np.ndarray:
        n = samples.shape[1]
        sort_samples = UniformityTest.sort_samples(samples)
        indices = np.arange(1.0, n + 1.0)
        term1 = np.log(sort_samples) / (n - indices + 0.5)
        term2 = np.log(1.0 - sort_samples) / (indices - 0.5)
        return -np.sum(term1 + term2, axis=1)


class GreenwoodTest(UniformityTest):
    """Class for Greenwood test of uniformity."""

    def __init__(self) -> None:
        """Initialize GreenwoodTest."""
        super().__init__()
        self.alternative = "less"
        self.name = "greenwood"

    def _statistic(self, samples: np.ndarray) -> np.ndarray:
        spacing = UniformityTest.diff_samples(samples)
        return np.sum(spacing**2.0, axis=1)


class QuesenberryMillerTest(UniformityTest):
    """Class for Quesenberry-Miller test of uniformity."""

    def __init__(self) -> None:
        """Initialize QuesenberryMillerTest."""
        super().__init__()
        self.alternative = "less"
        self.name = "quesenberry_miller"

    def _statistic(self, samples: np.ndarray) -> np.ndarray:
        spacing = UniformityTest.diff_samples(samples)
        return np.sum(spacing**2.0, axis=1) + np.sum(
            spacing[:, 1:] * spacing[:, :-1],
            axis=1,
        )


class PearsonTest(UniformityTest):
    """Class for Pearson test of uniformity."""

    def __init__(self) -> None:
        """Initialize PearsonTest."""
        super().__init__()
        self.alternative = "less"
        self.name = "pearson"

    def _statistic(self, samples: np.ndarray) -> np.ndarray:
        return -2.0 * np.sum(np.log(samples), axis=1)

    def _sf(self, stats: np.ndarray) -> np.ndarray:
        return chi2.sf(stats, 2 * self._sample_size)


class SukhatmeTest(UniformityTest):
    """Class for Sukhatme test of uniformity."""

    def __init__(self) -> None:
        """Initialize SukhatmeTest."""
        super().__init__()
        self.alternative = "less"
        self.name = "sukhatme"

    def _statistic(self, samples: np.ndarray) -> np.ndarray:
        mode = 0.5
        mask = samples <= mode
        values = 2.0 * np.where(mask, samples, 1.0 - samples)
        return -2.0 * np.sum(np.log(values), axis=1)

    def _sf(self, stats: np.ndarray) -> np.ndarray:
        return chi2.sf(stats, 2 * self._sample_size)


class NeymanFirstOrderTest(UniformityTest):
    """Class for Neyman's first-order test of uniformity."""

    def __init__(self) -> None:
        """Initialize NeymanFirstOrderTest."""
        super().__init__()
        self.alternative = "less"
        self.name = "neyman_first_order"

    def _statistic(self, samples: np.ndarray) -> np.ndarray:
        n = samples.shape[1]
        polynomials = [Polynomial([0.0, 2.0 * np.sqrt(3.0)])]
        values = np.zeros(samples.shape[0])
        for poly in polynomials:
            values += np.sum(poly(samples - 0.5), axis=1) ** 2.0
        return values / n

    def _sf(self, stats: np.ndarray) -> np.ndarray:
        return chi2.sf(stats, 1)


class NeymanSecondOrderTest(UniformityTest):
    """Class for Neyman's second-order test of uniformity."""

    def __init__(self) -> None:
        """Initialize NeymanSecondOrderTest."""
        super().__init__()
        self.alternative = "less"
        self.name = "neyman_second_order"

    def _statistic(self, samples: np.ndarray) -> np.ndarray:
        n = samples.shape[1]
        polynomials = [
            Polynomial([0.0, 2.0 * np.sqrt(3.0)]),
            Polynomial([-0.5 * np.sqrt(5.0), 0.0, 6.0 * np.sqrt(5.0)]),
        ]
        values = np.zeros(samples.shape[0])
        for poly in polynomials:
            values += np.sum(poly(samples - 0.5), axis=1) ** 2.0
        return values / n

    def _sf(self, stats: np.ndarray) -> np.ndarray:
        return chi2.sf(stats, 2)


class NeymanThirdOrderTest(UniformityTest):
    """Class for Neyman's third-order test of uniformity."""

    def __init__(self) -> None:
        """Initialize NeymanThirdOrderTest."""
        super().__init__()
        self.alternative = "less"
        self.name = "neyman_third_order"

    def _statistic(self, samples: np.ndarray) -> np.ndarray:
        n = samples.shape[1]
        polynomials = [
            Polynomial([0.0, 2.0 * np.sqrt(3.0)]),
            Polynomial([-0.5 * np.sqrt(5.0), 0.0, 6.0 * np.sqrt(5.0)]),
            Polynomial([0.0, -3.0 * np.sqrt(7.0), 0.0, 20.0 * np.sqrt(7.0)]),
        ]
        values = np.zeros(samples.shape[0])
        for poly in polynomials:
            values += np.sum(poly(samples - 0.5), axis=1) ** 2.0
        return values / n

    def _sf(self, stats: np.ndarray) -> np.ndarray:
        return chi2.sf(stats, 3)


class NeymanFourthOrderTest(UniformityTest):
    """Class for Neyman's fourth-order test of uniformity."""

    def __init__(self) -> None:
        """Initialize NeymanFourthOrderTest."""
        super().__init__()
        self.alternative = "less"
        self.name = "neyman_fourth_order"

    def _statistic(self, samples: np.ndarray) -> np.ndarray:
        n = samples.shape[1]
        polynomials = [
            Polynomial([0.0, 2.0 * np.sqrt(3.0)]),
            Polynomial([-0.5 * np.sqrt(5.0), 0.0, 6.0 * np.sqrt(5.0)]),
            Polynomial([0.0, -3.0 * np.sqrt(7.0), 0.0, 20.0 * np.sqrt(7.0)]),
            Polynomial([1.125, 0.0, -45.0, 0.0, 210.0]),
        ]
        values = np.zeros(samples.shape[0])
        for poly in polynomials:
            values += np.sum(poly(samples - 0.5), axis=1) ** 2.0
        return values / n

    def _sf(self, stats: np.ndarray) -> np.ndarray:
        return chi2.sf(stats, 4)


class ShermanTest(UniformityTest):
    """Class for Sherman test of uniformity."""

    def __init__(self) -> None:
        """Initialize ShermanTest."""
        super().__init__()
        self.alternative = "less"
        self.name = "sherman"

    def _statistic(self, samples: np.ndarray) -> np.ndarray:
        n = samples.shape[1]
        spacing = UniformityTest.diff_samples(samples)
        omega = 0.5 * np.sum(np.abs(spacing - (1.0 / (n + 1.0))), axis=1)
        nominator = omega - 0.3679 * (1.0 - 1.0 / (2.0 * n))
        denominator = 0.2431 * (1.0 - 0.605 / n) / np.sqrt(n)
        v = nominator / denominator
        return v - 0.0955 * (v**2.0 - 1.0) / np.sqrt(n)

    def _sf(self, stats: np.ndarray) -> np.ndarray:
        return norm.sf(stats)


class KimballTest(UniformityTest):
    """Class for Kimball test of uniformity."""

    def __init__(self) -> None:
        """Initialize KimballTest."""
        super().__init__()
        self.alternative = "less"
        self.name = "kimball"

    def _statistic(self, samples: np.ndarray) -> np.ndarray:
        n = samples.shape[1]
        spacing = UniformityTest.diff_samples(samples)
        return np.sum((spacing - 1.0 / (n + 1.0)) ** 2.0, axis=1)


class ChengSpiringTest(UniformityTest):
    """Class for Cheng-Spiring test of uniformity."""

    def __init__(self) -> None:
        """Initialize ChengSpiringTest."""
        super().__init__()
        self.alternative = "two-sided"
        self.name = "cheng_spiring"

    def _statistic(self, samples: np.ndarray) -> np.ndarray:
        n = samples.shape[1]
        sample_ranges = np.max(samples, axis=1) - np.min(samples, axis=1)
        return (sample_ranges * (n + 1.0) / (n - 1.0)) ** 2.0 / (
            np.sum((samples - np.mean(samples, axis=1, keepdims=True)) ** 2.0, axis=1)
        )


class HegazyGreenAbsoluteTest(UniformityTest):
    """Class for Hegazy-Green absolute test of uniformity."""

    def __init__(self) -> None:
        """Initialize HegazyGreenAbsoluteTest."""
        super().__init__()
        self.alternative = "less"
        self.name = "hegazy_green_absolute"

    def _statistic(self, samples: np.ndarray) -> np.ndarray:
        n = samples.shape[1]
        sort_samples = UniformityTest.sort_samples(samples)
        indices = np.arange(1.0, n + 1.0)
        return np.sum(np.abs(sort_samples - indices / (n + 1.0)), axis=1) / n


class HegazyGreenModifiedAbsoluteTest(UniformityTest):
    """Class for Hegazy-Green modified absolute test of uniformity."""

    def __init__(self) -> None:
        """Initialize HegazyGreenModifiedAbsoluteTest."""
        super().__init__()
        self.alternative = "less"
        self.name = "hegazy_green_modified_absolute"

    def _statistic(self, samples: np.ndarray) -> np.ndarray:
        n = samples.shape[1]
        sort_samples = UniformityTest.sort_samples(samples)
        indices = np.arange(0.0, n)
        return np.sum(np.abs(sort_samples - indices / (n + 1.0)), axis=1) / n


class HegazyGreenQuadraticTest(UniformityTest):
    """Class for Hegazy-Green quadratic test of uniformity."""

    def __init__(self) -> None:
        """Initialize HegazyGreenQuadraticTest."""
        super().__init__()
        self.alternative = "less"
        self.name = "hegazy_green_quadratic"

    def _statistic(self, samples: np.ndarray) -> np.ndarray:
        n = samples.shape[1]
        sort_samples = UniformityTest.sort_samples(samples)
        indices = np.arange(1.0, n + 1.0)
        return np.sum((sort_samples - indices / (n + 1.0)) ** 2.0, axis=1) / n


class HegazyGreenModifiedQuadraticTest(UniformityTest):
    """Class for Hegazy-Green modified quadratic test of uniformity."""

    def __init__(self) -> None:
        """Initialize HegazyGreenModifiedQuadraticTest."""
        super().__init__()
        self.alternative = "less"
        self.name = "hegazy_green_modified_quadratic"

    def _statistic(self, samples: np.ndarray) -> np.ndarray:
        n = samples.shape[1]
        sort_samples = UniformityTest.sort_samples(samples)
        indices = np.arange(0.0, n)
        return np.sum((sort_samples - indices / (n + 1.0)) ** 2.0, axis=1) / n


class YangTest(UniformityTest):
    """Class for Yang test of uniformity."""

    def __init__(self) -> None:
        """Initialize YangTest."""
        super().__init__()
        self.alternative = "two-sided"
        self.name = "yang"

    def _statistic(self, samples: np.ndarray) -> np.ndarray:
        n = samples.shape[1]
        spacing = UniformityTest.diff_samples(samples)
        m = np.sum(np.minimum(spacing[:, :-1], spacing[:, 1:]), axis=1)
        return (2.0 * (n + 1.0) * m - n) * np.sqrt(3.0 / (2.0 * n - 1.0))

    def _sf(self, stats: np.ndarray) -> np.ndarray:
        return norm.sf(stats)


class FroziniTest(UniformityTest):
    """Class for Frozini test of uniformity."""

    def __init__(self) -> None:
        """Initialize FroziniTest."""
        super().__init__()
        self.alternative = "less"
        self._asymptotic_sample_size = 100
        self.name = "frozini"

    def _statistic(self, samples: np.ndarray) -> np.ndarray:
        n = samples.shape[1]
        sort_samples = UniformityTest.sort_samples(samples)
        return np.sum(
            np.abs(sort_samples - (np.arange(1.0, n + 1.0) - 0.5) / n),
            axis=1,
        ) / np.sqrt(n)


def uniformity_test(
    samples: np.ndarray | list[float],
    alpha: float = 0.05,
) -> UniformityTestResult:
    """Conduct 25 types of uniformity tests on the given samples.

    Parameters
    ----------
    samples : np.ndarray | list[float]
        The samples to be tested. Must be 1D array.
    alpha : float, optional
        The significance level. Defaults to 0.05.

    Returns
    -------
    UniformityTestResult
        A class containing the result of uniformity tests.
    """
    test_instances: list[UniformityTest] = [
        KolmogorovSmirnovTest(),
        KuiperTest(),
        CramerVonMisesTest(),
        AndersonDarlingTest(),
        WatsonTest(),
        ZhangKolmogorovSmirnovTest(),
        ZhangCramerVonMisesTest(),
        ZhangAndersonDarlingTest(),
        GreenwoodTest(),
        QuesenberryMillerTest(),
        PearsonTest(),
        SukhatmeTest(),
        NeymanFirstOrderTest(),
        NeymanSecondOrderTest(),
        NeymanThirdOrderTest(),
        NeymanFourthOrderTest(),
        ShermanTest(),
        KimballTest(),
        ChengSpiringTest(),
        HegazyGreenAbsoluteTest(),
        HegazyGreenModifiedAbsoluteTest(),
        HegazyGreenQuadraticTest(),
        HegazyGreenModifiedQuadraticTest(),
        YangTest(),
        FroziniTest(),
    ]
    results = {}
    for test in test_instances:
        results[test.name] = cast(bool, test.test(samples, alpha))
    return UniformityTestResult(is_rejected=results)


kolmogorov_smirnov_test = KolmogorovSmirnovTest().test
kuiper_test = KuiperTest().test
cramer_von_mises_test = CramerVonMisesTest().test
anderson_darling_test = AndersonDarlingTest().test
watson_test = WatsonTest().test
zhang_kolmogorov_smirnov_test = ZhangKolmogorovSmirnovTest().test
zhang_cramer_von_mises_test = ZhangCramerVonMisesTest().test
zhang_anderson_darling_test = ZhangAndersonDarlingTest().test
greenwood_test = GreenwoodTest().test
quesenberry_miller_test = QuesenberryMillerTest().test
pearson_test = PearsonTest().test
sukhatme_test = SukhatmeTest().test
neyman_first_order_test = NeymanFirstOrderTest().test
neyman_second_order_test = NeymanSecondOrderTest().test
neyman_third_order_test = NeymanThirdOrderTest().test
neyman_fourth_order_test = NeymanFourthOrderTest().test
sherman_test = ShermanTest().test
kimball_test = KimballTest().test
cheng_spiring_test = ChengSpiringTest().test
hegazy_green_absolute_test = HegazyGreenAbsoluteTest().test
hegazy_green_modified_absolute_test = HegazyGreenModifiedAbsoluteTest().test
hegazy_green_quadratic_test = HegazyGreenQuadraticTest().test
hegazy_green_modified_quadratic_test = HegazyGreenModifiedQuadraticTest().test
yang_test = YangTest().test
frozini_test = FroziniTest().test


QUANTILES_TABLE = {
    "kolmogorov_smirnov": {
        100: [
            0.03993408104087579,
            0.04299230766982066,
            0.04678143339427213,
            0.050353537929717265,
            0.0556307933552389,
            0.12020291013336006,
            0.13420925121222688,
            0.14718094334585416,
            0.16223900169077082,
            0.1735345444837151,
        ],
        200: [
            0.028617514320347873,
            0.030239315564026755,
            0.032876710200271726,
            0.035687005598027406,
            0.03931170816312958,
            0.08589367534852131,
            0.09532617574393623,
            0.10387016012845376,
            0.11482827292517407,
            0.12061718975323094,
        ],
        500: [
            0.018403229414751768,
            0.01948022917502993,
            0.021162223743099985,
            0.022770047318075522,
            0.025160284626206404,
            0.0543611829557418,
            0.06044472050666721,
            0.06551152024846367,
            0.0721325515703169,
            0.07672293899849465,
        ],
        1000: [
            0.012777254072519062,
            0.013660092477512195,
            0.0150914105006927,
            0.01630852743902078,
            0.017864390831709303,
            0.03864803332725106,
            0.04268986222221934,
            0.04626953425980666,
            0.05111219295861458,
            0.054523201530307826,
        ],
        2000: [
            0.009161367893211904,
            0.009764345146500316,
            0.010633956953796394,
            0.011576843124907421,
            0.01277076541064307,
            0.027025104892075276,
            0.02999521041951999,
            0.03290410859561062,
            0.03584422762596709,
            0.0380086449064961,
        ],
        5000: [
            0.005845891611759681,
            0.006140679489921456,
            0.006736430630262047,
            0.00729563114321542,
            0.007996297507178826,
            0.01730498794966463,
            0.019100975810537633,
            0.020966080698628705,
            0.022895250599575753,
            0.024527899955085454,
        ],
        10000: [
            0.00409175893403388,
            0.004360530909267547,
            0.00475964009665224,
            0.005162232067976715,
            0.005690138217773838,
            0.012171844761091727,
            0.01355508251336269,
            0.014731503393867963,
            0.01604740517683877,
            0.017231827961042896,
        ],
    },
    "kuiper": {
        100: [
            0.06960268533140083,
            0.07240827525137761,
            0.07819438819986096,
            0.0830002872037697,
            0.08980974385891083,
            0.1591799301061581,
            0.17149206690716584,
            0.18383352708993614,
            0.19783289934779477,
            0.210457364880315,
        ],
        200: [
            0.04888364822830948,
            0.051870868340341256,
            0.05548960923395977,
            0.059161710690940734,
            0.06416794665770804,
            0.11276228302184195,
            0.1215368079572293,
            0.13043476875568874,
            0.14148188179910326,
            0.14763794593855115,
        ],
        500: [
            0.031402173497709365,
            0.03286055342380186,
            0.03544347976211118,
            0.03769912099312442,
            0.04087963371657049,
            0.07159199893833441,
            0.07751182339909554,
            0.08264663279456713,
            0.08883070438025588,
            0.09262465683782141,
        ],
        1000: [
            0.022278806032248983,
            0.023476086860213647,
            0.025267164470011986,
            0.026869234039221607,
            0.028931170588157737,
            0.05110849356294187,
            0.05530256562857513,
            0.05863634187611529,
            0.06324776855225588,
            0.06657642695034614,
        ],
        2000: [
            0.015792318312545697,
            0.016648278005378974,
            0.01798689485429413,
            0.019129319475938468,
            0.02053091726955075,
            0.036040321366572974,
            0.038901932936441785,
            0.04152548851547266,
            0.044556832298823006,
            0.04660456085486059,
        ],
        5000: [
            0.01023421260266783,
            0.010699660061447744,
            0.011415529142495945,
            0.01208244060504235,
            0.013030602878943722,
            0.02290766320633208,
            0.024675972919524333,
            0.026438488409195774,
            0.028166510453827773,
            0.029795596592950613,
        ],
        10000: [
            0.007220716752846328,
            0.007476668636060854,
            0.008073534672650878,
            0.008631865231528906,
            0.009305567085976203,
            0.01619052423867949,
            0.017308059935812522,
            0.01841995365583051,
            0.01990702261096684,
            0.020973569410244444,
        ],
    },
    "cramer_von_mises": {
        100: [
            0.02301643138178555,
            0.02529123333287832,
            0.030661033365082583,
            0.03696906067300369,
            0.045990587650133724,
            0.3418299939464982,
            0.4602110176149178,
            0.5769523108589147,
            0.7702356694179894,
            0.8925605207060658,
        ],
        200: [
            0.022163824711199584,
            0.02472171502241654,
            0.0308260903321918,
            0.037217383529463036,
            0.04637792687581759,
            0.34762101623385344,
            0.46300471411807476,
            0.5906915258213193,
            0.7592496302557828,
            0.8956842046390598,
        ],
        500: [
            0.022264278304638754,
            0.025636529693145813,
            0.030393923913559943,
            0.03610623181570575,
            0.04503000156075714,
            0.3449141600997972,
            0.46161109664841204,
            0.5803625996173928,
            0.7423109557751073,
            0.8642354108577844,
        ],
        1000: [
            0.022278983403364747,
            0.025227266716020793,
            0.031153635498493547,
            0.03712305870078087,
            0.0465686115342007,
            0.34557109821614596,
            0.46305711121349036,
            0.5887015466887524,
            0.7597371560535754,
            0.9144250685023723,
        ],
        2000: [
            0.022225435486585652,
            0.025025702894035648,
            0.03048751796843996,
            0.03664517438249989,
            0.04651122020807485,
            0.3461581062012289,
            0.4533868147775829,
            0.5705241611251614,
            0.7222567728466945,
            0.8249255546437154,
        ],
        5000: [
            0.021375537342235077,
            0.024394136270000807,
            0.030164438354978514,
            0.03696646667566205,
            0.046088595906756506,
            0.3409411247030021,
            0.46765193900880675,
            0.5883756038222481,
            0.7531967520793521,
            0.8882447476342663,
        ],
        10000: [
            0.021500029164632697,
            0.024259698978075703,
            0.02961530087017865,
            0.034922297534120454,
            0.045101072325991184,
            0.3446073779647723,
            0.464067592192451,
            0.5897494048439511,
            0.7366370405270289,
            0.8783257361612765,
        ],
    },
    "anderson_darling": {
        100: [
            0.18175349687820258,
            0.20751755356560822,
            0.23965667707827193,
            0.2871404610269416,
            0.3530580200112425,
            1.9305827160036435,
            2.4723571352323006,
            3.0912020114016765,
            3.8301285955744144,
            4.546026380141072,
        ],
        200: [
            0.18215031181787922,
            0.2038042325590311,
            0.2415119685977679,
            0.28438812062204877,
            0.34798408994138297,
            1.9425819559642463,
            2.5571896485049983,
            3.162662690337637,
            4.031824102325441,
            4.648035921247211,
        ],
        500: [
            0.18118009085751852,
            0.20388750355015986,
            0.24504435445269052,
            0.28546488795642233,
            0.34347537589493415,
            1.9297845242515115,
            2.527114317076527,
            3.1656126930105697,
            3.843423769640938,
            4.349048214431142,
        ],
        1000: [
            0.18107861860025765,
            0.2063062770593865,
            0.24282147246681804,
            0.28488491548024514,
            0.35093079980964603,
            1.9340757913183555,
            2.460666764922512,
            3.0116046005897963,
            3.8051388585630335,
            4.372687097688329,
        ],
        2000: [
            0.175079434736831,
            0.19732012418165826,
            0.23652768931253831,
            0.27945600297539386,
            0.34248500909018276,
            1.9136388303224978,
            2.4582692764467073,
            3.069944726514177,
            3.892246934007208,
            4.5928703136767215,
        ],
        5000: [
            0.17792789335903764,
            0.19814867001717176,
            0.24332579568351778,
            0.28648417627764505,
            0.34826789661992735,
            1.9781663996102001,
            2.5375835984293182,
            3.084814131904499,
            3.892479111964251,
            4.478411726955937,
        ],
        10000: [
            0.18089547325059357,
            0.20488894451362286,
            0.24221547501811075,
            0.2821257972398598,
            0.34316738129909935,
            1.959345201700853,
            2.4915061508721004,
            3.138790360757728,
            4.005313153305124,
            4.425198873658445,
        ],
    },
    "watson": {
        100: [
            0.017957994072911983,
            0.019901507001089527,
            0.023312033770738035,
            0.02725517693223842,
            0.03293083896907871,
            0.1508105857971324,
            0.18567502074461,
            0.22413364517521073,
            0.27589020037258305,
            0.3129759351393476,
        ],
        200: [
            0.01794982691178576,
            0.020093746837252215,
            0.02348485230721083,
            0.027249760840013505,
            0.03292404085916633,
            0.15367557210537908,
            0.18863445127208853,
            0.2248079645875818,
            0.2680545302850231,
            0.302553989314794,
        ],
        500: [
            0.017401175674163636,
            0.018896193409086787,
            0.022827538308485613,
            0.026840065326152733,
            0.032517303935061054,
            0.1512503670780285,
            0.18619132178361214,
            0.21947158506102646,
            0.2674504256624829,
            0.303561765369726,
        ],
        1000: [
            0.017205093393904847,
            0.01928811928721277,
            0.0233446145390889,
            0.02722075446028527,
            0.03311125866138414,
            0.1542132570705324,
            0.1896086772378477,
            0.2245513353011624,
            0.2718189429796212,
            0.30226406090300073,
        ],
        2000: [
            0.017210873503350866,
            0.019337285314748195,
            0.02270391837723206,
            0.02664219544358714,
            0.03305332657104996,
            0.14964229119140024,
            0.18514745813648617,
            0.22286113092271184,
            0.2678076364103306,
            0.2975347221763319,
        ],
        5000: [
            0.017898857600136268,
            0.01996682123311377,
            0.023420446371630085,
            0.02707343908477888,
            0.03252614854815211,
            0.15458670365615176,
            0.18735443103042407,
            0.2223664474850546,
            0.2683558891576231,
            0.3048203894466979,
        ],
        10000: [
            0.01741505562872332,
            0.019527222488264055,
            0.02317532514202429,
            0.027159355906973177,
            0.03307917281965745,
            0.1510237355879818,
            0.18519093915628357,
            0.2224643997665438,
            0.2737693391358337,
            0.3194032961527773,
        ],
    },
    "zhang_kolmogorov_smirnov": {
        100: [
            0.5337290725531194,
            0.6004792540543806,
            0.7179332834797125,
            0.8391887612519333,
            1.0063327298664462,
            3.81662963552053,
            4.521689928130895,
            5.220052351801792,
            6.149464041895604,
            6.794695924337883,
        ],
        200: [
            0.6109044629035256,
            0.688953259543439,
            0.8325264274286894,
            0.9749173593729524,
            1.1651364724026796,
            4.077876039110803,
            4.855632793569236,
            5.557518371070718,
            6.4296095294018585,
            7.178252530651468,
        ],
        500: [
            0.7479110353329936,
            0.8359752889472581,
            0.9760179990736251,
            1.1303887923419766,
            1.3359403929926907,
            4.388948039489339,
            5.165228231631315,
            5.938314475478242,
            6.919417090400145,
            7.62594303103489,
        ],
        1000: [
            0.8485415660209293,
            0.939752561012704,
            1.089197813448755,
            1.2387261841793007,
            1.4600460828370416,
            4.521139552275857,
            5.285015062171582,
            5.993959760024193,
            6.85234864299637,
            7.602524468448724,
        ],
        2000: [
            0.9154488464729258,
            1.0126667958995343,
            1.1782370472901205,
            1.3441791313632523,
            1.5622038105850125,
            4.647074639049589,
            5.391644487627163,
            6.1446769460902555,
            7.125106171738884,
            7.739552262545128,
        ],
        5000: [
            1.0108998903523618,
            1.1147434288794278,
            1.2983066964253578,
            1.473941513692682,
            1.6913654586517772,
            4.875566614140568,
            5.623176021117961,
            6.403940298155613,
            7.380037536260317,
            8.086435070265713,
        ],
        10000: [
            1.0983899619610806,
            1.180558068242792,
            1.3650079384798854,
            1.5343728934910645,
            1.7631358564975905,
            4.926977921000609,
            5.650628910366381,
            6.411091558503943,
            7.45594184726191,
            8.173631732997864,
        ],
    },
    "zhang_cramer_von_mises": {
        100: [
            3.1890522418000256,
            3.6165407366067974,
            4.336337405411752,
            5.175977367951812,
            6.20551786473452,
            23.369727108864893,
            28.565349496466947,
            33.44035050448584,
            40.06607548531131,
            45.46418143342005,
        ],
        200: [
            3.681291501744011,
            4.167593039382306,
            5.073373326357408,
            5.937072009301718,
            7.186674489939422,
            25.19987622123353,
            30.355135973788908,
            35.68679885394122,
            42.684477322051414,
            49.32600636526631,
        ],
        500: [
            4.54623933359635,
            5.074039473842575,
            6.112021585876744,
            7.01371647212153,
            8.306534685277578,
            27.679134416318618,
            32.911241892338595,
            38.37978295129872,
            46.051587390866395,
            52.36875592745258,
        ],
        1000: [
            5.155886225389471,
            5.794743325069839,
            6.8703044369422015,
            7.978983815206094,
            9.34378177945142,
            29.98426866540649,
            35.007594600181605,
            40.27567248197499,
            48.143391491538644,
            53.42145139397988,
        ],
        2000: [
            6.047371932628939,
            6.709081117496186,
            7.657961821191725,
            8.812413206971447,
            10.311563586504112,
            31.651439800579766,
            37.237478352438785,
            42.37077387596518,
            49.30193451901861,
            54.97237066457136,
        ],
        5000: [
            6.775914008854736,
            7.500602776234008,
            8.680163851271555,
            9.973251844967619,
            11.658902894090643,
            33.70866986716533,
            39.39643008163259,
            45.504742957014685,
            52.75802173697799,
            57.71771050111869,
        ],
        10000: [
            7.86638101704843,
            8.585199191772228,
            9.867703657228931,
            10.988977338329907,
            12.726603591487816,
            35.37690669458382,
            40.569784722391226,
            46.28617929195064,
            52.828793370571496,
            57.78921174437653,
        ],
    },
    "zhang_anderson_darling": {
        100: [
            3.2955376701586947,
            3.297539950670104,
            3.3007120472152685,
            3.3039839780069737,
            3.3086554305795066,
            3.393035325618884,
            3.4163731766544503,
            3.444006704644719,
            3.4757890126771267,
            3.5008518528743116,
        ],
        200: [
            3.294095276377137,
            3.295103697986357,
            3.2970791622441236,
            3.299099671503369,
            3.301727844285991,
            3.3469845674793857,
            3.359883463257752,
            3.3739402231619273,
            3.391351382965628,
            3.4026666001450727,
        ],
        500: [
            3.292318324490558,
            3.292859615121757,
            3.2936950781944065,
            3.294537024884152,
            3.295711784710381,
            3.3154283394008024,
            3.320738282700158,
            3.3263750695357803,
            3.3339046295030843,
            3.339557331121335,
        ],
        1000: [
            3.291327202934995,
            3.291610878339082,
            3.292110443834653,
            3.2925499630429007,
            3.293174726293346,
            3.303540825072151,
            3.306487992191091,
            3.3092625229703283,
            3.3133759827199505,
            3.3158582206826717,
        ],
        2000: [
            3.2908194561246242,
            3.2909544510561006,
            3.2912183843174074,
            3.2914845097230545,
            3.291797278326552,
            3.297059312694727,
            3.298438165224011,
            3.2999566260453848,
            3.302021806757578,
            3.303330247810432,
        ],
        5000: [
            3.290319959944314,
            3.2903960975166298,
            3.2904870644751525,
            3.2905961924881355,
            3.290763660619487,
            3.2930380874449643,
            3.2936157441869733,
            3.29426007626636,
            3.2949731597726184,
            3.295526102993093,
        ],
        10000: [
            3.2901396008526795,
            3.2901751164454356,
            3.2902289921938284,
            3.290286876325664,
            3.290361465030496,
            3.291531406064208,
            3.291817065430337,
            3.2920787362054824,
            3.2925442482574403,
            3.292875067256293,
        ],
    },
    "greenwood": {
        100: [
            0.015914265846703673,
            0.01618681202546344,
            0.016578943543348233,
            0.01694719228392654,
            0.01744063032192682,
            0.022096626912681182,
            0.02309074484785195,
            0.024070806537982232,
            0.025469926532382857,
            0.026258672849002976,
        ],
        200: [
            0.008462118572957146,
            0.008594070333116198,
            0.008757324661907437,
            0.008906470669232943,
            0.009092417448436077,
            0.010776393010518526,
            0.01111812274537184,
            0.011425623181985355,
            0.011777227286413911,
            0.012089895714764407,
        ],
        500: [
            0.003587947535913559,
            0.0036203086368674026,
            0.003668470056351973,
            0.0037160529473404024,
            0.0037692500157111082,
            0.004221632923235028,
            0.004302304761541931,
            0.004376824393019125,
            0.004466982880689434,
            0.00452167334722211,
        ],
        1000: [
            0.0018526516672232771,
            0.0018632041460765305,
            0.0018804008779380523,
            0.0018981459322840392,
            0.0019173845479867543,
            0.0020788507936522088,
            0.002103178096196305,
            0.0021260474541496786,
            0.0021523864437264467,
            0.0021746327312372623,
        ],
        2000: [
            0.0009467675070690051,
            0.0009507416385341252,
            0.0009576951424879746,
            0.0009630159443014661,
            0.0009707247028065896,
            0.0010285401905800835,
            0.0010373609047259825,
            0.001046023863477363,
            0.0010556586550618716,
            0.0010624293224379,
        ],
        5000: [
            0.0003860795980381503,
            0.0003874329647389262,
            0.00038932558027412323,
            0.00039095461553395876,
            0.0003927723090704896,
            0.0004069494938453625,
            0.00040916951995397226,
            0.00041096449804716264,
            0.0004133120975364641,
            0.0004148986098575974,
        ],
        10000: [
            0.00019490888516566552,
            0.00019545297049003744,
            0.00019614013815582858,
            0.00019669542771273655,
            0.00019740952248038497,
            0.00020256220519733516,
            0.000203286037235974,
            0.00020406812183975038,
            0.00020478768483259547,
            0.00020526357532972255,
        ],
    },
    "quesenberry_miller": {
        100: [
            0.02516095245978705,
            0.025491315894749518,
            0.025909688540755406,
            0.026371995146286912,
            0.026888424916223678,
            0.03199389499075429,
            0.033113094476841506,
            0.03419763031362231,
            0.035562745778166975,
            0.03667399968317194,
        ],
        200: [
            0.013214738478088257,
            0.013372935776632577,
            0.01354086396235609,
            0.013702049484866785,
            0.013903646795442986,
            0.015829585674439747,
            0.01618403575652595,
            0.016538638721370118,
            0.016959372677635746,
            0.017293573440312765,
        ],
        500: [
            0.00553137413089348,
            0.005569960305534348,
            0.005616950660007761,
            0.005668774000084539,
            0.005731054853879594,
            0.006228630521381463,
            0.006314942333964606,
            0.006396311922718415,
            0.006492541903006323,
            0.00654911234619492,
        ],
        1000: [
            0.0028280577947270395,
            0.0028439990716849052,
            0.0028630862132516784,
            0.002883031010034334,
            0.0029060678417678746,
            0.003083557258945027,
            0.0031119112324662014,
            0.0031403776329687904,
            0.003172377484537131,
            0.0031943873809805786,
        ],
        2000: [
            0.0014377121136281085,
            0.001443225380443454,
            0.001451706050927237,
            0.00145895288022327,
            0.0014675350176422438,
            0.001529938242849728,
            0.0015397722918989352,
            0.001549246447471491,
            0.0015610386009746195,
            0.001567302066912815,
        ],
        5000: [
            0.0005839926327543862,
            0.00058557882999717,
            0.0005876469156642541,
            0.0005895246033456919,
            0.0005916510009354145,
            0.0006079541446543989,
            0.000610549391731278,
            0.0006128248886884818,
            0.0006151695622989998,
            0.0006171375018077062,
        ],
        10000: [
            0.0002942016443442007,
            0.00029478918936878075,
            0.0002955996309903972,
            0.00029628544034157643,
            0.00029706867512920986,
            0.000302801138532371,
            0.00030367765103009396,
            0.00030435088660801877,
            0.0003051659750010697,
            0.0003058273763934283,
        ],
    },
    "pearson": {
        100: [
            151.32707920192536,
            155.75548528259597,
            161.84692221686998,
            167.78586837350852,
            174.27326643119164,
            225.49207285872694,
            234.17745781183368,
            242.23648369601665,
            249.74837315254527,
            254.24851892297988,
        ],
        200: [
            329.54416899555895,
            335.83873480790095,
            345.47960898798664,
            353.38335949617397,
            363.99614182225315,
            436.31030888056466,
            446.36383559143565,
            455.97544537379315,
            466.19575072452494,
            476.0370245376848,
        ],
        500: [
            885.6801636307556,
            897.6542625462451,
            913.539121015304,
            925.5889886490781,
            941.428386970378,
            1058.8155310656107,
            1075.9943983428855,
            1090.989004983177,
            1108.7649440049263,
            1122.4994870208373,
        ],
        1000: [
            1838.7330388203407,
            1854.1792213280212,
            1878.5340622525157,
            1897.9096604781284,
            1920.2294900053284,
            2080.9377213144685,
            2105.0160521555986,
            2124.0306738271884,
            2149.331466709414,
            2169.714776958837,
        ],
        2000: [
            3770.2822795030934,
            3788.0172073086264,
            3821.8551782971817,
            3851.2486040045383,
            3884.0064123893135,
            4114.298910765312,
            4145.753257888211,
            4180.243045896816,
            4217.8205577934195,
            4239.81696224796,
        ],
        5000: [
            9647.529852000653,
            9681.501874287003,
            9728.600552194772,
            9770.606895053406,
            9821.25997502782,
            10186.689264862334,
            10237.8206561172,
            10280.421359273692,
            10336.800298039358,
            10372.62087317698,
        ],
        10000: [
            19483.416434300878,
            19532.384240336072,
            19615.61528784627,
            19673.244193111605,
            19744.292488672076,
            20260.420180065594,
            20335.391800399284,
            20396.680856869887,
            20470.64169197132,
            20531.743867229965,
        ],
    },
    "sukhatme": {
        100: [
            152.98903884570453,
            156.8596266127973,
            162.9981790783032,
            168.54631107092638,
            175.46927505510877,
            226.1535968186534,
            234.15320144335578,
            241.77500570468817,
            250.004329656905,
            255.27429349153434,
        ],
        200: [
            330.67334159517776,
            338.29436579874283,
            345.8391581934259,
            355.14741162784094,
            364.12664673034857,
            436.44189719064286,
            447.6952455801882,
            458.06935969038994,
            469.27197851838054,
            476.48050325819975,
        ],
        500: [
            889.7970148919561,
            898.4736050846404,
            915.528334514557,
            928.8192088852552,
            944.1763117214917,
            1060.383998360363,
            1077.8683434675497,
            1093.1815784934715,
            1109.5327423349102,
            1121.3595991803961,
        ],
        1000: [
            1835.0307403548038,
            1852.1282196060467,
            1877.2920572825792,
            1896.8574177425971,
            1918.1378077786649,
            2082.077644814179,
            2105.6561447752442,
            2126.8903156055917,
            2151.0868892731655,
            2163.466201837175,
        ],
        2000: [
            3782.0960049793657,
            3796.618461847017,
            3826.9376973731014,
            3851.805961338502,
            3883.2102905454653,
            4113.432747162447,
            4146.955771014782,
            4176.448254546717,
            4207.548874429864,
            4232.645780699603,
        ],
        5000: [
            9633.397000300363,
            9669.000146991591,
            9722.242915037836,
            9765.413991007674,
            9815.498963456344,
            10183.765240876737,
            10234.773460894838,
            10280.112245252383,
            10326.461475422246,
            10353.57086404678,
        ],
        10000: [
            19493.658624744192,
            19539.34606331609,
            19615.14628434687,
            19673.336864732908,
            19743.575665753524,
            20259.022943787004,
            20332.23857936432,
            20395.054125775412,
            20468.597641955235,
            20527.4277017475,
        ],
    },
    "neyman_first_order": {
        100: [
            4.3602519680977925e-05,
            0.00018335302380860407,
            0.0009994182094453057,
            0.003960647338079484,
            0.015489794596627379,
            2.6859482446845946,
            3.877291030049377,
            5.085559929243807,
            6.464906089014077,
            7.472020638027237,
        ],
        200: [
            5.2545965457723586e-05,
            0.0001883205027030237,
            0.0009475037544987518,
            0.004262372083565575,
            0.017723712678968793,
            2.6352770708624362,
            3.8082008379005985,
            5.081742738653257,
            6.746122635852135,
            8.271167772417277,
        ],
        500: [
            3.2594499692846115e-05,
            0.00012917105232648258,
            0.001020271256462485,
            0.003710541469757529,
            0.014303861512574133,
            2.6676340151271805,
            3.8364646927050097,
            4.982273733459392,
            6.631520659051279,
            7.68045866339121,
        ],
        1000: [
            4.057800730246014e-05,
            0.00019355145095933986,
            0.0010922440988839317,
            0.004699958961972702,
            0.016126150529874716,
            2.735633887556104,
            3.8969062538032233,
            5.124871768723213,
            6.819101022731426,
            8.272920801632717,
        ],
        2000: [
            3.100584267213924e-05,
            0.00013595827448092984,
            0.0009569553214908675,
            0.004037601661371128,
            0.017126511240988605,
            2.7173762926368723,
            3.8709967489622024,
            4.977750504430246,
            6.331672169978264,
            7.670936569121943,
        ],
        5000: [
            4.268587785780403e-05,
            0.0002229892764095769,
            0.0010971778200152109,
            0.004360831875720702,
            0.016506486766388145,
            2.7191111109766792,
            3.869531250835468,
            5.178009386156432,
            6.856486277198851,
            8.039410000078261,
        ],
        10000: [
            3.441048600179934e-05,
            0.0001664949018647216,
            0.0008248793762916824,
            0.003934184700059319,
            0.016689520987157206,
            2.723452359061618,
            3.876630643176326,
            5.097941793241269,
            6.5467043442240005,
            7.731107702582155,
        ],
    },
    "neyman_second_order": {
        100: [
            0.008493090676034648,
            0.019248526398606202,
            0.05160149242985679,
            0.09910157536028777,
            0.20691552128420362,
            4.572490058804866,
            5.9573745700523215,
            7.35174146797517,
            9.24612761369314,
            10.619961211433676,
        ],
        200: [
            0.010433100153145093,
            0.018881610804822122,
            0.05139872696050406,
            0.10555684945799795,
            0.2114255347815725,
            4.608502349325124,
            5.97604781239203,
            7.23228407412274,
            9.349932337423468,
            11.02079778497221,
        ],
        500: [
            0.009275351854883993,
            0.01834052776163748,
            0.04689493728726988,
            0.10306341883439307,
            0.21373075132464026,
            4.590506225852606,
            5.933701633765057,
            7.2398333962499315,
            8.973556323503471,
            10.669912192319467,
        ],
        1000: [
            0.008612973618646042,
            0.01583702619304429,
            0.0446037687420032,
            0.0937511124645871,
            0.2004609596123473,
            4.64278262476306,
            5.9741090140542745,
            7.348707100963957,
            9.09234950179675,
            10.347445627279564,
        ],
        2000: [
            0.011799144497506403,
            0.020227387356981964,
            0.04703340662915452,
            0.10164492220212475,
            0.2076992760959272,
            4.597233157534149,
            5.928551301163611,
            7.473713585902365,
            9.187918678806021,
            10.680060486163748,
        ],
        5000: [
            0.008921791865338095,
            0.0176236905718436,
            0.043175477226995794,
            0.0884494724239254,
            0.19602866831679328,
            4.604436920223497,
            6.101807539361336,
            7.328126375994069,
            9.141679071853138,
            10.566737476760899,
        ],
        10000: [
            0.009623299617675739,
            0.018443262145631783,
            0.04568485294969446,
            0.10054792348689685,
            0.2165360638010939,
            4.648365464681114,
            6.1126427899023446,
            7.477336745180383,
            9.237535477364622,
            10.45892607707295,
        ],
    },
    "neyman_third_order": {
        100: [
            0.07170762789880498,
            0.11264945436444422,
            0.22554365500096432,
            0.37157446721821147,
            0.6229258839586411,
            6.243526479226885,
            7.792794324532484,
            9.325604197287255,
            11.358307557275774,
            12.81408276664958,
        ],
        200: [
            0.07811642721616366,
            0.11615656877720291,
            0.2226452688329671,
            0.36678812647507814,
            0.58945372767276,
            6.109155389703072,
            7.60195591136269,
            9.052779094279845,
            11.009792626329988,
            12.434447591420673,
        ],
        500: [
            0.06739174368139696,
            0.1092239915315769,
            0.21941176529854411,
            0.34237138399121886,
            0.5715877813200326,
            6.239996882930418,
            7.849184089620858,
            9.309813093317278,
            11.48152228892954,
            12.969803498057466,
        ],
        1000: [
            0.0701609280255696,
            0.1181466952439768,
            0.2181002218478563,
            0.3457273500796159,
            0.5949372237058478,
            6.252894755231794,
            7.772996734941521,
            9.328524173446013,
            11.24942841184385,
            12.494195205547637,
        ],
        2000: [
            0.07090690975761192,
            0.11387933072222367,
            0.21911227412428985,
            0.3613285795852577,
            0.6063659015178369,
            6.183167498832893,
            7.751013735080073,
            9.271872223632075,
            11.329682638386373,
            12.976958531510618,
        ],
        5000: [
            0.08016043476015577,
            0.12024617543782543,
            0.22736694260242224,
            0.37240650126800384,
            0.6016650299505761,
            6.206376215131254,
            7.827352772680449,
            9.474664662755382,
            11.237277125862434,
            12.982350081731914,
        ],
        10000: [
            0.08355659346508505,
            0.12956945621517466,
            0.2582286931551461,
            0.3765156626275433,
            0.5933116313083485,
            6.250964479445527,
            7.649105103645692,
            9.335038489009717,
            11.121174817171116,
            12.83859833476252,
        ],
    },
    "neyman_fourth_order": {
        100: [
            0.20807793962166884,
            0.30036812378642647,
            0.48682120890761105,
            0.7120954362964961,
            1.0892754927891266,
            7.7197074702976085,
            9.568097392664589,
            11.205577032338407,
            13.4156638979495,
            15.060402846740532,
        ],
        200: [
            0.18669098627733996,
            0.28576105638421595,
            0.4760202107995773,
            0.6830056926780373,
            1.023993566289106,
            7.664873247304066,
            9.29770530438287,
            11.038205474905983,
            13.254656299952007,
            14.720404666364512,
        ],
        500: [
            0.18348361975886768,
            0.28287862473773023,
            0.46792816329466147,
            0.6913503731965447,
            1.0562410248614245,
            7.7008954261287235,
            9.472365163875631,
            11.036221313926552,
            13.018081078582473,
            14.551064336910347,
        ],
        1000: [
            0.18955993300128648,
            0.2768604792436523,
            0.4900009680723881,
            0.7188240436683778,
            1.0780556997572266,
            7.923863001737646,
            9.639784222830436,
            11.250097992162758,
            13.500802134130323,
            15.256022817160353,
        ],
        2000: [
            0.17250961068087753,
            0.2474384246415882,
            0.46106398432352896,
            0.6760838665797504,
            1.0600300811967076,
            7.904600686792621,
            9.527940751510672,
            11.117081664449895,
            12.9602429154354,
            14.751407944693097,
        ],
        5000: [
            0.22691105419255808,
            0.32411186510281875,
            0.5035130290492278,
            0.7291394411775216,
            1.0861982639936172,
            7.855728292205867,
            9.339826245827341,
            11.008521979743579,
            12.945324317578208,
            14.368351178578349,
        ],
        10000: [
            0.23109062217431683,
            0.3137450715427652,
            0.4915720438269801,
            0.730414629228495,
            1.0737077865907556,
            7.822691774782688,
            9.562300407557721,
            11.11448795336878,
            13.25905011241899,
            15.023734888885711,
        ],
    },
    "sherman": {
        100: [
            -2.5936342488897512,
            -2.341504233620437,
            -1.9556634008241807,
            -1.6447781257373364,
            -1.2567616511433706,
            1.276664560271058,
            1.6520611449390814,
            2.0034972030740636,
            2.3534413749384915,
            2.565786769136402,
        ],
        200: [
            -2.5018117254671943,
            -2.300177360641878,
            -1.9543585861787593,
            -1.6424679309669916,
            -1.2837857185337747,
            1.3131242258927018,
            1.7010926714597563,
            2.0075427518812368,
            2.3409446420154443,
            2.6055244733848,
        ],
        500: [
            -2.5333186227571005,
            -2.2291998580188284,
            -1.90617574811247,
            -1.601130943444054,
            -1.2347648276635914,
            1.2694669969389556,
            1.6359612872864517,
            1.9421973447084206,
            2.2642852030511427,
            2.515961341850448,
        ],
        1000: [
            -2.6922941429804954,
            -2.3579175616633963,
            -2.003098115499478,
            -1.6732593563932665,
            -1.301832224809084,
            1.3013432730487042,
            1.6387012443516642,
            1.9584224238629253,
            2.3385944055922696,
            2.5896252076210473,
        ],
        2000: [
            -2.66336845915229,
            -2.389064163397217,
            -1.9630219075729236,
            -1.6377562555731544,
            -1.2653620241270922,
            1.283475615326003,
            1.6609435050713786,
            1.9641746029045275,
            2.3268121131208543,
            2.5457916902573556,
        ],
        5000: [
            -2.5890822010183787,
            -2.2926749474577037,
            -1.9774604491275156,
            -1.6926903746141175,
            -1.3336920934888485,
            1.2557183736148347,
            1.6214238597407276,
            1.9427059372964826,
            2.289085087306925,
            2.498197786163232,
        ],
        10000: [
            -2.5603010097296575,
            -2.2872229326965896,
            -1.91126711810587,
            -1.6087292676290037,
            -1.2847492938574558,
            1.2630519017230184,
            1.6401359918303833,
            1.9521558721024246,
            2.404861219834438,
            2.623285952632725,
        ],
    },
    "kimball": {
        100: [
            0.005978877369579283,
            0.006313928020661,
            0.006709965569731685,
            0.007082817786277865,
            0.007548902314273798,
            0.012080216075443963,
            0.013106565029249556,
            0.014017994518144039,
            0.015244330846759123,
            0.016216756934263055,
        ],
        200: [
            0.003471083887773483,
            0.003595191522053097,
            0.0037489549198650956,
            0.003913208183237525,
            0.004099304724210029,
            0.005819435923618659,
            0.006146293427354493,
            0.006426934842028089,
            0.00685212938752676,
            0.007264261474508478,
        ],
        500: [
            0.0015995120598042555,
            0.0016248981028945187,
            0.001672060291681208,
            0.001718648395700499,
            0.001770488150293904,
            0.0022214612625754758,
            0.0022964290242166356,
            0.00235933952674809,
            0.0024587028592157147,
            0.002527652954966387,
        ],
        1000: [
            0.0008516311775880618,
            0.0008635324911096614,
            0.0008814250146110047,
            0.0008992974471177435,
            0.0009188063292216203,
            0.0010796436116805408,
            0.0011056679191909517,
            0.0011278805927245883,
            0.0011544192205957334,
            0.001175952837095205,
        ],
        2000: [
            0.0004464804184603699,
            0.0004516148650126348,
            0.00045835665801906316,
            0.0004642804777316379,
            0.00047168666443884923,
            0.0005289791401355925,
            0.0005378453709603282,
            0.0005457606223577157,
            0.0005561565777086496,
            0.0005614542246937085,
        ],
        5000: [
            0.0001854520769972574,
            0.0001868278871272504,
            0.00018899723569940877,
            0.00019087600788602458,
            0.00019268531238453903,
            0.00020706629238967822,
            0.00020933391624661528,
            0.00021127560143162688,
            0.00021361992255386066,
            0.0002153736386229945,
        ],
        10000: [
            9.49620782398211e-05,
            9.546911071458452e-05,
            9.623103680790665e-05,
            9.68142958496817e-05,
            9.748114262119327e-05,
            0.00010252466051180142,
            0.00010324472175582167,
            0.00010389053753385735,
            0.00010460015433483577,
            0.0001051901345523548,
        ],
    },
    "cheng_spiring": {
        100: [
            0.09864823950660621,
            0.10077786766992491,
            0.10350560920001799,
            0.10608602283320487,
            0.10921062370851752,
            0.13625271152236332,
            0.14100926969124442,
            0.1452204659627845,
            0.15082716167450858,
            0.1547704171233715,
        ],
        200: [
            0.05197864109495319,
            0.052631400895089636,
            0.053675132863779715,
            0.05465275535759691,
            0.05580376684113937,
            0.06567144926688548,
            0.06722981196082749,
            0.06874568521732834,
            0.07030897817727629,
            0.0713488315311192,
        ],
        500: [
            0.021838181176335313,
            0.022001448546057845,
            0.022307650805727327,
            0.02258744348417041,
            0.022882078027967243,
            0.025347235605770742,
            0.025726110758444765,
            0.0260796209471367,
            0.02648489412728252,
            0.026802223408121415,
        ],
        1000: [
            0.011208126683957296,
            0.011272596547080706,
            0.011380858225057584,
            0.011477746555188852,
            0.011594613271101556,
            0.012459062948573808,
            0.012587372302199126,
            0.012704539222509076,
            0.01284785527497571,
            0.012951143365166074,
        ],
        2000: [
            0.005707152452373274,
            0.005734706783060561,
            0.005780238787768585,
            0.005812085353739317,
            0.005850377799103882,
            0.006159269984182506,
            0.006204430960077924,
            0.00624105717295445,
            0.0062861138067712855,
            0.0063210484910031,
        ],
        5000: [
            0.0023232016755556865,
            0.0023317597182050645,
            0.002342122894481744,
            0.002351298411411845,
            0.0023617208243021095,
            0.002440354492948649,
            0.0024522105895757053,
            0.0024616513903802833,
            0.002471718328632028,
            0.0024791004670328197,
        ],
        10000: [
            0.0011728779924946393,
            0.0011750906762306784,
            0.0011793178397610637,
            0.0011825787779660812,
            0.0011864916528455264,
            0.0012140334404957258,
            0.0012180745167923786,
            0.001221300225885966,
            0.0012249650210934388,
            0.0012274690042021713,
        ],
    },
    "hegazy_green_absolute": {
        100: [
            0.011476323500102923,
            0.012271026186790747,
            0.013643785427014964,
            0.015102242622096957,
            0.017063275243183032,
            0.05025758541426075,
            0.058547561224031326,
            0.0663731988707411,
            0.07524486880221376,
            0.08108054250219536,
        ],
        200: [
            0.008339060547686699,
            0.00887541854934689,
            0.00987050631590869,
            0.01088406601537292,
            0.01224523146855103,
            0.03524382340834208,
            0.041254226993665044,
            0.046953759256468525,
            0.053512674231990125,
            0.05636583059380678,
        ],
        500: [
            0.005360873441941274,
            0.005725956173663658,
            0.006269031445923205,
            0.00681953598413775,
            0.007716359110707951,
            0.022240153016066175,
            0.02605951950335425,
            0.02945682757583445,
            0.03356204682612405,
            0.036443718327808405,
        ],
        1000: [
            0.003738522757910505,
            0.004072318517296631,
            0.004463092027432638,
            0.004876217121571186,
            0.00545691468154747,
            0.01576203948085612,
            0.01829506130554086,
            0.020643148838191145,
            0.023814530322443444,
            0.025769093834916754,
        ],
        2000: [
            0.002608117277705106,
            0.0028046007207211308,
            0.0031304353458199057,
            0.003432827263481652,
            0.0038760777465525077,
            0.011056838991164532,
            0.012858683306663492,
            0.014727144258164368,
            0.016801528510817818,
            0.018327600514804396,
        ],
        5000: [
            0.0016828596079040727,
            0.001807773702206123,
            0.0019768827279793007,
            0.002171371600771633,
            0.0024564260305075854,
            0.007098312721506352,
            0.008278035786222753,
            0.009347185460912039,
            0.010657938032373929,
            0.01155327228248703,
        ],
        10000: [
            0.0011762426897401564,
            0.001241595737321937,
            0.001380001208161546,
            0.001514105794422314,
            0.0017112672811988303,
            0.00499770383913785,
            0.005790257221266044,
            0.006500972291163154,
            0.007343505713407419,
            0.007899572190038862,
        ],
    },
    "hegazy_green_modified_absolute": {
        100: [
            0.011954971349142224,
            0.012830673034528088,
            0.014237835163673147,
            0.015578880823975802,
            0.01773863392792616,
            0.05219584204706662,
            0.06093891011924229,
            0.0696668689484571,
            0.07947798843851556,
            0.08644785708195218,
        ],
        200: [
            0.008407242652507183,
            0.008936213266928741,
            0.010008791381870736,
            0.010897946866163969,
            0.012369184626345094,
            0.036073562389261196,
            0.042037983471929656,
            0.04725552294437804,
            0.053550783352938766,
            0.057706418864391484,
        ],
        500: [
            0.005318563730073473,
            0.005692872450237338,
            0.0062355122317175925,
            0.006831669187145818,
            0.0077116463212433895,
            0.022751013474805234,
            0.026651313886894058,
            0.030200190545192025,
            0.034509398934061936,
            0.03791608340543647,
        ],
        1000: [
            0.0037582160546947435,
            0.004007926513123273,
            0.004420457745785463,
            0.004876932825207076,
            0.005429420367479304,
            0.016032604375000863,
            0.018758516059415305,
            0.020972605900327127,
            0.023631918469393377,
            0.025730910967582125,
        ],
        2000: [
            0.002667709877225298,
            0.0028515999651918976,
            0.003147229350119878,
            0.003427452393455119,
            0.003860517209499871,
            0.011184200626257475,
            0.013005967644396122,
            0.014648132753840973,
            0.01667161835249517,
            0.018009192438789304,
        ],
        5000: [
            0.0016731418423946813,
            0.0017685863871504746,
            0.0019611261339217245,
            0.002170742799088326,
            0.0024558491494326407,
            0.007068330898604352,
            0.008239486492258682,
            0.009399455673899675,
            0.010484414019935417,
            0.011301267668511734,
        ],
        10000: [
            0.0011797287126323486,
            0.001252839214762405,
            0.0013889369725129616,
            0.0015226340190762717,
            0.0017141944558256785,
            0.00494304492005699,
            0.005755480910496221,
            0.0065751065451190825,
            0.007614923484600148,
            0.008309641075612023,
        ],
    },
    "hegazy_green_quadratic": {
        100: [
            0.0002163728335925484,
            0.0002441840754256556,
            0.00029162337374083534,
            0.00034796456362179656,
            0.0004502301270240926,
            0.003384433961985482,
            0.004476344392177371,
            0.0056421220359449105,
            0.007182988610844892,
            0.008569017876589032,
        ],
        200: [
            0.00010773181355853121,
            0.00012364494387195326,
            0.00014989272116618435,
            0.00017964200820120457,
            0.00022940797852131624,
            0.0017360701172404907,
            0.002268684872926941,
            0.002799924284888883,
            0.0034942437443262722,
            0.004000838356078608,
        ],
        500: [
            4.4228029902320946e-05,
            5.048413025081634e-05,
            6.0929083867859345e-05,
            7.260879264017823e-05,
            9.150303549310957e-05,
            0.0006861417184784637,
            0.0009288989932985537,
            0.0011841009677880786,
            0.0014807284035545359,
            0.0017522123420542267,
        ],
        1000: [
            2.0839750962117362e-05,
            2.434464108874219e-05,
            3.0025182839202693e-05,
            3.677657820269862e-05,
            4.6521073384822874e-05,
            0.0003476381743886592,
            0.0004611807328971059,
            0.0005849282299614924,
            0.0007433632016269796,
            0.0008707429483455965,
        ],
        2000: [
            1.0772488387665067e-05,
            1.2474806268290741e-05,
            1.4992802968426462e-05,
            1.829440769926797e-05,
            2.2732960453783188e-05,
            0.0001746867049436718,
            0.00023290072792424556,
            0.0002900152942419035,
            0.00036812882402216274,
            0.00042766570216723444,
        ],
        5000: [
            4.127257133295734e-06,
            4.9328045388225645e-06,
            6.013912118874541e-06,
            7.229061609493196e-06,
            9.092130382832486e-06,
            6.970522920679512e-05,
            9.095825209185361e-05,
            0.00011497016586320581,
            0.00014629453237451678,
            0.0001771573014107806,
        ],
        10000: [
            2.2311451751639214e-06,
            2.5050235921487726e-06,
            3.039278268411629e-06,
            3.6378373595799857e-06,
            4.5795140981566825e-06,
            3.499926667545245e-05,
            4.623929260160182e-05,
            5.779723350974818e-05,
            7.465499341936738e-05,
            8.893458978070359e-05,
        ],
    },
    "hegazy_green_modified_quadratic": {
        100: [
            0.00021785361610021078,
            0.00025003643122154635,
            0.000299736982602691,
            0.0003635691200366354,
            0.00046079594646585286,
            0.0037052601110697265,
            0.004920350187121652,
            0.006096575178987157,
            0.007813016032302942,
            0.009472296388896509,
        ],
        200: [
            0.0001120662717515161,
            0.00012644969032664702,
            0.00015599058316368997,
            0.00018645360917892616,
            0.00023444745475272814,
            0.0018182338678463944,
            0.0024227476349664152,
            0.003043913653740688,
            0.003842154297547392,
            0.004503460668722604,
        ],
        500: [
            4.343433320109753e-05,
            5.130744935643122e-05,
            6.147633184177424e-05,
            7.363913999326556e-05,
            9.251969649422108e-05,
            0.0006784856686120029,
            0.0008897851236211334,
            0.001148456840391739,
            0.0014353578383187301,
            0.001693709930620876,
        ],
        1000: [
            2.2034045593005767e-05,
            2.4619711574770694e-05,
            3.0344432160092173e-05,
            3.6673029990411324e-05,
            4.612004163065488e-05,
            0.0003446525480828288,
            0.0004621959345595778,
            0.0005803031336399273,
            0.0007474532851813936,
            0.0008503228626798859,
        ],
        2000: [
            1.0783371381156774e-05,
            1.2239496005831323e-05,
            1.5225478553806927e-05,
            1.8123290565487407e-05,
            2.2774819840931453e-05,
            0.00017347477140027337,
            0.00022816712656421068,
            0.00028690685580762336,
            0.000360326528965557,
            0.0004377351614175118,
        ],
        5000: [
            4.101420145220453e-06,
            4.8361165795458436e-06,
            5.977563949070282e-06,
            7.234821890008962e-06,
            9.090054081265404e-06,
            7.052089204112433e-05,
            9.313982804350668e-05,
            0.00011686491629626552,
            0.0001442278307024435,
            0.0001683639083187596,
        ],
        10000: [
            2.2451914257373907e-06,
            2.5348084904449595e-06,
            3.0866365530887383e-06,
            3.66965877590444e-06,
            4.688426907304536e-06,
            3.485056348852568e-05,
            4.593965968401373e-05,
            5.7190337134919936e-05,
            7.456311945771318e-05,
            8.836927081971781e-05,
        ],
    },
    "yang": {
        100: [
            -2.673531884560701,
            -2.357877489738843,
            -1.960074440273532,
            -1.6561472337626346,
            -1.2944051770190832,
            1.2946843205868417,
            1.633950038064104,
            1.9068473285891094,
            2.2436784892104926,
            2.4725694628822223,
        ],
        200: [
            -2.4690347508787043,
            -2.2857473419921046,
            -1.9217756994166628,
            -1.6471485930594392,
            -1.2535297990333505,
            1.3045938040350455,
            1.6407976326997311,
            1.954516265836644,
            2.2768920011079175,
            2.522001915734335,
        ],
        500: [
            -2.5967403129957267,
            -2.3250818428020668,
            -1.9605034202695522,
            -1.652353474587339,
            -1.2871692614053634,
            1.2818967692828236,
            1.6272541134781051,
            1.945975120775847,
            2.3404871590395357,
            2.5900023545775808,
        ],
        1000: [
            -2.611901553770722,
            -2.3431026440684986,
            -1.9620891606697155,
            -1.6443931695808833,
            -1.2846800019411653,
            1.2862696223307337,
            1.6612075978181569,
            1.9956514474275653,
            2.355334354761594,
            2.652076691065366,
        ],
        2000: [
            -2.527114886522198,
            -2.286307617536561,
            -1.8997640144877395,
            -1.6288849944351416,
            -1.2779724252480473,
            1.3076126160692025,
            1.6721466950525752,
            2.007830672411752,
            2.369050367272708,
            2.6257266266120047,
        ],
        5000: [
            -2.6151381186011804,
            -2.343209277473279,
            -1.9878666362414836,
            -1.647769870155106,
            -1.273447020410826,
            1.2965374126229503,
            1.668523105232724,
            1.966526134476308,
            2.344977660436385,
            2.624115929429571,
        ],
        10000: [
            -2.5714923400986027,
            -2.3245345458025914,
            -1.9629953209276225,
            -1.6498195016451371,
            -1.268648163717182,
            1.2985723516663648,
            1.6604196550352484,
            2.003113209971988,
            2.3388814557284308,
            2.5558144064023414,
        ],
    },
    "frozini": {
        100: [
            0.11826684437029897,
            0.12552671309048347,
            0.13970431714858506,
            0.1547503102798813,
            0.1727436007560333,
            0.498491179437998,
            0.5818486647761267,
            0.6589606838700204,
            0.7484179889736676,
            0.8219933480923926,
        ],
        200: [
            0.11502989491894602,
            0.12286294210278155,
            0.13888623659587465,
            0.1529189958790568,
            0.17175337650825664,
            0.49676840472063505,
            0.5795762396378009,
            0.6579125488539961,
            0.7491977420628034,
            0.8110806988968754,
        ],
        500: [
            0.11591084049545414,
            0.12318049205843608,
            0.1390613951329023,
            0.15370771764742694,
            0.17244226012603972,
            0.49703656214461367,
            0.5757388414859339,
            0.6592218141415481,
            0.7386623447176551,
            0.7980360392551971,
        ],
        1000: [
            0.116264205386782,
            0.12417525785541969,
            0.13823749959638276,
            0.1531176821951099,
            0.17205798224304905,
            0.5012983666145538,
            0.5880593298580347,
            0.6526969193905476,
            0.7472605524485266,
            0.8075597639634028,
        ],
        2000: [
            0.1191765000261825,
            0.12616909751366517,
            0.14060436186782044,
            0.15409075281861156,
            0.17345029924964972,
            0.4987424072704739,
            0.5791518331292206,
            0.6524507574028391,
            0.7522184918800956,
            0.8183440504043333,
        ],
        5000: [
            0.11532792737211711,
            0.12391842337061394,
            0.13825488550866658,
            0.15157943245647562,
            0.17117987167049087,
            0.49267469446758155,
            0.5749685202181178,
            0.6558083345213124,
            0.7554212417682281,
            0.8125675620842057,
        ],
        10000: [
            0.12085615572084461,
            0.1272040044351735,
            0.14163165231799021,
            0.15479922347458738,
            0.1726384991717926,
            0.49843610935491,
            0.5830438003217503,
            0.663876469544962,
            0.7616782389077073,
            0.8201368158715489,
        ],
    },
}
