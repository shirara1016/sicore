"""Module providing tests for uniformity of given samples."""

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Callable

import numpy as np
from numpy.polynomial import Polynomial
from scipy.stats import chi2, ecdf, kstwo, norm  # type: ignore[import]
from scipy.stats._hypotests import _cdf_cvm  # type: ignore[import]


class UniformityTest:
    """Class for testing uniformity of given samples."""

    def __init__(self) -> None:
        """Initialize UniformityTest."""
        self.alternative: Literal["two-sided", "less"]
        self._asymptotic_sample_size: int | None = None
        self._monte_carlo_sf: Callable[[np.ndarray], np.ndarray] | None = None
        self._sample_size: int

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
        """Computed differences between samples."""
        samples = UniformityTest.sort_samples(samples)
        samples = np.hstack(
            [np.zeros((samples.shape[0], 1)), samples, np.ones((samples.shape[0], 1))],
        )
        return np.diff(samples, axis=1)

    def test(self, samples: np.ndarray | list[float]) -> np.ndarray:
        """Test uniformity of given samples."""
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

    def _statistic(self, samples: np.ndarray) -> np.ndarray:
        spacing = UniformityTest.diff_samples(samples)
        return np.sum(spacing**2.0, axis=1)


class QuesenberryMillerTest(UniformityTest):
    """Class for Quesenberry-Miller test of uniformity."""

    def __init__(self) -> None:
        """Initialize QuesenberryMillerTest."""
        super().__init__()
        self.alternative = "less"

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

    def _statistic(self, samples: np.ndarray) -> np.ndarray:
        n = samples.shape[1]
        sort_samples = UniformityTest.sort_samples(samples)
        return np.sum(
            np.abs(sort_samples - (np.arange(1.0, n + 1.0) - 0.5) / n),
            axis=1,
        ) / np.sqrt(n)


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
