"""Module providing function for generating non-gaussian random variables."""

from enum import Enum
from typing import Literal

import numpy as np
from scipy.integrate import quad  # type: ignore[import]
from scipy.optimize import brentq  # type: ignore[import]
from scipy.stats import (  # type: ignore[import]
    exponnorm,
    gennorm,
    norm,
    rv_continuous,
    skewnorm,
    t,
)

gennormflat = gennorm
gennormsteep = gennorm


class RandomVariable(Enum):
    """An enumeration class for random variables."""

    SKEWNORM = (skewnorm, (1e-4, 30.0))
    EXPONNORM = (exponnorm, (1e-4, 15.0))
    GENNORMSTEEP = (gennormsteep, (1e-1, 2.0 - 1e-4))
    GENNORMFLAT = (gennormflat, (2.0 + 1e-4, 50.0))
    T = (t, (3.0, 200.0))

    def __init__(
        self,
        rv: rv_continuous,
        interval: tuple[float, float],
    ) -> None:
        """Initialize the RandomVariable class."""
        self.rv = rv
        self.interval = interval

    @classmethod
    def rv_of(cls, name: str) -> rv_continuous:
        """Return the rv_continuous object of the specified random variable name (str)."""
        return cls[name.upper()].rv

    @classmethod
    def interval_of(cls, name: str) -> tuple[float, float]:
        """Return the interval of the specified random variable name (str)."""
        return cls[name.upper()].interval

    @classmethod
    def load_parameter(cls, name: str, distance: float) -> float:
        """Return the parameter of the specified random variable name (str) and distance."""
        return _PARAMETERS_CACHE[cls[name.upper()]][Distance(distance)]


class Distance(Enum):
    """An enumeration class for Wasserstein distance."""

    DISTANCE_01 = 0.01
    DISTANCE_02 = 0.02
    DISTANCE_03 = 0.03
    DISTANCE_04 = 0.04
    DISTANCE_05 = 0.05
    DISTANCE_06 = 0.06
    DISTANCE_07 = 0.07
    DISTANCE_08 = 0.08
    DISTANCE_09 = 0.09
    DISTANCE_10 = 0.10
    DISTANCE_11 = 0.11
    DISTANCE_12 = 0.12
    DISTANCE_13 = 0.13
    DISTANCE_14 = 0.14
    DISTANCE_15 = 0.15


_PARAMETERS_CACHE = {
    RandomVariable.SKEWNORM: {
        Distance.DISTANCE_01: 0.7133823849123759,
        Distance.DISTANCE_02: 0.9487870025277886,
        Distance.DISTANCE_03: 1.1416795358936562,
        Distance.DISTANCE_04: 1.3193929754458082,
        Distance.DISTANCE_05: 1.4927298765329757,
        Distance.DISTANCE_06: 1.6680276466576924,
        Distance.DISTANCE_07: 1.8501475264662277,
        Distance.DISTANCE_08: 2.0436599544835636,
        Distance.DISTANCE_09: 2.253555993157257,
        Distance.DISTANCE_10: 2.485897962704065,
        Distance.DISTANCE_11: 2.748650507304287,
        Distance.DISTANCE_12: 3.0529774424630536,
        Distance.DISTANCE_13: 3.4155352787388704,
        Distance.DISTANCE_14: 3.8629595354460173,
        Distance.DISTANCE_15: 4.441693019741314,
    },
    RandomVariable.EXPONNORM: {
        Distance.DISTANCE_01: 0.33721067022973705,
        Distance.DISTANCE_02: 0.44364780788055175,
        Distance.DISTANCE_03: 0.5274333543183709,
        Distance.DISTANCE_04: 0.6013600549954201,
        Distance.DISTANCE_05: 0.6701454396644267,
        Distance.DISTANCE_06: 0.7361945074944093,
        Distance.DISTANCE_07: 0.8009838942565699,
        Distance.DISTANCE_08: 0.8655507998712546,
        Distance.DISTANCE_09: 0.9307079975416189,
        Distance.DISTANCE_10: 0.9971545143701969,
        Distance.DISTANCE_11: 1.0655411969089221,
        Distance.DISTANCE_12: 1.1365153042836744,
        Distance.DISTANCE_13: 1.2107561109521647,
        Distance.DISTANCE_14: 1.2890078085580212,
        Distance.DISTANCE_15: 1.3721145981608385,
    },
    RandomVariable.GENNORMSTEEP: {
        Distance.DISTANCE_01: 1.884529311374642,
        Distance.DISTANCE_02: 1.780220838761154,
        Distance.DISTANCE_03: 1.6854863473825552,
        Distance.DISTANCE_04: 1.5990318279832734,
        Distance.DISTANCE_05: 1.519791362513657,
        Distance.DISTANCE_06: 1.446878209855903,
        Distance.DISTANCE_07: 1.3795480570165637,
        Distance.DISTANCE_08: 1.3171713741228466,
        Distance.DISTANCE_09: 1.2592111500312941,
        Distance.DISTANCE_10: 1.2052069662522373,
        Distance.DISTANCE_11: 1.154761138415992,
        Distance.DISTANCE_12: 1.1075283854229743,
        Distance.DISTANCE_13: 1.0632072877139103,
        Distance.DISTANCE_14: 1.0215335724227814,
        Distance.DISTANCE_15: 0.9822742249914626,
    },
    RandomVariable.GENNORMFLAT: {
        Distance.DISTANCE_01: 2.1286064765091752,
        Distance.DISTANCE_02: 2.2728355452507434,
        Distance.DISTANCE_03: 2.43587090975357,
        Distance.DISTANCE_04: 2.6218635329945905,
        Distance.DISTANCE_05: 2.8363423901938267,
        Distance.DISTANCE_06: 3.086857432939011,
        Distance.DISTANCE_07: 3.384033347805173,
        Distance.DISTANCE_08: 3.743376866148876,
        Distance.DISTANCE_09: 4.188574703249481,
        Distance.DISTANCE_10: 4.757986594979181,
        Distance.DISTANCE_11: 5.518778187988543,
        Distance.DISTANCE_12: 6.602235272399702,
        Distance.DISTANCE_13: 8.312152448317505,
        Distance.DISTANCE_14: 11.597773675426838,
        Distance.DISTANCE_15: 23.02101817049859,
    },
    RandomVariable.T: {
        Distance.DISTANCE_01: 39.08912539655782,
        Distance.DISTANCE_02: 20.208567806145904,
        Distance.DISTANCE_03: 13.911718115375214,
        Distance.DISTANCE_04: 10.761011502283745,
        Distance.DISTANCE_05: 8.868942181417832,
        Distance.DISTANCE_06: 7.606345474941096,
        Distance.DISTANCE_07: 6.703581036685421,
        Distance.DISTANCE_08: 6.025828458349993,
        Distance.DISTANCE_09: 5.4981866258441165,
        Distance.DISTANCE_10: 5.075713304670316,
        Distance.DISTANCE_11: 4.729807111306054,
        Distance.DISTANCE_12: 4.441398730634292,
        Distance.DISTANCE_13: 4.197284379084112,
        Distance.DISTANCE_14: 3.9880311024769033,
        Distance.DISTANCE_15: 3.8067196925899176,
    },
}


def _standardize(rv: rv_continuous, param: float) -> rv_continuous:
    """Standardize a random variable.

    Args:
        rv (rv_continuous): Random variable object to be standardized.
        param (float): Parameter of a given random variable to be standardized.

    Returns:
        rv_continuous: Standardized random variable object with the given parameter.
    """
    mean = rv.mean(param)
    std = rv.std(param)
    return rv(param, loc=-mean / std, scale=1 / std)


def generate_non_gaussian_rv(
    rv_name: Literal["skewnorm", "exponnorm", "gennormsteep", "gennormflat", "t"],
    distance: float,
) -> rv_continuous:
    """Generate a random variable from the specified random variable family.

    Generate a standardized random variable from
    the specified random variable family which has the specified
    Wasserstein distance from the standard gaussian distribution.

    Args:
        rv_name (Literal["skewnorm","exponnorm", "gennormsteep", "gennormflat", "t"]):
            Random variable name to be generated.
        distance (float): Wasserstein distance between the generated random variable
            and the standard gaussian distribution. It is strongly
            recommended to set a value between 0.01 and 0.15.

    Returns:
        rv_continuous:
            Generated standardized random variable from the
            specified random variable family which has the specified
            Wasserstein distance from the standard gaussian distribution.
    """
    rv = RandomVariable.rv_of(rv_name)
    try:
        param = RandomVariable.load_parameter(rv_name, distance)
    except (KeyError, ValueError):
        param = brentq(
            lambda param: (
                quad(
                    lambda z: np.abs(_standardize(rv, param).cdf(z) - norm.cdf(z)),
                    -np.inf,
                    np.inf,
                )[0]
                - distance
            ),
            *RandomVariable.interval_of(rv_name),
        )
    return _standardize(rv, param)
