"""Module providing function for generating non-gaussian random variables."""

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


def _standardize(rv: rv_continuous, param: float) -> rv_continuous:
    """Standardize a random variable.

    Parameters
    ----------
    rv : rv_continuous
        Random variable object to be standardized.
    param : float
        Parameter of a given random variable to be standardized.

    Returns
    -------
    rv_continuous
        Standardized random variable object with the given parameter.
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

    Parameters
    ----------
    rv_name : Literal["skewnorm","exponnorm", "gennormsteep", "gennormflat", "t"]
        Random variable name to be generated.
    distance : float
        Wasserstein distance between the generated random variable and the standard gaussian distribution.
        It is strongly recommended to set a value between 0.01 and 0.15.

    Returns
    -------
    rv_continuous:
        Generated standardized random variable from the specified random variable family
        which has the specified Wasserstein distance from the standard gaussian distribution.
    """
    rv = rv_dict[rv_name]
    try:
        param = param_dict[rv_name][f"{distance:.2f}"]
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
            *range_dict[rv_name],
        )
    return _standardize(rv, param)


rv_dict = {
    "skewnorm": skewnorm,
    "exponnorm": exponnorm,
    "gennormsteep": gennorm,
    "gennormflat": gennorm,
    "t": t,
}


range_dict = {
    "skewnorm": (1e-4, 30.0),
    "exponnorm": (1e-4, 15.0),
    "gennormsteep": (1e-1, 2.0 - 1e-4),
    "gennormflat": (2.0 + 1e-4, 100.0),
    "t": (3.0, 200.0),
}


param_dict = {
    "skewnorm": {
        "0.01": 0.7133823849123759,
        "0.02": 0.9487870025277886,
        "0.03": 1.1416795358936562,
        "0.04": 1.3193929754458082,
        "0.05": 1.4927298765329757,
        "0.06": 1.6680276466576924,
        "0.07": 1.8501475264662277,
        "0.08": 2.0436599544835636,
        "0.09": 2.253555993157257,
        "0.10": 2.485897962704065,
        "0.11": 2.748650507304287,
        "0.12": 3.0529774424630536,
        "0.13": 3.4155352787388704,
        "0.14": 3.8629595354460173,
        "0.15": 4.441693019741314,
    },
    "exponnorm": {
        "0.01": 0.33721067022973705,
        "0.02": 0.44364780788055175,
        "0.03": 0.5274333543183709,
        "0.04": 0.6013600549954201,
        "0.05": 0.6701454396644267,
        "0.06": 0.7361945074944093,
        "0.07": 0.8009838942565699,
        "0.08": 0.8655507998712546,
        "0.09": 0.9307079975416189,
        "0.10": 0.9971545143701969,
        "0.11": 1.0655411969089221,
        "0.12": 1.1365153042836744,
        "0.13": 1.2107561109521647,
        "0.14": 1.2890078085580212,
        "0.15": 1.3721145981608385,
    },
    "gennormsteep": {
        "0.01": 1.884529311374642,
        "0.02": 1.780220838761154,
        "0.03": 1.6854863473825552,
        "0.04": 1.5990318279832734,
        "0.05": 1.519791362513657,
        "0.06": 1.446878209855903,
        "0.07": 1.3795480570165637,
        "0.08": 1.3171713741228466,
        "0.09": 1.2592111500312941,
        "0.10": 1.2052069662522373,
        "0.11": 1.154761138415992,
        "0.12": 1.1075283854229743,
        "0.13": 1.0632072877139103,
        "0.14": 1.0215335724227814,
        "0.15": 0.9822742249914626,
    },
    "gennormflat": {
        "0.01": 2.1286064765091752,
        "0.02": 2.2728355452507434,
        "0.03": 2.43587090975357,
        "0.04": 2.6218635329945905,
        "0.05": 2.8363423901938267,
        "0.06": 3.086857432939011,
        "0.07": 3.384033347805173,
        "0.08": 3.743376866148876,
        "0.09": 4.188574703249481,
        "0.10": 4.757986594979181,
        "0.11": 5.518778187988543,
        "0.12": 6.602235272399702,
        "0.13": 8.312152448317505,
        "0.14": 11.597773675426838,
        "0.15": 23.02101817049859,
    },
    "t": {
        "0.01": 39.08912539655782,
        "0.02": 20.208567806145904,
        "0.03": 13.911718115375214,
        "0.04": 10.761011502283745,
        "0.05": 8.868942181417832,
        "0.06": 7.606345474941096,
        "0.07": 6.703581036685421,
        "0.08": 6.025828458349993,
        "0.09": 5.4981866258441165,
        "0.10": 5.075713304670316,
        "0.11": 4.729807111306054,
        "0.12": 4.441398730634292,
        "0.13": 4.197284379084112,
        "0.14": 3.9880311024769033,
        "0.15": 3.8067196925899176,
    },
}
