import numpy as np
from scipy.stats import rv_continuous, norm, t, skewnorm, gennorm, exponnorm  # type: ignore
from scipy.integrate import quad  # type: ignore
from scipy.optimize import brentq  # type: ignore

gennormflat = gennorm
gennormsteep = gennorm

params_dict = {
    "skewnorm": {
        "0.03": 1.141679535895037,
        "0.06": 1.668027646656356,
        "0.09": 2.253555993158534,
        "0.12": 3.052977442461724,
        "0.15": 4.441693019739707,
    },
    "exponnorm": {
        "0.03": 0.5274333543184184,
        "0.06": 0.7361945074942922,
        "0.09": 0.9307079975424131,
        "0.12": 1.1365153042836023,
        "0.15": 1.372114598160624,
    },
    "gennormsteep": {
        "0.03": 1.685486347382175,
        "0.06": 1.446878209856004,
        "0.09": 1.2592111500311147,
        "0.12": 1.1075283854228473,
        "0.15": 0.9822742249929434,
    },
    "gennormflat": {
        "0.03": 2.4358709097539135,
        "0.06": 3.0868574329392504,
        "0.09": 4.188574703248306,
        "0.12": 6.60223527240027,
        "0.15": 23.021018170499307,
    },
    "t": {
        "0.03": 13.911718115376004,
        "0.06": 7.606345474941293,
        "0.09": 5.498186625845221,
        "0.12": 4.441398730633352,
        "0.15": 3.8067196925891835,
    },
}

range_dict = {
    "skewnorm": (1e-4, 30.0),
    "exponnorm": (1e-4, 15.0),
    "gennormsteep": (1e-1, 2.0 - 1e-4),
    "gennormflat": (2.0 + 1e-4, 50.0),
    "t": (3.0, 200.0),
}


def _standardize(rv_name: str, param: float) -> rv_continuous:
    """Standardize a random variable.

    Args:
        rv (rv_continuous): Random variable name to be standardized.
        param (float): Parameter of a given random variable to be standardized.

    Returns:
        rv_continuous: Standardized random variable object with the given parameter.
    """
    rv = eval(rv_name)
    mean = rv.mean(param)
    std = rv.std(param)
    return rv(param, loc=-mean / std, scale=1 / std)


def _wasserstein_distance(rv: rv_continuous) -> float:
    """Compute the Wasserstein distance between a given random variable and the standard gaussian distribution.

    Args:
        rv (rv_continuous): Random variable object.

    Returns:
        float: Wasserstein distance between the given random variable and the standard gaussian distribution.
    """

    def func(x):
        return np.abs(rv.cdf(x) - norm.cdf(x))

    return quad(func, -np.inf, np.inf)[0]


def _binary_search(rv_name: str, distance: float) -> float:
    """Binary search for the parameter where a given random variable has a given Wasserstein distance from the standard gaussian distribution.

    Args:
        rv_name (str): Random variable name.
        distance (float): Wasserstein distance from the standard gaussian distribution.

    Returns:
        float: Parameter of the random variable where the given random variable has the given
            Wasserstein distance from the standard gaussian distribution.
    """

    f = lambda param: _wasserstein_distance(_standardize(rv_name, param)) - distance
    return brentq(f, *range_dict[rv_name])


def generate_non_gaussian_rv(rv_name: str, distance: float):
    """Generate a standard random variable in a given rv_name family with a given Wasserstein distance from the standard gaussian distribution.

    Args:
        rv_name (str): Random variable name.
        distance (float): Wasserstein distance between the random variable
            and the standard gaussian distribution.

    Returns:
        rv_continuous: Generated standardized random variable object in the given rv_name family
        with the given Wasserstein distance from the standard gaussian distribution.
    """
    try:
        param = params_dict[rv_name][str(distance)]
    except KeyError:
        param = _binary_search(rv_name, distance)
    return _standardize(rv_name, param)
