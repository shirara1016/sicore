import numpy as np


def false_positive_rate(p_values: list[float] | np.ndarray, alpha: float = 0.05):
    """
    Compute false positive rate of p-values under null.

    Args:
        p_values (list[float] | np.ndarray): List of p-values.
        alpha (float, optional): Significance level.

    Returns:
        float: False positive rate.
    """
    p_values = np.array(p_values)
    return np.count_nonzero(p_values <= alpha) / len(p_values)


def false_negative_rate(p_values: list[float] | np.ndarray, alpha: float = 0.05):
    """
    Compute false negative rate of p-values under alternative.

    Args:
        p_values (list[float] | np.ndarray): List of p-values.
        alpha (float, optional): Significance level.

    Returns:
        float: False negative rate.
    """
    p_values = np.array(p_values)
    return np.count_nonzero(p_values > alpha) / len(p_values)


def true_negative_rate(p_values: list[float] | np.ndarray, alpha: float = 0.05):
    """
    Compute true negative rate of p-values under null.

    Args:
        p_values (list[float] | np.ndarray): List of p-values.
        alpha (float, optional): Significance level.

    Returns:
        float: True negative rate.
    """
    return 1 - false_positive_rate(p_values, alpha=alpha)


def true_positive_rate(p_values: list[float] | np.ndarray, alpha: float = 0.05):
    """
    Compute true positive rate of p-values under alternative.

    Args:
        p_values (list[float] | np.ndarray): List of p-values.
        alpha (float, optional): Significance level.

    Returns:
        float: True positive rate.
    """
    return 1 - false_negative_rate(p_values, alpha=alpha)


def type1_error_rate(p_values: list[float] | np.ndarray, alpha: float = 0.05):
    """
    Compute type I error rate of p-values under null.

    Args:
        p_values (list[float] | np.ndarray): List of p-values.
        alpha (float, optional): Significance level.

    Returns:
        float: Type I error rate.
    """
    return false_positive_rate(p_values, alpha=alpha)


def type2_error_rate(p_values: list[float] | np.ndarray, alpha: float = 0.05):
    """
    Compute type II error rate of p-values under alternative.

    Args:
        p_values (list[float] | np.ndarray): List of p-values.
        alpha (float, optional): Significance level.

    Returns:
        float: Type II error rate.
    """
    return false_negative_rate(p_values, alpha=alpha)


def power(p_values: list[float] | np.ndarray, alpha: float = 0.05):
    """
    Compute power of p-values under alternative.

    Args:
        p_values (list[float] | np.ndarray): List of p-values.
        alpha (float, optional): Significance level.

    Returns:
        float: Power.
    """
    return true_positive_rate(p_values, alpha=alpha)
