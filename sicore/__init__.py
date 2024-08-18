"""
Core package for selective inference.
"""

from .core.base import SelectiveInference, SelectiveInferenceResult
from .core.real_subset import RealSubset
from .main.inference import SelectiveInferenceNorm, SelectiveInferenceChi
from .utils.constructor import OneVec, construct_projection_matrix
from .utils.evaluation import reject_rate
from .utils.intervals import (
    complement,
    union,
    intersection,
    difference,
    symmetric_difference,
    polynomial_below_zero,
    polytope_below_zero,
    degree_one_polynomials_below_zero,
)
from .utils.figure import FprFigure, TprFigure, pvalues_hist, pvalues_qqplot
from .utils.non_gaussian import generate_non_gaussian_rv

__all__ = [
    "SelectiveInference",
    "SelectiveInferenceNorm",
    "SelectiveInferenceChi",
    "SelectiveInferenceResult",
    "reject_rate",
    "pvalues_hist",
    "pvalues_qqplot",
    "FprFigure",
    "TprFigure",
    "RealSubset",
    "complement",
    "union",
    "intersection",
    "difference",
    "symmetric_difference",
    "polynomial_below_zero",
    "polytope_below_zero",
    "degree_one_polynomials_below_zero",
    "generate_non_gaussian_rv",
    "OneVec",
    "construct_projection_matrix",
]
