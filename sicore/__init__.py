"""Core package for selective inference."""

from .core.base import SelectiveInference, SelectiveInferenceResult
from .core.real_subset import RealSubset
from .main.inference import SelectiveInferenceChi, SelectiveInferenceNorm
from .utils.constructor import OneVector, construct_projection_matrix
from .utils.evaluation import rejection_rate
from .utils.figure import (
    FprFigure,
    SummaryFigure,
    TprFigure,
    pvalues_hist,
    pvalues_qqplot,
)
from .utils.intervals import (
    complement,
    difference,
    intersection,
    linear_polynomials_below_zero,
    polynomial_below_zero,
    polytope_below_zero,
    symmetric_difference,
    union,
)
from .utils.non_gaussian import generate_non_gaussian_rv

__all__ = [
    "SelectiveInference",
    "SelectiveInferenceNorm",
    "SelectiveInferenceChi",
    "SelectiveInferenceResult",
    "rejection_rate",
    "pvalues_hist",
    "pvalues_qqplot",
    "SummaryFigure",
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
    "linear_polynomials_below_zero",
    "generate_non_gaussian_rv",
    "OneVector",
    "construct_projection_matrix",
]
