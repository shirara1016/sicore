"""
Core package for selective inference.
"""
from . import intervals
from .intervals import polytope_to_interval, poly_lt_zero
from .utils import OneVec, construct_projection_matrix
from .evaluation import (
    power,
    type1_error_rate,
)
from .figures import (
    pvalues_hist,
    pvalues_qqplot,
)
from .inference.base import (
    SelectiveInferenceResult,
    InfiniteLoopError,
)
from .inference.norm import (
    NaiveInferenceNorm,
    SelectiveInferenceNorm,
)
from .inference.chi import (
    NaiveInferenceChi,
    SelectiveInferenceChi,
)

__all__ = [
    "NaiveInferenceNorm",
    "SelectiveInferenceNorm",
    "NaiveInferenceChi",
    "SelectiveInferenceChi",
    "SelectiveInferenceResult",
    "InfiniteLoopError",
    "type1_error_rate",
    "power",
    "pvalues_hist",
    "pvalues_qqplot",
    "intervals",
    "polytope_to_interval",
    "poly_lt_zero",
    "construct_projection_matrix",
    "OneVec",
]
