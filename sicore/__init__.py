"""Core package for selective inference.

.. image:: https://img.shields.io/pypi/v/sicore
    :alt: PyPI - Version
    :target: https://pypi.org/project/sicore/
.. image:: https://img.shields.io/pypi/pyversions/sicore
    :alt: PyPI - Python Version
    :target: https://pypi.org/project/sicore/
.. image:: https://img.shields.io/pypi/l/sicore
   :alt: PyPI - License
   :target: https://opensource.org/license/MIT

============
Installation
============
This package requires python 3.10 or higher and automatically installs any dependent packages. If you want to use tensorflow and pytorch's tensors, please install them manually.

.. code-block:: bash

    $ pip install sicore
"""

from .core.base import (
    InfiniteLoopError,
    SelectiveInference,
    SelectiveInferenceResult,
)
from .core.cdf import truncated_cdf
from .core.real_subset import RealSubset
from .main.inference import SelectiveInferenceChi, SelectiveInferenceNorm
from .utils.constructor import OneVector, construct_projection_matrix
from .utils.evaluation import rejection_rate
from .utils.figure import (
    SummaryFigure,
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
from .utils.uniformity_test import uniformity_test

__all__ = [
    "SelectiveInference",
    "SelectiveInferenceNorm",
    "SelectiveInferenceChi",
    "SelectiveInferenceResult",
    "InfiniteLoopError",
    "rejection_rate",
    "pvalues_hist",
    "pvalues_qqplot",
    "SummaryFigure",
    "RealSubset",
    "complement",
    "union",
    "intersection",
    "difference",
    "symmetric_difference",
    "polynomial_below_zero",
    "polytope_below_zero",
    "linear_polynomials_below_zero",
    "truncated_cdf",
    "generate_non_gaussian_rv",
    "uniformity_test",
    "OneVector",
    "construct_projection_matrix",
]
