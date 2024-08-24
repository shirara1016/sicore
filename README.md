# sicore package

This package provides core functions for selective inference.

Detailed API reference is [here](https://shirara1016.github.io/sicore/).

## Installation

This package requires python 3.10 or higher and automatically installs any dependent packages. If you want to use tensorflow and pytorch's tensors, please install them manually.
```
$ pip install sicore
```
Uninstall :
```
$ pip uninstall sicore
```

## Module Contents
The following modules can be imported by `from sicore import *`.

**Selective Inference**
- SelectiveInferenceNorm : Selective inference for the normal distribution.
- SelectiveInferenceChi : Selective inference for the chi distribution.
- SelectiveInferenceResult: Data class for the result of selective inference.

**Evaluation**
- rejection_rate(): Computes rejection rate from the list of SelectiveInferenceResult objects or p-values.

**Figure**
- pvalues_hist() : Draws a histogram of p-values.
- pvalues_qqplot() : Draws a uniform Q-Q plot of p-values.
- FprFigure: Draws a figure of the false positive rate.
- TprFigure: Draws a figure of the true positive rate.

**Interval Operations**
- RealSubset : Class for representing a subset of real numbers, which provides many operations with intuitive syntax.
- complement() : Take the complement of intervals.
- union() : Take the union of two intervals.
- intersection() : Take the intersection of two intervals.
- difference() : Take the difference of first intervals with second intervals.
- symmetric_difference() : Take the symmetric difference of two intervals.

**Inequalities Solver**
- polynomial_below_zero() : Compute intervals where a given polynomial is below zero.
- polytope_below_zero() : Compute intervals where a given polytope is below zero.
- degree_one_polynomials_below_zero: Compute intervals where given degree-one polynomials are all below zero.

**Non-Gaussian Random Variables**
- generate_non_gaussian_rv(): Generate a standardized random variable in a given rv_name family with a given Wasserstein distance from the standard gaussian distribution.

**Constructor**
- OneVector : Vector whose elements at specified positions are set to 1, and 0 otherwise.
- construct_projection_matrix() : Construct projection matrix from basis.

## Others
Execute code test :
```
$ pytest tests/
```
