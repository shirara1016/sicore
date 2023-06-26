# sicore package

This package consists of core functions commonly used in selective inference.

The Japanese version README is [here](/README_ja.md).

## Installation

This package requires python 3.10 or higher and automatically installs any dependent packages. If you want to use tensorflow and pytorch's tensor, please install the framework manually.
```
$ pip install sicore
```
Uninstall :
```
$ pip uninstall sicore
```

## API Reference
Deteiled API reference is [here](https://shirara1016.github.io/sicore/).

## List of functions
The following fuctions are imported by `from sicore import *`

**Statistical Inference**
- NaiveInferenceNorm : Naive statistical inference for the test statistic following a normal distribution.
- SelectiveInferenceNorm : Selective statistical inference for the test statistic following a normal distribution.
    - Parametric SI and Over-Conditioning provided.
    - Parametric SI offers the following three types of methods.
        - Calculation of p-value with specified guaranteed accuracy.
        - Determining if the null hypothesis is rejected or not.
        - Performing a parametric search of the entire specified range.
    - Inference results are returned as a data class.
- NaiveInferenceChi : Naive statistical inference for the test statistic following a chi distribution.
- SelectiveInferenceChi : Selective statistical inference for the test statistic following a chi distribution.
    - Parametric SI and Over-Conditioning provided.
    - Parametric SI offers the following three types of methods.
        - Calculation of p-value with specified guaranteed accuracy.
        - Determining if the null hypothesis is rejected or not.
        - Performing a parametric search of the entire specified range.
    - Inference results are returned as a data class.

**Truncated Distribution**
Provides computation with arbitrary precision using mpmath for multiple truncated intervals.
- tn_cdf() : truncated standard normal distribution
- tc_cdf() : truncated chi distribution

**Evaluation Function**
- type1_error_rate()
- power()

**Figure Drawing**
- pvalues_hist() : Draws a histogram of p-values.
- pvalues_qqplot() : Draws a uniform Q-Q plot of p-values.

**Interval Operations**
- intervals.intersection() : Computes the intersection of two sets of intervals.
- intervals.intersection_all() : Computes the intersection of set of intervals.
- intervals.union_all() : Computes the union of set of intervals.
- intervals.not_() : Computes the complement of set of intervals with real numbers as the whole set.

**Utility**
- OneVec : Generates a vector that is 1 at the specified index and 0 otherwise.
- poly_lt_zero() : Calculation of the intervals for which the polynomial is less than or equal to 0.
- polytope_to_interval() : Converts a selection event given in quadratic form into truncated intervals.
- construct_projection_matrix() : Constructs a projection matrix from a basis given as a list of vectors to the subspace it spans.

## Others
Execute code test :
```
$ pytest tests/
```
