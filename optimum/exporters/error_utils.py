# coding=utf-8
"""
This module provides utility classes for error handling within the optimum-export framework.

The following custom exceptions are defined:
- `ShapeError`: Raised when tensor shapes are incompatible or do not match expected values.
- `AtolError`: Raised when the absolute tolerance of numerical comparison is exceeded.
- `OutputMatchError`: Raised when model outputs do not match expected values.
- `NumberOfInputsMatchError`: Raised when the number of inputs to the model does not match the expected number.
- `NumberOfOutputsMatchError`: Raised when the number of outputs from the model does not match the expected number.
- `MinimumVersionError`: Raised when the version of a library or framework does not meet the minimum required version.
"""



class ShapeError(ValueError):
    pass


class AtolError(ValueError):
    pass


class OutputMatchError(ValueError):
    pass


class NumberOfInputsMatchError(ValueError):
    pass


class NumberOfOutputsMatchError(ValueError):
    pass


class MinimumVersionError(ValueError):
    pass
