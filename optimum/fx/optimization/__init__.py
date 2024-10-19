"""
This module provides a variety of transformations aimed at optimizing models for better performance.
These transformations focus on fusing certain layers and altering operations to reduce complexity,
increase efficiency, and potentially improve inference speed. The transformations can be composed
together to apply multiple optimizations in sequence.

Available Transformations:
--------------------------

- **ChangeTrueDivToMulByInverse**: Replaces division operations with multiplication by the inverse to optimize performance.
- **FuseBatchNorm1dInLinear**: Fuses 1D Batch Normalization into Linear layers for optimization.
- **FuseBatchNorm2dInConv2d**: Fuses 2D Batch Normalization into Convolution layers, reducing redundant operations.
- **FuseBiasInLinear**: Fuses bias terms directly into Linear layers, simplifying the computation graph.
- **MergeLinears**: Merges consecutive Linear layers into a single operation to reduce overhead.
- **ReversibleTransformation**: Provides a framework for reversible transformations, enabling efficient optimization that can be reverted if necessary.
- **Transformation**: Base class for implementing various transformations.
- **compose**: Utility to compose multiple transformations into a single optimization pass.

Usage:
------
Import the desired transformations and apply them to your model using the `compose` function for sequential optimizations.

Example:
    from .transformations import ChangeTrueDivToMulByInverse, compose

    optimized_model = compose(ChangeTrueDivToMulByInverse(), another_transformation)(model)

"""

from .transformations import (  # noqa
    ChangeTrueDivToMulByInverse,
    FuseBatchNorm1dInLinear,
    FuseBatchNorm2dInConv2d,
    FuseBiasInLinear,
    MergeLinears,
    ReversibleTransformation,
    Transformation,
    compose,
)
