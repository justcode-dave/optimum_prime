"""
This module provides distributed operations tailored for parallel execution within
a model's graph. These operations are essential for efficient data distribution and
aggregation across multiple devices (such as GPUs) in a distributed setup, and they
are differentiable, meaning they integrate smoothly with the backpropagation process.

The functions included allow for seamless gathering, reducing, scattering, and identity
operations across distributed processes, ensuring correct gradient propagation in training.

Exports:
    - differentiable_all_gather: Collects tensors from multiple devices while maintaining gradient flow.
    - differentiable_all_reduce_sum: Sums tensors from multiple devices and supports backpropagation.
    - differentiable_identity: A differentiable identity operation, typically used in distributed contexts.
    - differentiable_scatter: Distributes input data across devices, supporting gradient propagation.
    - scatter: Standard operation to distribute data across devices without gradients.

"""

from .dist_ops import (
    differentiable_all_gather,
    differentiable_all_reduce_sum,
    differentiable_identity,
    differentiable_scatter,
    scatter,
)
