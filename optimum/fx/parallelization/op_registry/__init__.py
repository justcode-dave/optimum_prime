"""
This module initializes the operation registry and handlers for parallelization.

It provides:
    - `REGISTRY`: A registry mapping operations (such as PyTorch functions or layers) to their respective handlers
      responsible for propagating parallelization strategies during execution.
    - `FallbackParallelAxisPropagateHandler`: A fallback handler that is used for operations without specialized
      handlers, ensuring that parallelization propagation can still occur with a default behavior.

These components are essential for defining how different operators propagate their parallelization axes within the
parallelized computation graph.

Modules:
    - `op_handlers.py`: Contains the core definitions for handling parallel axis propagation for different
      operators.

Exports:
    - `REGISTRY`: The registry object that maps supported operations to their handlers.
    - `FallbackParallelAxisPropagateHandler`: A fallback handler used when no specific handler exists for an operation.
"""

from .op_handlers import REGISTRY, FallbackParallelAxisPropagateHandler
