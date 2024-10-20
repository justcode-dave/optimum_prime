"""
This module provides functionalities to enable and manage model parallelization 
within the `optimum.fx` framework. It includes key tools to facilitate the 
parallelization of models and backends, ensuring efficient execution across 
multiple devices or distributed systems.

Key Components:
    - `parallelize_backend`: A function to parallelize the backend operations.
    - `parallelize_model`: A function to enable parallelization of models across
      multiple devices or nodes.
    - `Config`: A configuration class to manage and store parallelization settings.
    - `ParallelExecutionCtx`: A context manager to handle parallel execution, 
      ensuring correct resource allocation and synchronization across devices.
"""

from .api import parallelize_backend, parallelize_model
from .core import Config, ParallelExecutionCtx
