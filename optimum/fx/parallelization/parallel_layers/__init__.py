"""
This module handles the initialization of core parallel layers and operations within the parallelization framework. 
It exposes key classes and functions related to parallelized versions of common neural network layers, 
such as embeddings, linear layers, and loss functions. These parallelized components enable the efficient 
distribution of large models across multiple devices, enhancing computational performance and scalability.

Imports:
    - VocabParallelEmbedding: Embedding layer that is distributed across multiple devices on the vocabulary dimension.
    - ColumnParallelLinear: A parallelized linear layer where the weight matrix is split column-wise across devices.
    - RowParallelLinear: A parallelized linear layer where the weight matrix is split row-wise across devices.
    - VocabParallelCrossEntropyLoss: A loss function tailored for models parallelized across the vocabulary dimension.
    - sharded_cross_entropy_wrapper_fn: A wrapper for cross-entropy loss to accommodate sharded vocabularies.

These components allow seamless integration of model parallelism in transformer models and other architectures, 
facilitating large-scale model training across multi-device setups.
"""

from .embedding import VocabParallelEmbedding
from .linear import ColumnParallelLinear, RowParallelLinear
from .loss import VocabParallelCrossEntropyLoss, sharded_cross_entropy_wrapper_fn
