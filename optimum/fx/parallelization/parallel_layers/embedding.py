"""
This module defines a parallelized version of the embedding layer, specifically designed to distribute the embedding 
matrix across multiple devices by dividing it along the vocabulary dimension. This enables efficient training of 
large-scale models with embeddings that exceed single-device memory limits. 

Key Components:
    - VocabParallelEmbedding: 
        A parallelized `torch.nn.Embedding` layer that partitions the embedding weights across devices. 
        This class modifies the embedding's internal representation to handle distribution across multiple 
        tensor parallel devices and ensures correct parameter initialization and distribution.

Features:
    - Vocabulary partitioning: The embedding matrix is split across the vocabulary dimension, with each 
      device handling a subset of the total embeddings.
    - Efficient gradient synchronization: Gradients are aggregated across devices using differentiable 
      collective communication operations.
    - Meta information modification: Updates metadata of embedding weights to handle parameter initialization 
      and tracking of the parallelization strategy.
    
Usage:
    This class is instantiated by replacing an existing `torch.nn.Embedding` layer in a model during the 
    parallelization setup and is integrated into the overall parallelism framework.
    
Imports:
    - torch: PyTorch core library for tensor computations.
    - torch.distributed: For handling distributed process groups and communication.
    - torch.nn: Provides the base `torch.nn.Module` class and neural network layers.
    - torch.nn.functional: Provides functional operations like embedding lookups.
    - core, distributed, utils: Internal modules for handling parallel execution contexts, distributed 
      communication, and utility functions.

"""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from ..core import ParallelExecutionCtx, ParameterMeta
from ..distributed import differentiable_all_reduce_sum
from ..utils import ensure_divisibility


class VocabParallelEmbedding(nn.Module):
    """
    Embedding layer parallelized in vocabulary dimension.

    Arguments:
        ctx(`ParallelExecutionCtx`): parallel execution context which contains runtime information.
        embedding(`torch.nn.Embedding`): the original embedding module being replaced.
    """

    def __init__(self, ctx: ParallelExecutionCtx, embedding: nn.Embedding):
        super(VocabParallelEmbedding, self).__init__()
        self.process_group = ctx.tp_group
        world_size = dist.get_world_size(self.process_group)
        tp_rank = dist.get_rank(self.process_group)
        ensure_divisibility(embedding.num_embeddings, world_size)

        num_embeddings = embedding.num_embeddings // world_size

        self.padding_idx = embedding.padding_idx
        self.max_norm = embedding.max_norm
        self.norm_type = embedding.norm_type
        self.scale_grad_by_freq = embedding.scale_grad_by_freq
        self.sparse = embedding.sparse
        self.vocab_start_idx = tp_rank * num_embeddings
        self.vocab_end_idx = (tp_rank + 1) * num_embeddings

        # modify meta information
        weight_meta = getattr(embedding.weight, "meta", None)
        assert isinstance(
            weight_meta, ParameterMeta
        ), "should have run `initialize_parameter_meta` after moving model to current device"
        if weight_meta.is_modified_meta:
            assert weight_meta.is_tied, "only tied parameters could already have modified meta"
        else:
            weight_meta.need_initialize = True
            weight_meta.is_parallel = True
            weight_meta.dim = 0
            for _, Slice in weight_meta.mapping.items():
                Slice.index = slice(self.vocab_start_idx, self.vocab_end_idx)
            weight_meta.is_modified_meta = True

        # skip creating actual parameters
        self.weight = embedding.weight

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input_mask = (input < self.vocab_start_idx) | (input >= self.vocab_end_idx)
        masked_input = input.clone() - self.vocab_start_idx
        masked_input[input_mask] = 0

        output = F.embedding(
            masked_input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

        output[input_mask, :] = 0.0
        output = differentiable_all_reduce_sum(output, self.process_group)
        return output
