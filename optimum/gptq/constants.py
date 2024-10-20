# coding=utf-8
#
# GPTQ Constants
#
# This module defines constant values and patterns that are used across the GPTQ (Quantization) package for
# configuring and handling model transformations. These constants ensure consistency and reduce the need for
# hardcoded values when working with sequence lengths, model architecture patterns, and configuration files
# related to GPTQ-based quantization.
#
# Constants:
# - `SEQLEN_KEYS_TRANFORMERS`: List of sequence length keys commonly found in transformer-based models. These keys
#   are used to identify and manipulate sequence-related parameters such as the maximum number of positions a model
#   can handle.
#
# - `BLOCK_PATTERNS`: List of common layer/block patterns within transformer models. This is useful for identifying
#   key architectural components during model parsing or transformation.
#
# - `GPTQ_CONFIG`: The default filename for the GPTQ quantization configuration file. This file contains settings
#   specific to quantizing a given model.
#
# Example Usage:
#
# ```python
# from .constants import SEQLEN_KEYS_TRANFORMERS, BLOCK_PATTERNS, GPTQ_CONFIG
# # Access predefined sequence length keys and layer patterns for transformation.
# ```
#

SEQLEN_KEYS_TRANFORMERS = ["max_position_embeddings", "seq_length", "n_positions"]
BLOCK_PATTERNS = [
    "transformer.h",
    "model.decoder.layers",
    "gpt_neox.layers",
    "model.layers",
]

GPTQ_CONFIG = "quantize_config.json"
