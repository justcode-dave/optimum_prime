"""
This module defines the base class for integrating models with the BetterTransformer framework, providing functionality 
for validating model configurations and applying transformations to improve inference performance. 

The `BetterTransformerBaseLayer` class serves as a base layer for transforming specific layers or attributes of 
pretrained models to optimize their performance with PyTorch's BetterTransformer enhancements. 

This module handles:
- Activation function selection and validation for model layers.
- Validation of essential parameters such as embedding dimensions, attention heads, and normalization constants.
- Reverting the BetterTransformer-optimized model back to its original architecture, ensuring compatibility for saving or fine-tuning.

The transformation process checks for supported activation functions, position embeddings, and layer mappings, 
allowing users to harness the efficiency of BetterTransformer without modifying the core model design.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PretrainedConfig

import torch
from ...utils import logging, recurse_getattr, recurse_setattr

# Known activation, positional embedding, and layer attributes to check during transformation
KNOWN_ACTIVATION_ATTRIBUTES = ["hidden_act", "activation", "act_fn", "activation_function"]
KNOWN_POS_EMB_ATTRIBUTES = ["position_embedding_type"]
KNOWN_NUM_LAYERS = ["num_hidden_layers", "num_layers", "encoder_layers", "n_layers"]

# Supported and risky activation functions for BetterTransformer
SUPPORTED_ACTIVATION_FUNCTIONS = ["gelu", "relu", "gelu_new"]
USE_AT_OWN_RISK_ACTIVATION_FUNCTIONS = ["quick_gelu"]

# Set up logger
logger = logging.get_logger(__name__)

class BetterTransformerBaseLayer:
    """
    Base layer for BetterTransformer integration. This class wraps necessary components for 
    transforming a model to work with BetterTransformer.

    Args:
        config (`transformers.PretrainedConfig`):
            The configuration of the pretrained model.
    """
    
    def __init__(self, config: "PretrainedConfig"):
        self.norm_first = False
        self.use_gelu = False
        self.act_fn = None
        self.pos_emb_type = None
        self.num_heads = None
        self.embed_dim = None
        self.num_layers = None
        self.original_layers_mapping = {}
        self.module_mapping = None
        self.keys_to_ignore = []  # Attributes to ignore if not present in the model

        # Determine activation function from config
        for attr in KNOWN_ACTIVATION_ATTRIBUTES:
            if hasattr(config, attr):
                self.act_fn = getattr(config, attr)
                break

        # Fallback to private method if activation not found in config
        if self.act_fn is None and hasattr(self, "_get_activation_function"):
            self.act_fn = self._get_activation_function(config)

        # Determine positional embedding type from config
        for attr in KNOWN_POS_EMB_ATTRIBUTES:
            if hasattr(config, attr):
                self.pos_emb_type = getattr(config, attr)
                break

        # Determine the number of layers in the model from config
        for attr in KNOWN_NUM_LAYERS:
            if hasattr(config, attr):
                self.num_layers = getattr(config, attr)
                break

    def validate_bettertransformer(self):
        """
        Validate the BetterTransformer integration for the current model configuration.
        This function performs checks on model parameters, including the number of heads,
        embedding dimensions, positional embeddings, and activation functions, ensuring compatibility.
        """
        
        # Ensure num_heads, embed_dim, and normalization are set
        if self.num_heads is None:
            raise ValueError("Number of heads not set for BetterTransformer integration.")
        if self.embed_dim is None:
            raise ValueError("Embedding dimension not set for BetterTransformer integration.")
        if self.norm2_eps is None or self.norm1_eps is None:
            raise ValueError("norm2_eps and norm1_eps not set for BetterTransformer integration.")
        
        # Check positional embedding type is supported
        if self.pos_emb_type is not None and self.pos_emb_type != "absolute":
            raise ValueError(
                f"Positional embedding type {self.pos_emb_type} is not supported for BetterTransformer integration."
            )
        
        # Ensure norm1 and norm2 epsilons are equal
        if self.norm1_eps != self.norm2_eps:
            raise ValueError("norm1_eps and norm2_eps must be equal for BetterTransformer integration.")
        
        # Check if activation function is supported
        if self.act_fn in USE_AT_OWN_RISK_ACTIVATION_FUNCTIONS:
            logger.warning(
                f"Overriding {self.act_fn} activation with gelu. The output logits could differ significantly. Use with caution."
            )
            self.act_fn = "gelu"
        elif self.act_fn not in SUPPORTED_ACTIVATION_FUNCTIONS:
            raise ValueError(f"Activation function {self.act_fn} is not supported for BetterTransformer integration.")
        self.use_gelu = (self.act_fn == "gelu") or (self.act_fn == "gelu_new")
        
        # Ensure number of heads is even
        if self.num_heads % 2 == 1:
            raise ValueError(
                f"Number of heads {self.num_heads} is not supported for BetterTransformer integration. Must be even."
            )

    def _revert(self, module: torch.nn.Module) -> torch.nn.Module:
        """
        Revert a transformed module back to its original state by restoring its parameters 
        and attributes to their initial configurations.

        Args:
            module (`torch.nn.Module`):
                The transformed module to revert.

        Returns:
            `torch.nn.Module`: The original module with restored attributes.
        """
        if self.module_mapping is not None:
            if "" in self.module_mapping.values():
                for bt_module_attr_name, value in self.module_mapping.items():
                    if value == "":
                        module = getattr(self, bt_module_attr_name)
                        return module
            else:
                raise NotImplementedError("Reverting submodules in module_mapping is not supported.")

        # Revert the modified layers based on the original layer mappings
        for modified_layer_key_names, original_layer_key_names in self.original_layers_mapping.items():
            if isinstance(original_layer_key_names, list):
                current_weight = getattr(self, modified_layer_key_names)
                split_index = current_weight.shape[0] // len(original_layer_key_names)

                for i, subparam_name in enumerate(original_layer_key_names):
                    if recurse_getattr(module, subparam_name) is None:
                        continue
                    if module not in self.keys_to_ignore:
                        parameter = current_weight[i * split_index : (i + 1) * split_index].clone()
                        if isinstance(recurse_getattr(module, subparam_name), torch.nn.Parameter):
                            parameter = torch.nn.Parameter(parameter)
                        recurse_setattr(module, subparam_name, parameter)
            elif isinstance(original_layer_key_names, str):
                if recurse_getattr(module, original_layer_key_names) is None:
                    continue
                parameter = getattr(self, modified_layer_key_names)
                if isinstance(recurse_getattr(module, original_layer_key_names), torch.nn.Parameter):
                    parameter = torch.nn.Parameter(parameter)
                recurse_setattr(module, original_layer_key_names, parameter)
            else:
                raise ValueError(
                    f"Invalid type {type(modified_layer_key_names)} for `original_layers_mapping`. "
                    "Please use either `str` or `list`."
                )
        return module
