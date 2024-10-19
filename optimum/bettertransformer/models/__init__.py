"""
This module defines the mapping and management of supported model architectures for 
the BetterTransformer implementation in Optimum. BetterTransformer optimizes Hugging Face 
Transformers models for faster inference using PyTorch’s efficient implementations.

The `BetterTransformerManager` class maps supported model types to their respective 
BetterTransformer attention layer implementations. It also handles model-specific 
overrides, exclusions, and unsupported cases. Additionally, this module provides utilities 
for checking if a model can be converted to BetterTransformer and handles 
warning messages for unsupported saving operations.

Imports attention layers and encoder models required for BetterTransformer to perform 
replacements on transformer models during the conversion process.
"""

import warnings

from ...utils.import_utils import check_if_transformers_greater
from .decoder_models import (
    BarkAttentionLayerBetterTransformer,
    BartAttentionLayerBetterTransformer,
    BlenderbotAttentionLayerBetterTransformer,
    BloomAttentionLayerBetterTransformer,
    CodegenAttentionLayerBetterTransformer,
    GPT2AttentionLayerBetterTransformer,
    GPTJAttentionLayerBetterTransformer,
    GPTNeoAttentionLayerBetterTransformer,
    GPTNeoXAttentionLayerBetterTransformer,
    M2M100AttentionLayerBetterTransformer,
    MarianAttentionLayerBetterTransformer,
    OPTAttentionLayerBetterTransformer,
    PegasusAttentionLayerBetterTransformer,
    T5AttentionLayerBetterTransformer,
)
from .encoder_models import (
    AlbertLayerBetterTransformer,
    BartEncoderLayerBetterTransformer,
    BertLayerBetterTransformer,
    CLIPLayerBetterTransformer,
    DistilBertLayerBetterTransformer,
    FSMTEncoderLayerBetterTransformer,
    MBartEncoderLayerBetterTransformer,
    ProphetNetEncoderLayerBetterTransformer,
    ViltLayerBetterTransformer,
    ViTLayerBetterTransformer,
    Wav2Vec2EncoderLayerBetterTransformer,
)


class BetterTransformerManager:
    """
    Manages the mapping of Hugging Face model types to their respective 
    BetterTransformer-compatible layers for faster inference.

    This class provides static methods for checking if a model is supported by BetterTransformer,
    whether it requires nested tensors, and if strict validation is necessary. 
    It also holds information about models that are explicitly excluded from the transformation process.

    Attributes:
        MODEL_MAPPING (dict): Maps model types to their respective BetterTransformer layers.
        OVERWRITE_METHODS (dict): Defines methods that need to be overwritten for certain models during transformation.
        EXCLUDE_FROM_TRANSFORM (dict): Models or submodules explicitly excluded from BetterTransformer transformation.
        CAN_NOT_BE_SUPPORTED (dict): Lists model types that cannot be supported by BetterTransformer.
        NOT_REQUIRES_NESTED_TENSOR (set): Models that do not require nested tensors for their BetterTransformer implementation.
        NOT_REQUIRES_STRICT_VALIDATION (set): Models that do not require strict validation checks.
    """

    MODEL_MAPPING = {
        # Defines model type mappings to BetterTransformer layers
        "albert": {"AlbertLayer": AlbertLayerBetterTransformer},
        "bark": {"BarkSelfAttention": BarkAttentionLayerBetterTransformer},
        "bart": {
            "BartEncoderLayer": BartEncoderLayerBetterTransformer,
            "BartAttention": BartAttentionLayerBetterTransformer,
        },
        # Additional models mapped to BetterTransformer layers
        # ...
    }

    OVERWRITE_METHODS = {
        # Defines specific methods that need to be overwritten for certain models (if applicable)
    }

    EXCLUDE_FROM_TRANSFORM = {
        # Defines models or submodules explicitly excluded from the BetterTransformer transformation process
    }

    CAN_NOT_BE_SUPPORTED = {
        # Lists models that cannot be supported by BetterTransformer
        "deberta-v2": "DeBERTa v2 does not use a regular attention mechanism, which is not supported in PyTorch's BetterTransformer.",
        "glpn": "GLPN has a convolutional layer in the FFN network, which is not supported by BetterTransformer.",
    }

    NOT_REQUIRES_NESTED_TENSOR = {
        # Models that do not require nested tensors for BetterTransformer
    }

    NOT_REQUIRES_STRICT_VALIDATION = {
        # Models that do not require strict validation checks
    }

    @staticmethod
    def cannot_support(model_type: str) -> bool:
        """
        Checks whether a given model type cannot be supported by PyTorch’s BetterTransformer.

        Args:
            model_type (str): The model type to check.

        Returns:
            bool: True if the model type is unsupported, otherwise False.
        """
        return model_type in BetterTransformerManager.CAN_NOT_BE_SUPPORTED

    @staticmethod
    def supports(model_type: str) -> bool:
        """
        Checks if a given model type is supported by BetterTransformer and integrated into Optimum.

        Args:
            model_type (str): The model type to check.

        Returns:
            bool: True if the model type is supported, otherwise False.
        """
        return model_type in BetterTransformerManager.MODEL_MAPPING

    @staticmethod
    def requires_nested_tensor(model_type: str) -> bool:
        """
        Determines whether the BetterTransformer implementation for a model type requires nested tensors.

        Args:
            model_type (str): The model type to check.

        Returns:
            bool: True if nested tensors are required, otherwise False.
        """
        return model_type not in BetterTransformerManager.NOT_REQUIRES_NESTED_TENSOR

    @staticmethod
    def requires_strict_validation(model_type: str) -> bool:
        """
        Determines whether a model type requires strict validation during BetterTransformer conversion.

        Args:
            model_type (str): The model type to check.

        Returns:
            bool: True if strict validation is required, otherwise False.
        """
        return model_type not in BetterTransformerManager.NOT_REQUIRES_STRICT_VALIDATION


class warn_uncompatible_save(object):
    """
    A decorator to emit a warning when calling `save_pretrained` on a model 
    transformed using BetterTransformer. The warning informs the user that unexpected 
    behavior may occur when saving a BetterTransformer model without reverting it.
    """

    def __init__(self, callback):
        self.callback = callback

    def __enter__(self):
        return self

    def __exit__(self, ex_typ, ex_val, traceback):
        return True

    def __call__(self, *args, **kwargs):
        warnings.warn(
            "You are calling `save_pretrained` on a `BetterTransformer`-converted model, which may cause unexpected behavior.",
            UserWarning,
        )
        return self.callback(*args, **kwargs)
