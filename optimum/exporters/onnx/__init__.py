# Utilities for ONNX configuration, conversion, and export

"""
This module provides lazy-loaded utilities to handle ONNX exportation of models within the Hugging Face ecosystem.
It defines structures and methods that facilitate configuring and converting model architectures to the ONNX format.

The ONNX export functionality is designed to improve the deployment and integration of transformer-based models
by converting them into a format that can be executed in a broad range of environments, including mobile devices,
cloud platforms, and edge applications.

The key functionalities include:
- Configuring the model for export to ONNX, including loss functions, attention mechanisms, and sequence-to-sequence support.
- Conversion of pre-trained transformer models into ONNX format using specified configurations.
- Validation of model outputs to ensure consistency and correctness after conversion.
- Support for exporting models designed for text generation, text encoding, and other tasks specific to transformer models.
- Providing utilities to extract submodels (e.g., encoder-decoder) from more complex architectures for export.

Components:
- `base`: Contains the base classes and configurations used for exporting models to ONNX, such as `OnnxConfig`, `OnnxConfigWithLoss`, and others.
- `config`: Provides specific configurations for different types of transformer models, including text decoders, encoders, and sequence-to-sequence models.
- `convert`: Offers the core functions for exporting models to ONNX format, as well as functions to validate the exported models.
- `utils`: Includes utility functions to facilitate model preparation for ONNX export, including support for various model types.
- `__main__`: Command-line utilities for exporting models, intended to be run as standalone scripts for model export tasks.

This module is designed to be lazily loaded, meaning that individual components are only loaded when they are accessed. 
This reduces the initial load time and resource consumption when the full suite of utilities is not required.
"""

from typing import TYPE_CHECKING

from transformers.utils import _LazyModule

_import_structure = {
    "base": ["OnnxConfig", "OnnxConfigWithLoss", "OnnxConfigWithPast", "OnnxSeq2SeqConfigWithPast"],
    "config": ["TextDecoderOnnxConfig", "TextEncoderOnnxConfig", "TextSeq2SeqOnnxConfig"],
    "convert": [
        "export",
        "export_models",
        "validate_model_outputs",
        "validate_models_outputs",
        "onnx_export_from_model",
    ],
    "utils": [
        "get_decoder_models_for_export",
        "get_encoder_decoder_models_for_export",
        "get_diffusion_models_for_export",
        "MODEL_TYPES_REQUIRING_POSITION_IDS",
    ],
    "__main__": ["main_export"],
}

if TYPE_CHECKING:
    from .base import OnnxConfig, OnnxConfigWithLoss, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast  # noqa
    from .config import TextDecoderOnnxConfig, TextEncoderOnnxConfig, TextSeq2SeqOnnxConfig  # noqa
    from .convert import (
        export,
        export_models,
        validate_model_outputs,
        validate_models_outputs,
        onnx_export_from_model,
    )  # noqa
    from .utils import (
        get_decoder_models_for_export,
        get_encoder_decoder_models_for_export,
        get_diffusion_models_for_export,
        MODEL_TYPES_REQUIRING_POSITION_IDS,
    )
    from .__main__ import main_export
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
