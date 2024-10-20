"""
TensorFlow Lite (TFLite) Export and Quantization Utilities.

This module provides the infrastructure for exporting Hugging Face models to TensorFlow Lite (TFLite) format, 
with support for model quantization. TFLite is a popular format for deploying machine learning models on 
edge devices such as mobile phones, microcontrollers, and embedded systems, where lightweight and optimized 
inference is critical.

Key functionalities of this module:
- **Model Export**: Tools to export models from the Hugging Face Transformers library into the TFLite format 
  for efficient on-device inference.
- **Quantization Configurations**: Classes to manage different quantization approaches and configurations, 
  allowing models to be optimized for reduced size and faster inference with minimal accuracy loss.
- **Lazy Loading**: Modules are loaded lazily to optimize memory usage and improve startup time. This is 
  particularly useful when the entire set of export tools is not required.

The `_import_structure` dictionary defines the structure of this module, specifying the available submodules 
and the key classes and functions they expose:
- **base**: Contains configurations for TFLite export and quantization strategies (e.g., `TFLiteQuantizationConfig`, 
  `QuantizationApproach`).
- **convert**: Handles the core logic for exporting models to TFLite and validating the outputs for correctness.

This structure allows users to only load the components they need, reducing memory footprint during execution.

Classes and Functions:
- `TFLiteQuantizationConfig`: Defines the quantization settings for TFLite models, including supported quantization 
  approaches like dynamic and post-training quantization.
- `export`: Exports a Hugging Face model to the TFLite format.
- `validate_model_outputs`: Verifies that the exported TFLite model produces valid outputs compared to the original 
  model, ensuring correctness.

This module is essential for developers looking to deploy transformer models on mobile and edge devices using 
the TensorFlow Lite framework.
"""


from typing import TYPE_CHECKING

from transformers.utils import _LazyModule


_import_structure = {
    "base": ["QuantizationApproach", "TFLiteQuantizationConfig", "TFLiteConfig"],
    "convert": ["export", "validate_model_outputs"],
}

if TYPE_CHECKING:
    from .base import QuantizationApproach, TFLiteQuantizationConfig, TFLiteConfig  # noqa
    from .convert import export, validate_model_outputs  # noqa
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
