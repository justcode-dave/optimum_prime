"""
Utils Initialization Module for Promise Optimizer

This module serves as the entry point for all utility functions and constants used across the 
Promise Optimizer (Optimum) core. The utilities provided here include constants, input generators, version 
and dependency checks, and utility functions for model configuration and manipulation. These utilities 
are essential for ensuring that the various submodules within Promise Optimizer are able to work 
together seamlessly and efficiently.

The module imports the following:
1. Constants for configuration and model subfolders.
2. Utility functions for version checking and dependency management.
3. Dummy input generators for handling different model input types in testing and development.
4. Helper functions for recursively accessing and setting attributes.
5. Normalized configuration classes for consistent handling of model configurations across different architectures.
"""

# Importing constants for configuration and model subfolders
from .constant import (
    CONFIG_NAME,  # Default configuration file name
    DIFFUSION_MODEL_TEXT_ENCODER_2_SUBFOLDER,  # Subfolder for text encoder 2 in diffusion models
    DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER,  # Subfolder for text encoder in diffusion models
    DIFFUSION_MODEL_UNET_SUBFOLDER,  # Subfolder for UNet in diffusion models
    DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER,  # Subfolder for VAE decoder in diffusion models
    DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER,  # Subfolder for VAE encoder in diffusion models
    ONNX_WEIGHTS_NAME,  # Default file name for ONNX weights
)

# Importing utilities for checking version compatibility and available dependencies
from .import_utils import (
    DIFFUSERS_MINIMUM_VERSION,  # Minimum required version for diffusers
    ORT_QUANTIZE_MINIMUM_VERSION,  # Minimum required version for ONNX Runtime quantization
    TORCH_MINIMUM_VERSION,  # Minimum required version for PyTorch
    TRANSFORMERS_MINIMUM_VERSION,  # Minimum required version for Hugging Face Transformers
    check_if_diffusers_greater,  # Function to check if the installed diffusers version is greater than required
    check_if_pytorch_greater,  # Function to check if the installed PyTorch version is greater than required
    check_if_transformers_greater,  # Function to check if the installed Transformers version is greater than required
    is_accelerate_available,  # Check if accelerate package is available
    is_auto_gptq_available,  # Check if Auto GPTQ is available
    is_diffusers_available,  # Check if diffusers package is available
    is_onnx_available,  # Check if ONNX is available
    is_onnxruntime_available,  # Check if ONNX Runtime is available
    is_pydantic_available,  # Check if Pydantic is available
    is_sentence_transformers_available,  # Check if sentence-transformers is available
    is_timm_available,  # Check if TIMM (PyTorch image models) is available
    is_torch_onnx_support_available,  # Check if PyTorch has ONNX support enabled
    require_numpy_strictly_lower,  # Check if NumPy version is strictly lower than a given version
    torch_version,  # Get the current PyTorch version
)

# Importing dummy input generators for testing models with various input types
from .input_generators import (
    DEFAULT_DUMMY_SHAPES,  # Default shapes for dummy inputs
    DTYPE_MAPPER,  # Maps data types to corresponding Torch types
    BloomDummyPastKeyValuesGenerator,  # Dummy generator for Bloom model past key values
    DummyAudioInputGenerator,  # Dummy generator for audio inputs
    DummyBboxInputGenerator,  # Dummy generator for bounding box inputs
    DummyCodegenDecoderTextInputGenerator,  # Dummy generator for Codegen decoder text inputs
    DummyDecoderTextInputGenerator,  # Dummy generator for decoder text inputs
    DummyEncodecInputGenerator,  # Dummy generator for Encodec inputs
    DummyInputGenerator,  # General-purpose dummy input generator
    DummyIntGenerator,  # Dummy generator for integer inputs
    DummyLabelsGenerator,  # Dummy generator for label inputs
    DummyPastKeyValuesGenerator,  # Dummy generator for past key values
    DummyPix2StructInputGenerator,  # Dummy generator for Pix2Struct model inputs
    DummyPointsGenerator,  # Dummy generator for points-based inputs
    DummySeq2SeqDecoderTextInputGenerator,  # Dummy generator for Seq2Seq decoder text inputs
    DummySeq2SeqPastKeyValuesGenerator,  # Dummy generator for Seq2Seq past key values
    DummySpeechT5InputGenerator,  # Dummy generator for SpeechT5 inputs
    DummyTextInputGenerator,  # Dummy generator for text inputs
    DummyTimestepInputGenerator,  # Dummy generator for timestep inputs
    DummyVisionEmbeddingsGenerator,  # Dummy generator for vision embeddings
    DummyVisionEncoderDecoderPastKeyValuesGenerator,  # Dummy generator for vision encoder-decoder past key values
    DummyVisionInputGenerator,  # Dummy generator for vision inputs
    DummyXPathSeqInputGenerator,  # Dummy generator for XPath sequence inputs
    FalconDummyPastKeyValuesGenerator,  # Dummy generator for Falcon model past key values
    GemmaDummyPastKeyValuesGenerator,  # Dummy generator for Gemma model past key values
    GPTBigCodeDummyPastKeyValuesGenerator,  # Dummy generator for GPT BigCode model past key values
    MistralDummyPastKeyValuesGenerator,  # Dummy generator for Mistral model past key values
    MultiQueryPastKeyValuesGenerator,  # Dummy generator for MultiQuery model past key values
)

# Importing utility functions for recursively getting and setting attributes in model configurations
from .modeling_utils import recurse_getattr, recurse_setattr

# Importing normalized configurations for various model architectures
from .normalized_config import (
    NormalizedConfig,  # Base class for normalized configurations
    NormalizedConfigManager,  # Manager class for handling normalized configurations
    NormalizedEncoderDecoderConfig,  # Normalized configuration for encoder-decoder models
    NormalizedSeq2SeqConfig,  # Normalized configuration for sequence-to-sequence models
    NormalizedTextAndVisionConfig,  # Normalized configuration for text and vision models
    NormalizedTextConfig,  # Normalized configuration for text-based models
    NormalizedTextConfigWithGQA,  # Normalized configuration for text models with GQA (Generalized Question Answering)
    NormalizedVisionConfig,  # Normalized configuration for vision models
)
