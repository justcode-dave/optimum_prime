"""
This module defines constants used throughout the ONNX export process for Hugging Face models.

These constants include:
- **File size limits**: For managing external data formats during the ONNX export process.
- **Model component names**: Standard names for encoder and decoder parts of models during export.
- **Model architecture lists**:
  - **Unpickable architectures**: A list of models that cannot be exported due to specific technical constraints.
  - **Architectures unsupported for ONNX export with SDPA**: A list of model architectures where the 
    ONNX export process is not supported due to limitations with specific attention mechanisms.

These constants help ensure consistency and prevent issues during the ONNX export process for various model 
architectures.
"""


# 2 GB
EXTERNAL_DATA_FORMAT_SIZE_LIMIT = 2 * 1024 * 1024 * 1024

ONNX_ENCODER_NAME = "encoder_model"
ONNX_DECODER_NAME = "decoder_model"
ONNX_DECODER_WITH_PAST_NAME = "decoder_with_past_model"
ONNX_DECODER_MERGED_NAME = "decoder_model_merged"

UNPICKABLE_ARCHS = [
    "encodec",
    "hubert",
    "sew",
    "sew-d",
    "speecht5",
    "unispeech",
    "unispeech-sat",
    "wav2vec2",
    "wav2vec2-conformer",
    "wavlm",
]

SDPA_ARCHS_ONNX_EXPORT_NOT_SUPPORTED = [
    "bart",
    "musicgen",
    "whisper",
]
