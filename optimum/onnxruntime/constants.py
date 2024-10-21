"""
This module defines constants that are used as patterns for matching specific ONNX model file names. These patterns are 
useful for identifying the different parts of encoder-decoder models, such as encoder files, decoder files, decoders with 
past, and merged decoders during the export process in ONNX Runtime.

Constants:

1. **ENCODER_ONNX_FILE_PATTERN**: 
    - Regex pattern used to match ONNX files corresponding to the encoder part of a model.
    - Pattern: `r"(.*)?encoder(.*)?\.onnx"`

2. **DECODER_ONNX_FILE_PATTERN**: 
    - Regex pattern used to match ONNX files for the decoder part of a model, excluding those with "with_past" or "merged".
    - Pattern: `r"(.*)?decoder((?!(with_past|merged)).)*?\.onnx"`

3. **DECODER_WITH_PAST_ONNX_FILE_PATTERN**: 
    - Regex pattern used to identify ONNX decoder files that include "with_past" in the filename, indicating 
      the model supports past key/value cache for faster inference.
    - Pattern: `r"(.*)?decoder(.*)?with_past(.*)?\.onnx"`

4. **DECODER_MERGED_ONNX_FILE_PATTERN**: 
    - Regex pattern used to match ONNX decoder files that include "merged" in the filename, which implies that 
      the decoder has merged its past key/value support.
    - Pattern: `r"(.*)?decoder(.*)?merged(.*)?\.onnx"`

These patterns facilitate the proper organization and recognition of model files during export and inference, 
particularly for models that have separate encoder and decoder components.
"""


ENCODER_ONNX_FILE_PATTERN = r"(.*)?encoder(.*)?\.onnx"
DECODER_ONNX_FILE_PATTERN = r"(.*)?decoder((?!(with_past|merged)).)*?\.onnx"
DECODER_WITH_PAST_ONNX_FILE_PATTERN = r"(.*)?decoder(.*)?with_past(.*)?\.onnx"
DECODER_MERGED_ONNX_FILE_PATTERN = r"(.*)?decoder(.*)?merged(.*)?\.onnx"
