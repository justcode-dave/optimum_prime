# coding=utf-8
#
# GPTQ Package Initialization
#
# This module initializes the GPTQ (Gradient-based Post-training Quantization) package, which provides tools
# for quantizing transformer models. The package includes efficient quantization techniques that help reduce
# the memory footprint and improve inference speed while maintaining accuracy. The key components exposed in
# this initialization file are:
#
# - `GPTQQuantizer`: A class that implements GPTQ-based quantization for transformer models. It enables users to
#   perform post-training quantization by adjusting the weights of models to lower precision formats.
#
# - `load_quantized_model`: A utility function for loading a pre-trained model that has already undergone
#   quantization, allowing for quick integration into inference pipelines.
#
# These functionalities are critical for deploying models in environments with limited resources, enabling the use
# of large-scale models on smaller devices without sacrificing performance.
#
# Example Usage:
#
# ```python
# from optimum.gptq import GPTQQuantizer, load_quantized_model
#
# # Quantize a model
# quantizer = GPTQQuantizer(model)
# quantized_model = quantizer.quantize()
#
# # Load a quantized model
# model = load_quantized_model("path_to_quantized_model")
# ```
#

from .quantizer import GPTQQuantizer, load_quantized_model
