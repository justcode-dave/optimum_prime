# coding=utf-8
#
# FX Package Initialization
#
# This module initializes the FX (Functional Transformations) package, which provides a framework for transforming
# and optimizing models through graph manipulation. It integrates various techniques for model optimization and
# quantization, enabling more efficient execution of transformer models across diverse hardware platforms.
#
# Key Components:
# - `optimization`: A submodule that focuses on various optimization techniques such as graph transformations,
#   operator fusion, and other strategies that improve runtime performance, memory usage, and execution speed.
#
# - `quantization`: A submodule that implements quantization-aware training (QAT) and post-training quantization (PTQ),
#   allowing for the reduction of model precision (e.g., from FP32 to INT8) while maintaining accuracy.
#
# These tools are vital for users who want to push the performance boundaries of large-scale models when deploying
# them in constrained environments such as mobile devices or edge computing hardware.
#
# Example Usage:
#
# ```python
# from optimum.fx import optimization, quantization
#
# # Apply optimizations
# optimized_model = optimization.apply_optimizations(model)
#
# # Quantize a model
# quantized_model = quantization.quantize_model(model)
# ```
#

from . import optimization, quantization
