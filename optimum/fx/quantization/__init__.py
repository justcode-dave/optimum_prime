"""
This module provides the interface for key quantization functionalities in the FX (Functional Transformations) 
submodule of `optimum`.

It re-exports the following functions:
- `fuse_fx`: A method to fuse supported operations (such as convolution and batch normalization layers) 
  to improve model efficiency, which is typically required before quantization.
- `prepare_fx`: Prepares a model for static quantization by inserting observers and other necessary components 
  to collect statistics for later quantization.
- `prepare_qat_fx`: Prepares a model for Quantization-Aware Training (QAT) by adding fake quantization operations 
  to simulate quantized inference during training.

These functions are built on top of PyTorch's FX API and are critical steps in the quantization process, 
helping models achieve efficient inference, especially in resource-constrained environments.
"""

from .functions import fuse_fx, prepare_fx, prepare_qat_fx
