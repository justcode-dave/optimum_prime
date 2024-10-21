"""
This module provides a utility function to find fully connected layers in an ONNX model using ONNX Runtime's 
transformer optimizations. The primary function identifies layers that consist of a combination of `Add` and `MatMul` 
operations, which typically represent fully connected (dense) layers in neural networks.

Functions:

1. **find_fully_connected_layers_nodes(model: OnnxModel) -> List[List[str]]**:
    - This function identifies and returns the nodes corresponding to fully connected layers in the given ONNX model.
    - It searches for `Add` nodes and matches them with parent `MatMul` nodes. A combination of an `Add` followed by 
      a `MatMul` is commonly found in fully connected layers.
    - Parameters:
        - `model (OnnxModel)`: The ONNX model object to analyze.
    - Returns:
        - A list of fully connected layer node pairs, where each element is a list containing the `Add` node and its 
          corresponding parent `MatMul` node.

This function is useful for tasks like optimization, quantization, or analysis of ONNX models, particularly when working 
with models exported from deep learning frameworks.
"""

from typing import List

from onnxruntime.transformers.onnx_model import OnnxModel


def find_fully_connected_layers_nodes(model: OnnxModel) -> List[List[str]]:
    adds = model.get_nodes_by_op_type("Add")
    fc = list(filter(lambda graph: graph[1] is not None, ((add, model.match_parent(add, "MatMul")) for add in adds)))

    return fc
