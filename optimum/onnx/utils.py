"""
Utility Functions for Handling ONNX External Data and Model Operations.

This module provides utilities to manage ONNX models, especially focusing on models that make use of external data.
It includes functions for extracting, verifying, and managing external data paths from ONNX models, as well as utility
methods to check the presence of specific inputs in models.

Main features:
    - **External Data Management**: Functions to retrieve paths to external data referenced by the ONNX model, check if a model uses external data, and help in managing and copying the necessary external files.
    - **Input Checking**: Utility to check whether a specific input exists in the ONNX model, useful for model verification and preparation tasks.
    - **Model Loading without External Data**: Methods to load ONNX models while ignoring external data, providing flexibility for custom manipulation of model tensors.
    - **Helper for ONNX Initializers**: Functionality to assist with accessing and handling ONNX initializer tensors.

These utilities are designed to facilitate working with large ONNX models that may have external data dependencies and ensure smooth handling of ONNX inputs and graph nodes.

Example usage:
    These functions are typically used when preparing ONNX models for deployment, where external data may need to be managed or verified, 
    or when ensuring that models have the correct input configuration for specific tasks or hardware accelerators.
"""


from pathlib import Path
from typing import List, Tuple, Union

import onnx
from onnx.external_data_helper import ExternalDataInfo, _get_initializer_tensors


def _get_onnx_external_constants(model: onnx.ModelProto) -> List[str]:
    external_constants = []

    for node in model.graph.node:
        if node.op_type == "Constant":
            for attribute in node.attribute:
                external_datas = attribute.t.external_data
                for external_data in external_datas:
                    external_constants.append(external_data.value)

    return external_constants


def _get_onnx_external_data_tensors(model: onnx.ModelProto) -> List[str]:
    """
    Gets the paths of the external data tensors in the model.
    Note: make sure you load the model with load_external_data=False.
    """
    model_tensors = _get_initializer_tensors(model)
    model_tensors_ext = [
        ExternalDataInfo(tensor).location
        for tensor in model_tensors
        if tensor.HasField("data_location") and tensor.data_location == onnx.TensorProto.EXTERNAL
    ]
    return model_tensors_ext


def _get_external_data_paths(src_paths: List[Path], dst_paths: List[Path]) -> Tuple[List[Path], List[str]]:
    """
    Gets external data paths from the model and add them to the list of files to copy.
    """
    model_paths = src_paths.copy()
    for idx, model_path in enumerate(model_paths):
        model = onnx.load(str(model_path), load_external_data=False)
        model_tensors = _get_initializer_tensors(model)
        # filter out tensors that are not external data
        model_tensors_ext = [
            ExternalDataInfo(tensor).location
            for tensor in model_tensors
            if tensor.HasField("data_location") and tensor.data_location == onnx.TensorProto.EXTERNAL
        ]
        if len(set(model_tensors_ext)) == 1:
            # if external data was saved in a single file
            src_paths.append(model_path.parent / model_tensors_ext[0])
            dst_paths.append(dst_paths[idx].parent / model_tensors_ext[0])
        else:
            # if external data doesnt exist or was saved in multiple files
            src_paths.extend([model_path.parent / tensor_name for tensor_name in model_tensors_ext])
            dst_paths.extend(dst_paths[idx].parent / tensor_name for tensor_name in model_tensors_ext)
    return src_paths, dst_paths


def _get_model_external_data_paths(model_path: Path) -> List[Path]:
    """
    Gets external data paths from the model.
    """

    onnx_model = onnx.load(str(model_path), load_external_data=False)
    model_tensors = _get_initializer_tensors(onnx_model)
    # filter out tensors that are not external data
    model_tensors_ext = [
        ExternalDataInfo(tensor).location
        for tensor in model_tensors
        if tensor.HasField("data_location") and tensor.data_location == onnx.TensorProto.EXTERNAL
    ]
    return [model_path.parent / tensor_name for tensor_name in model_tensors_ext]


def check_model_uses_external_data(model: onnx.ModelProto) -> bool:
    """
    Checks if the model uses external data.
    """
    model_tensors = _get_initializer_tensors(model)
    return any(
        tensor.HasField("data_location") and tensor.data_location == onnx.TensorProto.EXTERNAL
        for tensor in model_tensors
    )


def has_onnx_input(model: Union[onnx.ModelProto, Path, str], input_name: str) -> bool:
    """
    Checks if the model has a specific input.
    """
    if isinstance(model, (str, Path)):
        model = Path(model).as_posix()
        model = onnx.load(model, load_external_data=False)

    for input in model.graph.input:
        if input.name == input_name:
            return True
    return False
