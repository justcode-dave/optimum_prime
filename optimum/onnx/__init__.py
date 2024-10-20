"""
ONNX Model Graph Transformations.

This module provides functionality for applying a variety of graph transformations to ONNX models,
enabling optimizations and modifications for deployment and inference.

Features:
    - cast_slice_nodes_inputs_to_int32: Ensure that the inputs to slice nodes are cast to int32.
    - merge_decoders: Combine multiple decoder components into a single structure.
    - remove_duplicate_weights: Detect and remove redundant weights in the model to optimize size.
    - replace_atenops_to_gather: Replace specific PyTorch ATen operations with ONNX gather operations.
    - remove_duplicate_weights_from_tied_info: Clean up redundant weight references in models with tied weights.

Usage:
    This module uses lazy loading to delay importing until the specific functions are accessed. 
    This improves initial load performance when the module is not used.
"""

from typing import TYPE_CHECKING

from transformers.utils import _LazyModule

_import_structure = {
    "graph_transformations": [
        "cast_slice_nodes_inputs_to_int32",
        "merge_decoders",
        "remove_duplicate_weights",
        "replace_atenops_to_gather",
        "remove_duplicate_weights_from_tied_info",
    ],
}

if TYPE_CHECKING:
    from .graph_transformations import (
        cast_slice_nodes_inputs_to_int32,
        merge_decoders,
        remove_duplicate_weights,
        remove_duplicate_weights_from_tied_info,
        replace_atenops_to_gather,
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
