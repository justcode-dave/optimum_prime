"""
This module initializes key components for handling pipelines in Optimum. It imports the essential functions and 
mappings required to support tasks such as loading ORT pipelines, using BetterTransformer, and defining pipeline behaviors.

Imports:
    - `MAPPING_LOADING_FUNC`: A dictionary mapping tasks to their corresponding loading functions.
    - `ORT_SUPPORTED_TASKS`: A list of tasks that are supported by the ONNX Runtime (ORT) pipeline.
    - `load_bettertransformer`: Function to load the BetterTransformer for optimized transformer performance.
    - `load_ort_pipeline`: Function to load a pipeline compatible with ONNX Runtime (ORT) for inference.
    - `pipeline`: The general-purpose pipeline function used for various tasks like text classification, translation, etc.
"""


from .pipelines_base import (
    MAPPING_LOADING_FUNC,
    ORT_SUPPORTED_TASKS,
    load_bettertransformer,
    load_ort_pipeline,
    pipeline,
)
