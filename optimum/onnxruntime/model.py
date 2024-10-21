"""
This module provides a temporary class `ORTModel` for performing evaluation on ONNX models using ONNX Runtime (ORT). 
It allows loading an ONNX model, running inference on a dataset, and computing evaluation metrics. 
The class is marked for deprecation and will be replaced in future releases.

Classes:

1. **ORTModel**:
    - A temporary class to evaluate ONNX models using ONNX Runtime.
    - Attributes:
        - `compute_metrics (Optional[Callable[[EvalPrediction], Dict]])`: Function to compute evaluation metrics.
        - `label_names (List[str])`: Names of the labels in the dataset.
        - `session (InferenceSession)`: The ONNX Runtime session for running model inference.
        - `onnx_input_names (Dict[str, int])`: A dictionary mapping input names to their index in the ONNX model.
    - Methods:
        - `__init__(self, model_path: Union[str, os.PathLike], execution_provider: Optional[str] = "CPUExecutionProvider", compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None, label_names: Optional[List[str]] = None)`:
            - Initializes the ORT model, loads the ONNX model, and sets up the evaluation configuration.
            - Parameters:
                - `model_path (Union[str, os.PathLike])`: Path to the ONNX model file.
                - `execution_provider (Optional[str])`: ONNX Runtime execution provider, defaults to "CPUExecutionProvider".
                - `compute_metrics (Optional[Callable[[EvalPrediction], Dict]])`: Function to compute evaluation metrics, optional.
                - `label_names (Optional[List[str]])`: List of label names, defaults to `["labels"]`.
        - `evaluation_loop(self, dataset: Dataset)`:
            - Runs the evaluation loop on the provided dataset, generating predictions and computing metrics.
            - Parameters:
                - `dataset (Dataset)`: The dataset to evaluate.
            - Returns:
                - An `EvalLoopOutput` containing predictions, label IDs, metrics, and the number of samples evaluated.

This class will be removed in future versions and replaced with a more robust ONNX model evaluation system.
"""


import logging
import os
from typing import Callable, Dict, List, Optional, Union

import numpy as np
from datasets import Dataset
from transformers import EvalPrediction
from transformers.trainer_pt_utils import nested_concat
from transformers.trainer_utils import EvalLoopOutput

from onnxruntime import InferenceSession


logger = logging.getLogger(__name__)


# TODO : Temporary class, added to perform ONNX models evaluation, will be replaced with ONNXModel class
class ORTModel:
    def __init__(
        self,
        model_path: Union[str, os.PathLike],
        execution_provider: Optional[str] = "CPUExecutionProvider",
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        label_names: Optional[List[str]] = None,
    ):
        """
        Args:
            model_path (`Union[str, os.PathLike]`):
                The path to the model ONNX Intermediate Representation (IR).
            execution_provider (:obj:`str`, `optional`):
                ONNX Runtime execution provider to use.
            compute_metrics (`Callable[[EvalPrediction], Dict]`, `optional`):
                The function that will be used to compute metrics at evaluation. Must take an `EvalPrediction` and
                return a dictionary string to metric values.
            label_names (`List[str]`, `optional`):
                The list of keys in your dictionary of inputs that correspond to the labels.
        """

        logger.warning(
            "The class `optimum.onnxruntime.model.ORTModel` is deprecated and will be removed in the next release."
        )

        self.compute_metrics = compute_metrics
        self.label_names = ["labels"] if label_names is None else label_names
        self.session = InferenceSession(str(model_path), providers=[execution_provider])
        self.onnx_input_names = {input_key.name: idx for idx, input_key in enumerate(self.session.get_inputs())}

    def evaluation_loop(self, dataset: Dataset):
        """
        Run evaluation and returns metrics and predictions.

        Args:
            dataset (`datasets.Dataset`):
                Dataset to use for the evaluation step.
        """
        logger.info("***** Running evaluation *****")
        all_preds = None
        all_labels = None
        for step, inputs in enumerate(dataset):
            has_labels = all(inputs.get(k) is not None for k in self.label_names)
            if has_labels:
                labels = tuple(np.array([inputs.get(name)]) for name in self.label_names)
                if len(labels) == 1:
                    labels = labels[0]
            else:
                labels = None
            onnx_inputs = {key: np.array([inputs[key]]) for key in self.onnx_input_names if key in inputs}
            preds = self.session.run(None, onnx_inputs)
            if len(preds) == 1:
                preds = preds[0]
            all_preds = preds if all_preds is None else nested_concat(all_preds, preds, padding_index=-100)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}
        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=len(dataset))
