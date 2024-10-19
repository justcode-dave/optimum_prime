"""
runs.py

This module provides configuration classes, enums, and utility functions for managing model tasks, calibration, and 
framework arguments. It includes mechanisms for checking supported tasks, handling platform-specific CPU information, 
and managing configurations for quantization approaches and post-training calibrations.

Key Components:
---------------
- `APIFeaturesManager`: Manages supported model-task pairs and tasks.
- `Frameworks`: Enum for specifying supported machine learning frameworks (e.g., onnxruntime).
- `CalibrationMethods`, `QuantizationApproach`: Enums for specifying calibration methods and quantization approaches.
- `cpu_info_command`: Utility to return the correct CPU info command based on the operating system.
- Dataclass configurations for calibration, framework arguments, datasets, and task-specific parameters.
"""

import platform  # Handles system-specific commands and information
from dataclasses import field  # Enables dataclass fields with metadata and defaults
from enum import Enum  # Provides enumerations for quantization approaches, calibration methods, and frameworks
from typing import Dict, List, Optional, Union  # Provides type hints for flexible type annotations

from . import is_pydantic_available  # Helper to check if pydantic is available in the environment
from .doc import generate_doc_dataclass  # Decorator to generate documentation for dataclasses

# Conditionally import dataclass based on pydantic availability
if is_pydantic_available():
    from pydantic.dataclasses import dataclass  # Import pydantic's dataclass if available
else:
    from dataclasses import dataclass  # Fall back to standard dataclass if pydantic is not available


class APIFeaturesManager:
    """
    Manager for supported tasks and model-task pair validation. Provides static methods to check if a model-task
    combination or task is supported.
    """
    _SUPPORTED_TASKS = ["text-classification", "token-classification", "question-answering", "image-classification"]

    @staticmethod
    def check_supported_model_task_pair(model_type: str, task: str):
        """
        Validates if the given model type and task combination is supported.

        Args:
            model_type (str): The type of the model (e.g., 'bert', 'gpt').
            task (str): The task (e.g., 'text-classification', 'question-answering').

        Raises:
            KeyError: If the model type or task is not supported.
        """
        model_type = model_type.lower()  # Normalize model type to lowercase
        if model_type not in APIFeaturesManager._SUPPORTED_MODEL_TYPE:
            raise KeyError(
                f"{model_type} is not supported yet. "
                f"Only {list(APIFeaturesManager._SUPPORTED_MODEL_TYPE.keys())} are supported. "
                f"If you want to support {model_type} please propose a PR or open up an issue."
            )
        elif task not in APIFeaturesManager._SUPPORTED_MODEL_TYPE[model_type]:
            raise KeyError(
                f"{task} is not supported yet for model {model_type}. "
                f"Only {APIFeaturesManager._SUPPORTED_MODEL_TYPE[model_type]} are supported. "
                f"If you want to support {task} please propose a PR or open up an issue."
            )

    @staticmethod
    def check_supported_task(task: str):
        """
        Checks if a task is supported by the API.

        Args:
            task (str): The task to check (e.g., 'text-classification').

        Raises:
            KeyError: If the task is not supported.
        """
        if task not in APIFeaturesManager._SUPPORTED_TASKS:
            raise KeyError(
                f"{task} is not supported yet. "
                f"Only {APIFeaturesManager._SUPPORTED_TASKS} are supported. "
                f"If you want to support {task} please propose a PR or open up an issue."
            )


class Frameworks(str, Enum):
    """Enumeration for supported machine learning frameworks."""
    onnxruntime = "onnxruntime"  # ONNX Runtime framework


class CalibrationMethods(str, Enum):
    """Enumeration of supported calibration methods for quantization."""
    minmax = "minmax"  # Min-max calibration method
    percentile = "percentile"  # Percentile-based calibration method
    entropy = "entropy"  # Entropy-based calibration method


class QuantizationApproach(str, Enum):
    """Enumeration for quantization approaches."""
    static = "static"  # Static quantization approach
    dynamic = "dynamic"  # Dynamic quantization approach


def cpu_info_command():
    """
    Returns the appropriate command to retrieve CPU information based on the operating system.

    Returns:
        str: The command to retrieve CPU info (e.g., 'lscpu' for Linux, 'sysctl' for macOS).

    Raises:
        NotImplementedError: If the operating system is not supported.
    """
    if platform.system() == "Linux":
        return "lscpu"  # Linux CPU info command
    elif platform.system() == "Darwin":
        return "sysctl -a | grep machdep.cpu"  # macOS CPU info command
    else:
        raise NotImplementedError("OS not supported.")  # Raise an error for unsupported OS


@generate_doc_dataclass
@dataclass
class Calibration:
    """
    Parameters for post-training calibration with static quantization.

    Attributes:
        method (CalibrationMethods): 
            The calibration method used ('minmax', 'entropy', 'percentile').
        num_calibration_samples (int): 
            Number of examples used for calibration during static quantization.
        calibration_histogram_percentile (Optional[float]): 
            Percentile used for the percentile calibration method.
        calibration_moving_average (Optional[bool]): 
            Whether to compute moving averages for the minmax calibration method.
        calibration_moving_average_constant (Optional[float]): 
            Constant factor used when computing the moving average for minmax calibration.
    """
    method: CalibrationMethods = field(
        metadata={"description": 'Calibration method used, either "minmax", "entropy" or "percentile".'}
    )
    num_calibration_samples: int = field(
        metadata={"description": "Number of examples to use for the calibration step resulting from static quantization."}
    )
    calibration_histogram_percentile: Optional[float] = field(
        default=None, metadata={"description": "The percentile used for the percentile calibration method."}
    )
    calibration_moving_average: Optional[bool] = field(
        default=None,
        metadata={"description": "Whether to compute the moving average of the minimum and maximum values for the minmax calibration method."}
    )
    calibration_moving_average_constant: Optional[float] = field(
        default=None,
        metadata={"description": "Constant smoothing factor used when computing the moving average of the minimum and maximum values for minmax calibration."}
    )


@generate_doc_dataclass
@dataclass
class FrameworkArgs:
    """
    Arguments for configuring the ONNX framework during model export.

    Attributes:
        opset (Optional[int]): 
            ONNX opset version to export the model with (default: 11).
        optimization_level (Optional[int]): 
            ONNX optimization level for model export (default: 0).
    """
    opset: Optional[int] = field(default=11, metadata={"description": "ONNX opset version to export the model with."})
    optimization_level: Optional[int] = field(default=0, metadata={"description": "ONNX optimization level."})

    def __post_init__(self):
        """Validate the `opset` and `optimization_level` values."""
        assert self.opset <= 15, f"Unsupported OnnxRuntime opset: {self.opset}"  # Validate opset version
        assert self.optimization_level in [0, 1, 2, 99], f"Unsupported OnnxRuntime optimization level: {self.optimization_level}"  # Validate optimization level


@generate_doc_dataclass
@dataclass
class DatasetArgs:
    """
    Parameters related to the dataset used in model evaluation or calibration.

    Attributes:
        path (str): 
            Path to the dataset (used with `datasets.load_dataset`).
        eval_split (str): 
            Dataset split used for evaluation (e.g., "test").
        data_keys (Dict[str, Union[None, str]]): 
            Columns used as input data, indicated as "primary" and "secondary".
        ref_keys (List[str]): 
            Columns used for references during evaluation.
        name (Optional[str]): 
            Name of the dataset (used with `datasets.load_dataset`).
        calibration_split (Optional[str]): 
            Dataset split used for calibration (e.g., "train").
    """
    path: str = field(metadata={"description": "Path to the dataset, as in `datasets.load_dataset(path)`."})
    eval_split: str = field(metadata={"description": 'Dataset split used for evaluation (e.g. "test").'})
    data_keys: Dict[str, Union[None, str]] = field(
        metadata={"description": 'Dataset columns used as input data. At most two, indicated with "primary" and "secondary".'}
    )
    ref_keys: List[str] = field(metadata={"description": "Dataset column used for references during evaluation."})
    name: Optional[str] = field(default=None, metadata={"description": "Name of the dataset, as in `datasets.load_dataset(path, name)`."})
    calibration_split: Optional[str] = field(default=None, metadata={"description": 'Dataset split used for calibration (e.g. "train").'})


@generate_doc_dataclass
@dataclass
class TaskArgs:
    """
    Task-specific parameters for model training or evaluation.

    Attributes:
        is_regression (Optional[bool]): 
            Text classification-specific. Set to True if the task is regression (output = one float).
    """
    is_regression: Optional[bool] = field(
        default=None,
        metadata={"description": "Text classification specific. Set whether the task is regression (output = one float)."}
    )


@generate_doc_dataclass
@dataclass
class BenchmarkTimeArgs:
    """Parameters related to time benchmark."""
    
    duration: Optional[int] = field(
        default=30, metadata={"description": "Duration in seconds of the time evaluation."}
    )
    warmup_runs: Optional[int] = field(
        default=10, metadata={"description": "Number of warmup calls to forward() before the time evaluation."}
    )


@dataclass
class _RunBase:
    """
    Base configuration class for a model run.

    Attributes:
        model_name_or_path (str): Name or path of the model hosted on the Hub to use for the run.
        task (str): Task performed by the model (e.g., 'text-classification').
        quantization_approach (QuantizationApproach): Quantization approach (e.g., dynamic or static).
        dataset (DatasetArgs): Dataset arguments, including split and data keys for evaluation.
        framework (Frameworks): Framework used for the run (e.g., 'onnxruntime').
        framework_args (FrameworkArgs): Framework-specific arguments such as opset version and optimization level.
    """
    model_name_or_path: str = field(
        metadata={"description": "Name of the model hosted on the Hub to use for the run."}
    )
    task: str = field(metadata={"description": "Task performed by the model."})
    quantization_approach: QuantizationApproach = field(
        metadata={"description": "Whether to use dynamic or static quantization."}
    )
    dataset: DatasetArgs = field(
        metadata={"description": "Dataset to use. Several keys must be set on top of the dataset name."}
    )
    framework: Frameworks = field(metadata={"description": 'Name of the framework used (e.g. "onnxruntime").'})
    framework_args: FrameworkArgs = field(metadata={"description": "Framework-specific arguments."})


@dataclass
class _RunDefaults:
    """
    Default configuration for a run, defining optional parameters such as quantization and calibration.

    Attributes:
        operators_to_quantize (Optional[List[str]]): List of operators to quantize (default: ["Add", "MatMul"]).
        node_exclusion (Optional[List[str]]): Nodes excluded from quantization (default: ["layernorm", "gelu"]).
        per_channel (Optional[bool]): Whether to apply per-channel quantization (default: False).
        calibration (Optional[Calibration]): Calibration parameters for static quantization (default: None).
        task_args (Optional[TaskArgs]): Task-specific arguments (default: None).
        aware_training (Optional[bool]): Whether Quantization-Aware Training is enabled (default: False).
        max_eval_samples (Optional[int]): Maximum number of evaluation samples (default: None).
        time_benchmark_args (Optional[BenchmarkTimeArgs]): Parameters for time benchmarking (default: 30 sec).
    """
    operators_to_quantize: Optional[List[str]] = field(
        default_factory=lambda: ["Add", "MatMul"],
        metadata={"description": 'Operators to quantize, doing no modifications to others (default: `["Add", "MatMul"]`).'}
    )
    node_exclusion: Optional[List[str]] = field(
        default_factory=lambda: ["layernorm", "gelu", "residual", "gather", "softmax"],
        metadata={"description": "Specific nodes to exclude from being quantized (default: `['layernorm', 'gelu', 'residual', 'gather', 'softmax']`)."}
    )
    per_channel: Optional[bool] = field(
        default=False, metadata={"description": "Whether to quantize per channel (default: `False`)."}
    )
    calibration: Optional[Calibration] = field(
        default=None, metadata={"description": "Calibration parameters, in case static quantization is used."}
    )
    task_args: Optional[TaskArgs] = field(
        default=None, metadata={"description": "Task-specific arguments (default: `None`)."}
    )
    aware_training: Optional[bool] = field(
        default=False, metadata={"description": "Whether the quantization is to be done with Quantization-Aware Training (not supported)."}
    )
    max_eval_samples: Optional[int] = field(
        default=None, metadata={"description": "Maximum number of samples to use from the evaluation dataset for evaluation."}
    )
    time_benchmark_args: Optional[BenchmarkTimeArgs] = field(
        default=BenchmarkTimeArgs(), metadata={"description": "Parameters related to time benchmark."}
    )


@dataclass
class _RunConfigBase:
    """
    Base class for run configuration, including metrics used in evaluation.

    Attributes:
        metrics (List[str]): List of metrics to evaluate during the run (e.g., 'accuracy', 'f1-score').
    """
    metrics: List[str] = field(metadata={"description": "List of metrics to evaluate on."})


@dataclass
class _RunConfigDefaults(_RunDefaults):
    """
    Run configuration defaults including batch sizes and input lengths for benchmarking.

    Attributes:
        batch_sizes (Optional[List[int]]): Batch sizes to include in the run (default: [4, 8]).
        input_lengths (Optional[List[int]]): Input lengths to include in the run for benchmarking time metrics (default: 128).
    """
    batch_sizes: Optional[List[int]] = field(
        default_factory=lambda: [4, 8],
        metadata={"description": "Batch sizes to include in the run to measure time metrics."}
    )
    input_lengths: Optional[List[int]] = field(
        default_factory=lambda: [128],
        metadata={"description": "Input lengths to include in the run to measure time metrics."}
    )


@dataclass
class Run(_RunDefaults, _RunBase):
    """
    Class representing a model run configuration, including task, quantization, calibration, and dataset parameters.
    """
    def __post_init__(self):
        """Validates task, dataset, calibration, and other settings during initialization."""
        # Validate the task
        APIFeaturesManager.check_supported_task(task=self.task)

        # Validate task-specific arguments
        if self.task == "text-classification":
            message = "For text classification, whether the task is regression should be explicitly specified in the task_args.is_regression key."
            assert self.task_args is not None, message
            assert self.task_args["is_regression"] is not None, message

        # Validate dataset for static quantization
        if self.quantization_approach == "static":
            assert self.dataset["calibration_split"], "Calibration split should be passed for static quantization in the dataset.calibration_split key."

        # Validate calibration for static quantization
        if self.quantization_approach == "static":
            assert self.calibration, "Calibration parameters should be passed for static quantization in the calibration key."

        # Check that Quantization-Aware Training is not supported
        assert self.aware_training is False, "Quantization-Aware Training not supported."


@generate_doc_dataclass
@dataclass
class RunConfig(Run, _RunConfigDefaults, _RunConfigBase):
    """
    Class holding the parameters to launch a run, combining configurations for datasets, calibration, benchmarking, and task-specific parameters.
    """
    def __post_init__(self):
        """Extends initialization to handle dictionary-based configurations and support for Python 3.8."""
        super().__post_init__()

        # Support Python 3.8: Convert dict-based configurations to dataclasses
        if isinstance(self.dataset, dict):
            self.dataset = DatasetArgs(**self.dataset)
        if isinstance(self.framework_args, dict):
            self.framework_args = FrameworkArgs(**self.framework_args)
        if isinstance(self.calibration, dict):
            self.calibration = Calibration(**self.calibration)
        if isinstance(self.task_args, dict):
            self.task_args = TaskArgs(**self.task_args)
        if isinstance(self.time_benchmark_args, dict):
            self.time_benchmark_args = BenchmarkTimeArgs(**self.time_benchmark_args)
