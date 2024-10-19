"""
RunsBase Module

This module serves as the foundational implementation for handling model runs in the Promise Optimizer project. 
It provides base classes and methods required to manage and execute optimization runs across various algorithms 
and libraries integrated within the Promise Optimizer. This base class is designed for flexibility and extensibility, 
allowing developers to define custom optimization strategies while leveraging standardized workflows.

The module supports:
- Calibration for quantization using a dataset.
- Running model inference and evaluation benchmarks.
- Measuring latency and throughput of models.

Classes:
    - Calibrator: Base class to handle the model calibration process for quantization.
    - Run: Manages and executes model runs, including benchmarking and evaluation.
    - TimeBenchmark: Tracks latency and throughput for models.

Functions:
    - get_autoclass_name(task): Returns the appropriate autoclass name based on the task.
    - ns_to_ms(ns_time): Converts nanoseconds to milliseconds.
"""

# Importing necessary modules and libraries for various tasks

import os  # For handling environment variables and file system operations
import subprocess  # For executing shell commands to retrieve system information
from contextlib import contextmanager  # For implementing context management (e.g., benchmarking time)
from time import perf_counter_ns  # High-resolution timer for performance tracking
from typing import Set  # Type hinting for Sets in function arguments

import numpy as np  # Numerical operations, specifically for array processing
import optuna  # Optuna for hyperparameter tuning and optimization
import torch  # PyTorch for deep learning model management
import transformers  # Hugging Face transformers library for model handling
from datasets import Dataset  # Handling datasets for calibration and evaluation
from tqdm import trange  # Progress bar for iterations during evaluation

# Importing custom modules from the project
from . import version as optimum_version  # Project version tracking for reproducibility
from .utils.preprocessing import (  # Utilities for dataset preprocessing across multiple tasks
    ImageClassificationProcessing,
    QuestionAnsweringProcessing,
    TextClassificationProcessing,
    TokenClassificationProcessing,
)
from .utils.runs import RunConfig, cpu_info_command  # Utility for run configuration and CPU info command

# Disable tokenizers parallelism to avoid concurrency issues during multi-threading
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_autoclass_name(task):
    """
    Get the autoclass name based on the task.

    Args:
        task (str): Task name (e.g., 'text-classification', 'audio-classification').

    Returns:
        str: Autoclass name (e.g., 'sequence-classification').
    """
    if task in ["text-classification", "audio-classification"]:
        autoclass_name = "sequence-classification"
    else:
        autoclass_name = task
    return autoclass_name


class Calibrator:
    """
    Class responsible for handling model calibration for quantization.

    Args:
        calibration_dataset (Dataset): The dataset used for calibration.
        quantizer: The quantization algorithm or framework used.
        model_path (str): Path to the model to be calibrated.
        qconfig: Configuration for quantization.
        calibration_params: Parameters for calibration.
        node_exclusion: Nodes to exclude from quantization.
    """

    def __init__(self, calibration_dataset: Dataset, quantizer, model_path, qconfig, calibration_params, node_exclusion):
        self.calibration_dataset = calibration_dataset
        self.quantizer = quantizer
        self.model_path = model_path
        self.qconfig = qconfig
        self.calibration_params = calibration_params
        self.node_exclusion = node_exclusion

    def fit(self):
        """
        Perform the calibration process. Must be implemented by subclasses.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError()


class Run:
    """
    Class to manage and execute model runs including inference and evaluation.

    Args:
        run_config (dict): Parameters for running the optimization and evaluation process. Validated by RunConfig.

    Attributes:
        task (str): Task type for the run.
        study (optuna.Study): Optuna study for parameter optimization.
        return_body (dict): Dictionary to store results and metadata of the run.
    """

    def __init__(self, run_config: dict):
        """Initialize the Run class holding methods to perform inference and evaluation given a config.

        A run compares a transformers model and an optimized model on latency/throughput, model size, and provided metrics.

        Args:
            run_config (dict): Parameters to use for the run. See [`~utils.runs.RunConfig`] for the expected keys.
        """
        RunConfig(**run_config)  # Validate the data (useful if used as standalone)

        self.task = run_config["task"]

        if run_config["quantization_approach"] == "static":
            self.static_quantization = True
        else:
            self.static_quantization = False

        search_space = {"batch_size": run_config["batch_sizes"], "input_length": run_config["input_lengths"]}

        self.study = optuna.create_study(
            directions=["maximize", "minimize"],
            sampler=optuna.samplers.GridSampler(search_space),
        )

        cpu_info = subprocess.check_output([cpu_info_command()], shell=True).decode("utf-8")

        optimum_hash = None
        if "dev" in optimum_version.__version__:
            optimum_hash = subprocess.check_output(
                "git ls-remote https://github.com/huggingface/optimum.git HEAD | awk '{ print $1}'", shell=True
            )
            optimum_hash = optimum_hash.decode("utf-8").strip("\n")

        self.return_body = {
            "model_name_or_path": run_config["model_name_or_path"],
            "task": self.task,
            "task_args": run_config["task_args"],
            "dataset": run_config["dataset"],
            "quantization_approach": run_config["quantization_approach"],
            "operators_to_quantize": run_config["operators_to_quantize"],
            "node_exclusion": run_config["node_exclusion"],
            "aware_training": run_config["aware_training"],
            "per_channel": run_config["per_channel"],
            "calibration": run_config["calibration"],
            "framework": run_config["framework"],
            "framework_args": run_config["framework_args"],
            "hardware": cpu_info,  # Store CPU info
            "versions": {
                "transformers": transformers.__version__,
                "optimum": optimum_version.__version__,
                "optimum_hash": optimum_hash,
            },
            "evaluation": {
                "time": [],
                "others": {"baseline": {}, "optimized": {}},
            },
            "max_eval_samples": run_config["max_eval_samples"],
            "time_benchmark_args": run_config["time_benchmark_args"],
        }

    def launch(self):
        """Launch inference to compare metrics between the original and optimized model.

        These metrics are latency, throughput, model size, and user-provided metrics.

        Returns:
            dict: Finalized run data with metrics stored in the 'evaluation' key.
        """
        try:
            self.study.optimize(self._launch_time)
            self.launch_eval()
        finally:
            self.finalize()
            print("Finished run.")

        return self.return_body

    def _launch_time(self, trial):
        """Optuna objective function to measure latency/throughput.

        Populates the `["evaluation"]["time"]` list of the run for various batch sizes and input lengths.

        Args:
            trial: Optuna trial object for parameter tuning.

        Returns:
            dict: Dummy data (to be customized by subclass).
        """
        raise NotImplementedError()

    def launch_eval(self):
        """Run evaluation on the original and optimized model.

        Populates the `["evaluation"]["others"]` subdictionary of the run.
        """
        raise NotImplementedError()

    def load_datasets(self):
        """Load evaluation dataset and calibration dataset (if needed for static quantization)."""
        datasets_dict = self.task_processor.load_datasets()

        self._eval_dataset = datasets_dict["eval"]
        if self.static_quantization:
            self._calibration_dataset = datasets_dict["calibration"]

    def get_calibration_dataset(self):
        """Get the calibration dataset for static quantization.

        The dataset needs to be loaded first with [`~optimum.runs_base.Run.load_datasets`].

        Returns:
            Dataset: Calibration dataset.

        Raises:
            KeyError: If no calibration dataset is loaded.
        """
        if not hasattr(self, "_calibration_dataset"):
            raise KeyError("No calibration dataset defined for this run.")
        return self._calibration_dataset

    def get_eval_dataset(self):
        """Get the evaluation dataset.

        The dataset needs to be loaded first with [`~optimum.runs_base.Run.load_datasets`].

        Returns:
            Dataset: Evaluation dataset.

        Raises:
            KeyError: If no evaluation dataset is loaded.
        """
        if not hasattr(self, "_eval_dataset"):
            raise KeyError("No evaluation dataset defined for this run.")
        return self._eval_dataset

    def finalize(self):
        """Clean up intermediary files and resources.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError()


SEC_TO_NS_SCALE = 1000000000
NS_TO_MS_SCALE = 1e6


def ns_to_ms(ns_time):
    """Convert time from nanoseconds to milliseconds.

    Args:
        ns_time (int): Time in nanoseconds.

    Returns:
        float: Time in milliseconds.
    """
    return ns_time / NS_TO_MS_SCALE


class TimeBenchmark:
    """
    Benchmark class to track latency and throughput for models.

    This class handles the benchmarking of models by tracking their latency and throughput 
    over multiple forward passes. It supports warmup runs and can generate dummy inputs 
    for specific input types such as 'input_ids', 'attention_mask', 'token_type_ids', and 'pixel_values'.

    Args:
        model (torch.nn.Module): The model to benchmark.
        batch_size (int): The batch size for inputs.
        input_length (int): The input length (sequence length for text or image size).
        model_input_names (Set[str]): The set of input names used by the model.
        warmup_runs (int): The number of warmup runs before actual benchmarking.
        duration (float): The total duration (in seconds) to run the benchmark.

    Attributes:
        latencies (list): Stores the latencies (in nanoseconds) for each forward pass.
        throughput (float): Tracks the throughput of the model (forward passes per second).
    """

    def __init__(self, model, batch_size: int, input_length: int, model_input_names: Set[str], warmup_runs: int, duration: float):
        self.batch_size = batch_size
        self.input_length = input_length
        self.model = model

        # Duration for warmup runs (in seconds)
        self.warmup_runs = warmup_runs
        self.benchmark_duration = duration

        # To store latencies and throughput
        self.latencies = []
        self.throughput = float("-inf")

        self.model_input_names = model_input_names

    @property
    def num_runs(self) -> int:
        """
        Get the number of forward runs performed during the benchmark.

        Returns:
            int: Number of forward passes tracked.
        """
        return len(self.latencies)

    @contextmanager
    def track(self):
        """
        Context manager to track the duration of a forward pass and store the latency.

        This method uses a high-resolution timer to track the time taken by the model 
        to perform a forward pass. The result is appended to the latencies list.
        """
        start = perf_counter_ns()
        yield
        end = perf_counter_ns()

        # Append the time to the buffer
        self.latencies.append(end - start)
        print(f"Tracked function took: {(end - start)}ns ({(end - start) / 1e6:.3f}ms)")

    def finalize(self, duration_ns: int):
        """
        Finalize the benchmarking by calculating throughput.

        Args:
            duration_ns (int): Total duration of the benchmarking (in nanoseconds).
        """
        self.throughput = round((len(self.latencies) / duration_ns) * SEC_TO_NS_SCALE, 2)

    def to_dict(self) -> dict:
        """
        Compute and return benchmark statistics as a dictionary.

        The statistics include the number of forward passes, throughput, 
        and various latency percentiles (50th, 90th, 95th, 99th, 99.9th).

        Returns:
            dict: Dictionary containing benchmark statistics.
        """
        # Compute statistics for latencies (converted to milliseconds)
        benchmarks_stats = {
            "nb_forwards": len(self.latencies),
            "throughput": self.throughput,
            "latency_mean": ns_to_ms(np.mean(self.latencies)),
            "latency_std": ns_to_ms(np.std(self.latencies)),
            "latency_50": ns_to_ms(np.quantile(self.latencies, 0.5)),
            "latency_90": ns_to_ms(np.quantile(self.latencies, 0.9)),
            "latency_95": ns_to_ms(np.quantile(self.latencies, 0.95)),
            "latency_99": ns_to_ms(np.quantile(self.latencies, 0.99)),
            "latency_999": ns_to_ms(np.quantile(self.latencies, 0.999)),
        }

        return benchmarks_stats

    def execute(self) -> dict:
        """
        Execute the benchmark by running the model with dummy inputs.

        This method generates dummy inputs based on the model input names and tracks 
        the performance over the specified duration or number of runs. Warmup runs are 
        executed first to ensure accurate measurement.

        Returns:
            dict: Dictionary containing benchmark statistics.
        """
        inputs = {}

        # Set of recognized model inputs
        checked_inputs = {"input_ids", "attention_mask", "token_type_ids", "pixel_values"}

        # Generate dummy inputs based on the model's input names
        if "input_ids" in self.model_input_names:
            inputs["input_ids"] = torch.randint(high=1000, size=(self.batch_size, self.input_length))
        if "attention_mask" in self.model_input_names:
            inputs["attention_mask"] = torch.ones(self.batch_size, self.input_length, dtype=torch.int64)
        if "token_type_ids" in self.model_input_names:
            inputs["token_type_ids"] = torch.ones(self.batch_size, self.input_length, dtype=torch.int64)
        if "pixel_values" in self.model_input_names:
            # Handle RGB images (default). TODO: add grayscale support if needed.
            inputs["pixel_values"] = torch.rand(
                self.batch_size, 3, self.model.config.image_size, self.model.config.image_size, dtype=torch.float32
            )

        # Check if any input in the model_input_names is not handled
        if np.any([k not in checked_inputs for k in self.model_input_names]):
            raise NotImplementedError(
                f"At least one input in {self.model_input_names} is not supported for dummy input generation."
            )

        # Warmup phase to ensure accurate measurements
        for _ in trange(self.warmup_runs, desc="Warming up"):
            self.model.forward(**inputs)

        # Execute benchmark for the specified duration
        if self.benchmark_duration != 0:
            benchmark_duration_ns = self.benchmark_duration * SEC_TO_NS_SCALE
            print(f"Running time tracking for {self.benchmark_duration:.1f}s.")
            while sum(self.latencies) < benchmark_duration_ns:
                # Track the forward pass duration
                with self.track():
                    self.model.forward(**inputs)

            self.finalize(benchmark_duration_ns)
            return self.to_dict()

        # Return default stats if no benchmarking is performed
        else:
            benchmarks_stats = {
                "nb_forwards": 0,
                "throughput": -1,
                "latency_mean": -1,
            }
            return benchmarks_stats


# Map tasks to the appropriate processing classes
task_processing_map = {
    "text-classification": TextClassificationProcessing,
    "token-classification": TokenClassificationProcessing,
    "question-answering": QuestionAnsweringProcessing,
    "image-classification": ImageClassificationProcessing,
}
