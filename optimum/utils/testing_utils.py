"""
testing_utils.py

This module provides utilities for testing various functionalities, models, and configurations. It includes helper 
functions for running tests, checking required dependencies, and handling parameters for testing grid configurations.

Functions:
    - flatten_dict: Flattens a nested dictionary into a flat dictionary.
    - require_accelerate: Decorator to skip tests if Accelerate is not available.
    - require_auto_gptq: Decorator to skip tests if AutoGPTQ is not available.
    - require_torch_gpu: Decorator to skip tests if CUDA and PyTorch are not available.
    - require_ort_rocm: Decorator to skip tests if ONNX Runtime ROCM provider is not available.
    - require_hf_token: Decorator to skip tests if a Hugging Face token is not set.
    - require_sigopt_token_and_project: Decorator to skip tests if SigOpt API token and project are not set.
    - is_ort_training_available: Check if ONNX Runtime Training is available.
    - require_ort_training: Decorator to skip tests if ONNX Runtime Training is not available.
    - require_diffusers: Decorator to skip tests if Diffusers library is not available.
    - require_timm: Decorator to skip tests if Timm library is not available.
    - require_sentence_transformers: Decorator to skip tests if Sentence Transformers are not available.
    - grid_parameters: Generates all combinations of parameters from a dictionary.
    - remove_directory: Cross-platform function to remove a directory and its contents.
"""

import importlib.util  # For checking if a module is available
import itertools  # To handle parameter grid combinations
import os  # For handling file system paths and environment variables
import shutil  # To remove directories and files
import subprocess  # To run shell commands
import sys  # System-specific parameters and functions
import unittest  # For unit testing
from collections.abc import MutableMapping  # For flattening nested dictionaries
from typing import Any, Callable, Dict, Iterable, Optional, Tuple  # Type hinting

import torch  # PyTorch for GPU checks

from . import (
    is_accelerate_available,  # Check if Accelerate is installed
    is_auto_gptq_available,  # Check if AutoGPTQ is installed
    is_diffusers_available,  # Check if Diffusers library is installed
    is_sentence_transformers_available,  # Check if Sentence Transformers library is installed
    is_timm_available,  # Check if Timm library is installed
)

# Dummy user for testing the Hugging Face hub
USER = "__DUMMY_OPTIMUM_USER__"


def flatten_dict(dictionary: Dict) -> Dict:
    """
    Flatten a nested dictionary into a flat dictionary.

    Args:
        dictionary (Dict): The dictionary to flatten.

    Returns:
        Dict: A flattened dictionary.
    """
    items = []
    for k, v in dictionary.items():
        if isinstance(v, MutableMapping):
            # Recursively flatten if the value is another dictionary
            items.extend(flatten_dict(v).items())
        else:
            items.append((k, v))
    return dict(items)


def require_accelerate(test_case):
    """Decorator marking a test that requires Accelerate. These tests are skipped if Accelerate is not installed."""
    return unittest.skipUnless(is_accelerate_available(), "test requires Accelerate")(test_case)


def require_auto_gptq(test_case):
    """Decorator marking a test that requires AutoGPTQ. These tests are skipped if AutoGPTQ is not installed."""
    return unittest.skipUnless(is_auto_gptq_available(), "test requires AutoGPTQ")(test_case)


def require_torch_gpu(test_case):
    """Decorator marking a test that requires CUDA and PyTorch."""
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    return unittest.skipUnless(torch_device == "cuda", "test requires CUDA")(test_case)


def require_ort_rocm(test_case):
    """Decorator marking a test that requires ROCMExecutionProvider for ONNX Runtime."""
    import onnxruntime as ort

    providers = ort.get_available_providers()
    return unittest.skipUnless("ROCMExecutionProvider" in providers, "test requires ROCMExecutionProvider")(test_case)


def require_hf_token(test_case):
    """Decorator marking a test that requires a Hugging Face token to be set in the environment."""
    hf_token = os.environ.get("HF_AUTH_TOKEN", None)
    if hf_token is None:
        return unittest.skip("test requires Hugging Face token set as `HF_AUTH_TOKEN`")(test_case)
    return test_case


def require_sigopt_token_and_project(test_case):
    """Decorator marking a test that requires SigOpt API token and project to be set in the environment."""
    sigopt_api_token = os.environ.get("SIGOPT_API_TOKEN", None)
    has_sigopt_project = os.environ.get("SIGOPT_PROJECT", None)
    if sigopt_api_token is None or has_sigopt_project is None:
        return unittest.skip("test requires `SIGOPT_API_TOKEN` and `SIGOPT_PROJECT` environment variables")(test_case)
    return test_case


def is_ort_training_available() -> bool:
    """
    Check if ONNX Runtime Training and torch_ort are available and configured.

    Returns:
        bool: True if ONNX Runtime Training and torch_ort are available and configured, else False.
    """
    is_ort_train_available = importlib.util.find_spec("onnxruntime.training") is not None
    is_torch_ort_configured = False
    if importlib.util.find_spec("torch_ort") is not None:
        try:
            subprocess.run([sys.executable, "-m", "torch_ort.configure"], check=True)
            is_torch_ort_configured = True
        except subprocess.CalledProcessError:
            is_torch_ort_configured = False
    return is_ort_train_available and is_torch_ort_configured


def require_ort_training(test_case):
    """Decorator marking a test that requires ONNX Runtime Training and torch_ort to be correctly installed."""
    return unittest.skipUnless(
        is_ort_training_available(),
        "test requires ONNX Runtime Training and torch_ort",
    )(test_case)


def require_diffusers(test_case):
    """Decorator marking a test that requires the Diffusers library."""
    return unittest.skipUnless(is_diffusers_available(), "test requires Diffusers")(test_case)


def require_timm(test_case):
    """Decorator marking a test that requires the Timm library."""
    return unittest.skipUnless(is_timm_available(), "test requires Timm")(test_case)


def require_sentence_transformers(test_case):
    """Decorator marking a test that requires Sentence Transformers."""
    return unittest.skipUnless(is_sentence_transformers_available(), "test requires Sentence Transformers")(test_case)


def grid_parameters(
    parameters: Dict[str, Iterable[Any]],
    yield_dict: bool = False,
    add_test_name: bool = True,
    filter_params_func: Optional[Callable[[Tuple], Tuple]] = None,
) -> Iterable:
    """
    Generates an iterable over the grid of all combinations of parameters.

    Args:
        parameters (Dict[str, Iterable[Any]]): Dictionary of multiple values to generate a grid from.
        yield_dict (bool, optional): If True, returns a dictionary of sampled parameters. Otherwise, a list. Default is False.
        add_test_name (bool, optional): Whether to add the test name to the yielded list or dictionary. Default is True.
        filter_params_func (Optional[Callable[[Tuple], Tuple]], optional): Function to modify or exclude parameter sets. Default is None.

    Yields:
        Iterable: Grid of parameter combinations as either dictionaries or lists.
    """
    for params in itertools.product(*parameters.values()):
        if filter_params_func is not None:
            params = filter_params_func(list(params))
            if params is None:
                continue

        test_name = "_".join([str(param) for param in params])
        if yield_dict:
            res_dict = {key: params[i] for i, key in enumerate(parameters.keys())}
            if add_test_name:
                res_dict["test_name"] = test_name
            yield res_dict
        else:
            yield [test_name] + list(params) if add_test_name else list(params)


def remove_directory(dirpath: str) -> None:
    """
    Remove a directory and its contents.

    Args:
        dirpath (str): The path to the directory to remove.

    This is a cross-platform solution that avoids using shutil.rmtree on Windows.
    """
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        if os.name == "nt":
            os.system(f"rmdir /S /Q {dirpath}")
        else:
            shutil.rmtree(dirpath)
