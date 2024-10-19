"""Import utilities."""
"""
Utility Functions for Checking Package Availability in Promise Optimizer

This module provides utility functions to check the availability of various packages, including ONNX, PyTorch, 
Transformers, and others. It also defines minimum required versions for these packages and checks whether the 
installed versions meet the requirements.

Functions:
    - is_torch_onnx_support_available: Checks if the installed version of PyTorch supports ONNX.
    - is_onnx_available: Checks if the ONNX package is available.
    - is_onnxruntime_available: Checks if ONNX Runtime is installed and available.
    - is_pydantic_available: Checks if Pydantic is available.
    - is_accelerate_available: Checks if Accelerate is available.
    - is_diffusers_available: Checks if Diffusers is available.
    - is_timm_available: Checks if TIMM is available.
    - is_sentence_transformers_available: Checks if Sentence Transformers is available.
    - is_auto_gptq_available: Checks if Auto GPTQ is available and meets the required version.
"""

import importlib.util  # Utility functions for package import handling
import inspect  # To inspect live objects and modules
import sys  # System-specific parameters and functions
from collections import OrderedDict  # Ordered dictionary for maintaining order of entries
from contextlib import contextmanager  # Context manager for resource management
from typing import Tuple, Union  # Type hinting for functions

import numpy as np  # NumPy for numerical operations
from packaging import version  # Version comparison utility from packaging
from transformers.utils import is_torch_available  # Utility function to check if PyTorch is available


def _is_package_available(pkg_name: str, return_version: bool = False) -> Union[Tuple[bool, str], bool]:
    """
    Checks whether the specified package is available and optionally returns its version.

    Args:
        pkg_name (str): Name of the package to check.
        return_version (bool, optional): Whether to return the package version. Defaults to False.

    Returns:
        Union[Tuple[bool, str], bool]: If `return_version` is True, returns a tuple of (availability, version). 
        Otherwise, returns just the availability as a boolean.
    """
    # Check we're not importing a "pkg_name" directory somewhere but the actual library by trying to grab the version
    package_exists = importlib.util.find_spec(pkg_name) is not None
    package_version = "N/A"
    if package_exists:
        try:
            package_version = importlib.metadata.version(pkg_name)
            package_exists = True
        except importlib.metadata.PackageNotFoundError:
            package_exists = False
    if return_version:
        return package_exists, package_version
    else:
        return package_exists


# The package importlib_metadata is in a different place, depending on the python version.
if sys.version_info < (3, 8):
    import importlib_metadata  # For Python versions < 3.8, importlib_metadata is in a separate package
else:
    import importlib.metadata as importlib_metadata  # For Python >= 3.8, importlib.metadata is available in the stdlib


# Minimum required versions for various packages
TORCH_MINIMUM_VERSION = version.parse("1.11.0")
TRANSFORMERS_MINIMUM_VERSION = version.parse("4.25.0")
DIFFUSERS_MINIMUM_VERSION = version.parse("0.22.0")
AUTOGPTQ_MINIMUM_VERSION = version.parse("0.4.99")  # Allows 0.5.0.dev0


# Minimum required version to support ONNX Runtime features
ORT_QUANTIZE_MINIMUM_VERSION = version.parse("1.4.0")


# Check the availability of key packages
_onnx_available = _is_package_available("onnx")
_onnxruntime_available = importlib.util.find_spec("onnxruntime") is not None  # ONNX Runtime availability check
_pydantic_available = _is_package_available("pydantic")
_accelerate_available = _is_package_available("accelerate")
_diffusers_available = _is_package_available("diffusers")
_auto_gptq_available = _is_package_available("auto_gptq")
_timm_available = _is_package_available("timm")
_sentence_transformers_available = _is_package_available("sentence_transformers")


# Check PyTorch version if available
torch_version = None
if is_torch_available():
    torch_version = version.parse(importlib_metadata.version("torch"))

# Check if PyTorch supports ONNX by comparing versions
_is_torch_onnx_support_available = is_torch_available() and (
    TORCH_MINIMUM_VERSION.major,
    TORCH_MINIMUM_VERSION.minor,
) <= (
    torch_version.major,
    torch_version.minor,
)


# Check the Diffusers package version if available
_diffusers_version = None
if _diffusers_available:
    try:
        _diffusers_version = importlib_metadata.version("diffusers")
    except importlib_metadata.PackageNotFoundError:
        _diffusers_available = False


def is_torch_onnx_support_available():
    """
    Checks if PyTorch ONNX support is available, based on the installed PyTorch version.

    Returns:
        bool: True if ONNX support is available, False otherwise.
    """
    return _is_torch_onnx_support_available


def is_onnx_available():
    """
    Checks if the ONNX package is installed.

    Returns:
        bool: True if ONNX is available, False otherwise.
    """
    return _onnx_available


def is_onnxruntime_available():
    """
    Checks if the ONNX Runtime package is installed and functional.

    Returns:
        bool: True if ONNX Runtime is available, False otherwise.
    """
    try:
        # Try to import the source file of onnxruntime - if you run the tests from `tests` the function gets
        # confused since there a folder named `onnxruntime` in `tests`. Therefore, `_onnxruntime_available`
        # will be set to `True` even if not installed.
        mod = importlib.import_module("onnxruntime")
        inspect.getsourcefile(mod)
    except Exception:
        return False
    return _onnxruntime_available


def is_pydantic_available():
    """
    Checks if the Pydantic package is installed.

    Returns:
        bool: True if Pydantic is available, False otherwise.
    """
    return _pydantic_available


def is_accelerate_available():
    """
    Checks if the Accelerate package is installed.

    Returns:
        bool: True if Accelerate is available, False otherwise.
    """
    return _accelerate_available


def is_diffusers_available():
    """
    Checks if the Diffusers package is installed.

    Returns:
        bool: True if Diffusers is available, False otherwise.
    """
    return _diffusers_available


def is_timm_available():
    """
    Checks if the TIMM package is installed.

    Returns:
        bool: True if TIMM is available, False otherwise.
    """
    return _timm_available


def is_sentence_transformers_available():
    """
    Checks if the Sentence Transformers package is installed.

    Returns:
        bool: True if Sentence Transformers is available, False otherwise.
    """
    return _sentence_transformers_available


def is_auto_gptq_available():
    """
    Checks if the Auto GPTQ package is installed and meets the required version.

    Returns:
        bool: True if Auto GPTQ is available and meets the minimum version, False otherwise.

    Raises:
        ImportError: If the installed version of Auto GPTQ is lower than the minimum required version.
    """
    if _auto_gptq_available:
        version_autogptq = version.parse(importlib_metadata.version("auto_gptq"))
        if AUTOGPTQ_MINIMUM_VERSION < version_autogptq:
            return True
        else:
            raise ImportError(
                f"Found an incompatible version of auto-gptq. Found version {version_autogptq}, but only versions above {AUTOGPTQ_MINIMUM_VERSION} are supported."
            )

"""
Utility Functions for Backend Checks and Version Compatibility in Promise Optimizer

This module provides utility functions for checking the availability of specific backends and ensuring the
required versions of libraries are installed. It also defines context managers to enforce version requirements
for key dependencies like PyTorch, Transformers, and Diffusers.

Functions:
    - check_if_pytorch_greater: Context manager that checks if PyTorch is at least the specified version.
    - check_if_transformers_greater: Checks if Transformers is at least the target version.
    - check_if_diffusers_greater: Checks if Diffusers is at least the target version.
    - require_numpy_strictly_lower: Context manager that enforces a specific upper bound on NumPy versions.
    - requires_backends: Ensures that required backends are installed and raises an ImportError if not.
    - DummyObject: A metaclass that raises ImportError for dummy objects when the required backend is unavailable.
"""

import importlib.util  # Utility for handling package imports
import inspect  # To inspect modules and functions
import sys  # For system-specific functionality
from collections import OrderedDict  # Ordered dictionary to maintain insertion order
from contextlib import contextmanager  # For context management functionality
from typing import Tuple, Union  # Type hinting support

import numpy as np  # NumPy for numerical operations
from packaging import version  # To parse and compare version strings
from transformers.utils import is_torch_available  # Check if PyTorch is installed


@contextmanager
def check_if_pytorch_greater(target_version: str, message: str):
    """
    A context manager that checks if the installed PyTorch version is greater than or equal to `target_version`.
    If not, an ImportError is raised with the provided message.

    Args:
        target_version (str): The minimum required version of PyTorch.
        message (str): Error message to display if the version check fails.

    Raises:
        ImportError: If the installed version of PyTorch is lower than `target_version`.
    """
    import torch

    if not version.parse(torch.__version__) >= version.parse(target_version):
        raise ImportError(
            f"Found an incompatible version of PyTorch. Found version {torch.__version__}, but only {target_version} and above are supported. {message}"
        )
    try:
        yield  # Proceed with the context
    finally:
        pass


def check_if_transformers_greater(target_version: Union[str, version.Version]) -> bool:
    """
    Checks if the installed version of Transformers is greater than or equal to the specified target version.

    Args:
        target_version (Union[str, version.Version]): The minimum required version of Transformers.

    Returns:
        bool: True if the installed version meets or exceeds the target version, False otherwise.
    """
    import transformers

    if isinstance(target_version, str):
        target_version = version.parse(target_version)

    return version.parse(transformers.__version__) >= target_version


def check_if_diffusers_greater(target_version: str) -> bool:
    """
    Checks if the installed version of Diffusers is greater than or equal to the specified target version.

    Args:
        target_version (str): The minimum required version of Diffusers.

    Returns:
        bool: True if the installed version meets or exceeds the target version, False otherwise.
    """
    if not _diffusers_available:
        return False

    return version.parse(_diffusers_version) >= version.parse(target_version)


@contextmanager
def require_numpy_strictly_lower(package_version: str, message: str):
    """
    A context manager that checks if the installed NumPy version is strictly lower than the specified version.
    If not, an ImportError is raised with the provided message.

    Args:
        package_version (str): The upper bound version of NumPy.
        message (str): Error message to display if the version check fails.

    Raises:
        ImportError: If the installed version of NumPy is greater than or equal to `package_version`.
    """
    if not version.parse(np.__version__) < version.parse(package_version):
        raise ImportError(
            f"Found an incompatible version of NumPy. Found version {np.__version__}, but expected NumPy < {package_version}. {message}"
        )
    try:
        yield  # Proceed with the context
    finally:
        pass


# Error messages for missing backends
DIFFUSERS_IMPORT_ERROR = """
{0} requires the diffusers library, but it was not found in your environment. You can install it with pip: `pip install diffusers`. Please note that you may need to restart your runtime after installation.
"""

TRANSFORMERS_IMPORT_ERROR = """
{0} requires the transformers>={1} library, but it was not found in your environment. You can install it with pip: `pip install -U transformers`. Please note that you may need to restart your runtime after installation.
"""

# Backend mapping for easy checks of required versions
BACKENDS_MAPPING = OrderedDict(
    [
        ("diffusers", (is_diffusers_available, DIFFUSERS_IMPORT_ERROR)),
        (
            "transformers_431",
            (lambda: check_if_transformers_greater("4.31"), "{0} " + TRANSFORMERS_IMPORT_ERROR.format("4.31")),
        ),
        (
            "transformers_432",
            (lambda: check_if_transformers_greater("4.32"), "{0} " + TRANSFORMERS_IMPORT_ERROR.format("4.32")),
        ),
        (
            "transformers_434",
            (lambda: check_if_transformers_greater("4.34"), "{0} " + TRANSFORMERS_IMPORT_ERROR.format("4.34")),
        ),
    ]
)


def requires_backends(obj, backends):
    """
    Ensures that the required backends are installed, otherwise raises an ImportError with an informative message.

    Args:
        obj: The object (class or function) that requires the specified backends.
        backends (Union[str, list, tuple]): The backend(s) required by the object.

    Raises:
        ImportError: If any of the required backends is not installed.
    """
    if not isinstance(backends, (list, tuple)):
        backends = [backends]

    name = obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__
    checks = (BACKENDS_MAPPING[backend] for backend in backends)
    failed = [msg.format(name) for available, msg in checks if not available()]
    if failed:
        raise ImportError("".join(failed))


class DummyObject(type):
    """
    Metaclass for dummy objects. Any class inheriting from this metaclass will raise an ImportError if the required
    backend is not available when any method of that class is accessed.

    This metaclass is used to generate placeholder objects for missing backends, ensuring that the user gets a
    meaningful error message if they try to use functionality requiring an unavailable backend.
    """

    def __getattr__(cls, key):
        if key.startswith("_"):
            return super().__getattr__(cls, key)
        requires_backends(cls, cls._backends)
