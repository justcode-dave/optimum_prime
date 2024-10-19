#!/usr/bin/env python

"""
Subpackages Loader for Promise Optimizer

This module is responsible for dynamically loading subpackages inside the Promise Optimizer project, 
specifically for loading modules in namespace packages like `optimum`. The main purpose of this file 
is to identify and load subpackages that provide additional backend functionality such as ONNX Runtime, 
register their commands, and make them available for use within the broader framework.

The module also supports loading custom submodules from other libraries within the same namespace. 
It ensures flexibility and modularity by allowing different subpackages to register commands dynamically.

Functions:
    - load_namespace_modules(namespace: str, module: str): Loads a specific module from all subpackages in a namespace.
    - load_subpackages(): Automatically loads the required subpackages for optimum integration.

Dependencies:
    - importlib: For dynamic importing of modules.
    - importlib_metadata: For extracting metadata from installed distributions.
    - logging: To log the loading process and any errors encountered.
    - sys: To check the Python version and manage loaded modules.
"""

import importlib  # For dynamically importing modules
import logging  # For logging information and errors
import sys  # To check the Python version and manage system modules

# Use different metadata imports depending on the Python version
if sys.version_info >= (3, 8):
    from importlib import metadata as importlib_metadata  # Python 3.8+ metadata handling
else:
    import importlib_metadata  # Legacy metadata handling for Python < 3.8

from importlib.util import find_spec, module_from_spec  # Tools for finding and loading module specs

from .utils import is_onnxruntime_available  # Check if ONNX Runtime is available

# Set up logging
logger = logging.getLogger(__name__)

def load_namespace_modules(namespace: str, module: str):
    """
    Load all modules with a specific name inside a namespace package.

    This function goes through each installed distribution in the specified `namespace` and tries to load
    a module with the provided `module` name inside that namespace package. It ensures that submodules 
    are dynamically loaded if they are not already present in `sys.modules`.

    Args:
        namespace (str): The namespace containing packages that need to be loaded.
        module (str): The name of the module to load inside each package in the namespace.

    Example:
        load_namespace_modules("optimum", "subpackage")
    """
    for dist in importlib_metadata.distributions():  # Iterate over all installed distributions
        dist_name = dist.metadata["Name"]
        if not dist_name.startswith(f"{namespace}-"):
            continue  # Skip distributions that are not part of the namespace

        package_import_name = dist_name.replace("-", ".")  # Replace '-' with '.' for proper import path
        module_import_name = f"{package_import_name}.{module}"  # Create full module path
        if module_import_name in sys.modules:
            continue  # Skip if the module is already loaded

        backend_spec = find_spec(module_import_name)  # Find the module specification
        if backend_spec is None:
            continue  # Skip if no module spec is found

        try:
            imported_module = module_from_spec(backend_spec)  # Load the module from spec
            sys.modules[module_import_name] = imported_module  # Register the module in sys.modules
            backend_spec.loader.exec_module(imported_module)  # Execute the module code
            logger.debug(f"Successfully loaded {module_import_name}")
        except Exception as e:
            logger.error(f"An exception occurred while loading {module_import_name}: {e}")

def load_subpackages():
    """
    Load Promise Optimizer subpackages.

    This function automatically loads all subpackages inside the `optimum` namespace, specifically the `subpackage` 
    module for each subpackage. It ensures that additional functionality provided by the subpackages, such as 
    ONNX Runtime support, is made available. It also registers subpackage commands dynamically.

    If ONNX Runtime is available, it also loads the `optimum.onnxruntime` subpackage.
    """
    SUBPACKAGE_LOADER = "subpackage"  # Define the name of the subpackage loader module
    load_namespace_modules("optimum", SUBPACKAGE_LOADER)  # Load subpackages inside the optimum namespace

    # Check for internal modules and load ONNX Runtime subpackage if available
    loader_name = "." + SUBPACKAGE_LOADER
    if is_onnxruntime_available():
        importlib.import_module(loader_name, package="optimum.onnxruntime")  # Load ONNX Runtime-specific subpackage
