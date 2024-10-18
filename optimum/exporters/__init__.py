"""
__init__.py

This file initializes the exporters module within the Hugging Face Optimum library. It imports key components,
such as the ONNX export module and the TasksManager class, which handles task management for model export.

Imports:
    - onnx: The submodule for exporting models in ONNX format.
    - TasksManager: The task management class for handling various model export operations.

"""

# Importing the ONNX export submodule for handling ONNX model exports.
from . import onnx  # noqa: This comment disables linter warnings for unused imports.

# Importing TasksManager, which is responsible for managing export tasks (such as exporting models to various formats).
from .tasks import TasksManager  # noqa: Disabling linter warnings for unused imports.

