"""
This module defines the `EnvironmentCommand` for the Optimum command-line interface (CLI).

The `EnvironmentCommand` gathers and displays information about the current environment, including:
- The versions of `optimum`, `transformers`, `huggingface_hub`, and the Python interpreter.
- Details about the platform being used (e.g., OS and Python version).
- Information on installed machine learning frameworks like PyTorch and TensorFlow, along with their GPU availability.

This command is useful for debugging and reporting issues as it provides a concise snapshot of the environment setup.
"""


import platform

import huggingface_hub
from transformers import __version__ as transformers_version
from transformers.utils import is_tf_available, is_torch_available

from ..version import __version__ as version
from . import BaseOptimumCLICommand, CommandInfo


class EnvironmentCommand(BaseOptimumCLICommand):
    COMMAND = CommandInfo(name="env", help="Get information about the environment used.")

    @staticmethod
    def format_dict(d):
        return "\n".join([f"- {prop}: {val}" for prop, val in d.items()]) + "\n"

    def run(self):
        pt_version = "not installed"
        pt_cuda_available = "NA"
        if is_torch_available():
            import torch

            pt_version = torch.__version__
            pt_cuda_available = torch.cuda.is_available()

        tf_version = "not installed"
        tf_cuda_available = "NA"
        if is_tf_available():
            import tensorflow as tf

            tf_version = tf.__version__
            try:
                # deprecated in v2.1
                tf_cuda_available = tf.test.is_gpu_available()
            except AttributeError:
                # returns list of devices, convert to bool
                tf_cuda_available = bool(tf.config.list_physical_devices("GPU"))

        info = {
            "`optimum` version": version,
            "`transformers` version": transformers_version,
            "Platform": platform.platform(),
            "Python version": platform.python_version(),
            "Huggingface_hub version": huggingface_hub.__version__,
            "PyTorch version (GPU?)": f"{pt_version} (cuda availabe: {pt_cuda_available})",
            "Tensorflow version (GPU?)": f"{tf_version} (cuda availabe: {tf_cuda_available})",
        }

        print("\nCopy-and-paste the text below in your GitHub issue:\n")
        print(self.format_dict(info))

        return info
