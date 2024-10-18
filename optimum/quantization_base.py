"""
quantization_base.py

This module defines the abstract base class (ABC) `OptimumQuantizer` which serves as the foundation for implementing
quantization strategies in the Hugging Face Optimum library. It provides method placeholders for loading models from
pretrained checkpoints and performing model quantization, which are intended to be implemented by subclasses.
"""

# --- Import necessary modules for logging, abstract base classes (ABC), and handling file paths ---
import logging  # Provides logging functionality for debugging and informational purposes.
from abc import ABC, abstractmethod  # Imports the base class for abstract methods, used to define mandatory methods in subclasses.
from pathlib import Path  # Path object to handle file system paths in an OS-independent way.
from typing import Optional, Union  # Utility types for handling optional parameters and supporting multiple data types.

# --- Initialize a logger for logging important events or errors during quantization ---
logger = logging.getLogger(__name__)

class OptimumQuantizer(ABC):
    """
    Abstract Base Class (ABC) for model quantization in the Hugging Face Optimum library.
    
    This class defines the structure that all quantizer implementations should follow. It provides the necessary 
    methods for loading a pretrained model and performing quantization, which must be implemented in subclasses.
    
    Methods:
        from_pretrained: Class method to load a pretrained model for quantization (to be implemented in subclasses).
        quantize: Abstract method to perform quantization on a model (to be implemented in subclasses).
    """

    @classmethod
    def from_pretrained(
        cls,
        model_or_path: Union[str, Path],
        file_name: Optional[str] = None,
    ) -> "OptimumQuantizer":
        """
        Load a pretrained model for quantization.

        This method is intended to be overridden in subclasses to define how to load a pretrained model
        for quantization purposes. The model could be loaded from a local path or a remote repository.

        Args:
            model_or_path (Union[str, Path]): The name of the model or path to the model's directory on disk.
            file_name (Optional[str]): Optional file name for the pretrained model.

        Returns:
            OptimumQuantizer: An instance of the quantizer with the loaded pretrained model.

        Raises:
            NotImplementedError: If this method is not implemented in a subclass.
        """
        raise NotImplementedError(
            "Overwrite this method in subclass to define how to load your model from pretrained for quantization"
        )

    @abstractmethod
    def quantize(self, save_dir: Union[str, Path], file_prefix: Optional[str] = None, **kwargs) -> None:
        """
        Perform model quantization.

        This abstract method is intended to be implemented in subclasses to define the quantization
        process for a model. The method should save the quantized model to the specified directory.

        Args:
            save_dir (Union[str, Path]): Directory to save the quantized model.
            file_prefix (Optional[str]): Optional prefix for the quantized model's file name.
            **kwargs: Additional arguments that may be required for specific quantization processes.

        Raises:
            NotImplementedError: If this method is not implemented in a subclass.
        """
        raise NotImplementedError(
            "Overwrite this method in subclass to define how to quantize your model for quantization"
        )
