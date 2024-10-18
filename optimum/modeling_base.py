"""
modeling_base.py

This file provides the base class (`OptimizedModel`) for optimized model inference wrapping in the Hugging Face Optimum library. 
It defines abstract methods for saving, loading, and pushing optimized models to the Hugging Face Hub.

Classes:
    PreTrainedModel (abstract): A placeholder class for compatibility between optimized models and the Hugging Face transformers.
    OptimizedModel (abstract): A base class for optimized models, providing methods for inference, saving/loading, and pushing to the Hugging Face Hub.

Functions:
    Various methods for loading, saving, and pushing models, as well as abstract methods that must be implemented by subclasses.

"""

# Import necessary modules for logging, file handling, subprocess management, and abstract base class definition
import logging
import os
import subprocess
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

# Hugging Face Hub imports for repository management and model uploading
from huggingface_hub import(
       create_repo, #Creates a repository on the Hugging Face hub
       upload_file, #Handles file transfer (including model weights, configuration files) to the repository on Hugging face 
)
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE #Specifies the cache location containing cached model files 
from transformers import(
       AutoConfig, #Automatically loads the correct configuration class based on loaded model 
       PretrainedConfig, #Base class for model configurations in the Hugging Face library
       #Every model has a corresponding PretrainedConfig subclass defining its architecture and parameter
       add_start_docstrings #Utility function to programatically add shared Docstring, ensuring clarity and consistency
) 

# Local imports for handling specific tasks and configurations
from .exporters import TasksManager #A class or utility that manages model export, configuration and optimization tasks  
from .utils import CONFIG_NAME #Used to identify the configuration file name 

# Conditional imports for avoiding runtime errors when the classes aren't present
if TYPE_CHECKING: #Ensures that imports are ONLY active when type hinting is being evaluated 
    from transformers import (
        PreTrainedModel, #Base class for PyTorch-based models and shared functionality across pre-trained models 
        TFPreTrainedModel #Base class for TensorFlow-based models and shared functionality across pre-trained models 
    )

# Set up a logger for logging messages from this module
logger = logging.getLogger(__name__)

# A docstring template for the `from_pretrained` method to ensure consistency across models.
FROM_PRETRAINED_START_DOCSTRING = r"""
    Instantiate a pretrained model from a pre-trained model configuration.

    Args:
        model_id (`Union[str, Path]`): Can be either:
            - A string (model ID) of a pretrained model hosted on huggingface.co.
            - A path to a directory containing a model saved using the `save_pretrained` method.
        export (`bool`, defaults to `False`): Defines whether the provided `model_id` needs to be exported.
        force_download (`bool`, defaults to `True`): Whether to force download of the model, overriding cached versions.
        token (`Optional[Union[bool, str]]`, defaults to `None`): Token to use for HTTP bearer authorization.
        cache_dir (`Optional[str]`, defaults to `None`): Directory to cache downloaded model configurations.
        subfolder (`str`, defaults to `""`): Specify if files are located inside a subfolder of the model repo.
        config (`Optional[transformers.PretrainedConfig]`, defaults to `None`): Model configuration.
        local_files_only (`Optional[bool]`, defaults to `False`): Use only local files.
        trust_remote_code (`bool`, defaults to `False`): Whether to trust custom code in remote repos.
        revision (`Optional[str]`, defaults to `None`): Model version or git revision (branch, tag, or commit).
"""

# This class is a placeholder for optimized models.
class PreTrainedModel(ABC):
    """
    A placeholder class for compatibility between optimized models and Hugging Face transformers.
    This abstract base class is used as a foundation for creating optimized model classes.
    """
    pass


# Base class for optimized models that handles inference, saving/loading, and pushing to the Hub.
class OptimizedModel(PreTrainedModel):
    """
    OptimizedModel is a base class for models that have undergone optimization for inference.

    Attributes:
        config_class (AutoConfig): Used to automatically load model configurations from Hugging Face.
        base_model_prefix (str): A prefix used to identify the optimized model.
        config_name (str): The configuration name, typically referring to a JSON configuration file.

    Methods:
        __call__: Calls the forward pass of the model.
        forward: Abstract method that must be implemented for model inference.
        save_pretrained: Saves the optimized model to a directory.
        push_to_hub: Pushes the saved model to the Hugging Face Hub.
        _save_pretrained: Abstract method for saving model weights (must be implemented).
        _save_config: Saves model configuration.
        from_pretrained: Loads the model from pretrained weights and configurations.
    """

    config_class = AutoConfig  # Use AutoConfig to load model configurations automatically.
    base_model_prefix = "optimized_model"  # The prefix for optimized models.
    config_name = CONFIG_NAME  # Refers to the configuration file name, such as 'config.json'.

    def __init__(self, model: Union["PreTrainedModel", "TFPreTrainedModel"], config: PretrainedConfig):
        """
        Initializes the OptimizedModel.

        Args:
            model (Union["PreTrainedModel", "TFPreTrainedModel"]): The pre-trained model.
            config (PretrainedConfig): The model configuration.
        """
        super().__init__() #Constructor from the parent (PreTrainedModel) class 
        self.model = model  # Stores the loaded model.
        self.config = config  # Stores the configuration.
        self.preprocessors = []  # List to store model preprocessors.

    def __call__(self, *args, **kwargs):
        """
        Enables calling the model instance as a function, triggering the forward pass.

        Args:
            *args: Positional arguments for the forward method.
            **kwargs: Keyword arguments for the forward method.

        Returns:
            The output of the model's forward pass.
        """
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Abstract method for performing the forward pass of the model.
        Must be implemented in subclasses.

        Args:
            *args: Positional arguments for the model's forward pass.
            **kwargs: Keyword arguments for the model's forward pass.

        Raises:
            NotImplementedError: This method must be overridden by subclasses.
        """
        raise NotImplementedError("Forward pass must be implemented.")

    def save_pretrained(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
        """
        Saves the model and configuration files to a directory.

        Args:
            save_directory (Union[str, os.PathLike]): Directory where the model and config will be saved.
            push_to_hub (bool, optional): If True, pushes the model to the Hugging Face Hub.
            **kwargs: Additional arguments for `push_to_hub`.

        Raises:
            ValueError: If the provided path is a file instead of a directory.
        """
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file.")
            return

        # Create the save directory if it doesn't exist.
        os.makedirs(save_directory, exist_ok=True)

        # Save model configuration and preprocessor information.
        self._save_config(save_directory)
        for preprocessor in self.preprocessors:
            preprocessor.save_pretrained(save_directory)

        # Save the model itself.
        self._save_pretrained(save_directory)

        # If specified, push the model to Hugging Face Hub.
        if push_to_hub:
            return self.push_to_hub(save_directory, **kwargs)

    @abstractmethod
    def _save_pretrained(self, save_directory):
        """
        Abstract method for saving model weights.

        Args:
            save_directory (Union[str, os.PathLike]): Directory where the model weights will be saved.

        Raises:
            NotImplementedError: This method must be overridden by subclasses.
        """
        raise NotImplementedError("This method should be implemented in subclasses to save model weights.")

    def _save_config(self, save_directory):
        """
        Saves the model configuration to the specified directory.

        Args:
            save_directory (Union[str, os.PathLike]): Directory where the configuration will be saved.
        """
        self.config.save_pretrained(save_directory)

    def push_to_hub(self, save_directory: str, repository_id: str, private: Optional[bool] = None, token: Optional[Union[bool, str]] = None):
        """
        Pushes the saved model to the Hugging Face Hub.

        Args:
            save_directory (str): Directory where the model is saved.
            repository_id (str): Repository ID on Hugging Face Hub.
            private (Optional[bool]): If True, the repository will be private.
            token (Optional[Union[bool, str]]): Authentication token for pushing the model to the Hub.
        """
        # Create a repository on Hugging Face Hub.
        create_repo(token=token, repo_id=repository_id, exist_ok=True, private=private)

        # Upload files in the save directory to the repository.
        for path, subdirs, files in os.walk(save_directory):
            for name in files:
                local_file_path = os.path.join(path, name)
                _, hub_file_path = os.path.split(local_file_path)
                try:
                    upload_file(token=token, repo_id=repository_id, path_or_fileobj=os.path.join(os.getcwd(), local_file_path), path_in_repo=hub_file_path)
                except (KeyError, NameError):
                    pass

    @classmethod
    def from_pretrained(cls, model_id: Union[str, Path], **kwargs):
        """
        Loads an optimized model from a pre-trained configuration.

        Args:
            model_id (Union[str, Path]): The model ID or path to a local model directory.
            **kwargs: Additional arguments for loading the model.

        Returns:
            OptimizedModel: The loaded optimized model.
        """
        return cls._from_pretrained(model_id, **kwargs)
