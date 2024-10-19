# coding=utf-8
"""
Configuration Utilities for Promise Optimizer

This module defines the base configuration class (`BaseConfig`) used to handle model configuration files
in the Promise Optimizer project. The `BaseConfig` class extends Hugging Face's `PretrainedConfig` class
but provides additional functionalities and customization for saving, loading, and managing configurations,
particularly with version compatibility checks for the Transformers library.

Key functionalities:
- Compatibility with different versions of Hugging Face Transformers (handling version threshold).
- Support for saving configuration files to a directory and optionally pushing them to the Hugging Face Hub.
- Custom regular expressions for matching and managing configuration files.
- Handling of both old and new versions of the Transformers library (below and above version 4.22).

Classes:
    - BaseConfig: Extends `PretrainedConfig` to handle custom configurations and saving/loading processes.

"""

import copy  # For deep copying of dictionaries/objects
import json  # For saving/loading configurations in JSON format
import os  # File system operations (e.g., creating directories, saving files)
import re  # Regular expressions for file matching and version checks
import warnings  # Issuing deprecation warnings
from typing import Any, Dict, List, Tuple, Union  # Type hinting utilities

from packaging import version  # For version parsing and comparison
from transformers import PretrainedConfig  # Hugging Face's base class for configuration management
from transformers import __version__ as transformers_version_str  # Get the version of the installed Transformers library

from .utils import logging  # Custom logging for Promise Optimizer
from .version import __version__  # Version of Promise Optimizer for comparison

# Check the Transformers library version and set a compatibility threshold (4.22)
_transformers_version = version.parse(transformers_version_str)
_transformers_version_threshold = (4, 22)
_transformers_version_is_below_threshold = (
    _transformers_version.major,
    _transformers_version.minor,
) < _transformers_version_threshold

# Import different utilities based on the Transformers version
if _transformers_version_is_below_threshold:
    from transformers.utils import cached_path, hf_bucket_url  # Pre-4.22 utilities
else:
    from transformers.dynamic_module_utils import custom_object_save  # Post-4.22 utility
    from transformers.utils import cached_file, download_url, extract_commit_hash, is_remote_url  # Newer utilities

# Set up a logger for this module
logger = logging.get_logger(__name__)

class BaseConfig(PretrainedConfig):
    """
    Base class for configuration classes in Promise Optimizer that extend `PretrainedConfig` but use a different
    configuration file naming convention. This class supports both old and new versions of the Hugging Face Transformers
    library and includes custom behavior for saving and managing configuration files.

    Attributes:
        CONFIG_NAME (str): The name of the primary configuration file (default: 'config.json').
        FULL_CONFIGURATION_FILE (str): Full file name of the configuration file (default: 'config.json').

    Methods:
        - save_pretrained: Saves the configuration to a directory and optionally pushes it to the Hugging Face Hub.
        - _re_configuration_file: Returns a regular expression pattern for matching configuration files.
    """

    CONFIG_NAME = "config.json"  # Default name for configuration files
    FULL_CONFIGURATION_FILE = "config.json"  # Full name for configuration files

    @classmethod
    def _re_configuration_file(cls):
        """
        Generates a regular expression for matching configuration file names based on the class's
        `FULL_CONFIGURATION_FILE` attribute.

        Returns:
            re.Pattern: A compiled regular expression pattern that matches configuration file names.
        """
        return re.compile(rf"{cls.FULL_CONFIGURATION_FILE.split('.')[0]}(.*)\.json")

    def save_pretrained(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
        """
        Save the current configuration to the directory `save_directory`, ensuring that it can be reloaded
        later using the `from_pretrained` class method.

        Args:
            save_directory (`str` or `os.PathLike`): The directory where the configuration JSON file will be saved.
                The directory will be created if it does not exist.
            push_to_hub (`bool`, optional): Whether to push the configuration to the Hugging Face Model Hub. Defaults to `False`.
            kwargs: Additional arguments for the Hugging Face Hub push (e.g., commit message).

        Raises:
            AssertionError: If the provided `save_directory` is a file, rather than a directory.
        """
        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file.")

        # Create the directory if it doesn't exist (for transformers version >= 4.22)
        if not _transformers_version_is_below_threshold:
            os.makedirs(save_directory, exist_ok=True)

        # Handle pushing to the Hugging Face Model Hub, if enabled
        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)

            if _transformers_version_is_below_threshold:
                repo = self._create_or_get_repo(save_directory, **kwargs)
            else:
                repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
                repo_id = self._create_repo(repo_id, **kwargs)
                token = kwargs.get("token", None)

                if "use_auth_token" in kwargs:
                    warnings.warn(
                        "The `use_auth_token` argument is deprecated. Please use `token` instead.",
                        FutureWarning,
                    )
                    if token is not None:
                        raise ValueError("Cannot use both `use_auth_token` and `token` simultaneously.")
                    kwargs["token"] = kwargs.pop("use_auth_token")

                files_timestamps = self._get_files_timestamps(save_directory)

        # Ensure the directory exists for pre-4.22 versions
        if _transformers_version_is_below_threshold:
            os.makedirs(save_directory, exist_ok=True)

        # If we have a custom configuration, save the custom config in the directory
        if self._auto_class is not None:
            custom_object_save(self, save_directory, config=self)

        # Save the configuration using the default file name
        output_config_file = os.path.join(save_directory, self.CONFIG_NAME)
        self.to_json_file(output_config_file, use_diff=True)
        logger.info(f"Configuration saved in {output_config_file}")

        # Handle pushing to the Model Hub if enabled
        if push_to_hub:
            if _transformers_version_is_below_threshold:
                url = self._push_to_hub(repo, commit_message=commit_message)
                logger.info(f"Configuration pushed to the hub in this commit: {url}")
            else:
                self._upload_modified_files(
                    save_directory, repo_id, files_timestamps, commit_message=commit_message, token=token
                )
    # Adapted from transformers.configuration_utils.PretrainedConfig.get_configuration_file
    @classmethod
    def get_configuration_file(cls, configuration_files: List[str]) -> str:
        """
        Determines the appropriate configuration file to use based on the version of Transformers or Promise Optimizer.

        Args:
            configuration_files (List[str]): A list of available configuration files to choose from.

        Returns:
            str: The name of the configuration file that matches the current version of Transformers or Promise Optimizer.
        """
        # Map to store configuration files and their corresponding versions
        configuration_files_map = {}
        _re_configuration_file = cls._re_configuration_file()  # Get regex pattern to identify valid config files
        for file_name in configuration_files:
            search = _re_configuration_file.search(file_name)  # Match configuration file with regex
            if search is not None:
                v = search.groups()[0]  # Extract the version part from the file name
                configuration_files_map[v] = file_name
        available_versions = sorted(configuration_files_map.keys())  # Sort versions

        # Default to the class's config file and check for newer versions
        configuration_file = cls.CONFIG_NAME
        optimum_version = version.parse(__version__)  # Get the version of Promise Optimizer
        for v in available_versions:
            if version.parse(v) <= optimum_version:
                configuration_file = configuration_files_map[v]
            else:
                break  # Stop once we find a newer version that's not applicable

        return configuration_file

    # Adapted from transformers.configuration_utils.PretrainedConfig.get_config_dict
    @classmethod
    def get_config_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Load the configuration dictionary from a given model checkpoint or directory.

        Args:
            pretrained_model_name_or_path (str or os.PathLike): The name or path of the pre-trained model or directory 
                containing the configuration.

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: A tuple containing the configuration dictionary and any additional arguments.
        """
        original_kwargs = copy.deepcopy(kwargs)  # Preserve the original kwargs
        # Retrieve the base config dictionary using helper method `_get_config_dict`
        config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, **kwargs)
        if "_commit_hash" in config_dict:
            original_kwargs["_commit_hash"] = config_dict["_commit_hash"]  # Save the commit hash

        # If the config file points to another config file, retrieve it
        if "configuration_files" in config_dict:
            configuration_file = cls.get_configuration_file(config_dict["configuration_files"])
            config_dict, kwargs = cls._get_config_dict(
                pretrained_model_name_or_path, _configuration_file=configuration_file, **original_kwargs
            )

        return config_dict, kwargs

    # Adapted from transformers.configuration_utils.PretrainedConfig._get_config_dict
    @classmethod
    def _get_config_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Helper method to retrieve the configuration dictionary, either from a local file, cache, or remote repository.

        Args:
            pretrained_model_name_or_path (str or os.PathLike): The path or identifier of the pre-trained model.
            kwargs: Additional arguments for file retrieval, such as cache settings and authentication tokens.

        Returns:
            Tuple[Dict, Dict]: A tuple containing the configuration dictionary and any remaining kwargs.
        """
        # Handle optional arguments
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        use_auth_token = kwargs.pop("use_auth_token", None)
        token = kwargs.pop("token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        subfolder = kwargs.pop("subfolder", "")
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        commit_hash = kwargs.pop("_commit_hash", None)

        # Deprecation warning for `use_auth_token`
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed soon. Please use the `token` argument instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError("You cannot use both `use_auth_token` and `token` arguments at the same time.")
            token = use_auth_token

        if trust_remote_code is True:
            logger.warning(
                "The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is ignored."
            )

        # Create user agent metadata
        user_agent = {"file_type": "config", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)  # Ensure string format for the path

        is_local = os.path.isdir(pretrained_model_name_or_path)
        # Handle case when the model path is a local file
        if os.path.isfile(os.path.join(subfolder, pretrained_model_name_or_path)):
            resolved_config_file = pretrained_model_name_or_path
            is_local = True
        # Handle pre-4.22 versions of Transformers
        elif _transformers_version_is_below_threshold and os.path.isdir(pretrained_model_name_or_path):
            configuration_file = kwargs.pop("_configuration_file", cls.CONFIG_NAME)
            resolved_config_file = os.path.join(pretrained_model_name_or_path, configuration_file)
            if not os.path.isfile(resolved_config_file):
                raise EnvironmentError(f"Could not locate {configuration_file} inside {pretrained_model_name_or_path}.")
        # Handle remote file resolution
        elif not _transformers_version_is_below_threshold and is_remote_url(pretrained_model_name_or_path):
            configuration_file = pretrained_model_name_or_path
            resolved_config_file = download_url(pretrained_model_name_or_path)
        else:
            configuration_file = kwargs.pop("_configuration_file", cls.CONFIG_NAME)

            try:
                if _transformers_version_is_below_threshold:
                    # Load from URL or cache (pre-4.22)
                    config_file = hf_bucket_url(
                        pretrained_model_name_or_path,
                        filename=configuration_file,
                        revision=revision,
                        subfolder=subfolder if len(subfolder) > 0 else None,
                    )
                    resolved_config_file = cached_path(
                        config_file,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        proxies=proxies,
                        resume_download=resume_download,
                        local_files_only=local_files_only,
                        token=token,
                        user_agent=user_agent,
                    )
                else:
                    # Load from cache or download (4.22 and above)
                    resolved_config_file = cached_file(
                        pretrained_model_name_or_path,
                        configuration_file,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        proxies=proxies,
                        resume_download=resume_download,
                        local_files_only=local_files_only,
                        token=token,
                        user_agent=user_agent,
                        revision=revision,
                        subfolder=subfolder,
                        _commit_hash=commit_hash,
                    )
                    commit_hash = extract_commit_hash(resolved_config_file, commit_hash)
            except EnvironmentError:
                # Raise the specific environment error for caching
                raise
            except Exception:
                # Generic error handling for unexpected failures
                raise EnvironmentError(
                    f"Can't load the configuration of '{pretrained_model_name_or_path}'. "
                    "If you were trying to load it from 'https://huggingface.co/models', ensure no local directory exists with the same name. "
                    f"Otherwise, verify that '{pretrained_model_name_or_path}' contains a {configuration_file} file."
                )

        try:
            # Load the configuration dictionary from the JSON file
            config_dict = cls._dict_from_json_file(resolved_config_file)
            if _transformers_version_is_below_threshold:
                config_dict["_commit_hash"] = commit_hash  # Add commit hash for old versions
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise EnvironmentError(f"It appears the config file at '{resolved_config_file}' is not a valid JSON file.")

        # Log where the configuration file was loaded from
        if is_local:
            logger.info(f"Loading configuration file {resolved_config_file}")
        else:
            logger.info(f"Loading configuration file {configuration_file} from cache at {resolved_config_file}")

        return config_dict, kwargs
    # Adapted from transformers.configuration_utils.PretrainedConfig.from_dict
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> "PretrainedConfig":
        """
        Instantiates a [`PretrainedConfig`] object from a Python dictionary of parameters.

        This method allows you to load a configuration object from a dictionary that could have been
        obtained from a pre-trained model checkpoint using the `get_config_dict` method. The method also supports 
        additional keyword arguments (`kwargs`) for initializing or overriding specific configuration attributes.

        Args:
            config_dict (Dict[str, Any]): The dictionary containing configuration parameters. This can come from a
                checkpoint or manually created.
            kwargs (Dict[str, Any]): Additional keyword arguments for initializing or modifying the configuration object.

        Returns:
            PretrainedConfig: An instance of the configuration class populated with the parameters from `config_dict`.

        Raises:
            ValueError: If the number of labels (`num_labels`) does not match the `id2label` mapping in `kwargs`.
        """
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)
        
        # Remove internal telemetry arguments to avoid exposing them as unused kwargs
        kwargs.pop("_from_auto", None)
        kwargs.pop("_from_pipeline", None)

        # Preserve any commit hash present in the config_dict
        if "_commit_hash" in kwargs and "_commit_hash" in config_dict:
            kwargs["_commit_hash"] = config_dict["_commit_hash"]

        # Create a configuration instance using the parameters from config_dict
        config = cls(**config_dict)

        # Handle pruned heads, ensuring the dictionary keys are integers
        if hasattr(config, "pruned_heads"):
            config.pruned_heads = {int(key): value for key, value in config.pruned_heads.items()}

        # Ensure consistency between num_labels and id2label in kwargs, raising an error if they are incompatible
        if "num_labels" in kwargs and "id2label" in kwargs:
            num_labels = kwargs["num_labels"]
            id2label = kwargs["id2label"] if kwargs["id2label"] is not None else []
            if len(id2label) != num_labels:
                raise ValueError(
                    f"Inconsistent arguments: `num_labels={num_labels }` does not match `id2label` length: {len(id2label)}."
                    " Please ensure that `num_labels` and `id2label` are consistent."
                )

        # Update the config with any additional kwargs and remove used kwargs
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                if key != "torch_dtype":
                    to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        # Log the final configuration object
        logger.info(config)

        # Return config and unused kwargs if `return_unused_kwargs` is True
        if return_unused_kwargs:
            return config, kwargs
        else:
            return config

    # Adapted from transformers.configuration_utils.PretrainedConfig.to_dict
    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the configuration instance into a Python dictionary.

        This method converts the configuration object into a deep-copied dictionary representation, 
        which can be used to save the configuration to a file, transfer it over a network, or further 
        modify it programmatically.

        Returns:
            Dict[str, Any]: A dictionary containing all the configuration parameters and their values.
        """
        # Create a deep copy of the configuration's attributes (to avoid modifying the original object)
        output = copy.deepcopy(self.__dict__)

        # Add the model type to the dictionary if it exists in the class
        if hasattr(self.__class__, "model_type"):
            output["model_type"] = self.__class__.model_type

        # Remove internal attributes that should not be serialized
        if "_auto_class" in output:
            del output["_auto_class"]
        if "_commit_hash" in output:
            del output["_commit_hash"]

        # Add the version of Transformers and Promise Optimizer at the time of serialization
        output["transformers_version"] = transformers_version_str
        output["optimum_version"] = __version__

        # Ensure torch dtype attributes are correctly serialized as strings
        self.dict_torch_dtype_to_str(output)

        return output
