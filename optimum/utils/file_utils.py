"""
Utility Functions for File Handling in Promise Optimizer

This module provides utility functions for handling files both locally and on the Hugging Face Hub. It includes 
functions for checking file existence, finding files that match specific patterns, and ensuring compatibility 
between local and remote resources.

Functions:
    - validate_file_exists: Checks if a specified file exists in a local directory or a remote Hugging Face repository.
    - find_files_matching_pattern: Searches for files that match a given pattern in a local directory or model repo.
"""

import re  # For regular expression matching
import warnings  # To handle deprecation warnings
from pathlib import Path  # For handling file paths
from typing import List, Optional, Union  # For type hints

import huggingface_hub  # To interact with Hugging Face Hub
from huggingface_hub import get_hf_file_metadata, hf_hub_url  # Specific functions to fetch metadata from the hub

from ..utils import logging  # Local logging utility from the utils module


logger = logging.get_logger(__name__)  # Initialize logger for this module


def validate_file_exists(
    model_name_or_path: Union[str, Path], filename: str, subfolder: str = "", revision: Optional[str] = None
) -> bool:
    """
    Checks that the file called `filename` exists in the `model_name_or_path` directory or model repo.

    This function checks both local directories and remote Hugging Face repositories. It returns a boolean value 
    indicating whether the file exists at the specified path.

    Args:
        model_name_or_path (`Union[str, Path]`): The local directory path or model repository name.
        filename (`str`): The name of the file to check for existence.
        subfolder (`str`, *optional*): Subfolder inside the model directory or repo. Defaults to `""`.
        revision (`Optional[str]`, *optional*): The specific model version to check, such as branch or commit. Defaults to `None`.

    Returns:
        `bool`: Returns `True` if the file exists, `False` otherwise.
    """
    model_path = Path(model_name_or_path) if isinstance(model_name_or_path, str) else model_name_or_path
    if model_path.is_dir():
        return (model_path / subfolder / filename).is_file()  # Check local file existence
    succeeded = True
    try:
        # Check remote file existence on Hugging Face Hub
        get_hf_file_metadata(hf_hub_url(model_name_or_path, filename, subfolder=subfolder, revision=revision))
    except Exception:
        succeeded = False
    return succeeded


def find_files_matching_pattern(
    model_name_or_path: Union[str, Path],
    pattern: str,
    glob_pattern: str = "**/*",
    subfolder: str = "",
    use_auth_token: Optional[Union[bool, str]] = None,
    token: Optional[Union[bool, str]] = None,
    revision: Optional[str] = None,
) -> List[Path]:
    """
    Scans either a model repo or a local directory to find filenames matching the pattern.

    This function supports both local directories and Hugging Face repositories. It looks for files that match a 
    specified pattern and returns a list of `Path` objects representing the matching files.

    Args:
        model_name_or_path (`Union[str, Path]`): The name of the model repo on the Hugging Face Hub or the path to a local directory.
        pattern (`str`): The regex pattern to use to look for matching files.
        glob_pattern (`str`, defaults to `"**/*"`): The glob pattern to list all files that need to be checked.
        subfolder (`str`, defaults to `""`): Subfolder within the model directory/repo to narrow the search.
        use_auth_token (`Optional[Union[bool,str]]`, defaults to `None`): Deprecated. Use the `token` argument instead.
        token (`Optional[Union[bool,str]]`, defaults to `None`): The token to use for HTTP bearer authorization for remote files.
        revision (`Optional[str]`, defaults to `None`): The specific model version to use (branch, tag, or commit id).

    Returns:
        `List[Path]`: A list of paths to the files matching the specified pattern.
    
    Raises:
        ValueError: If both `use_auth_token` and `token` arguments are provided.
    """

    if use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed soon. Please use the `token` argument instead.",
            FutureWarning,
        )
        if token is not None:
            raise ValueError("You cannot use both `use_auth_token` and `token` arguments at the same time.")
        token = use_auth_token

    model_path = Path(model_name_or_path) if isinstance(model_name_or_path, str) else model_name_or_path
    pattern = re.compile(f"{subfolder}/{pattern}" if subfolder != "" else pattern)  # Compile regex pattern
    if model_path.is_dir():
        # Search local directory
        path = model_path
        files = model_path.glob(glob_pattern)  # Search with the provided glob pattern
        files = [p for p in files if re.search(pattern, str(p))]  # Filter files by matching the regex pattern
    else:
        # Search remote Hugging Face repo
        path = model_name_or_path
        repo_files = map(Path, huggingface_hub.list_repo_files(model_name_or_path, revision=revision, token=token))
        if subfolder != "":
            path = f"{path}/{subfolder}"
        files = [Path(p) for p in repo_files if re.match(pattern, str(p))]

    return files
