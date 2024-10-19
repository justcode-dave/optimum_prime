"""
save_utils.py

This module provides utilities for loading and saving model preprocessors such as tokenizers, processors, 
and feature extractors from Hugging Face's Transformers library. It allows for loading these components 
from a source directory or model name, and saving them to a specified destination directory.

Functions:
    - maybe_load_preprocessors: Attempts to load a tokenizer, processor, and feature extractor from a given 
      source path or model.
    - maybe_save_preprocessors: Saves available preprocessors from the source path to the destination directory.

Typical usage example:
    preprocessors = maybe_load_preprocessors("path/to/model")
    maybe_save_preprocessors("path/to/model", "path/to/save")
"""

import logging  # Logging utility to capture events and debug messages
from pathlib import Path  # Path object to handle file system paths
from typing import List, Union  # Type hints for function arguments and return types

from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer  # Pretrained components from Hugging Face

# Set up logger for this module
logger = logging.getLogger(__name__)

def maybe_load_preprocessors(
    src_name_or_path: Union[str, Path], subfolder: str = "", trust_remote_code: bool = False
) -> List:
    """
    Attempts to load various preprocessors (AutoTokenizer, AutoProcessor, AutoFeatureExtractor) from a given source path.

    Args:
        src_name_or_path (Union[str, Path]): The source directory or model name from which to load the preprocessors.
        subfolder (str, optional): Subfolder within the source directory where the preprocessor files might reside. Default is "".
        trust_remote_code (bool, optional): If set to True, it will allow loading models that may execute arbitrary code. Default is False.

    Returns:
        List: A list of successfully loaded preprocessors.
    """
    preprocessors = []  # List to hold the loaded preprocessors

    # Try to load the tokenizer
    try:
        preprocessors.append(
            AutoTokenizer.from_pretrained(src_name_or_path, subfolder=subfolder, trust_remote_code=trust_remote_code)
        )
    except Exception:
        pass  # Silently skip if loading fails

    # Try to load the processor
    try:
        preprocessors.append(
            AutoProcessor.from_pretrained(src_name_or_path, subfolder=subfolder, trust_remote_code=trust_remote_code)
        )
    except Exception:
        pass  # Silently skip if loading fails

    # Try to load the feature extractor
    try:
        preprocessors.append(
            AutoFeatureExtractor.from_pretrained(
                src_name_or_path, subfolder=subfolder, trust_remote_code=trust_remote_code
            )
        )
    except Exception:
        pass  # Silently skip if loading fails
    
    return preprocessors  # Return the list of successfully loaded preprocessors


def maybe_save_preprocessors(
    src_name_or_path: Union[str, Path],
    dest_dir: Union[str, Path],
    src_subfolder: str = "",
    trust_remote_code: bool = False,
):
    """
    Saves available preprocessors (tokenizer, processor, feature extractor) from the source path to the destination directory.

    Args:
        src_name_or_path (Union[str, Path]): The source directory or model name from which to load the preprocessors.
        dest_dir (Union[str, Path]): The destination directory where the preprocessors will be saved.
        src_subfolder (str, optional): Subfolder within the source directory where the preprocessor files might reside. Default is "".
        trust_remote_code (bool, optional): If set to True, it will allow loading models that may execute arbitrary code. Default is False.
    """
    # Ensure dest_dir is a Path object, create the destination directory if it doesn't exist
    if not isinstance(dest_dir, Path):
        dest_dir = Path(dest_dir)

    dest_dir.mkdir(exist_ok=True)  # Create destination directory if it doesn't exist

    # Load and save each preprocessor found at the source
    for preprocessor in maybe_load_preprocessors(
        src_name_or_path, subfolder=src_subfolder, trust_remote_code=trust_remote_code
    ):
        preprocessor.save_pretrained(dest_dir)  # Save each preprocessor to the destination directory
