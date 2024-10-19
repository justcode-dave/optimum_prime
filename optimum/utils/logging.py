"""
logging.py

This module provides logging utilities for managing and customizing log outputs for the Hugging Face Transformers library.
It encapsulates functionalities for configuring loggers, setting log levels, enabling or disabling handlers, 
controlling log message propagation, and formatting log outputs. 

Key Functions:
--------------
- `get_logger(name: Optional[str] = None) -> logging.Logger`:
    Returns a logger with the specified name, with default configuration applied if needed.
    
- `get_verbosity() -> int`:
    Returns the current verbosity level of the root logger.

- `set_verbosity(verbosity: int) -> None`:
    Sets the verbosity level of the root logger to one of the supported log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL).

- `disable_default_handler() -> None`:
    Disables the default log handler, stopping log messages from being displayed.

- `enable_default_handler() -> None`:
    Enables the default log handler for the library, allowing log messages to be displayed.

- `add_handler(handler: logging.Handler) -> None`:
    Adds a custom log handler to the root logger for fine-grained control of log outputs.

- `remove_handler(handler: logging.Handler) -> None`:
    Removes a specified log handler from the root logger.

- `enable_propagation() -> None` and `disable_propagation() -> None`:
    Controls log message propagation to parent loggers. Propagation is disabled by default.

- `enable_explicit_format() -> None` and `reset_format() -> None`:
    Controls the log format for all handlers. The explicit format includes detailed log message components 
    like the log level, filename, line number, and timestamp.

Logging Levels:
---------------
Logging levels supported are mapped to standard Python logging levels:
- DEBUG (10)
- INFO (20)
- WARNING (30)
- ERROR (40)
- CRITICAL (50)

Environment Variables:
----------------------
- `TRANSFORMERS_VERBOSITY`:
    Can be set to one of the valid log levels ("debug", "info", "warning", "error", "critical") to override 
    the default log level globally for the library.

Thread Safety:
--------------
The module ensures thread safety by using locks when configuring or modifying the root logger.

Caching:
--------
- `warn_once`: Uses `lru_cache` to ensure that certain warnings are only logged once, preventing redundant messages.
"""


import logging  # Core Python logging module
import os  # For environment variable access
import sys  # For system-level operations
import threading  # For thread-safe logging configuration
from functools import lru_cache  # Caching mechanism to avoid repeated function calls
from logging import (  # Import different logging levels
    CRITICAL,  # NOQA
    DEBUG,  # NOQA
    ERROR,  # NOQA
    FATAL,  # NOQA
    INFO,  # NOQA
    NOTSET,  # NOQA
    WARN,  # NOQA
    WARNING,  # NOQA
)
from typing import TYPE_CHECKING, Optional  # Typing support for type hints

# Optional type-checking logger
if TYPE_CHECKING:
    from logging import Logger

# Thread lock for thread-safe logging operations
_lock = threading.Lock()
# Default handler, initialized as None
_default_handler: Optional[logging.Handler] = None

# Mapping of log levels to their respective numeric values
log_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

# Default logging level set to WARNING
_default_log_level = logging.WARNING


def _get_default_logging_level():
    """
    Determines the default logging level based on the environment variable `TRANSFORMERS_VERBOSITY`.

    If the `TRANSFORMERS_VERBOSITY` variable is set to a valid log level, return that as the default level.
    If not set or invalid, it returns the default log level defined in `_default_log_level`.

    Returns:
        int: The appropriate logging level (e.g., DEBUG, INFO, etc.)
    """
    env_level_str = os.getenv("TRANSFORMERS_VERBOSITY", None)
    if env_level_str:
        if env_level_str in log_levels:
            return log_levels[env_level_str]
        else:
            logging.getLogger().warning(
                f"Unknown option TRANSFORMERS_VERBOSITY={env_level_str}, "
                f"has to be one of: {', '.join(log_levels.keys())}"
            )
    return _default_log_level


def _get_library_name() -> str:
    """
    Returns the name of the library for which the logger is configured.

    This typically corresponds to the base module name (e.g., 'transformers').

    Returns:
        str: The name of the library.
    """
    return __name__.split(".")[0]  # Split by '.' to get the base module name


def _get_library_root_logger() -> logging.Logger:
    """
    Fetches the root logger for the library.

    This logger is used as the main logger for all library-related log messages.

    Returns:
        logging.Logger: The root logger instance.
    """
    return logging.getLogger(_get_library_name())  # Returns logger based on library name


def _configure_library_root_logger() -> None:
    """
    Configures the root logger for the library if it's not already configured.

    It sets up a default logging handler and applies the default logging level from environment variables or
    falls back to the default log level.
    """
    global _default_handler

    with _lock:  # Ensure thread-safe operations
        if _default_handler:
            # If already configured, return
            return
        _default_handler = logging.StreamHandler()  # Assign sys.stderr as the stream handler
        _default_handler.flush = sys.stderr.flush  # Ensure stderr flushes correctly

        # Fetch and configure the library root logger
        library_root_logger = _get_library_root_logger()
        library_root_logger.addHandler(_default_handler)  # Attach the default handler
        library_root_logger.setLevel(_get_default_logging_level())  # Set the log level
        library_root_logger.propagate = False  # Disable log propagation to avoid duplicate logging


def _reset_library_root_logger() -> None:
    """
    Resets the root logger configuration by removing the default handler and restoring the log level to NOTSET.
    """
    global _default_handler

    with _lock:  # Ensure thread-safe operations
        if not _default_handler:
            return

        # Remove the handler and reset logging configuration
        library_root_logger = _get_library_root_logger()
        library_root_logger.removeHandler(_default_handler)
        library_root_logger.setLevel(logging.NOTSET)
        _default_handler = None  # Clear the default handler


def get_log_levels_dict():
    """
    Returns the dictionary of available log levels.

    Returns:
        dict: A dictionary mapping log level names to their numeric values.
    """
    return log_levels  # Returns the log level mappings


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Returns a logger instance for the specified name.

    This is the primary logger for the library, and if no name is specified, it defaults to the library's root logger.

    Args:
        name (Optional[str], defaults to None): The name of the logger. If None, returns the root logger.

    Returns:
        logging.Logger: The logger instance.
    """
    if name is None:
        name = _get_library_name()  # Default to the library name

    _configure_library_root_logger()  # Ensure the logger is configured
    return logging.getLogger(name)  # Return the logger


def get_verbosity() -> int:
    """
    Retrieves the current logging verbosity level for the library's root logger.

    This function fetches the effective logging level.

    Returns:
        int: The current logging level (e.g., INFO, WARNING, DEBUG).
    """
    _configure_library_root_logger()  # Ensure logger is configured
    return _get_library_root_logger().getEffectiveLevel()  # Return the current logging level


def set_verbosity(verbosity: int) -> None:
    """
    Sets the logging verbosity level for the library's root logger.

    Args:
        verbosity (int): The desired logging level (e.g., logging.DEBUG, logging.INFO).
    """
    _configure_library_root_logger()  # Ensure logger is configured
    _get_library_root_logger().setLevel(verbosity)  # Set the desired verbosity level


def set_verbosity_info():
    """Set the logging verbosity to the INFO level."""
    return set_verbosity(INFO)


def set_verbosity_warning():
    """Set the logging verbosity to the WARNING level."""
    return set_verbosity(WARNING)
def set_verbosity_debug():
    """Set the logging verbosity to the DEBUG level."""
    return set_verbosity(DEBUG)


def set_verbosity_error():
    """Set the logging verbosity to the ERROR level."""
    return set_verbosity(ERROR)


def disable_default_handler() -> None:
    """
    Disable the default handler of the Hugging Face Transformers root logger.

    This function removes the default logging handler, preventing the library's log messages from being displayed.
    """
    _configure_library_root_logger()  # Ensure logger is configured

    assert _default_handler is not None  # Ensure the default handler exists
    _get_library_root_logger().removeHandler(_default_handler)  # Remove the default handler


def enable_default_handler() -> None:
    """
    Enable the default handler for the Hugging Face Transformers root logger.

    This function re-enables the default handler if it has been disabled previously.
    """
    _configure_library_root_logger()  # Ensure logger is configured

    assert _default_handler is not None  # Ensure the default handler exists
    _get_library_root_logger().addHandler(_default_handler)  # Add the default handler back


def add_handler(handler: logging.Handler) -> None:
    """
    Add a custom logging handler to the Hugging Face Transformers root logger.

    Args:
        handler (logging.Handler): The handler to be added to the logger.
    """
    _configure_library_root_logger()  # Ensure logger is configured

    assert handler is not None  # Ensure a valid handler is provided
    _get_library_root_logger().addHandler(handler)  # Add the provided handler


def remove_handler(handler: logging.Handler) -> None:
    """
    Remove a specific logging handler from the Hugging Face Transformers root logger.

    Args:
        handler (logging.Handler): The handler to be removed from the logger.
    """
    _configure_library_root_logger()  # Ensure logger is configured

    assert handler is not None and handler in _get_library_root_logger().handlers  # Ensure the handler exists in logger
    _get_library_root_logger().removeHandler(handler)  # Remove the specified handler


def disable_propagation() -> None:
    """
    Disable log message propagation for the library.

    Propagation is disabled by default. This function ensures log messages don't propagate to parent loggers.
    """
    _configure_library_root_logger()  # Ensure logger is configured
    _get_library_root_logger().propagate = False  # Disable propagation


def enable_propagation() -> None:
    """
    Enable log message propagation for the library.

    Use this function with caution. If the root logger is configured with its own handlers, you may experience double logging.
    """
    _configure_library_root_logger()  # Ensure logger is configured
    _get_library_root_logger().propagate = True  # Enable propagation


def enable_explicit_format() -> None:
    """
    Enable explicit log formatting for all Hugging Face Transformers loggers.

    The log format is set as:
    ::

        [LEVELNAME|FILENAME|LINE NUMBER] TIME >> MESSAGE

    This function affects all handlers currently bound to the root logger.
    """
    handlers = _get_library_root_logger().handlers  # Retrieve all handlers

    for handler in handlers:
        # Define the new formatter
        formatter = logging.Formatter("[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s")
        handler.setFormatter(formatter)  # Apply the formatter to each handler


def reset_format() -> None:
    """
    Reset the log formatting for all Hugging Face Transformers loggers.

    This function removes the explicit formatting for all handlers, resetting them to their default format.
    """
    handlers = _get_library_root_logger().handlers  # Retrieve all handlers

    for handler in handlers:
        handler.setFormatter(None)  # Reset the formatter to default for each handler


@lru_cache(None)
def warn_once(logger: "Logger", msg: str):
    """
    Issue a warning message only once per unique warning, regardless of how many times it is triggered.

    Args:
        logger (Logger): The logger instance used to issue the warning.
        msg (str): The warning message to be logged.
    """
    logger.warning(msg)  # Log the warning
