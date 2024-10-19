"""
Conftest Module for Promise Optimizer Tests

This module contains directory-specific settings for the test suite of Promise Optimizer. 
It is automatically executed by pytest before any tests are run. The primary goal is to 
configure the test environment, modify the sys path for multi-repository setups, and 
customize doctest behavior for test output validation.

Key functionalities:
- Modifies `sys.path` to ensure tests can run across multiple checkouts of the repository.
- Registers a custom option flag for doctest (`IGNORE_RESULT`) to ignore specific output checks.
- Implements a `CustomOutputChecker` class that overrides doctest's default output checker 
  behavior to allow ignoring certain results during doctest runs.
"""

import doctest  # For running and customizing doctest behavior
import sys  # For modifying the system path
from os.path import abspath, dirname, join  # For handling file paths

# Modify sys.path to prioritize the current repository's "src" folder, allowing
# tests to run across multiple checkouts without needing to reinstall dependencies.
git_repo_path = abspath(join(dirname(__file__), "src"))
sys.path.insert(1, git_repo_path)

# Custom doctest flag for ignoring specific output in tests.
IGNORE_RESULT = doctest.register_optionflag("IGNORE_RESULT")

# Inherit the default OutputChecker from doctest
OutputChecker = doctest.OutputChecker


class CustomOutputChecker(OutputChecker):
    """
    Custom OutputChecker for doctests.

    This class overrides the default `check_output` method of doctest's `OutputChecker`.
    It allows test cases marked with the `IGNORE_RESULT` flag to bypass result checking,
    which is useful for cases where the output is expected to vary but should not affect
    the test result.

    Methods:
        - check_output: Overrides default behavior to check the IGNORE_RESULT flag.
    """
    def check_output(self, want, got, optionflags):
        """
        Checks whether the test output matches the expected output.

        Args:
            want (str): The expected output in the doctest.
            got (str): The actual output produced by the test.
            optionflags (int): The option flags for doctest, including custom flags.

        Returns:
            bool: True if the output matches or if IGNORE_RESULT is set, False otherwise.
        """
        # Skip output comparison if IGNORE_RESULT is set
        if IGNORE_RESULT & optionflags:
            return True
        # Otherwise, use the default output checking behavior
        return OutputChecker.check_output(self, want, got, optionflags)


# Override doctest's OutputChecker with the custom checker.
doctest.OutputChecker = CustomOutputChecker
