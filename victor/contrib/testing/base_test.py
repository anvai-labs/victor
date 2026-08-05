# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Base test case class for Victor verticals.

This module provides VerticalTestCase, a reusable base class for
testing verticals. It provides common test utilities and setup
that verticals can use to avoid duplicating test code.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Optional
from unittest import mock

import pytest


class VerticalTestCase:
    """Base test case class for Victor verticals.

    Provides common test utilities:
    - Temporary directory management
    - Environment variable isolation
    - Mock fixture access
    - Common assertions

    Usage:
        class TestMyVertical(VerticalTestCase):
            vertical_name = "myvertical"

            def test_something(self):
                # Use self.temp_dir for temporary files
                # Use self.assert_* methods for assertions
                pass
    """

    # Vertical-specific configuration (override in subclasses)
    vertical_name: str = ""
    assistant_class: Optional[type] = None

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp(prefix=f"victor_{self.vertical_name}_")

        # Store original environment
        self.original_environ = os.environ.copy()

    def teardown_method(self):
        """Clean up after each test method."""
        import shutil

        # Remove temporary directory
        if hasattr(self, "temp_dir") and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

        # Restore environment
        if hasattr(self, "original_environ"):
            os.environ.clear()
            os.environ.update(self.original_environ)

    # ==========================================================================
    # Environment Helpers
    # ==========================================================================

    def set_env_var(self, key: str, value: str) -> None:
        """Set an environment variable for testing.

        Args:
            key: Environment variable name
            value: Environment variable value
        """
        os.environ[key] = value

    def unset_env_var(self, key: str) -> None:
        """Unset an environment variable.

        Args:
            key: Environment variable name to unset
        """
        os.environ.pop(key, None)

    def clear_env_vars(self, *keys: str) -> None:
        """Clear multiple environment variables.

        Args:
            *keys: Environment variable names to clear
        """
        for key in keys:
            self.unset_env_var(key)

    # ==========================================================================
    # File System Helpers
    # ==========================================================================

    def create_temp_file(self, content: str, filename: str = "test.txt") -> str:
        """Create a temporary file with content.

        Args:
            content: Content to write to the file
            filename: Name for the file

        Returns:
            Path to the created file
        """
        filepath = os.path.join(self.temp_dir, filename)
        with open(filepath, "w") as f:
            f.write(content)
        return filepath

    def create_temp_dir(self, dirname: str = "test_dir") -> str:
        """Create a temporary subdirectory.

        Args:
            dirname: Name for the directory

        Returns:
            Path to the created directory
        """
        dirpath = os.path.join(self.temp_dir, dirname)
        os.makedirs(dirpath, exist_ok=True)
        return dirpath

    def read_file(self, filepath: str) -> str:
        """Read a file's content.

        Args:
            filepath: Path to the file

        Returns:
            File content as string
        """
        with open(filepath, "r") as f:
            return f.read()

    # ==========================================================================
    # Assertion Helpers
    # ==========================================================================

    def assert_in_logs(self, message: str, logs: str) -> None:
        """Assert that a message appears in logs.

        Args:
            message: Message to look for
            logs: Log content to search

        Raises:
            AssertionError: If message not found in logs
        """
        assert message in logs, f"Message '{message}' not found in logs"

    def assert_not_in_logs(self, message: str, logs: str) -> None:
        """Assert that a message does not appear in logs.

        Args:
            message: Message to look for
            logs: Log content to search

        Raises:
            AssertionError: If message found in logs
        """
        assert message not in logs, f"Message '{message}' unexpectedly found in logs"

    def assert_dict_contains(self, actual: dict, expected: dict) -> None:
        """Assert that actual dict contains all key-value pairs from expected.

        Args:
            actual: Actual dictionary
            expected: Expected key-value pairs

        Raises:
            AssertionError: If expected keys not in actual or values don't match
        """
        for key, value in expected.items():
            assert key in actual, f"Key '{key}' not found in actual dict"
            assert (
                actual[key] == value
            ), f"Value mismatch for key '{key}': expected {value}, got {actual[key]}"

    def assert_has_keys(self, data: dict, *keys: str) -> None:
        """Assert that a dict has all specified keys.

        Args:
            data: Dictionary to check
            *keys: Keys that must be present

        Raises:
            AssertionError: If any key is missing
        """
        missing_keys = [key for key in keys if key not in data]
        assert not missing_keys, f"Keys missing from dict: {missing_keys}"


__all__ = [
    "VerticalTestCase",
]
