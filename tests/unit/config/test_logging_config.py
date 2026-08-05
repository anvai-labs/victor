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

"""Tests for centralized logging configuration."""

import logging
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from victor.config.config_loaders import (
    LoggingConfig,
    get_logging_config,
    get_default_logging_config,
    _apply_env_overrides,
)


class TestLoggingConfig:
    """Test LoggingConfig dataclass."""

    def test_default_values(self):
        """Test default values are correct."""
        config = LoggingConfig()
        assert config.console_level == "WARNING"
        assert config.file_level == "INFO"
        assert config.file_enabled is True
        assert "victor.log" in config.file_path
        assert config.file_max_bytes == 10 * 1024 * 1024  # 10MB
        assert config.file_backup_count == 5
        assert config.event_logging is True
        assert config.module_levels == {}

    def test_expanded_file_path(self):
        """Test that ~ is expanded in file path."""
        config = LoggingConfig(file_path="~/.victor/logs/test.log")
        expanded = config.expanded_file_path
        assert "~" not in str(expanded)
        assert expanded.name == "test.log"

    def test_get_console_level_int(self):
        """Test console level conversion to int."""
        config = LoggingConfig(console_level="DEBUG")
        assert config.get_console_level_int() == logging.DEBUG

        config = LoggingConfig(console_level="INFO")
        assert config.get_console_level_int() == logging.INFO

        config = LoggingConfig(console_level="WARNING")
        assert config.get_console_level_int() == logging.WARNING

        config = LoggingConfig(console_level="ERROR")
        assert config.get_console_level_int() == logging.ERROR

    def test_get_file_level_int(self):
        """Test file level conversion to int."""
        config = LoggingConfig(file_level="DEBUG")
        assert config.get_file_level_int() == logging.DEBUG

        config = LoggingConfig(file_level="INFO")
        assert config.get_file_level_int() == logging.INFO

    def test_invalid_level_fallback(self):
        """Test that invalid levels fall back to defaults."""
        config = LoggingConfig(console_level="INVALID")
        assert config.get_console_level_int() == logging.WARNING

        config = LoggingConfig(file_level="INVALID")
        assert config.get_file_level_int() == logging.INFO


class TestGetLoggingConfig:
    """Test get_logging_config function."""

    def test_default_config(self):
        """Test getting default config."""
        config = get_logging_config()
        assert config.console_level == "WARNING"
        assert config.file_level == "INFO"
        assert config.file_enabled is True

    def test_command_override(self):
        """Test command-specific overrides from package config."""
        # 'serve' command should have INFO console level
        config = get_logging_config(command="serve")
        assert config.console_level == "INFO"

        # 'debug' command should have DEBUG level
        config = get_logging_config(command="debug")
        assert config.console_level == "DEBUG"
        assert config.file_level == "DEBUG"

    def test_cli_override(self):
        """Test CLI argument overrides."""
        config = get_logging_config(cli_console_level="DEBUG")
        assert config.console_level == "DEBUG"

        config = get_logging_config(cli_file_level="ERROR")
        assert config.file_level == "ERROR"

    def test_cli_overrides_command(self):
        """Test that CLI overrides command-specific settings."""
        # 'serve' has INFO default, but CLI DEBUG should win
        config = get_logging_config(command="serve", cli_console_level="DEBUG")
        assert config.console_level == "DEBUG"

    def test_module_levels(self):
        """Test that module levels are loaded from config."""
        config = get_logging_config()
        # These should be loaded from logging_config.yaml
        assert "httpx" in config.module_levels
        assert config.module_levels["httpx"] == "WARNING"


class TestEnvOverrides:
    """Test environment variable overrides."""

    def test_victor_log_level_override(self):
        """Test VICTOR_LOG_LEVEL env var override."""
        config = LoggingConfig()
        with patch.dict(os.environ, {"VICTOR_LOG_LEVEL": "DEBUG"}):
            config = _apply_env_overrides(config)
            assert config.console_level == "DEBUG"

    def test_victor_log_file_level_override(self):
        """Test VICTOR_LOG_FILE_LEVEL env var override."""
        config = LoggingConfig()
        with patch.dict(os.environ, {"VICTOR_LOG_FILE_LEVEL": "ERROR"}):
            config = _apply_env_overrides(config)
            assert config.file_level == "ERROR"

    def test_victor_log_file_override(self):
        """Test VICTOR_LOG_FILE env var override."""
        config = LoggingConfig()
        with patch.dict(os.environ, {"VICTOR_LOG_FILE": "/tmp/custom.log"}):
            config = _apply_env_overrides(config)
            assert config.file_path == "/tmp/custom.log"

    def test_victor_log_disabled(self):
        """Test VICTOR_LOG_DISABLED env var."""
        config = LoggingConfig()
        with patch.dict(os.environ, {"VICTOR_LOG_DISABLED": "true"}):
            config = _apply_env_overrides(config)
            assert config.file_enabled is False

    def test_env_priority_in_get_logging_config(self):
        """Test that env vars are applied in get_logging_config."""
        with patch.dict(os.environ, {"VICTOR_LOG_LEVEL": "ERROR"}):
            config = get_logging_config()
            assert config.console_level == "ERROR"


class TestGetDefaultLoggingConfig:
    """Test get_default_logging_config convenience function."""

    def test_returns_config(self):
        """Test that it returns a LoggingConfig."""
        config = get_default_logging_config()
        assert isinstance(config, LoggingConfig)

    def test_same_as_get_logging_config(self):
        """Test that it's equivalent to get_logging_config with no args."""
        default = get_default_logging_config()
        explicit = get_logging_config()
        assert default.console_level == explicit.console_level
        assert default.file_level == explicit.file_level
