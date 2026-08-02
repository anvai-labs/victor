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

"""Tests for model existence validation."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from victor.config.settings import Settings, validate_default_model


class TestModelValidation:
    """Test model existence validation logic."""

    def test_validate_non_ollama_provider(self):
        """Validation should pass for non-Ollama providers."""
        settings = Settings(provider={"default_provider": "anthropic"})
        is_valid, warning = validate_default_model(settings)

        assert is_valid is True
        assert warning is None

    @patch("subprocess.run")
    def test_validate_ollama_not_running(self, mock_run: MagicMock):
        """Should show warning when Ollama is not running."""
        # Mock Ollama command failure
        mock_run.return_value = MagicMock(returncode=1)

        settings = Settings(
            provider={"default_provider": "ollama", "default_model": "qwen2.5-coder:7b"}
        )
        is_valid, warning = validate_default_model(settings)

        assert is_valid is False
        assert warning is not None
        assert "Ollama is not running" in warning
        assert "ollama serve" in warning

    @patch("subprocess.run")
    def test_validate_model_not_installed(self, mock_run: MagicMock):
        """Should show warning when default model is not installed."""
        # Mock successful Ollama list but model not in list
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="MODEL ID\tSIZE\tMODIFIED\nqwen2.5:7b\t4.7 GB\t2025-01-15\n",
        )

        settings = Settings(
            provider={"default_provider": "ollama", "default_model": "qwen3-coder:30b"}
        )
        is_valid, warning = validate_default_model(settings)

        assert is_valid is False
        assert warning is not None
        assert "is not installed" in warning
        assert "qwen3-coder:30b" in warning
        assert "ollama pull qwen3-coder:30b" in warning

    @patch("subprocess.run")
    def test_validate_model_exists(self, mock_run: MagicMock):
        """Should pass validation when model exists."""
        # Mock successful Ollama list with model present
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="MODEL ID\tSIZE\tMODIFIED\nqwen2.5-coder:7b\t4.7 GB\t2025-01-15\n",
        )

        settings = Settings(
            provider={"default_provider": "ollama", "default_model": "qwen2.5-coder:7b"}
        )
        is_valid, warning = validate_default_model(settings)

        assert is_valid is True
        assert warning is None

    @patch("subprocess.run")
    def test_validate_ollama_binary_not_found(self, mock_run: MagicMock):
        """Should show installation guide when Ollama binary is missing."""
        # Mock FileNotFoundError
        mock_run.side_effect = FileNotFoundError()

        settings = Settings(
            provider={"default_provider": "ollama", "default_model": "qwen2.5-coder:7b"}
        )
        is_valid, warning = validate_default_model(settings)

        assert is_valid is False
        assert warning is not None
        assert "Ollama binary not found" in warning
        assert "https://ollama.com" in warning

    @patch("subprocess.run")
    def test_validate_ollama_timeout(self, mock_run: MagicMock):
        """Should show timeout message when Ollama command times out."""
        # Mock timeout
        mock_run.side_effect = subprocess.TimeoutExpired("ollama", 5)

        settings = Settings(
            provider={"default_provider": "ollama", "default_model": "qwen2.5-coder:7b"}
        )
        is_valid, warning = validate_default_model(settings)

        assert is_valid is False
        assert warning is not None
        assert "timed out" in warning

    @patch("subprocess.run")
    def test_validate_unexpected_error_passes(self, mock_run: MagicMock):
        """Should not block startup on unexpected errors."""
        # Mock unexpected exception
        mock_run.side_effect = RuntimeError("Unexpected error")

        settings = Settings(
            provider={"default_provider": "ollama", "default_model": "qwen2.5-coder:7b"}
        )
        is_valid, warning = validate_default_model(settings)

        # Should pass to avoid blocking startup
        assert is_valid is True
        assert warning is None
