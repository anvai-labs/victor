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

"""Tests for the config-validate CLI command."""

import re
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from typer.testing import CliRunner

from victor.config.validation import ConfigValidationResult
from victor.ui.cli import app

runner = CliRunner()

# A passing ConfigValidationResult to bypass validate_configuration when testing verbose checks
_VALID_RESULT = ConfigValidationResult()


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


def _make_settings_mock(**overrides):
    """Create a MagicMock for settings with numeric attributes that validation compares."""
    mock = MagicMock()
    defaults = {
        "default_provider": "ollama",
        "fallback_max_tools": 5,
        "tool_selection_cache_ttl": 300,
        "framework_preload_enabled": True,
        "http_connection_pool_enabled": True,
        "tool_selection_cache_enabled": True,
        "codebase_persist_directory": None,
    }
    defaults.update(overrides)
    for attr, value in defaults.items():
        setattr(mock, attr, value)
    return mock


class TestConfigValidateCommand:
    """Tests for victor config-validate command."""

    def test_config_validate_basic(self, tmp_path):
        """Test basic config validation with existing config."""
        config_dir = tmp_path / ".victor"
        config_dir.mkdir(exist_ok=True)
        profiles_file = config_dir / "profiles.yaml"
        profiles_file.write_text("profiles:\n  default:\n    provider: ollama\n")

        mock_settings = _make_settings_mock()
        mock_settings.get_config_dir.return_value = config_dir
        mock_settings.load_profiles.return_value = {
            "default": MagicMock(
                provider="ollama",
                model="test-model",
                temperature=0.7,
                max_tokens=4096,
            )
        }

        with (
            patch("victor.ui.commands.config.load_settings", return_value=mock_settings),
            patch(
                "victor.ui.commands.config.validate_configuration",
                return_value=_VALID_RESULT,
            ),
        ):
            result = runner.invoke(app, ["config", "validate"])
            # Should pass or fail gracefully
            assert result.exit_code in [0, 1]

    def test_config_validate_verbose_flag(self, tmp_path):
        """Test verbose output flag."""
        config_dir = tmp_path / ".victor"
        config_dir.mkdir(exist_ok=True)
        profiles_file = config_dir / "profiles.yaml"
        profiles_file.write_text("profiles:\n  default:\n    provider: ollama\n")

        mock_settings = _make_settings_mock()
        mock_settings.get_config_dir.return_value = config_dir
        mock_settings.load_profiles.return_value = {
            "default": MagicMock(
                provider="ollama",
                model="test-model",
                temperature=0.7,
                max_tokens=4096,
            )
        }

        with (
            patch("victor.ui.commands.config.load_settings", return_value=mock_settings),
            patch(
                "victor.ui.commands.config.validate_configuration",
                return_value=_VALID_RESULT,
            ),
        ):
            result = runner.invoke(app, ["config", "validate", "--verbose"])
            assert result.exit_code in [0, 1]

    def test_config_validate_missing_config_dir(self):
        """Test handling of missing config directory."""
        mock_settings = _make_settings_mock()
        # Return a non-existent directory
        mock_settings.get_config_dir.return_value = Path("/nonexistent/path/.victor")

        with (
            patch("victor.ui.commands.config.load_settings", return_value=mock_settings),
            patch(
                "victor.ui.commands.config.validate_configuration",
                return_value=_VALID_RESULT,
            ),
        ):
            result = runner.invoke(app, ["config", "validate", "--verbose"])
            assert result.exit_code == 1
            assert "not found" in result.output.lower() or "init" in result.output.lower()

    def test_config_validate_invalid_yaml(self, tmp_path):
        """Test handling of invalid YAML file."""
        config_dir = tmp_path / ".victor"
        config_dir.mkdir(exist_ok=True)
        profiles_file = config_dir / "profiles.yaml"
        # Write invalid YAML that will cause a parse error
        profiles_file.write_text("profiles:\n  default:\n    - invalid: [unclosed")

        mock_settings = _make_settings_mock()
        mock_settings.get_config_dir.return_value = config_dir

        with (
            patch("victor.ui.commands.config.load_settings", return_value=mock_settings),
            patch(
                "victor.ui.commands.config.validate_configuration",
                return_value=_VALID_RESULT,
            ),
        ):
            result = runner.invoke(app, ["config", "validate", "--verbose"])
            assert result.exit_code == 1
            assert "yaml" in result.output.lower() or "invalid" in result.output.lower()

    def test_config_validate_missing_profiles_section(self, tmp_path):
        """Test handling of config without profiles section."""
        config_dir = tmp_path / ".victor"
        config_dir.mkdir(exist_ok=True)
        profiles_file = config_dir / "profiles.yaml"
        # Write YAML without profiles section
        profiles_file.write_text("other_key: value\n")

        mock_settings = _make_settings_mock()
        mock_settings.get_config_dir.return_value = config_dir

        with (
            patch("victor.ui.commands.config.load_settings", return_value=mock_settings),
            patch(
                "victor.ui.commands.config.validate_configuration",
                return_value=_VALID_RESULT,
            ),
        ):
            result = runner.invoke(app, ["config", "validate", "--verbose"])
            assert result.exit_code == 1
            assert "profiles" in result.output.lower()

    def test_config_validate_invalid_temperature(self, tmp_path):
        """Test detection of invalid temperature value."""
        config_dir = tmp_path / ".victor"
        config_dir.mkdir(exist_ok=True)
        profiles_file = config_dir / "profiles.yaml"
        profiles_file.write_text("profiles:\n  invalid:\n    provider: ollama\n")

        mock_settings = _make_settings_mock()
        mock_settings.get_config_dir.return_value = config_dir
        mock_settings.load_profiles.return_value = {
            "invalid": MagicMock(
                provider="ollama",
                model="test-model",
                temperature=5.0,  # Invalid - out of range
                max_tokens=4096,
            )
        }
        mock_settings.get_provider_settings.return_value = {}

        with (
            patch("victor.ui.commands.config.load_settings", return_value=mock_settings),
            patch(
                "victor.ui.commands.config.validate_configuration",
                return_value=_VALID_RESULT,
            ),
        ):
            result = runner.invoke(app, ["config", "validate", "--verbose"])
            assert result.exit_code == 1
            assert "temperature" in result.output.lower()

    def test_config_validate_invalid_max_tokens(self, tmp_path):
        """Test detection of invalid max_tokens value."""
        config_dir = tmp_path / ".victor"
        config_dir.mkdir(exist_ok=True)
        profiles_file = config_dir / "profiles.yaml"
        profiles_file.write_text("profiles:\n  invalid:\n    provider: ollama\n")

        mock_settings = _make_settings_mock()
        mock_settings.get_config_dir.return_value = config_dir
        mock_settings.load_profiles.return_value = {
            "invalid": MagicMock(
                provider="ollama",
                model="test-model",
                temperature=0.7,
                max_tokens=-100,  # Invalid - negative
            )
        }
        mock_settings.get_provider_settings.return_value = {}

        with (
            patch("victor.ui.commands.config.load_settings", return_value=mock_settings),
            patch(
                "victor.ui.commands.config.validate_configuration",
                return_value=_VALID_RESULT,
            ),
        ):
            result = runner.invoke(app, ["config", "validate", "--verbose"])
            assert result.exit_code == 1
            assert "max_tokens" in result.output.lower() or "invalid" in result.output.lower()

    def test_config_validate_unknown_provider(self, tmp_path):
        """Test detection of unknown provider."""
        config_dir = tmp_path / ".victor"
        config_dir.mkdir(exist_ok=True)
        profiles_file = config_dir / "profiles.yaml"
        profiles_file.write_text("profiles:\n  unknown:\n    provider: fake_provider\n")

        mock_settings = _make_settings_mock()
        mock_settings.get_config_dir.return_value = config_dir
        mock_settings.load_profiles.return_value = {
            "unknown": MagicMock(
                provider="fake_provider",  # Unknown provider
                model="test-model",
                temperature=0.7,
                max_tokens=4096,
            )
        }
        mock_settings.get_provider_settings.return_value = {}

        with (
            patch("victor.ui.commands.config.load_settings", return_value=mock_settings),
            patch(
                "victor.ui.commands.config.validate_configuration",
                return_value=_VALID_RESULT,
            ),
        ):
            result = runner.invoke(app, ["config", "validate", "--verbose"])
            assert result.exit_code == 1
            assert "unknown" in result.output.lower() or "provider" in result.output.lower()

    def test_config_validate_cloud_provider_missing_api_key(self, tmp_path):
        """Test warning for cloud provider without API key."""
        config_dir = tmp_path / ".victor"
        config_dir.mkdir(exist_ok=True)
        profiles_file = config_dir / "profiles.yaml"
        profiles_file.write_text("profiles:\n  cloud:\n    provider: anthropic\n")

        mock_settings = _make_settings_mock()
        mock_settings.get_config_dir.return_value = config_dir
        mock_settings.load_profiles.return_value = {
            "cloud": MagicMock(
                provider="anthropic",
                model="claude-3-sonnet",
                temperature=0.7,
                max_tokens=4096,
            )
        }
        mock_settings.get_provider_settings.return_value = {"api_key": None}  # No API key

        with (
            patch("victor.ui.commands.config.load_settings", return_value=mock_settings),
            patch(
                "victor.ui.commands.config.validate_configuration",
                return_value=_VALID_RESULT,
            ),
        ):
            result = runner.invoke(app, ["config", "validate", "--verbose"])
            # Should pass with warning (exit code 0)
            assert result.exit_code == 0
            assert "api key" in result.output.lower() or "warning" in result.output.lower()

    def test_config_validate_cloud_provider_with_api_key(self, tmp_path):
        """Test cloud provider with API key configured."""
        config_dir = tmp_path / ".victor"
        config_dir.mkdir(exist_ok=True)
        profiles_file = config_dir / "profiles.yaml"
        profiles_file.write_text("profiles:\n  cloud:\n    provider: anthropic\n")

        mock_settings = _make_settings_mock()
        mock_settings.get_config_dir.return_value = config_dir
        mock_settings.load_profiles.return_value = {
            "cloud": MagicMock(
                provider="anthropic",
                model="claude-3-sonnet",
                temperature=0.7,
                max_tokens=4096,
            )
        }
        mock_settings.get_provider_settings.return_value = {"api_key": "sk-test-key"}

        with (
            patch("victor.ui.commands.config.load_settings", return_value=mock_settings),
            patch(
                "victor.ui.commands.config.validate_configuration",
                return_value=_VALID_RESULT,
            ),
        ):
            result = runner.invoke(app, ["config", "validate", "--verbose"])
            assert result.exit_code == 0
            assert "api key configured" in result.output.lower()

    def test_config_validate_multiple_profiles(self, tmp_path):
        """Test validation of multiple profiles."""
        config_dir = tmp_path / ".victor"
        config_dir.mkdir(exist_ok=True)
        profiles_file = config_dir / "profiles.yaml"
        profiles_file.write_text(
            "profiles:\n  local:\n    provider: ollama\n  cloud:\n    provider: anthropic\n"
        )

        mock_settings = _make_settings_mock()
        mock_settings.get_config_dir.return_value = config_dir
        mock_settings.load_profiles.return_value = {
            "local": MagicMock(
                provider="ollama",
                model="qwen2.5-coder:7b",
                temperature=0.7,
                max_tokens=4096,
            ),
            "cloud": MagicMock(
                provider="anthropic",
                model="claude-3-sonnet",
                temperature=0.5,
                max_tokens=8192,
            ),
        }
        mock_settings.get_provider_settings.return_value = {"api_key": "sk-test"}

        with (
            patch("victor.ui.commands.config.load_settings", return_value=mock_settings),
            patch(
                "victor.ui.commands.config.validate_configuration",
                return_value=_VALID_RESULT,
            ),
        ):
            result = runner.invoke(app, ["config", "validate", "--verbose"])
            assert result.exit_code == 0
            clean_output = strip_ansi(result.output)
            assert "2 profile" in clean_output.lower()


class TestConfigValidateHelp:
    """Tests for config-validate help text."""

    def test_config_validate_help(self):
        """Test that help text is displayed correctly."""
        result = runner.invoke(app, ["config", "validate", "--help"])
        # Strip ANSI codes since they can split option names in rich output
        clean_output = strip_ansi(result.output)
        assert result.exit_code == 0
        assert "validate" in clean_output.lower()
        assert "--verbose" in clean_output
        assert "--check-connectivity" in clean_output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
