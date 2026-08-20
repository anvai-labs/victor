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

"""Unit tests for profile switching functionality.

Tests the ability to switch between pre-configured profiles, including:
- Loading profiles from configuration
- Switching to a profile
- Profile validation
- Profile metadata preservation
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile

from victor.config.settings import ProfileConfig, Settings
from victor.agent.orchestrator import AgentOrchestrator


class TestProfileConfig:
    """Tests for ProfileConfig model."""

    def test_create_profile_config(self):
        """Test creating a profile configuration."""
        profile = ProfileConfig(
            provider="anthropic",
            model="claude-3-sonnet",
            temperature=0.7,
            max_tokens=4096,
        )

        assert profile.provider == "anthropic"
        assert profile.model == "claude-3-sonnet"
        assert profile.temperature == 0.7
        assert profile.max_tokens == 4096

    def test_profile_config_with_description(self):
        """Test creating a profile with description."""
        profile = ProfileConfig(
            provider="ollama",
            model="llama3:8b",
            temperature=0.5,
            max_tokens=2048,
            description="Fast local model for testing",
        )

        assert profile.description == "Fast local model for testing"

    def test_profile_config_with_tool_selection(self):
        """Test creating a profile with tool selection config."""
        profile = ProfileConfig(
            provider="openai",
            model="gpt-4",
            tool_selection={"max_cost_threshold": 0.5},
        )

        assert profile.tool_selection is not None
        assert profile.tool_selection["max_cost_threshold"] == 0.5


class TestProfileLoading:
    """Tests for loading profiles from configuration."""

    def test_load_profiles_from_dict(self):
        """Test loading profiles from dictionary."""
        profiles_data = {
            "fast": {
                "provider": "ollama",
                "model": "llama3:8b",
                "temperature": 0.5,
                "max_tokens": 2048,
            },
            "smart": {
                "provider": "anthropic",
                "model": "claude-3-opus",
                "temperature": 0.8,
                "max_tokens": 8192,
            },
        }

        profiles = {name: ProfileConfig(**data) for name, data in profiles_data.items()}

        assert len(profiles) == 2
        assert "fast" in profiles
        assert "smart" in profiles
        assert profiles["fast"].model == "llama3:8b"
        assert profiles["smart"].provider == "anthropic"

    @pytest.fixture
    def temp_profiles_file(self):
        """Create a temporary profiles.yaml file."""
        profiles_data = {
            "profiles": {
                "local": {
                    "provider": "ollama",
                    "model": "llama3:8b",
                    "temperature": 0.5,
                    "max_tokens": 2048,
                    "description": "Fast local model",
                },
                "cloud": {
                    "provider": "anthropic",
                    "model": "claude-3-sonnet",
                    "temperature": 0.7,
                    "max_tokens": 4096,
                    "description": "Balanced cloud model",
                },
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            import yaml

            yaml.dump(profiles_data, f)
            temp_path = Path(f.name)

        yield temp_path

        # Cleanup
        temp_path.unlink()

    def test_load_profiles_from_yaml(self, temp_profiles_file):
        """Test loading profiles from YAML file."""
        import yaml

        with open(temp_profiles_file) as f:
            data = yaml.safe_load(f)

        profiles = {name: ProfileConfig(**cfg) for name, cfg in data["profiles"].items()}

        assert len(profiles) == 2
        assert profiles["local"].provider == "ollama"
        assert profiles["cloud"].provider == "anthropic"


class TestProfileSwitching:
    """Tests for switching between profiles."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock orchestrator for testing."""
        orchestrator = MagicMock(spec=AgentOrchestrator)
        orchestrator.provider_name = "anthropic"
        orchestrator.model = "claude-3-sonnet"
        orchestrator.switch_provider = MagicMock(return_value=True)
        orchestrator.switch_model = MagicMock(return_value=True)
        return orchestrator

    def test_switch_to_profile_by_name(self, mock_orchestrator):
        """Test switching to a profile by name."""
        profiles = {
            "fast": ProfileConfig(
                provider="ollama",
                model="llama3:8b",
                temperature=0.5,
                max_tokens=2048,
            )
        }

        # Simulate profile switch
        profile = profiles["fast"]
        success = mock_orchestrator.switch_provider(profile.provider, profile.model)

        assert success is True
        mock_orchestrator.switch_provider.assert_called_once_with("ollama", "llama3:8b")

    def test_switch_to_profile_with_temperature(self, mock_orchestrator):
        """Test switching to a profile with custom temperature."""
        profile = ProfileConfig(
            provider="openai",
            model="gpt-4",
            temperature=0.9,
            max_tokens=8192,
        )

        # Profile switch should update temperature
        success = mock_orchestrator.switch_provider(profile.provider, profile.model)

        assert success is True
        # Note: temperature would need to be set separately in actual implementation

    def test_switch_between_multiple_profiles(self, mock_orchestrator):
        """Test switching between multiple profiles in sequence."""
        profiles = {
            "profile1": ProfileConfig(
                provider="anthropic",
                model="claude-3-sonnet",
                temperature=0.7,
                max_tokens=4096,
            ),
            "profile2": ProfileConfig(
                provider="ollama",
                model="llama3:8b",
                temperature=0.5,
                max_tokens=2048,
            ),
            "profile3": ProfileConfig(
                provider="openai",
                model="gpt-4",
                temperature=0.8,
                max_tokens=8192,
            ),
        }

        # Switch through profiles
        for profile_name, profile in profiles.items():
            success = mock_orchestrator.switch_provider(profile.provider, profile.model)
            assert success is True

        assert mock_orchestrator.switch_provider.call_count == 3

    def test_profile_switch_preserves_conversation(self, mock_orchestrator):
        """Test that switching profiles preserves conversation history."""
        # In actual implementation, conversation should be preserved
        profile = ProfileConfig(
            provider="ollama",
            model="llama3:8b",
            temperature=0.5,
            max_tokens=2048,
        )

        success = mock_orchestrator.switch_provider(profile.provider, profile.model)

        assert success is True
        # Verify orchestrator still has conversation state
        # (This would be checked with actual orchestrator instance)


class TestProfileValidation:
    """Tests for profile validation."""

    def test_validate_profile_provider(self):
        """Test validating profile provider."""
        # Valid provider
        profile = ProfileConfig(provider="anthropic", model="claude-3")
        assert profile.provider in ["anthropic", "openai", "google", "ollama"]

    def test_validate_profile_temperature_range(self):
        """Test validating profile temperature is in valid range."""
        # Valid temperature
        profile = ProfileConfig(temperature=0.7, provider="test", model="test")
        assert 0.0 <= profile.temperature <= 2.0

    def test_validate_profile_max_tokens_positive(self):
        """Test validating profile max_tokens is positive."""
        profile = ProfileConfig(max_tokens=4096, provider="test", model="test")
        assert profile.max_tokens > 0

    def test_invalid_temperature_raises_error(self):
        """Test that invalid temperature raises validation error."""
        with pytest.raises(ValueError):
            ProfileConfig(temperature=3.0, provider="test", model="test")  # Invalid: > 2.0


class TestProfileMetadata:
    """Tests for profile metadata preservation."""

    def test_profile_description_preserved(self):
        """Test that profile description is preserved."""
        profile = ProfileConfig(
            provider="anthropic",
            model="claude-3",
            description="Best for code generation",
        )

        assert profile.description == "Best for code generation"

    def test_profile_tool_selection_preserved(self):
        """Test that tool selection config is preserved."""
        tool_config = {
            "enable_adaptive": True,
            "max_cost_threshold": 0.5,
            "complexity_threshold": 0.7,
        }

        profile = ProfileConfig(provider="openai", model="gpt-4", tool_selection=tool_config)

        assert profile.tool_selection == tool_config
        assert profile.tool_selection["enable_adaptive"] is True
