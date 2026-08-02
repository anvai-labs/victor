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

"""Tests for configuration profiles."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest

from victor.config.profiles import (
    ProfileLevel,
    ProfileTemplate,
    PROFILES,
    BASIC_PROFILE,
    ADVANCED_PROFILE,
    EXPERT_PROFILE,
    CODING_PROFILE,
    RESEARCH_PROFILE,
    list_profiles,
    get_profile,
    get_profiles_by_level,
    get_recommended_profile,
    generate_profile_yaml,
    _detect_provider,
    _detect_model_for_provider,
    install_profile,
    get_current_profile,
)


class TestProfileLevel:
    """Tests for ProfileLevel enum."""

    def test_level_values(self):
        """ProfileLevel has correct values."""
        assert ProfileLevel.BASIC.value == "basic"
        assert ProfileLevel.ADVANCED.value == "advanced"
        assert ProfileLevel.EXPERT.value == "expert"


class TestProfileTemplate:
    """Tests for ProfileTemplate dataclass."""

    def test_basic_profile_structure(self):
        """Basic profile has required fields."""
        assert BASIC_PROFILE.name == "basic"
        assert BASIC_PROFILE.display_name == "Basic"
        assert BASIC_PROFILE.level == ProfileLevel.BASIC
        assert isinstance(BASIC_PROFILE.settings, dict)
        assert isinstance(BASIC_PROFILE.provider_settings, dict)

    def test_advanced_profile_structure(self):
        """Advanced profile has required fields."""
        assert ADVANCED_PROFILE.name == "advanced"
        assert ADVANCED_PROFILE.level == ProfileLevel.ADVANCED

    def test_expert_profile_structure(self):
        """Expert profile has required fields."""
        assert EXPERT_PROFILE.name == "expert"
        assert EXPERT_PROFILE.level == ProfileLevel.EXPERT


class TestProfiles:
    """Tests for profile registry and lookup."""

    def test_all_profiles_registered(self):
        """All predefined profiles are registered."""
        assert "basic" in PROFILES
        assert "advanced" in PROFILES
        assert "expert" in PROFILES
        assert "coding" in PROFILES
        assert "research" in PROFILES

    def test_list_profiles(self):
        """Listing profiles returns all profiles."""
        profiles = list_profiles()
        assert len(profiles) == 6
        assert all(isinstance(p, ProfileTemplate) for p in profiles)

    def test_get_profile_by_name(self):
        """Getting profile by name works."""
        basic = get_profile("basic")
        assert basic is not None
        assert basic.name == "basic"

    def test_get_profile_invalid(self):
        """Getting invalid profile returns None."""
        invalid = get_profile("invalid")
        assert invalid is None

    def test_get_profiles_by_level(self):
        """Filtering profiles by level works."""
        basic_profiles = get_profiles_by_level(ProfileLevel.BASIC)
        assert len(basic_profiles) == 1
        assert basic_profiles[0].name == "basic"

        advanced_profiles = get_profiles_by_level(ProfileLevel.ADVANCED)
        # coding and research are advanced
        assert len(advanced_profiles) >= 2


class TestProfileSettings:
    """Tests for profile settings."""

    def test_basic_profile_settings(self):
        """Basic profile has conservative defaults."""
        settings = BASIC_PROFILE.settings
        assert settings["default_provider"] == "ollama"
        assert settings["fallback_max_tools"] == 5
        assert settings["framework_preload_enabled"] is True

    def test_advanced_profile_settings(self):
        """Advanced profile has higher tool budget."""
        settings = ADVANCED_PROFILE.settings
        assert settings["fallback_max_tools"] == 10
        assert settings["framework_preload_enabled"] is True

    def test_expert_profile_settings(self):
        """Expert profile has maximum flexibility."""
        settings = EXPERT_PROFILE.settings
        assert settings["fallback_max_tools"] == 20
        assert settings["tool_selection"]["adaptive"] is True


class TestProfileGeneration:
    """Tests for YAML generation."""

    def test_generate_basic_profile_yaml(self):
        """Generating basic profile YAML works."""
        yaml_content = generate_profile_yaml(BASIC_PROFILE)
        assert "provider:" in yaml_content
        assert "ollama" in yaml_content
        assert "fallback_max_tools:" in yaml_content
        assert "5" in yaml_content  # basic has 5 max tools

    def test_generate_advanced_profile_yaml(self):
        """Generating advanced profile YAML works."""
        yaml_content = generate_profile_yaml(ADVANCED_PROFILE)
        assert "fallback_max_tools:" in yaml_content
        assert "10" in yaml_content  # advanced has 10 max tools

    def test_generate_with_provider_override(self):
        """Provider override works in YAML generation."""
        yaml_content = generate_profile_yaml(BASIC_PROFILE, provider_override="anthropic")
        assert "provider: anthropic" in yaml_content

    def test_generate_with_model_override(self):
        """Model override works in YAML generation."""
        yaml_content = generate_profile_yaml(
            BASIC_PROFILE, model_override="claude-sonnet-4-5-20250514"
        )
        assert "model: claude-sonnet-4-5-20250514" in yaml_content


class TestProviderDetection:
    """Tests for provider detection."""

    def test_detect_provider_defaults_to_ollama(self, monkeypatch):
        """Provider detection defaults to Ollama when no API keys."""
        # Clear all API keys
        for key in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"]:
            monkeypatch.delenv(key, raising=False)

        provider = _detect_provider()
        assert provider == "ollama"

    def test_detect_model_for_provider(self):
        """Model detection returns appropriate model for provider."""
        anthropic_model = _detect_model_for_provider("anthropic")
        assert "claude" in anthropic_model.lower()

        ollama_model = _detect_model_for_provider("ollama")
        assert "qwen" in ollama_model.lower()


class TestSpecializedProfiles:
    """Tests for specialized profiles."""

    def test_coding_profile(self):
        """Coding profile is optimized for development."""
        assert CODING_PROFILE.name == "coding"
        assert "coder" in CODING_PROFILE.settings["default_model"].lower()
        assert CODING_PROFILE.settings["default_temperature"] == 0.3  # Lower temp for code

    def test_research_profile(self):
        """Research profile has higher context."""
        assert RESEARCH_PROFILE.name == "research"
        assert RESEARCH_PROFILE.settings["default_max_tokens"] == 16384
        assert RESEARCH_PROFILE.settings["default_temperature"] == 0.8  # Higher temp for creativity


class TestProfileConfigDirectoryResolution:
    """Tests for default config directory resolution."""

    def test_install_profile_uses_global_victor_dir_by_default(self, tmp_path):
        """install_profile should use centralized global Victor dir when config_dir is omitted."""
        global_dir = tmp_path / ".victor"
        fake_paths = SimpleNamespace(global_victor_dir=global_dir)

        with patch("victor.config.settings.get_project_paths", return_value=fake_paths):
            profiles_path = install_profile(BASIC_PROFILE)

        assert profiles_path == global_dir / "profiles.yaml"
        assert profiles_path.exists()

    def test_get_current_profile_uses_global_victor_dir_by_default(self, tmp_path):
        """get_current_profile should read from centralized global Victor dir by default."""
        global_dir = tmp_path / ".victor"
        fake_paths = SimpleNamespace(global_victor_dir=global_dir)

        with patch("victor.config.settings.get_project_paths", return_value=fake_paths):
            install_profile(BASIC_PROFILE)
            current = get_current_profile()

        assert current == "basic"
