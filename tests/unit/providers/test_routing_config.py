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

"""
Unit tests for routing configuration models.
"""

import pytest
from datetime import datetime
from pathlib import Path

from victor.providers.routing_config import (
    RoutingProfile,
    SmartRoutingConfig,
    load_routing_profiles,
    get_default_profiles,
    save_default_profiles,
)


class TestRoutingProfile:
    """Tests for RoutingProfile dataclass."""

    def test_routing_profile_creation(self):
        """Test creating a routing profile."""
        profile = RoutingProfile(
            name="test-profile",
            description="Test profile",
            fallback_chains={
                "default": ["ollama", "anthropic"],
                "coding": ["ollama", "deepseek"],
            },
            cost_preference="low",
            latency_preference="normal",
        )

        assert profile.name == "test-profile"
        assert profile.description == "Test profile"
        assert profile.cost_preference == "low"
        assert profile.latency_preference == "normal"

    def test_routing_profile_normalization(self):
        """Test provider name normalization to lowercase."""
        profile = RoutingProfile(
            name="test",
            description="Test",
            fallback_chains={
                "default": ["OLLAMA", "Anthropic", "OpenAI"],
            },
        )

        assert profile.fallback_chains["default"] == ["ollama", "anthropic", "openai"]

    def test_routing_profile_default_chain(self):
        """Test that default fallback chain is created if missing."""
        profile = RoutingProfile(
            name="test",
            description="Test",
            fallback_chains={},
        )

        assert "default" in profile.fallback_chains
        assert profile.fallback_chains["default"] == []

    def test_get_fallback_chain(self):
        """Test getting fallback chain for task type."""
        profile = RoutingProfile(
            name="test",
            description="Test",
            fallback_chains={
                "default": ["ollama", "anthropic"],
                "coding": ["ollama", "deepseek"],
            },
        )

        assert profile.get_fallback_chain("coding") == ["ollama", "deepseek"]
        assert profile.get_fallback_chain("unknown") == ["ollama", "anthropic"]

    def test_to_dict(self):
        """Test converting profile to dictionary."""
        profile = RoutingProfile(
            name="test",
            description="Test",
            fallback_chains={"default": ["ollama"]},
            cost_preference="low",
            latency_preference="high",
        )

        data = profile.to_dict()

        assert data["name"] == "test"
        assert data["description"] == "Test"
        assert data["fallback_chains"] == {"default": ["ollama"]}
        assert data["cost_preference"] == "low"
        assert data["latency_preference"] == "high"


class TestSmartRoutingConfig:
    """Tests for SmartRoutingConfig dataclass."""

    def test_config_creation(self):
        """Test creating smart routing config."""
        config = SmartRoutingConfig(
            enabled=True,
            profile_name="balanced",
            performance_window_size=50,
        )

        assert config.enabled is True
        assert config.profile_name == "balanced"
        assert config.performance_window_size == 50

    def test_config_defaults(self):
        """Test default configuration values."""
        config = SmartRoutingConfig()

        assert config.enabled is False
        assert config.profile_name == "balanced"
        assert config.performance_window_size == 100
        assert config.learning_enabled is True
        assert config.resource_awareness_enabled is True
        assert config.custom_fallback_chain is None

    def test_custom_fallback_chain_normalization(self):
        """Test custom fallback chain normalization."""
        config = SmartRoutingConfig(
            custom_fallback_chain=["OLLAMA", "Anthropic", "OpenAI"],
        )

        assert config.custom_fallback_chain == ["ollama", "anthropic", "openai"]

    def test_invalid_window_size(self):
        """Test that invalid window size raises error."""
        with pytest.raises(ValueError, match="performance_window_size must be at least 1"):
            SmartRoutingConfig(performance_window_size=0)


class TestDefaultProfiles:
    """Tests for default routing profiles."""

    def test_get_default_profiles(self):
        """Test getting default profiles."""
        profiles = get_default_profiles()

        assert "balanced" in profiles
        assert "cost-optimized" in profiles
        assert "performance" in profiles
        assert "local-first" in profiles

    def test_balanced_profile(self):
        """Test balanced profile configuration."""
        profiles = get_default_profiles()
        balanced = profiles["balanced"]

        assert balanced.name == "balanced"
        assert balanced.cost_preference == "normal"
        assert balanced.latency_preference == "normal"
        assert "ollama" in balanced.get_fallback_chain("default")

    def test_cost_optimized_profile(self):
        """Test cost-optimized profile configuration."""
        profiles = get_default_profiles()
        cost = profiles["cost-optimized"]

        assert cost.name == "cost-optimized"
        assert cost.cost_preference == "low"
        assert "ollama" in cost.get_fallback_chain("default")

    def test_performance_profile(self):
        """Test performance profile configuration."""
        profiles = get_default_profiles()
        perf = profiles["performance"]

        assert perf.name == "performance"
        assert perf.cost_preference == "high"
        assert perf.latency_preference == "low"
        assert "anthropic" in perf.get_fallback_chain("default")

    def test_local_first_profile(self):
        """Test local-first profile configuration."""
        profiles = get_default_profiles()
        local = profiles["local-first"]

        assert local.name == "local-first"
        assert local.cost_preference == "low"
        assert "ollama" in local.get_fallback_chain("default")


class TestProfileLoading:
    """Tests for profile loading and saving."""

    def test_load_nonexistent_file(self, tmp_path):
        """Test loading from non-existent file returns defaults."""
        profiles = load_routing_profiles(tmp_path / "nonexistent.yaml")

        # Should return default profiles
        assert "balanced" in profiles
        assert "cost-optimized" in profiles

    def test_save_and_load_profiles(self, tmp_path):
        """Test saving and loading profiles."""
        # Save defaults
        profile_path = tmp_path / "routing_profiles.yaml"
        save_default_profiles(profile_path)

        # Load back
        profiles = load_routing_profiles(profile_path)

        assert "balanced" in profiles
        assert "cost-optimized" in profiles
        assert "performance" in profiles
        assert "local-first" in profiles

    def test_load_invalid_yaml(self, tmp_path):
        """Test loading invalid YAML returns defaults."""
        # Write invalid YAML
        profile_path = tmp_path / "routing_profiles.yaml"
        profile_path.write_text("invalid: yaml: content: [")

        # Should return defaults without crashing
        profiles = load_routing_profiles(profile_path)

        assert "balanced" in profiles
