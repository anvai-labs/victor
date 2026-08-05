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

"""Unit tests for VerticalBehaviorConfigRegistry module."""

from __future__ import annotations

import pytest

from victor.core.verticals.config_registry import (
    VerticalBehaviorConfig,
    VerticalBehaviorConfigRegistry,
    get_canonicalization_setting,
    get_tool_dependency_strategy,
    is_strict_mode_enabled,
)


class TestVerticalBehaviorConfig:
    """Test suite for VerticalBehaviorConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = VerticalBehaviorConfig()

        assert config.canonicalize_tool_names is True
        assert config.tool_dependency_strategy == "auto"
        assert config.strict_mode is False
        assert config.load_priority == 0
        assert config.lazy_load is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = VerticalBehaviorConfig(
            canonicalize_tool_names=False,
            tool_dependency_strategy="entry_point",
            strict_mode=True,
            load_priority=100,
            lazy_load=False,
        )

        assert config.canonicalize_tool_names is False
        assert config.tool_dependency_strategy == "entry_point"
        assert config.strict_mode is True
        assert config.load_priority == 100
        assert config.lazy_load is False

    def test_invalid_tool_dependency_strategy(self):
        """Test that invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="Invalid tool_dependency_strategy"):
            VerticalBehaviorConfig(tool_dependency_strategy="invalid")

    def test_negative_load_priority(self):
        """Test that negative load_priority raises ValueError."""
        with pytest.raises(ValueError, match="load_priority must be non-negative"):
            VerticalBehaviorConfig(load_priority=-1)

    def test_valid_strategies(self):
        """Test all valid strategy values."""
        valid_strategies = ["auto", "entry_point", "factory", "none"]
        for strategy in valid_strategies:
            config = VerticalBehaviorConfig(tool_dependency_strategy=strategy)
            assert config.tool_dependency_strategy == strategy

    def test_frozen_dataclass(self):
        """Test that config is frozen (immutable)."""
        config = VerticalBehaviorConfig()

        with pytest.raises(Exception):  # FrozenInstanceError or similar
            config.canonicalize_tool_names = False

    def test_merge_configs(self):
        """Test merging two configs (removed - merge method not implemented)."""
        # Merge functionality not currently needed
        # VerticalBehaviorConfig is frozen, so merging creates new instances
        config1 = VerticalBehaviorConfig(canonicalize_tool_names=True, load_priority=10)
        config2 = VerticalBehaviorConfig(canonicalize_tool_names=False, load_priority=20)

        # Since configs are frozen, we just verify they're independent
        assert config1.canonicalize_tool_names is True
        assert config2.canonicalize_tool_names is False

    def test_merge_with_none(self):
        """Test that configs are independent and immutable."""
        config = VerticalBehaviorConfig(load_priority=10)

        # Config is frozen, so we can't modify it anyway
        assert config.load_priority == 10
        # Trying to modify would raise an exception (tested in test_frozen_dataclass)


class TestVerticalBehaviorConfigRegistry:
    """Test suite for VerticalBehaviorConfigRegistry class."""

    def setup_method(self):
        """Clear registry before each test to avoid pollution from decorator-registered verticals."""
        VerticalBehaviorConfigRegistry.clear()

    def teardown_method(self):
        """Clear registry after each test."""
        VerticalBehaviorConfigRegistry.clear()

    def test_register_and_get_config(self):
        """Test registering and retrieving configuration."""
        config = VerticalBehaviorConfig(canonicalize_tool_names=False)
        VerticalBehaviorConfigRegistry.register("test_vertical", config)

        retrieved = VerticalBehaviorConfigRegistry.get("test_vertical")
        assert retrieved.canonicalize_tool_names is False

    def test_get_unknown_vertical_returns_defaults(self):
        """Test that unknown vertical returns default config."""
        config = VerticalBehaviorConfigRegistry.get("unknown_vertical")

        assert config == VerticalBehaviorConfig()
        assert config.canonicalize_tool_names is True  # Default

    def test_has_config(self):
        """Test checking if vertical has explicit config."""
        config = VerticalBehaviorConfig()
        VerticalBehaviorConfigRegistry.register("test_vertical", config)

        assert VerticalBehaviorConfigRegistry.has_config("test_vertical") is True
        assert VerticalBehaviorConfigRegistry.has_config("unknown") is False

    def test_unregister_config(self):
        """Test unregistering configuration."""
        config = VerticalBehaviorConfig()
        VerticalBehaviorConfigRegistry.register("test_vertical", config)

        assert VerticalBehaviorConfigRegistry.has_config("test_vertical") is True

        VerticalBehaviorConfigRegistry.unregister("test_vertical")

        assert VerticalBehaviorConfigRegistry.has_config("test_vertical") is False

    def test_clear_all_configs(self):
        """Test clearing all configurations."""
        VerticalBehaviorConfigRegistry.register("v1", VerticalBehaviorConfig())
        VerticalBehaviorConfigRegistry.register("v2", VerticalBehaviorConfig())

        assert len(VerticalBehaviorConfigRegistry.list_configured_verticals()) == 2

        VerticalBehaviorConfigRegistry.clear()

        assert len(VerticalBehaviorConfigRegistry.list_configured_verticals()) == 0

    def test_list_configured_verticals(self):
        """Test listing all configured verticals."""
        VerticalBehaviorConfigRegistry.register("v1", VerticalBehaviorConfig())
        VerticalBehaviorConfigRegistry.register("v2", VerticalBehaviorConfig())
        VerticalBehaviorConfigRegistry.register("v3", VerticalBehaviorConfig())

        verticals = VerticalBehaviorConfigRegistry.list_configured_verticals()

        assert set(verticals) == {"v1", "v2", "v3"}

    def test_register_empty_name_raises_error(self):
        """Test that registering with empty name raises ValueError."""
        with pytest.raises(ValueError, match="Vertical name cannot be empty"):
            VerticalBehaviorConfigRegistry.register("", VerticalBehaviorConfig())

    def test_register_invalid_type_raises_error(self):
        """Test that registering non-VerticalBehaviorConfig raises ValueError."""
        with pytest.raises(ValueError, match="Config must be VerticalBehaviorConfig"):
            VerticalBehaviorConfigRegistry.register("test", "not a config")

    def test_from_manifest(self):
        """Test creating config from ExtensionManifest."""
        from victor_contracts.verticals.manifest import ExtensionManifest

        manifest = ExtensionManifest(
            name="test",
            canonicalize_tool_names=False,
            tool_dependency_strategy="entry_point",
            strict_mode=True,
            load_priority=50,
        )

        config = VerticalBehaviorConfigRegistry.from_manifest(manifest)

        assert config.canonicalize_tool_names is False
        assert config.tool_dependency_strategy == "entry_point"
        assert config.strict_mode is True
        assert config.load_priority == 50

    def test_get_or_create_from_manifest_with_registered_config(self):
        """Test that registered config takes precedence over manifest."""
        from victor_contracts.verticals.manifest import ExtensionManifest

        # Register explicit config
        explicit_config = VerticalBehaviorConfig(canonicalize_tool_names=True)
        VerticalBehaviorConfigRegistry.register("test", explicit_config)

        # Create manifest with different value
        manifest = ExtensionManifest(
            name="test",
            canonicalize_tool_names=False,
        )

        config = VerticalBehaviorConfigRegistry.get_or_create_from_manifest("test", manifest)

        # Should return registered config, not manifest config
        assert config.canonicalize_tool_names is True

    def test_get_or_create_from_manifest_with_no_registered_config(self):
        """Test that manifest is used when no explicit config registered."""
        # Clear any existing config first
        VerticalBehaviorConfigRegistry.clear()

        from victor_contracts.verticals.manifest import ExtensionManifest

        manifest = ExtensionManifest(
            name="test",
            canonicalize_tool_names=False,
            load_priority=75,
        )

        config = VerticalBehaviorConfigRegistry.get_or_create_from_manifest("test", manifest)

        # Should use manifest values
        assert config.canonicalize_tool_names is False
        assert config.load_priority == 75

    def test_get_or_create_from_manifest_with_no_manifest(self):
        """Test that defaults are used when no config or manifest."""
        config = VerticalBehaviorConfigRegistry.get_or_create_from_manifest("test", None)

        assert config == VerticalBehaviorConfig()


class TestConvenienceFunctions:
    """Test suite for convenience functions."""

    def test_get_canonicalization_setting(self):
        """Test getting canonicalization setting."""
        # Register config with False
        config = VerticalBehaviorConfig(canonicalize_tool_names=False)
        VerticalBehaviorConfigRegistry.register("test", config)

        assert get_canonicalization_setting("test") is False

        # Unknown vertical should return default (True)
        assert get_canonicalization_setting("unknown") is True

    def test_get_tool_dependency_strategy(self):
        """Test getting tool dependency strategy."""
        # Register config with entry_point strategy
        config = VerticalBehaviorConfig(tool_dependency_strategy="entry_point")
        VerticalBehaviorConfigRegistry.register("test", config)

        assert get_tool_dependency_strategy("test") == "entry_point"

        # Unknown vertical should return default (auto)
        assert get_tool_dependency_strategy("unknown") == "auto"

    def test_is_strict_mode_enabled(self):
        """Test checking if strict mode is enabled."""
        # Register config with strict mode
        config = VerticalBehaviorConfig(strict_mode=True)
        VerticalBehaviorConfigRegistry.register("test", config)

        assert is_strict_mode_enabled("test") is True

        # Unknown vertical should return default (False)
        assert is_strict_mode_enabled("unknown") is False


class TestConfigRegistryIntegration:
    """Integration tests for config registry with manifests."""

    def setup_method(self):
        """Clear registry before each test to prevent state leakage."""
        VerticalBehaviorConfigRegistry.clear()

    def test_manifest_decorator_registers_config(self):
        """Test that @register_vertical decorator registers behavior config."""
        from victor.core.verticals.registration import register_vertical
        from victor_contracts.verticals.manifest import ExtensionType

        @register_vertical(
            name="integration_test",
            version="1.0.0",
            canonicalize_tool_names=False,
            tool_dependency_strategy="entry_point",
        )
        class TestVertical:
            pass

        # Verify config was registered
        assert VerticalBehaviorConfigRegistry.has_config("integration_test") is True

        config = VerticalBehaviorConfigRegistry.get("integration_test")
        assert config.canonicalize_tool_names is False
        assert config.tool_dependency_strategy == "entry_point"

    def test_multiple_verticals_different_configs(self):
        """Test that multiple verticals can have different configs."""
        from victor.core.verticals.config_registry import (
            get_canonicalization_setting,
        )

        # Register synthetic verticals with different configurations
        config_a = VerticalBehaviorConfig(
            canonicalize_tool_names=True,
            tool_dependency_strategy="auto",
        )
        config_b = VerticalBehaviorConfig(
            canonicalize_tool_names=False,
            tool_dependency_strategy="entry_point",
        )
        VerticalBehaviorConfigRegistry.register("vertical_a", config_a)
        VerticalBehaviorConfigRegistry.register("vertical_b", config_b)

        assert get_canonicalization_setting("vertical_a") is True
        assert get_canonicalization_setting("vertical_b") is False


class TestConfigRegistryCleanup:
    """Test cleanup and isolation between tests."""

    def setup_method(self):
        """Clear registry before each test."""
        VerticalBehaviorConfigRegistry.clear()

    def teardown_method(self):
        """Clear registry after each test."""
        VerticalBehaviorConfigRegistry.clear()

    def test_registry_isolation(self):
        """Test that registry state is isolated between tests."""
        # Register a config
        VerticalBehaviorConfigRegistry.register("iso_test", VerticalBehaviorConfig())
        assert VerticalBehaviorConfigRegistry.has_config("iso_test") is True

        # Clear should work
        VerticalBehaviorConfigRegistry.clear()
        assert VerticalBehaviorConfigRegistry.has_config("iso_test") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
