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

"""Unit tests for PluginNamespaceManager."""

from __future__ import annotations

import pytest

from victor.core.verticals.namespace_manager import (
    NamespaceType,
    NamespaceConfig,
    NamespacedPluginKey,
    PluginNamespaceManager,
    get_namespace_manager,
    make_plugin_key,
    NAMESPACES,
)


class TestNamespaceType:
    """Test suite for NamespaceType enum."""

    def test_namespace_type_values(self):
        """Test that all expected namespace types exist."""
        assert NamespaceType.DEFAULT.value == "default"
        assert NamespaceType.CONTRIBUTED.value == "contrib"
        assert NamespaceType.EXTERNAL.value == "external"
        assert NamespaceType.EXPERIMENTAL.value == "experimental"


class TestNamespaceConfig:
    """Test suite for NamespaceConfig dataclass."""

    def test_create_config(self):
        """Test creating a namespace configuration."""
        config = NamespaceConfig(
            name="test",
            priority=50,
            allow_override=True,
            description="Test namespace",
            allowed_prefixes=["test_", "tmp_"],
        )

        assert config.name == "test"
        assert config.priority == 50
        assert config.allow_override is True
        assert config.description == "Test namespace"
        assert config.allowed_prefixes == ["test_", "tmp_"]

    def test_config_defaults(self):
        """Test default values for NamespaceConfig."""
        config = NamespaceConfig(name="test")

        assert config.priority == 50
        assert config.allow_override is True
        assert config.description == ""
        assert config.allowed_prefixes is None


class TestNamespacedPluginKey:
    """Test suite for NamespacedPluginKey dataclass."""

    def test_create_key_without_version(self):
        """Test creating a key without version."""
        key = NamespacedPluginKey(namespace="external", plugin_name="my_tool")

        assert key.namespace == "external"
        assert key.plugin_name == "my_tool"
        assert key.version is None
        assert key.full_key == "external:my_tool"

    def test_create_key_with_version(self):
        """Test creating a key with version."""
        key = NamespacedPluginKey(namespace="external", plugin_name="my_tool", version="1.0.0")

        assert key.namespace == "external"
        assert key.plugin_name == "my_tool"
        assert key.version == "1.0.0"
        assert key.full_key == "external:my_tool:1.0.0"

    def test_key_equality(self):
        """Test key equality comparison."""
        key1 = NamespacedPluginKey(namespace="external", plugin_name="my_tool")
        key2 = NamespacedPluginKey(namespace="external", plugin_name="my_tool")
        key3 = NamespacedPluginKey(namespace="default", plugin_name="my_tool")

        assert key1 == key2
        assert key1 != key3

    def test_key_hash(self):
        """Test key hashing for use in sets/dicts."""
        key1 = NamespacedPluginKey(namespace="external", plugin_name="my_tool")
        key2 = NamespacedPluginKey(namespace="external", plugin_name="my_tool")
        key3 = NamespacedPluginKey(namespace="default", plugin_name="my_tool")

        # Should be hashable
        key_set = {key1, key2, key3}
        assert len(key_set) == 2  # key1 and key2 are equal

    def test_key_str(self):
        """Test string representation of key."""
        key = NamespacedPluginKey(namespace="external", plugin_name="my_tool", version="1.0.0")

        assert str(key) == "external:my_tool:1.0.0"
        assert str(key) == key.full_key


class TestPluginNamespaceManager:
    """Test suite for PluginNamespaceManager."""

    def setup_method(self):
        """Create fresh manager for each test."""
        self.manager = PluginNamespaceManager()

    def test_singleton(self):
        """Test that get_instance returns singleton."""
        manager1 = PluginNamespaceManager.get_instance()
        manager2 = PluginNamespaceManager.get_instance()

        assert manager1 is manager2

    def test_make_key(self):
        """Test creating a namespaced key."""
        key = self.manager.make_key("external", "my_tool", "1.0.0")

        assert isinstance(key, NamespacedPluginKey)
        assert key.namespace == "external"
        assert key.plugin_name == "my_tool"
        assert key.version == "1.0.0"

    def test_register_plugin(self):
        """Test registering a plugin."""
        plugin = object()  # Mock plugin object

        self.manager.register_plugin("external", "my_tool", plugin, "1.0.0")

        # Verify plugin is registered
        plugins = self.manager.list_plugins()
        assert "external:my_tool:1.0.0" in plugins

    def test_register_plugin_duplicate_overwrites(self):
        """Test that registering duplicate plugin overwrites."""
        plugin1 = object()
        plugin2 = object()

        self.manager.register_plugin("external", "my_tool", plugin1, "1.0.0")
        self.manager.register_plugin("external", "my_tool", plugin2, "1.0.0")

        plugins = self.manager.list_plugins()
        # Should only have one entry
        assert plugins.count("external:my_tool:1.0.0") == 1

    def test_register_plugin_without_version(self):
        """Test registering plugin without version."""
        plugin = object()

        self.manager.register_plugin("default", "my_tool", plugin)

        plugins = self.manager.list_plugins()
        assert "default:my_tool" in plugins

    def test_resolve_plugin_by_priority(self):
        """Test resolving plugin by namespace priority."""
        plugin_external = "external_plugin"
        plugin_default = "default_plugin"

        # Register same plugin in different namespaces
        self.manager.register_plugin("external", "my_tool", plugin_external)
        self.manager.register_plugin("default", "my_tool", plugin_default)

        # External has higher priority (100) than default (0)
        result = self.manager.resolve(
            "my_tool", ["default", "external"], default_namespace="default"
        )

        # Should return external plugin (higher priority)
        assert result == plugin_external

    def test_resolve_plugin_not_found(self):
        """Test resolving non-existent plugin."""
        result = self.manager.resolve("nonexistent", ["default"])

        assert result is None

    def test_resolve_plugin_with_version(self):
        """Test resolving versioned plugin."""
        plugin = object()

        self.manager.register_plugin("external", "my_tool", plugin, "1.0.0")

        # Should find the versioned plugin and return the object
        result = self.manager.resolve("my_tool", ["external"])

        assert result is plugin

    def test_list_plugins_all(self):
        """Test listing all plugins."""
        self.manager.register_plugin("default", "tool1", object())
        self.manager.register_plugin("external", "tool2", object())
        self.manager.register_plugin("contrib", "tool3", object())

        plugins = self.manager.list_plugins()

        assert len(plugins) == 3
        assert "default:tool1" in plugins
        assert "external:tool2" in plugins
        assert "contrib:tool3" in plugins

    def test_list_plugins_by_namespace(self):
        """Test listing plugins filtered by namespace."""
        self.manager.register_plugin("default", "tool1", object())
        self.manager.register_plugin("external", "tool2", object())
        self.manager.register_plugin("external", "tool3", object())

        plugins = self.manager.list_plugins(namespace="external")

        assert len(plugins) == 2
        assert "external:tool2" in plugins
        assert "external:tool3" in plugins
        assert "default:tool1" not in plugins

    def test_unregister_plugin(self):
        """Test unregistering a plugin."""
        plugin = object()

        self.manager.register_plugin("external", "my_tool", plugin, "1.0.0")
        assert "external:my_tool:1.0.0" in self.manager.list_plugins()

        # Unregister
        success = self.manager.unregister_plugin("external", "my_tool", "1.0.0")
        assert success is True
        assert "external:my_tool:1.0.0" not in self.manager.list_plugins()

    def test_unregister_plugin_not_found(self):
        """Test unregistering non-existent plugin."""
        success = self.manager.unregister_plugin("external", "nonexistent")

        assert success is False

    def test_clear_all_plugins(self):
        """Test clearing all plugins."""
        self.manager.register_plugin("default", "tool1", object())
        self.manager.register_plugin("external", "tool2", object())

        count = self.manager.clear()

        assert count == 2
        assert len(self.manager.list_plugins()) == 0

    def test_clear_namespace_plugins(self):
        """Test clearing plugins in a namespace."""
        self.manager.register_plugin("default", "tool1", object())
        self.manager.register_plugin("external", "tool2", object())
        self.manager.register_plugin("external", "tool3", object())

        count = self.manager.clear(namespace="external")

        assert count == 2
        plugins = self.manager.list_plugins()
        assert "default:tool1" in plugins
        assert "external:tool2" not in plugins
        assert "external:tool3" not in plugins

    def test_get_namespace_config(self):
        """Test getting namespace configuration."""
        config = self.manager.get_namespace_config("external")

        assert config is not None
        assert config.name == "external"
        assert config.priority == 100

    def test_get_namespace_config_not_found(self):
        """Test getting non-existent namespace config."""
        config = self.manager.get_namespace_config("nonexistent")

        assert config is None

    def test_register_custom_namespace(self):
        """Test registering custom namespace."""
        custom_config = NamespaceConfig(
            name="custom",
            priority=75,
            allow_override=True,
            description="Custom namespace",
        )

        self.manager.register_namespace(custom_config)

        config = self.manager.get_namespace_config("custom")
        assert config is not None
        assert config.priority == 75
        assert config.description == "Custom namespace"

    def test_list_namespaces(self):
        """Test listing all namespaces."""
        namespaces = self.manager.list_namespaces()

        assert "default" in namespaces
        assert "contrib" in namespaces
        assert "external" in namespaces
        assert "experimental" in namespaces

    def test_priority_ordering_in_resolve(self):
        """Test that resolve respects priority ordering."""
        # Register plugins in different namespaces
        plugin_experimental = object()
        plugin_contrib = object()
        plugin_external = object()
        plugin_default = object()

        self.manager.register_plugin("experimental", "tool", plugin_experimental)  # priority 25
        self.manager.register_plugin("contrib", "tool", plugin_contrib)  # priority 50
        self.manager.register_plugin("external", "tool", plugin_external)  # priority 100
        self.manager.register_plugin("default", "tool", plugin_default)  # priority 0

        # Resolve with all namespaces available
        result = self.manager.resolve("tool", ["default", "contrib", "external", "experimental"])

        # Should pick external plugin (highest priority)
        assert result is plugin_external

    def test_find_keys_by_plugin_version_sorting(self):
        """Test that version keys are sorted correctly."""
        self.manager.register_plugin("external", "tool", object(), "1.0.0")
        self.manager.register_plugin("external", "tool", object(), "2.0.0")
        self.manager.register_plugin("external", "tool", object(), "0.5.0")

        keys = self.manager._find_keys_by_plugin("external", "tool")

        # Should be sorted by version (newest first)
        assert keys[0][0] == "2.0.0"
        assert keys[1][0] == "1.0.0"
        assert keys[2][0] == "0.5.0"


class TestConvenienceFunctions:
    """Test suite for convenience functions."""

    def test_get_namespace_manager(self):
        """Test get_namespace_manager convenience function."""
        manager1 = get_namespace_manager()
        manager2 = get_namespace_manager()

        assert manager1 is manager2
        assert isinstance(manager1, PluginNamespaceManager)

    def test_make_plugin_key(self):
        """Test make_plugin_key convenience function."""
        key = make_plugin_key("external", "my_tool", "1.0.0")

        assert isinstance(key, NamespacedPluginKey)
        assert key.namespace == "external"
        assert key.plugin_name == "my_tool"
        assert key.version == "1.0.0"


class TestPredefinedNamespaces:
    """Test suite for predefined namespace configurations."""

    def test_default_namespace_config(self):
        """Test default namespace configuration."""
        config = NAMESPACES["default"]

        assert config.name == "default"
        assert config.priority == 0
        assert config.allow_override is True

    def test_contrib_namespace_config(self):
        """Test contrib namespace configuration."""
        config = NAMESPACES["contrib"]

        assert config.name == "contrib"
        assert config.priority == 50
        assert config.allow_override is True

    def test_external_namespace_config(self):
        """Test external namespace configuration."""
        config = NAMESPACES["external"]

        assert config.name == "external"
        assert config.priority == 100
        assert config.allow_override is True

    def test_experimental_namespace_config(self):
        """Test experimental namespace configuration."""
        config = NAMESPACES["experimental"]

        assert config.name == "experimental"
        assert config.priority == 25
        assert config.allow_override is False  # Experimental doesn't allow override


class TestNamespaceIsolation:
    """Test suite for namespace isolation guarantees."""

    def test_different_namespaces_no_collision(self):
        """Test that same plugin name in different namespaces don't collide."""
        manager = PluginNamespaceManager()

        # Register same plugin name in different namespaces
        manager.register_plugin("default", "my_tool", "plugin1")
        manager.register_plugin("external", "my_tool", "plugin2")
        manager.register_plugin("contrib", "my_tool", "plugin3")

        # All three should coexist
        plugins = manager.list_plugins()
        assert "default:my_tool" in plugins
        assert "external:my_tool" in plugins
        assert "contrib:my_tool" in plugins
        assert len(plugins) == 3

    def test_versioned_plugins_coexist(self):
        """Test that different versions of same plugin coexist."""
        manager = PluginNamespaceManager()

        manager.register_plugin("external", "my_tool", "v1", "1.0.0")
        manager.register_plugin("external", "my_tool", "v2", "2.0.0")
        manager.register_plugin("external", "my_tool", "v3", "3.0.0")

        plugins = manager.list_plugins()
        assert "external:my_tool:1.0.0" in plugins
        assert "external:my_tool:2.0.0" in plugins
        assert "external:my_tool:3.0.0" in plugins
        assert len(plugins) == 3

    def test_resolve_prefers_higher_namespace(self):
        """Test that resolve prefers higher priority namespace."""
        manager = PluginNamespaceManager()

        plugin_default = "default_version"
        plugin_external = "external_version"

        manager.register_plugin("default", "tool", plugin_default)
        manager.register_plugin("external", "tool", plugin_external)

        # Even if default is listed first, external plugin should win
        result = manager.resolve("tool", ["default", "external"])
        assert result is plugin_external

    def test_clear_namespace_doesnt_affect_others(self):
        """Test that clearing one namespace doesn't affect others."""
        manager = PluginNamespaceManager()

        manager.register_plugin("default", "tool1", object())
        manager.register_plugin("external", "tool2", object())
        manager.register_plugin("external", "tool3", object())

        # Clear external only
        manager.clear(namespace="external")

        plugins = manager.list_plugins()
        assert len(plugins) == 1
        assert "default:tool1" in plugins


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
