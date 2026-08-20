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

"""Tests for the consolidated registry base implementation."""

import pytest

from victor.core.registry import BaseRegistry, IRegistry


class TestBaseRegistry:
    """Tests for BaseRegistry generic implementation."""

    def test_init_empty(self):
        """Test that registry initializes empty."""
        registry: BaseRegistry[str, int] = BaseRegistry()
        assert len(registry) == 0
        assert registry.list_all() == []

    def test_register_and_get(self):
        """Test basic register/get cycle."""
        registry: BaseRegistry[str, str] = BaseRegistry()
        registry.register("key1", "value1")

        assert registry.get("key1") == "value1"
        assert registry.get("nonexistent") is None

    def test_register_overwrites(self):
        """Test that registering with existing key overwrites."""
        registry: BaseRegistry[str, str] = BaseRegistry()
        registry.register("key", "value1")
        registry.register("key", "value2")

        assert registry.get("key") == "value2"
        assert len(registry) == 1

    def test_unregister_existing(self):
        """Test unregistering an existing item returns True."""
        registry: BaseRegistry[str, str] = BaseRegistry()
        registry.register("key", "value")

        result = registry.unregister("key")

        assert result is True
        assert registry.get("key") is None
        assert len(registry) == 0

    def test_unregister_nonexistent(self):
        """Test unregistering a nonexistent item returns False."""
        registry: BaseRegistry[str, str] = BaseRegistry()

        result = registry.unregister("nonexistent")

        assert result is False

    def test_list_all(self):
        """Test list_all returns all registered keys."""
        registry: BaseRegistry[str, int] = BaseRegistry()
        registry.register("a", 1)
        registry.register("b", 2)
        registry.register("c", 3)

        keys = registry.list_all()

        assert set(keys) == {"a", "b", "c"}
        assert len(keys) == 3

    def test_list_all_empty(self):
        """Test list_all on empty registry."""
        registry: BaseRegistry[str, int] = BaseRegistry()

        assert registry.list_all() == []

    def test_contains(self):
        """Test __contains__ operator."""
        registry: BaseRegistry[str, str] = BaseRegistry()
        registry.register("exists", "value")

        assert "exists" in registry
        assert "missing" not in registry

    def test_len(self):
        """Test __len__ returns correct count."""
        registry: BaseRegistry[str, int] = BaseRegistry()
        assert len(registry) == 0

        registry.register("a", 1)
        assert len(registry) == 1

        registry.register("b", 2)
        assert len(registry) == 2

        registry.unregister("a")
        assert len(registry) == 1

    def test_clear(self):
        """Test clear removes all items."""
        registry: BaseRegistry[str, int] = BaseRegistry()
        registry.register("a", 1)
        registry.register("b", 2)
        registry.register("c", 3)

        registry.clear()

        assert len(registry) == 0
        assert registry.list_all() == []
        assert registry.get("a") is None

    def test_iter(self):
        """Test __iter__ allows iteration over keys."""
        registry: BaseRegistry[str, int] = BaseRegistry()
        registry.register("a", 1)
        registry.register("b", 2)

        keys = list(registry)

        assert set(keys) == {"a", "b"}

    def test_values(self):
        """Test values() returns all registered values."""
        registry: BaseRegistry[str, int] = BaseRegistry()
        registry.register("a", 1)
        registry.register("b", 2)
        registry.register("c", 3)

        values = registry.values()

        assert set(values) == {1, 2, 3}

    def test_items(self):
        """Test items() returns all key-value pairs."""
        registry: BaseRegistry[str, int] = BaseRegistry()
        registry.register("a", 1)
        registry.register("b", 2)

        items = registry.items()

        assert set(items) == {("a", 1), ("b", 2)}


class TestIRegistryProtocol:
    """Tests for IRegistry protocol compliance."""

    def test_base_registry_implements_protocol(self):
        """Test that BaseRegistry is runtime checkable against IRegistry."""
        registry = BaseRegistry()
        assert isinstance(registry, IRegistry)

    def test_protocol_methods_exist(self):
        """Test that required protocol methods are callable."""
        registry: IRegistry[str, int] = BaseRegistry()

        # These should not raise
        registry.register("key", 42)
        assert registry.get("key") == 42
        assert registry.list_all() == ["key"]
        assert registry.unregister("key") is True
        registry.clear()


class TestRegistryWithCustomTypes:
    """Tests using custom types to verify generic behavior."""

    def test_with_object_values(self):
        """Test registry with object values."""

        class MockTool:
            def __init__(self, name: str):
                self.name = name

        registry: BaseRegistry[str, MockTool] = BaseRegistry()
        tool = MockTool("test_tool")

        registry.register("test_tool", tool)

        retrieved = registry.get("test_tool")
        assert retrieved is tool
        assert retrieved.name == "test_tool"

    def test_with_int_keys(self):
        """Test registry with integer keys."""
        registry: BaseRegistry[int, str] = BaseRegistry()

        registry.register(1, "one")
        registry.register(2, "two")

        assert registry.get(1) == "one"
        assert registry.get(2) == "two"
        assert 1 in registry
        assert 3 not in registry


class TestRegistryEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_string_key(self):
        """Test registry with empty string key."""
        registry: BaseRegistry[str, str] = BaseRegistry()
        registry.register("", "empty_key_value")

        assert registry.get("") == "empty_key_value"
        assert "" in registry

    def test_none_value(self):
        """Test registry storing None as a value."""
        registry: BaseRegistry[str, str | None] = BaseRegistry()
        registry.register("key", None)

        # get returns None for both missing and None-valued keys
        # but we can check presence via contains
        assert "key" in registry
        assert registry.get("key") is None

    def test_register_after_clear(self):
        """Test that registry works after being cleared."""
        registry: BaseRegistry[str, int] = BaseRegistry()
        registry.register("a", 1)
        registry.clear()
        registry.register("b", 2)

        assert registry.get("b") == 2
        assert len(registry) == 1

    def test_multiple_unregister_calls(self):
        """Test multiple unregister calls on same key."""
        registry: BaseRegistry[str, str] = BaseRegistry()
        registry.register("key", "value")

        assert registry.unregister("key") is True
        assert registry.unregister("key") is False
        assert registry.unregister("key") is False
