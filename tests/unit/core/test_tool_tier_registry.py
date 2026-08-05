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

"""Unit tests for ToolTierRegistry singleton.

Tests cover:
- Singleton behavior
- Registration and retrieval
- Tier inheritance and merging
- Tier extension
- Validation
- Thread safety
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from victor.core.tool_tier_registry import (
    ToolTierRegistry,
    TierRegistryEntry,
    get_tool_tier_registry,
)
from victor.core.vertical_types import TieredToolConfig


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the singleton before and after each test."""
    from victor.core.vertical_types import TieredToolTemplate

    ToolTierRegistry.reset_instance()
    TieredToolTemplate._registered_verticals.clear()
    yield
    TieredToolTemplate._registered_verticals.clear()
    ToolTierRegistry.reset_instance()


@pytest.fixture()
def seeded_registry():
    """Seed the registry with standard verticals for tests that need them.

    After the legacy _LEGACY_CORES removal, verticals must register
    dynamically via register_vertical_tools(). This fixture provides
    the same data for backward-compatible tests.
    """
    from victor.core.vertical_types import TieredToolTemplate

    TieredToolTemplate.register_vertical_tools(
        "coding",
        core_tools={"edit", "write", "shell", "git", "search", "overview"},
        readonly_for_analysis=False,
    )
    TieredToolTemplate.register_vertical_tools(
        "research",
        core_tools={"web_search", "web_fetch", "overview"},
        readonly_for_analysis=True,
    )
    TieredToolTemplate.register_vertical_tools(
        "devops",
        core_tools={"shell", "git", "docker", "overview"},
        readonly_for_analysis=False,
    )
    TieredToolTemplate.register_vertical_tools(
        "data_analysis",
        core_tools={"shell", "write", "overview"},
        readonly_for_analysis=False,
    )
    TieredToolTemplate.register_vertical_tools(
        "rag",
        core_tools={"rag_search", "rag_query", "rag_list", "rag_stats", "rag_delete"},
        readonly_for_analysis=True,
    )


class TestToolTierRegistrySingleton:
    """Test singleton behavior of ToolTierRegistry."""

    def test_get_instance_returns_same_instance(self):
        """get_instance() should return the same instance."""
        instance1 = ToolTierRegistry.get_instance()
        instance2 = ToolTierRegistry.get_instance()
        assert instance1 is instance2

    def test_reset_instance_clears_singleton(self):
        """reset_instance() should clear the singleton."""
        instance1 = ToolTierRegistry.get_instance()
        ToolTierRegistry.reset_instance()
        instance2 = ToolTierRegistry.get_instance()
        assert instance1 is not instance2

    def test_get_tool_tier_registry_function(self):
        """get_tool_tier_registry() convenience function works."""
        registry = get_tool_tier_registry()
        assert isinstance(registry, ToolTierRegistry)
        assert registry is ToolTierRegistry.get_instance()


class TestToolTierRegistryDefaults:
    """Test default tier registrations."""

    def test_base_tier_registered_by_default(self):
        """Base tier should be registered on initialization."""
        registry = ToolTierRegistry.get_instance()
        assert registry.has("base")

        base_config = registry.get("base")
        assert base_config is not None
        assert "read" in base_config.mandatory
        assert "ls" in base_config.mandatory
        assert "grep" in base_config.mandatory

    def test_coding_tier_registered_after_vertical_activation(self, seeded_registry):
        """Coding tier available after vertical registers its tools."""
        registry = ToolTierRegistry.get_instance()
        assert registry.has("coding")

        coding_config = registry.get("coding")
        assert coding_config is not None
        assert "edit" in coding_config.vertical_core
        assert "write" in coding_config.vertical_core

    def test_research_tier_registered_after_vertical_activation(self, seeded_registry):
        """Research tier available after vertical registers its tools."""
        registry = ToolTierRegistry.get_instance()
        assert registry.has("research")

        research_config = registry.get("research")
        assert research_config is not None
        assert research_config.readonly_only_for_analysis is True


class TestToolTierRegistryRegistration:
    """Test registration and unregistration."""

    def test_register_new_tier(self):
        """Can register a new tier."""
        registry = ToolTierRegistry.get_instance()

        config = TieredToolConfig(
            mandatory={"read", "ls"},
            vertical_core={"custom_tool"},
        )

        registry.register("custom", config, description="Custom tier")

        assert registry.has("custom")
        retrieved = registry.get("custom")
        assert retrieved is not None
        assert "custom_tool" in retrieved.vertical_core

    def test_register_duplicate_raises_error(self):
        """Registering duplicate name without overwrite raises error."""
        registry = ToolTierRegistry.get_instance()

        config = TieredToolConfig(mandatory={"read"})
        registry.register("duplicate_test", config)

        with pytest.raises(ValueError, match="already registered"):
            registry.register("duplicate_test", config)

    def test_register_with_overwrite(self):
        """Can overwrite existing registration with overwrite=True."""
        registry = ToolTierRegistry.get_instance()

        config1 = TieredToolConfig(mandatory={"read"}, vertical_core={"tool1"})
        config2 = TieredToolConfig(mandatory={"read"}, vertical_core={"tool2"})

        registry.register("overwrite_test", config1)
        registry.register("overwrite_test", config2, overwrite=True)

        retrieved = registry.get("overwrite_test")
        assert "tool2" in retrieved.vertical_core
        assert "tool1" not in retrieved.vertical_core

    def test_register_with_invalid_parent_raises_error(self):
        """Registering with non-existent parent raises error."""
        registry = ToolTierRegistry.get_instance()

        config = TieredToolConfig(mandatory={"read"})

        with pytest.raises(ValueError, match="Parent tier.*not found"):
            registry.register("child", config, parent="nonexistent_parent")

    def test_unregister_existing_tier(self):
        """Can unregister an existing tier."""
        registry = ToolTierRegistry.get_instance()

        config = TieredToolConfig(mandatory={"read"})
        registry.register("to_remove", config)

        assert registry.has("to_remove")
        result = registry.unregister("to_remove")

        assert result is True
        assert not registry.has("to_remove")

    def test_unregister_nonexistent_returns_false(self):
        """Unregistering non-existent tier returns False."""
        registry = ToolTierRegistry.get_instance()
        result = registry.unregister("nonexistent")
        assert result is False


class TestToolTierRegistryRetrieval:
    """Test retrieval methods."""

    def test_get_returns_none_for_nonexistent(self):
        """get() returns None for non-existent tier."""
        registry = ToolTierRegistry.get_instance()
        assert registry.get("nonexistent") is None

    def test_get_entry_returns_full_entry(self):
        """get_entry() returns the full TierRegistryEntry."""
        registry = ToolTierRegistry.get_instance()

        config = TieredToolConfig(mandatory={"read"})
        registry.register(
            "entry_test",
            config,
            description="Test entry",
            metadata={"key": "value"},
            version="2.0.0",
        )

        entry = registry.get_entry("entry_test")
        assert isinstance(entry, TierRegistryEntry)
        assert entry.name == "entry_test"
        assert entry.description == "Test entry"
        assert entry.metadata == {"key": "value"}
        assert entry.version == "2.0.0"

    def test_list_all_returns_all_names(self, seeded_registry):
        """list_all() returns all registered tier names."""
        registry = ToolTierRegistry.get_instance()
        names = registry.list_all()

        assert "base" in names
        assert "coding" in names
        assert "research" in names

    def test_list_entries_returns_all_entries(self):
        """list_entries() returns all TierRegistryEntry objects."""
        registry = ToolTierRegistry.get_instance()
        entries = registry.list_entries()

        assert len(entries) > 0
        assert all(isinstance(e, TierRegistryEntry) for e in entries)

    def test_get_vertical_tier_with_fallback(self):
        """get_vertical_tier() falls back to base tier."""
        registry = ToolTierRegistry.get_instance()

        # Existing vertical
        coding_config = registry.get_vertical_tier("coding")
        assert coding_config is not None

        # Non-existent vertical falls back to base
        unknown_config = registry.get_vertical_tier("unknown_vertical")
        assert unknown_config is not None
        assert unknown_config == registry.get("base")


class TestToolTierRegistryInheritance:
    """Test tier inheritance and merging."""

    def test_get_merged_includes_parent_tools(self):
        """get_merged() includes parent tier tools."""
        registry = ToolTierRegistry.get_instance()

        # Register a child tier
        child_config = TieredToolConfig(
            mandatory=set(),  # Empty mandatory
            vertical_core={"child_tool"},
        )
        registry.register("child_tier", child_config, parent="base")

        # Merged config should include base mandatory tools
        merged = registry.get_merged("child_tier")
        assert merged is not None
        assert "read" in merged.mandatory  # From base
        assert "ls" in merged.mandatory  # From base
        assert "child_tool" in merged.vertical_core  # From child

    def test_get_merged_handles_multi_level_inheritance(self):
        """get_merged() handles multi-level inheritance."""
        registry = ToolTierRegistry.get_instance()

        # Level 1: child of base
        level1_config = TieredToolConfig(
            mandatory=set(),
            vertical_core={"level1_tool"},
        )
        registry.register("level1", level1_config, parent="base")

        # Level 2: child of level1
        level2_config = TieredToolConfig(
            mandatory=set(),
            vertical_core={"level2_tool"},
        )
        registry.register("level2", level2_config, parent="level1")

        # Merged should include all ancestor tools
        merged = registry.get_merged("level2")
        assert "read" in merged.mandatory  # From base
        assert "level1_tool" in merged.vertical_core  # From level1
        assert "level2_tool" in merged.vertical_core  # From level2

    def test_get_merged_returns_none_for_nonexistent(self):
        """get_merged() returns None for non-existent tier."""
        registry = ToolTierRegistry.get_instance()
        assert registry.get_merged("nonexistent") is None


class TestToolTierRegistryExtension:
    """Test tier extension functionality."""

    def test_extend_adds_tools(self, seeded_registry):
        """extend() adds tools to base config."""
        registry = ToolTierRegistry.get_instance()

        extended = registry.extend(
            "coding",
            vertical_core={"new_tool"},
        )

        assert extended is not None
        assert "new_tool" in extended.vertical_core
        # Original tools still present
        assert "edit" in extended.vertical_core

    def test_extend_adds_mandatory_tools(self):
        """extend() can add mandatory tools."""
        registry = ToolTierRegistry.get_instance()

        extended = registry.extend(
            "base",
            mandatory={"new_mandatory"},
        )

        assert "new_mandatory" in extended.mandatory
        assert "read" in extended.mandatory  # Original still present

    def test_extend_overrides_readonly_setting(self, seeded_registry):
        """extend() can override readonly_only_for_analysis."""
        registry = ToolTierRegistry.get_instance()

        extended = registry.extend(
            "research",  # research is readonly by default
            readonly_only_for_analysis=False,
        )

        assert extended.readonly_only_for_analysis is False

    def test_extend_returns_none_for_nonexistent_base(self):
        """extend() returns None for non-existent base tier."""
        registry = ToolTierRegistry.get_instance()
        result = registry.extend("nonexistent", mandatory={"tool"})
        assert result is None


class TestToolTierRegistryValidation:
    """Test tier validation."""

    def test_validate_returns_empty_for_valid_tier(self, seeded_registry):
        """validate_tier() returns empty list for valid tier."""
        registry = ToolTierRegistry.get_instance()
        errors = registry.validate_tier("coding")
        assert errors == []

    def test_validate_detects_nonexistent_tier(self):
        """validate_tier() detects non-existent tier."""
        registry = ToolTierRegistry.get_instance()
        errors = registry.validate_tier("nonexistent")
        assert len(errors) == 1
        assert "not found" in errors[0]

    def test_validate_detects_empty_mandatory(self):
        """validate_tier() detects empty mandatory tools."""
        registry = ToolTierRegistry.get_instance()

        config = TieredToolConfig(
            mandatory=set(),  # Empty - should warn
            vertical_core={"tool"},
        )
        registry.register("empty_mandatory", config, parent="base")

        errors = registry.validate_tier("empty_mandatory")
        assert any("Empty mandatory" in e for e in errors)


class TestToolTierRegistryThreadSafety:
    """Test thread safety of the registry."""

    def test_concurrent_registration(self):
        """Concurrent registrations are thread-safe."""
        registry = ToolTierRegistry.get_instance()

        def register_tier(i: int):
            config = TieredToolConfig(
                mandatory={"read"},
                vertical_core={f"tool_{i}"},
            )
            registry.register(f"thread_tier_{i}", config, parent="base")

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(register_tier, i) for i in range(20)]
            for f in futures:
                f.result()  # Wait for all to complete

        # All tiers should be registered
        for i in range(20):
            assert registry.has(f"thread_tier_{i}")

    def test_concurrent_reads(self, seeded_registry):
        """Concurrent reads are thread-safe."""
        registry = ToolTierRegistry.get_instance()
        results = []

        def read_tier():
            config = registry.get("coding")
            results.append(config is not None)

        threads = [threading.Thread(target=read_tier) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(results)


class TestToolTierRegistrySerialization:
    """Test serialization functionality."""

    def test_entry_to_dict(self):
        """TierRegistryEntry.to_dict() serializes correctly."""
        config = TieredToolConfig(
            mandatory={"read", "ls"},
            vertical_core={"tool1"},
            semantic_pool={"tool2"},
            readonly_only_for_analysis=True,
        )

        entry = TierRegistryEntry(
            name="test",
            config=config,
            parent="base",
            description="Test entry",
            metadata={"key": "value"},
            version="1.0.0",
        )

        d = entry.to_dict()

        assert d["name"] == "test"
        assert set(d["config"]["mandatory"]) == {"read", "ls"}
        assert d["config"]["vertical_core"] == ["tool1"]
        assert d["parent"] == "base"
        assert d["description"] == "Test entry"
        assert d["metadata"] == {"key": "value"}

    def test_registry_to_dict(self, seeded_registry):
        """ToolTierRegistry.to_dict() serializes all entries."""
        registry = ToolTierRegistry.get_instance()
        d = registry.to_dict()

        assert "base" in d
        assert "coding" in d
        assert isinstance(d["base"], dict)
        assert d["base"]["name"] == "base"


class TestToolTierRegistryClear:
    """Test clear functionality."""

    def test_clear_removes_all_tiers(self):
        """clear() removes all registered tiers."""
        registry = ToolTierRegistry.get_instance()

        # Add some tiers
        config = TieredToolConfig(mandatory={"read"})
        registry.register("custom1", config, parent="base")
        registry.register("custom2", config, parent="base")

        registry.clear()

        # All tiers should be gone
        assert not registry.has("custom1")
        assert not registry.has("custom2")
        # Note: clear() doesn't re-register defaults
        assert not registry.has("base")
