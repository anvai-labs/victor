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

"""Tests for victor.framework.chain_registry module.

These tests verify the ChainRegistry functionality for cross-vertical
chain discovery and access.
"""

import pytest
from unittest.mock import MagicMock

from victor.framework.chain_registry import (
    ChainRegistry,
    ChainMetadata,
    get_chain_registry,
    register_chain,
    get_chain,
    create_chain,
    chain,
    reset_chain_registry,
)

# =============================================================================
# ChainMetadata Tests
# =============================================================================


class TestChainMetadata:
    """Tests for ChainMetadata dataclass."""

    def test_metadata_basic_attributes(self):
        """ChainMetadata should store basic attributes."""
        metadata = ChainMetadata(
            name="test_chain",
            vertical="coding",
            description="Test chain for testing",
            input_type="str",
            output_type="dict",
            tags=["test", "example"],
        )

        assert metadata.name == "test_chain"
        assert metadata.vertical == "coding"
        assert metadata.description == "Test chain for testing"
        assert metadata.input_type == "str"
        assert metadata.output_type == "dict"
        assert metadata.tags == ["test", "example"]

    def test_metadata_full_name_with_vertical(self):
        """ChainMetadata should return full name with vertical namespace."""
        metadata = ChainMetadata(name="my_chain", vertical="coding")
        assert metadata.full_name == "coding:my_chain"

    def test_metadata_full_name_without_vertical(self):
        """ChainMetadata should return name as full_name when no vertical."""
        metadata = ChainMetadata(name="my_chain")
        assert metadata.full_name == "my_chain"

    def test_metadata_defaults(self):
        """ChainMetadata should have sensible defaults."""
        metadata = ChainMetadata(name="simple")

        assert metadata.name == "simple"
        assert metadata.vertical is None
        assert metadata.description == ""
        assert metadata.input_type is None
        assert metadata.output_type is None
        assert metadata.tags == []


# =============================================================================
# ChainRegistry Basic Tests
# =============================================================================


class TestChainRegistry:
    """Tests for ChainRegistry class."""

    def test_registry_register_and_get(self):
        """Registry should register and retrieve chains."""
        registry = ChainRegistry()
        mock_chain = MagicMock(name="MockChain")

        registry.register("test_chain", mock_chain)
        result = registry.get("test_chain")

        assert result is mock_chain

    def test_registry_register_with_vertical(self):
        """Registry should register chains with vertical namespace."""
        registry = ChainRegistry()
        mock_chain = MagicMock(name="MockChain")

        registry.register("my_chain", mock_chain, vertical="coding")
        result = registry.get("my_chain", vertical="coding")

        assert result is mock_chain

    def test_registry_get_with_vertical_returns_namespaced(self):
        """Registry should return vertical-namespaced chain when vertical specified."""
        registry = ChainRegistry()
        global_chain = MagicMock(name="GlobalChain")
        coding_chain = MagicMock(name="CodingChain")

        registry.register("my_chain", global_chain)
        registry.register("my_chain", coding_chain, vertical="coding")

        # Without vertical, should get global
        assert registry.get("my_chain") is global_chain
        # With vertical, should get namespaced
        assert registry.get("my_chain", vertical="coding") is coding_chain

    def test_registry_register_duplicate_skips(self):
        """Registry should skip duplicate registration without replace=True."""
        registry = ChainRegistry()
        chain1 = MagicMock(name="Chain1")
        chain2 = MagicMock(name="Chain2")

        registry.register("test_chain", chain1)
        registry.register("test_chain", chain2)  # Should be skipped

        assert registry.get("test_chain") is chain1

    def test_registry_register_duplicate_with_replace(self):
        """Registry should allow replacing with replace=True."""
        registry = ChainRegistry()
        chain1 = MagicMock(name="Chain1")
        chain2 = MagicMock(name="Chain2")

        registry.register("test_chain", chain1)
        registry.register("test_chain", chain2, replace=True)

        assert registry.get("test_chain") is chain2

    def test_registry_unregister(self):
        """Registry should unregister chains."""
        registry = ChainRegistry()
        mock_chain = MagicMock(name="MockChain")

        registry.register("test_chain", mock_chain)
        result = registry.unregister("test_chain")

        assert result is True
        assert registry.get("test_chain") is None

    def test_registry_unregister_with_vertical(self):
        """Registry should unregister chains with vertical namespace."""
        registry = ChainRegistry()
        mock_chain = MagicMock(name="MockChain")

        registry.register("my_chain", mock_chain, vertical="coding")
        result = registry.unregister("my_chain", vertical="coding")

        assert result is True
        assert registry.get("my_chain", vertical="coding") is None

    def test_registry_unregister_nonexistent(self):
        """Registry should return False when unregistering nonexistent chain."""
        registry = ChainRegistry()
        result = registry.unregister("nonexistent_chain")
        assert result is False

    def test_registry_get_metadata(self):
        """Registry should return metadata for registered chains."""
        registry = ChainRegistry()
        mock_chain = MagicMock(name="MockChain")

        registry.register(
            "my_chain",
            mock_chain,
            vertical="coding",
            description="Test chain",
            input_type="str",
            output_type="dict",
            tags=["test"],
        )

        metadata = registry.get_metadata("my_chain", vertical="coding")

        assert metadata is not None
        assert metadata.name == "my_chain"
        assert metadata.vertical == "coding"
        assert metadata.description == "Test chain"
        assert metadata.input_type == "str"
        assert metadata.output_type == "dict"
        assert metadata.tags == ["test"]

    def test_registry_get_metadata_nonexistent(self):
        """Registry should return None for nonexistent chain metadata."""
        registry = ChainRegistry()
        metadata = registry.get_metadata("nonexistent")
        assert metadata is None


# =============================================================================
# ChainRegistry Listing Tests
# =============================================================================


class TestChainRegistryListing:
    """Tests for ChainRegistry listing methods."""

    def test_registry_list_chains(self):
        """Registry should list all chain names."""
        registry = ChainRegistry()
        registry.register("chain1", MagicMock())
        registry.register("chain2", MagicMock(), vertical="coding")
        registry.register("chain3", MagicMock(), vertical="devops")

        chains = registry.list_chains()

        assert len(chains) == 3
        assert "chain1" in chains
        assert "coding:chain2" in chains
        assert "devops:chain3" in chains

    def test_registry_list_chains_by_vertical(self):
        """Registry should list chains filtered by vertical."""
        registry = ChainRegistry()
        registry.register("chain1", MagicMock())
        registry.register("chain2", MagicMock(), vertical="coding")
        registry.register("chain3", MagicMock(), vertical="coding")
        registry.register("chain4", MagicMock(), vertical="devops")

        coding_chains = registry.list_chains(vertical="coding")

        assert len(coding_chains) == 2
        assert "coding:chain2" in coding_chains
        assert "coding:chain3" in coding_chains

    def test_registry_list_metadata(self):
        """Registry should list all chain metadata."""
        registry = ChainRegistry()
        registry.register("chain1", MagicMock(), description="First")
        registry.register("chain2", MagicMock(), vertical="coding", description="Second")

        metadata_list = registry.list_metadata()

        assert len(metadata_list) == 2
        assert any(m.name == "chain1" and m.description == "First" for m in metadata_list)
        assert any(m.name == "chain2" and m.description == "Second" for m in metadata_list)

    def test_registry_list_metadata_by_vertical(self):
        """Registry should list metadata filtered by vertical."""
        registry = ChainRegistry()
        registry.register("chain1", MagicMock())
        registry.register("chain2", MagicMock(), vertical="coding")
        registry.register("chain3", MagicMock(), vertical="devops")

        coding_metadata = registry.list_metadata(vertical="coding")

        assert len(coding_metadata) == 1
        assert coding_metadata[0].name == "chain2"


# =============================================================================
# ChainRegistry Find Tests
# =============================================================================


class TestChainRegistryFind:
    """Tests for ChainRegistry find methods."""

    def test_registry_find_by_vertical(self):
        """Registry should find chains by vertical."""
        registry = ChainRegistry()
        coding_chain = MagicMock(name="CodingChain")
        devops_chain = MagicMock(name="DevOpsChain")

        registry.register("chain1", coding_chain, vertical="coding")
        registry.register("chain2", devops_chain, vertical="devops")

        coding_chains = registry.find_by_vertical("coding")

        assert len(coding_chains) == 1
        assert "coding:chain1" in coding_chains
        assert coding_chains["coding:chain1"] is coding_chain

    def test_registry_find_by_tag(self):
        """Registry should find chains by tag."""
        registry = ChainRegistry()
        chain1 = MagicMock(name="Chain1")
        chain2 = MagicMock(name="Chain2")

        registry.register("chain1", chain1, tags=["review", "quality"])
        registry.register("chain2", chain2, tags=["implementation"])

        review_chains = registry.find_by_tag("review")

        assert len(review_chains) == 1
        assert "chain1" in review_chains

    def test_registry_find_by_tags_match_any(self):
        """Registry should find chains matching any of multiple tags."""
        registry = ChainRegistry()
        chain1 = MagicMock(name="Chain1")
        chain2 = MagicMock(name="Chain2")
        chain3 = MagicMock(name="Chain3")

        registry.register("chain1", chain1, tags=["review", "quality"])
        registry.register("chain2", chain2, tags=["implementation"])
        registry.register("chain3", chain3, tags=["quality", "testing"])

        chains = registry.find_by_tags(["quality", "testing"], match_all=False)

        assert len(chains) == 2
        assert "chain1" in chains  # Has quality
        assert "chain3" in chains  # Has both

    def test_registry_find_by_tags_match_all(self):
        """Registry should find chains matching all tags."""
        registry = ChainRegistry()
        chain1 = MagicMock(name="Chain1")
        chain2 = MagicMock(name="Chain2")

        registry.register("chain1", chain1, tags=["review", "quality"])
        registry.register("chain2", chain2, tags=["quality", "testing", "review"])

        chains = registry.find_by_tags(["quality", "review"], match_all=True)

        assert len(chains) == 2  # Both have quality and review

    def test_registry_replacement_updates_tag_indexes(self):
        """Replacing a chain should remove stale tag index entries."""
        registry = ChainRegistry()
        original = MagicMock(name="OriginalChain")
        replacement = MagicMock(name="ReplacementChain")

        registry.register("chain", original, tags=["old"], vertical="coding")
        registry.register(
            "chain",
            replacement,
            tags=["new"],
            vertical="coding",
            replace=True,
        )

        assert registry.find_by_tag("old") == {}
        assert registry.find_by_tag("new") == {"coding:chain": replacement}
        assert registry.find_by_vertical("coding") == {"coding:chain": replacement}

    def test_registry_unregister_clears_index_entries(self):
        """Unregistering a chain should remove all indexed lookup entries."""
        registry = ChainRegistry()
        chain = MagicMock(name="Chain")

        registry.register("chain", chain, tags=["review"], vertical="coding")
        assert registry.unregister("chain", vertical="coding") is True

        assert registry.find_by_tag("review") == {}
        assert registry.find_by_vertical("coding") == {}

    def test_registry_factory_metadata_is_indexed_by_vertical(self):
        """Factory metadata should remain discoverable through vertical indexes."""
        registry = ChainRegistry()

        registry.register_factory(
            "factory",
            lambda: MagicMock(name="FactoryChain"),
            vertical="coding",
            tags=["lazy"],
        )

        metadata = registry.list_metadata(vertical="coding")

        assert len(metadata) == 1
        assert metadata[0].name == "factory"
        assert metadata[0].is_factory is True


# =============================================================================
# ChainRegistry Bulk Operations Tests
# =============================================================================


class TestChainRegistryBulkOperations:
    """Tests for ChainRegistry bulk operations."""

    def test_registry_clear(self):
        """Registry should clear all chains."""
        registry = ChainRegistry()
        registry.register("chain1", MagicMock())
        registry.register("chain2", MagicMock())

        registry.clear()

        assert len(registry.list_chains()) == 0

    def test_registry_register_from_vertical(self):
        """Registry should register multiple chains from a vertical."""
        registry = ChainRegistry()
        chains = {
            "review_chain": MagicMock(name="ReviewChain"),
            "build_chain": MagicMock(name="BuildChain"),
        }

        count = registry.register_from_vertical("coding", chains)

        assert count == 2
        assert registry.get("review_chain", vertical="coding") is not None
        assert registry.get("build_chain", vertical="coding") is not None


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset the global registry before each test."""
        get_chain_registry().clear()
        yield
        get_chain_registry().clear()

    def test_register_chain_function(self):
        """register_chain should add to global registry."""
        mock_chain = MagicMock(name="MockChain")
        register_chain("test_chain", mock_chain, description="Test chain")

        result = get_chain("test_chain")
        assert result is mock_chain

    def test_register_chain_with_vertical(self):
        """register_chain should add to global registry with vertical."""
        mock_chain = MagicMock(name="MockChain")
        register_chain("test_chain", mock_chain, vertical="coding")

        result = get_chain("test_chain", vertical="coding")
        assert result is mock_chain

    def test_get_chain_function(self):
        """get_chain should retrieve from global registry."""
        mock_chain = MagicMock(name="MockChain")
        register_chain("test_chain", mock_chain)

        result = get_chain("test_chain")
        assert result is mock_chain

    def test_get_chain_returns_none_for_missing(self):
        """get_chain should return None for missing chain."""
        result = get_chain("nonexistent_chain")
        assert result is None

    def test_get_chain_registry_singleton(self):
        """get_chain_registry should return the same instance."""
        registry1 = get_chain_registry()
        registry2 = get_chain_registry()
        assert registry1 is registry2


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestChainRegistryThreadSafety:
    """Tests for ChainRegistry thread safety."""

    def test_concurrent_registration(self):
        """Registry should handle concurrent registrations safely."""
        import threading

        registry = ChainRegistry()
        errors = []

        def register_chains(prefix: str, count: int):
            try:
                for i in range(count):
                    registry.register(f"{prefix}_chain_{i}", MagicMock(), replace=True)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=register_chains, args=("thread1", 100)),
            threading.Thread(target=register_chains, args=("thread2", 100)),
            threading.Thread(target=register_chains, args=("thread3", 100)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(registry.list_chains()) == 300

    def test_concurrent_read_write(self):
        """Registry should handle concurrent reads and writes safely."""
        import threading

        registry = ChainRegistry()
        errors = []

        def writer():
            try:
                for i in range(50):
                    registry.register(f"chain_{i}", MagicMock(), replace=True)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(50):
                    registry.list_chains()
                    registry.get("chain_0")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# =============================================================================
# Singleton Pattern Tests
# =============================================================================


class TestChainRegistrySingleton:
    """Tests for ChainRegistry singleton behavior."""

    @pytest.fixture(autouse=True)
    def reset_registry_singleton(self):
        """Reset the singleton before and after each test."""
        reset_chain_registry()
        yield
        reset_chain_registry()

    def test_get_instance_returns_same_instance(self):
        """get_instance() should return the same instance."""
        instance1 = ChainRegistry.get_instance()
        instance2 = ChainRegistry.get_instance()
        assert instance1 is instance2

    def test_reset_instance_clears_singleton(self):
        """reset_instance() should clear the singleton."""
        instance1 = ChainRegistry.get_instance()
        ChainRegistry.reset_instance()
        instance2 = ChainRegistry.get_instance()
        assert instance1 is not instance2

    def test_reset_chain_registry_function(self):
        """reset_chain_registry() convenience function works."""
        instance1 = get_chain_registry()
        reset_chain_registry()
        instance2 = get_chain_registry()
        assert instance1 is not instance2


# =============================================================================
# Factory Registration Tests
# =============================================================================


class TestChainRegistryFactory:
    """Tests for ChainRegistry factory pattern."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset the registry before and after each test."""
        reset_chain_registry()
        yield
        reset_chain_registry()

    def test_register_factory_and_create(self):
        """Can register factory and create chain."""
        registry = get_chain_registry()
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return {"type": "chain", "count": call_count}

        registry.register_factory(
            "counter",
            factory,
            vertical="test",
            description="Counter chain",
        )

        chain1 = registry.create("counter", vertical="test")
        chain2 = registry.create("counter", vertical="test")

        assert chain1["count"] == 1
        assert chain2["count"] == 2  # Factory called again

    def test_register_factory_with_colon_name(self):
        """Factory supports 'vertical:name' format in name parameter."""
        registry = get_chain_registry()

        registry.register_factory(
            "coding:analyze",  # vertical:name format
            lambda: "analysis_chain",
        )

        chain_obj = registry.create("analyze", vertical="coding")
        assert chain_obj == "analysis_chain"

    def test_create_with_colon_name(self):
        """create() supports 'vertical:name' format."""
        registry = get_chain_registry()

        registry.register_factory(
            "devops:deploy",
            lambda: "deploy_chain",
        )

        chain_obj = registry.create("devops:deploy")
        assert chain_obj == "deploy_chain"

    def test_create_nonexistent_returns_none(self):
        """create() returns None for non-existent factory."""
        registry = get_chain_registry()
        result = registry.create("nonexistent")
        assert result is None

    def test_create_with_failing_factory_raises(self):
        """create() raises RuntimeError if factory fails."""
        registry = get_chain_registry()

        def failing_factory():
            raise ValueError("Factory error")

        registry.register_factory("failing", failing_factory)

        with pytest.raises(RuntimeError, match="Factory execution failed"):
            registry.create("failing")

    def test_create_chain_convenience_function(self):
        """create_chain() convenience function works."""
        registry = get_chain_registry()
        registry.register_factory(
            "coding:test",
            lambda: "test_chain",
        )

        chain_obj = create_chain("coding:test")
        assert chain_obj == "test_chain"

    def test_list_factories(self):
        """list_factories() returns factory names."""
        registry = get_chain_registry()
        registry.register_factory("coding:a", lambda: "a")
        registry.register_factory("coding:b", lambda: "b")
        registry.register_factory("devops:c", lambda: "c")

        all_factories = registry.list_factories()
        assert len(all_factories) == 3

        coding_factories = registry.list_factories(vertical="coding")
        assert set(coding_factories) == {"coding:a", "coding:b"}

    def test_has_returns_true_for_factory(self):
        """has() returns True for registered factories."""
        registry = get_chain_registry()
        registry.register_factory("test_factory", lambda: "chain")

        assert registry.has("test_factory") is True
        assert registry.has("nonexistent") is False

    def test_unregister_factory(self):
        """unregister() removes factories too."""
        registry = get_chain_registry()
        registry.register_factory("factory_to_remove", lambda: "chain")

        assert registry.has("factory_to_remove")
        result = registry.unregister("factory_to_remove")

        assert result is True
        assert not registry.has("factory_to_remove")
        assert registry.create("factory_to_remove") is None


# =============================================================================
# Chain Decorator Tests
# =============================================================================


class TestChainDecorator:
    """Tests for @chain decorator for declarative registration."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset the registry before and after each test."""
        reset_chain_registry()
        yield
        reset_chain_registry()

    def test_chain_decorator_registers_factory(self):
        """@chain decorator registers factory function."""

        @chain("test:decorated", description="Decorated chain")
        def my_chain():
            return "decorated_result"

        registry = get_chain_registry()
        assert registry.has("decorated", vertical="test")

        result = registry.create("decorated", vertical="test")
        assert result == "decorated_result"

    def test_chain_decorator_preserves_function(self):
        """@chain decorator preserves original function."""

        @chain("test:preserved")
        def original_func():
            return "original"

        # Function still callable directly
        assert original_func() == "original"

    def test_chain_decorator_with_tags(self):
        """@chain decorator passes tags to metadata."""

        @chain("test:tagged", tags=["tag1", "tag2"])
        def tagged_chain():
            return "tagged"

        registry = get_chain_registry()
        metadata = registry.get_metadata("tagged", vertical="test")

        assert metadata is not None
        assert set(metadata.tags) == {"tag1", "tag2"}

    def test_chain_decorator_metadata_is_factory(self):
        """@chain decorator sets is_factory=True in metadata."""

        @chain("test:factory_check")
        def factory_chain():
            return "factory"

        registry = get_chain_registry()
        metadata = registry.get_metadata("factory_check", vertical="test")

        assert metadata is not None
        assert metadata.is_factory is True

    def test_chain_decorator_with_version(self):
        """@chain decorator passes version to metadata."""

        @chain("test:versioned", version="2.0.0")
        def versioned_chain():
            return "v2"

        registry = get_chain_registry()
        metadata = registry.get_metadata("versioned", vertical="test")

        assert metadata is not None
        assert metadata.version == "2.0.0"


# =============================================================================
# ChainMetadata Serialization Tests
# =============================================================================


class TestChainMetadataSerialization:
    """Tests for ChainMetadata serialization."""

    def test_to_dict_serialization(self):
        """to_dict() serializes metadata correctly."""
        metadata = ChainMetadata(
            name="analyze",
            vertical="coding",
            description="Analyze code",
            input_type="str",
            output_type="dict",
            tags=["analysis", "code"],
            is_factory=True,
            version="2.0.0",
        )
        d = metadata.to_dict()

        assert d["name"] == "analyze"
        assert d["vertical"] == "coding"
        assert d["full_name"] == "coding:analyze"
        assert d["description"] == "Analyze code"
        assert d["input_type"] == "str"
        assert d["output_type"] == "dict"
        assert d["tags"] == ["analysis", "code"]
        assert d["is_factory"] is True
        assert d["version"] == "2.0.0"


# =============================================================================
# Registry Serialization Tests
# =============================================================================


class TestChainRegistrySerialization:
    """Tests for ChainRegistry serialization."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset the registry before and after each test."""
        reset_chain_registry()
        yield
        reset_chain_registry()

    def test_to_dict(self):
        """to_dict() serializes registry."""
        registry = get_chain_registry()
        registry.register("a", MagicMock(), description="Chain A")
        registry.register_factory("b", lambda: "b", description="Chain B")

        d = registry.to_dict()

        assert "a" in d
        assert "b" in d
        assert d["a"]["description"] == "Chain A"
        assert d["b"]["is_factory"] is True


# =============================================================================
# Clear Tests (Enhanced)
# =============================================================================


class TestChainRegistryClear:
    """Tests for clear functionality with factories."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset the registry before and after each test."""
        reset_chain_registry()
        yield
        reset_chain_registry()

    def test_clear_removes_chains_and_factories(self):
        """clear() removes all chains and factories."""
        registry = get_chain_registry()
        registry.register("a", MagicMock())
        registry.register_factory("b", lambda: "b")

        registry.clear()

        assert not registry.has("a")
        assert not registry.has("b")
        assert len(registry.list_chains()) == 0
        assert len(registry.list_factories()) == 0
