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

"""Tests for victor.framework.chains.registry module.

These tests verify the ChainRegistry functionality for versioned tool chains
with semantic versioning support and thread-safe singleton implementation.
"""

import threading
import pytest
from unittest.mock import MagicMock

from victor.framework.chains.registry import (
    ChainMetadata,
    ChainRegistry,
    get_chain_registry,
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
            version="1.0.0",
            description="Test chain",
            category="testing",
            tags=["test", "unit"],
        )

        assert metadata.name == "test_chain"
        assert metadata.version == "1.0.0"
        assert metadata.description == "Test chain"
        assert metadata.category == "testing"
        assert metadata.tags == ["test", "unit"]

    def test_metadata_with_author(self):
        """ChainMetadata should store author."""
        metadata = ChainMetadata(
            name="chain",
            version="1.0.0",
            description="Test",
            category="testing",
            author="Test Author",
        )

        assert metadata.author == "Test Author"

    def test_metadata_deprecated_flag(self):
        """ChainMetadata should store deprecated flag."""
        metadata1 = ChainMetadata(
            name="chain",
            version="1.0.0",
            description="Test",
            category="testing",
            deprecated=False,
        )

        metadata2 = ChainMetadata(
            name="chain",
            version="2.0.0",
            description="Test",
            category="testing",
            deprecated=True,
        )

        assert metadata1.deprecated is False
        assert metadata2.deprecated is True

    def test_metadata_defaults(self):
        """ChainMetadata should have sensible defaults."""
        metadata = ChainMetadata(
            name="test",
            version="1.0.0",
            description="Test",
            category="testing",
        )

        assert metadata.tags == []
        assert metadata.author is None
        assert metadata.deprecated is False


# =============================================================================
# ChainRegistry Singleton Tests
# =============================================================================


class TestChainRegistrySingleton:
    """Tests for ChainRegistry singleton pattern."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        ChainRegistry._instance = None
        ChainRegistry._initialized = False
        yield
        ChainRegistry._instance = None
        ChainRegistry._initialized = False

    def test_singleton_returns_same_instance(self):
        """Singleton should return same instance."""
        registry1 = ChainRegistry()
        registry2 = ChainRegistry()

        assert registry1 is registry2

    def test_singleton_thread_safe(self):
        """Singleton creation should be thread-safe."""
        instances = []
        lock = threading.Lock()

        def create_instance():
            instance = ChainRegistry()
            with lock:
                instances.append(instance)

        threads = [threading.Thread(target=create_instance) for _ in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All instances should be the same
        assert all(i is instances[0] for i in instances)

    def test_singleton_initializes_once(self):
        """Singleton should initialize only once."""
        registry1 = ChainRegistry()
        registry2 = ChainRegistry()

        # Both should have same initialized state
        assert registry1._initialized is True
        assert registry2._initialized is True
        assert registry1 is registry2


# =============================================================================
# ChainRegistry Registration Tests
# =============================================================================


class TestChainRegistryRegistration:
    """Tests for ChainRegistry registration methods."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        ChainRegistry._instance = None
        ChainRegistry._initialized = False
        yield
        ChainRegistry._instance = None
        ChainRegistry._initialized = False

    def test_register_chain_basic(self):
        """register_chain() should register a chain."""
        registry = ChainRegistry()
        mock_chain = MagicMock(name="MockChain")

        registry.register_chain(
            name="test_chain",
            version="1.0.0",
            chain=mock_chain,
            category="testing",
            description="Test chain",
        )

        retrieved = registry.get_chain("test_chain", version="1.0.0")
        assert retrieved is mock_chain

    def test_register_chain_with_tags(self):
        """register_chain() should register chain with tags."""
        registry = ChainRegistry()
        mock_chain = MagicMock()

        registry.register_chain(
            name="tagged_chain",
            version="1.0.0",
            chain=mock_chain,
            category="testing",
            tags=["tag1", "tag2"],
        )

        metadata = registry.get_chain_metadata("tagged_chain", version="1.0.0")
        assert metadata.tags == ["tag1", "tag2"]

    def test_register_chain_with_author(self):
        """register_chain() should register chain with author."""
        registry = ChainRegistry()
        mock_chain = MagicMock()

        registry.register_chain(
            name="authored_chain",
            version="1.0.0",
            chain=mock_chain,
            category="testing",
            author="Test Author",
        )

        metadata = registry.get_chain_metadata("authored_chain", version="1.0.0")
        assert metadata.author == "Test Author"

    def test_register_chain_deprecated(self):
        """register_chain() should register deprecated chain."""
        registry = ChainRegistry()
        mock_chain = MagicMock()

        registry.register_chain(
            name="deprecated_chain",
            version="1.0.0",
            chain=mock_chain,
            category="testing",
            deprecated=True,
        )

        metadata = registry.get_chain_metadata("deprecated_chain", version="1.0.0")
        assert metadata.deprecated is True

    def test_register_multiple_versions(self):
        """register_chain() should register multiple versions."""
        registry = ChainRegistry()
        chain_v1 = MagicMock(name="ChainV1")
        chain_v2 = MagicMock(name="ChainV2")

        registry.register_chain(
            name="multi_chain",
            version="1.0.0",
            chain=chain_v1,
            category="testing",
        )

        registry.register_chain(
            name="multi_chain",
            version="2.0.0",
            chain=chain_v2,
            category="testing",
        )

        assert registry.get_chain("multi_chain", version="1.0.0") is chain_v1
        assert registry.get_chain("multi_chain", version="2.0.0") is chain_v2


# =============================================================================
# ChainRegistry SemVer Validation Tests
# =============================================================================


class TestChainRegistrySemVerValidation:
    """Tests for ChainRegistry semantic versioning validation."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        ChainRegistry._instance = None
        ChainRegistry._initialized = False
        yield
        ChainRegistry._instance = None
        ChainRegistry._initialized = False

    def test_valid_semver_accepted(self):
        """Valid SemVer should be accepted."""
        registry = ChainRegistry()
        mock_chain = MagicMock()

        # Should not raise
        registry.register_chain(
            name="test",
            version="1.0.0",
            chain=mock_chain,
            category="testing",
        )

    def test_valid_semver_with_pre_release(self):
        """SemVer with pre-release should be accepted."""
        registry = ChainRegistry()
        mock_chain = MagicMock()

        registry.register_chain(
            name="test",
            version="1.0.0-alpha",
            chain=mock_chain,
            category="testing",
        )

        chain = registry.get_chain("test", version="1.0.0-alpha")
        assert chain is mock_chain

    def test_valid_semver_with_build_metadata(self):
        """SemVer with build metadata should be accepted."""
        registry = ChainRegistry()
        mock_chain = MagicMock()

        registry.register_chain(
            name="test",
            version="1.0.0+build.1",
            chain=mock_chain,
            category="testing",
        )

        chain = registry.get_chain("test", version="1.0.0+build.1")
        assert chain is mock_chain

    def test_invalid_semver_raises_error(self):
        """Invalid SemVer should raise ValueError."""
        registry = ChainRegistry()
        mock_chain = MagicMock()

        with pytest.raises(ValueError, match="Invalid SemVer"):
            registry.register_chain(
                name="test",
                version="invalid",
                chain=mock_chain,
                category="testing",
            )

    def test_invalid_semver_missing_patch(self):
        """SemVer missing patch should raise ValueError."""
        registry = ChainRegistry()
        mock_chain = MagicMock()

        with pytest.raises(ValueError, match="Invalid SemVer"):
            registry.register_chain(
                name="test",
                version="1.0",
                chain=mock_chain,
                category="testing",
            )

    def test_invalid_semver_missing_minor(self):
        """SemVer missing minor should raise ValueError."""
        registry = ChainRegistry()
        mock_chain = MagicMock()

        with pytest.raises(ValueError, match="Invalid SemVer"):
            registry.register_chain(
                name="test",
                version="1",
                chain=mock_chain,
                category="testing",
            )

    def test_invalid_semver_negative_numbers(self):
        """SemVer with negative numbers should raise ValueError."""
        registry = ChainRegistry()
        mock_chain = MagicMock()

        with pytest.raises(ValueError, match="Invalid SemVer"):
            registry.register_chain(
                name="test",
                version="1.0.-1",
                chain=mock_chain,
                category="testing",
            )


# =============================================================================
# ChainRegistry Category Validation Tests
# =============================================================================


class TestChainRegistryCategoryValidation:
    """Tests for ChainRegistry category validation."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        ChainRegistry._instance = None
        ChainRegistry._initialized = False
        yield
        ChainRegistry._instance = None
        ChainRegistry._initialized = False

    def test_valid_category_exploration(self):
        """Valid category 'exploration' should be accepted."""
        registry = ChainRegistry()
        mock_chain = MagicMock()

        registry.register_chain(
            name="test",
            version="1.0.0",
            chain=mock_chain,
            category="exploration",
        )

        assert registry.get_chain("test") is mock_chain

    def test_valid_category_editing(self):
        """Valid category 'editing' should be accepted."""
        registry = ChainRegistry()
        mock_chain = MagicMock()

        registry.register_chain(
            name="test",
            version="1.0.0",
            chain=mock_chain,
            category="editing",
        )

        assert registry.get_chain("test") is mock_chain

    def test_valid_category_analysis(self):
        """Valid category 'analysis' should be accepted."""
        registry = ChainRegistry()
        mock_chain = MagicMock()

        registry.register_chain(
            name="test",
            version="1.0.0",
            chain=mock_chain,
            category="analysis",
        )

        assert registry.get_chain("test") is mock_chain

    def test_valid_category_testing(self):
        """Valid category 'testing' should be accepted."""
        registry = ChainRegistry()
        mock_chain = MagicMock()

        registry.register_chain(
            name="test",
            version="1.0.0",
            chain=mock_chain,
            category="testing",
        )

        assert registry.get_chain("test") is mock_chain

    def test_valid_category_other(self):
        """Valid category 'other' should be accepted."""
        registry = ChainRegistry()
        mock_chain = MagicMock()

        registry.register_chain(
            name="test",
            version="1.0.0",
            chain=mock_chain,
            category="other",
        )

        assert registry.get_chain("test") is mock_chain

    def test_invalid_category_raises_error(self):
        """Invalid category should raise ValueError."""
        registry = ChainRegistry()
        mock_chain = MagicMock()

        with pytest.raises(ValueError, match="Invalid category"):
            registry.register_chain(
                name="test",
                version="1.0.0",
                chain=mock_chain,
                category="invalid_category",
            )


# =============================================================================
# ChainRegistry Retrieval Tests
# =============================================================================


class TestChainRegistryRetrieval:
    """Tests for ChainRegistry retrieval methods."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        ChainRegistry._instance = None
        ChainRegistry._initialized = False
        yield
        ChainRegistry._instance = None
        ChainRegistry._initialized = False

    def test_get_chain_returns_registered_chain(self):
        """get_chain() should return registered chain."""
        registry = ChainRegistry()
        mock_chain = MagicMock()

        registry.register_chain(
            name="test",
            version="1.0.0",
            chain=mock_chain,
            category="testing",
        )

        retrieved = registry.get_chain("test", version="1.0.0")
        assert retrieved is mock_chain

    def test_get_chain_returns_none_for_missing(self):
        """get_chain() should return None for missing chain."""
        registry = ChainRegistry()

        retrieved = registry.get_chain("nonexistent", version="1.0.0")
        assert retrieved is None

    def test_get_chain_latest_version(self):
        """get_chain() should return latest version when version not specified."""
        registry = ChainRegistry()
        chain_v1 = MagicMock(name="ChainV1")
        chain_v2 = MagicMock(name="ChainV2")

        registry.register_chain("test", "1.0.0", chain_v1, "testing")
        registry.register_chain("test", "2.0.0", chain_v2, "testing")

        retrieved = registry.get_chain("test")
        assert retrieved is chain_v2

    def test_get_chain_metadata(self):
        """get_chain_metadata() should return metadata."""
        registry = ChainRegistry()
        mock_chain = MagicMock()

        registry.register_chain(
            name="test",
            version="1.0.0",
            chain=mock_chain,
            category="testing",
            description="Test chain",
            tags=["tag1"],
            author="Author",
        )

        metadata = registry.get_chain_metadata("test", version="1.0.0")

        assert metadata is not None
        assert metadata.name == "test"
        assert metadata.version == "1.0.0"
        assert metadata.description == "Test chain"
        assert metadata.tags == ["tag1"]
        assert metadata.author == "Author"

    def test_get_chain_metadata_none_for_missing(self):
        """get_chain_metadata() should return None for missing chain."""
        registry = ChainRegistry()

        metadata = registry.get_chain_metadata("nonexistent")
        assert metadata is None

    def test_get_chain_version(self):
        """get_chain_version() should return latest version."""
        registry = ChainRegistry()
        mock_chain = MagicMock()

        registry.register_chain("test", "1.0.0", mock_chain, "testing")
        registry.register_chain("test", "2.0.0", mock_chain, "testing")
        registry.register_chain("test", "1.5.0", mock_chain, "testing")

        latest = registry.get_chain_version("test")
        assert latest == "2.0.0"

    def test_get_chain_version_none_for_missing(self):
        """get_chain_version() should return None for missing chain."""
        registry = ChainRegistry()

        latest = registry.get_chain_version("nonexistent")
        assert latest is None


# =============================================================================
# ChainRegistry Listing Tests
# =============================================================================


class TestChainRegistryListing:
    """Tests for ChainRegistry listing methods."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        ChainRegistry._instance = None
        ChainRegistry._initialized = False
        yield
        ChainRegistry._instance = None
        ChainRegistry._initialized = False

    def test_list_chains_all(self):
        """list_chains() should return all chain names."""
        registry = ChainRegistry()

        registry.register_chain("chain1", "1.0.0", MagicMock(), "testing")
        registry.register_chain("chain2", "1.0.0", MagicMock(), "editing")
        registry.register_chain("chain3", "1.0.0", MagicMock(), "analysis")

        chains = registry.list_chains()

        assert len(chains) == 3
        assert "chain1" in chains
        assert "chain2" in chains
        assert "chain3" in chains

    def test_list_chains_by_category(self):
        """list_chains() should filter by category."""
        registry = ChainRegistry()

        registry.register_chain("test1", "1.0.0", MagicMock(), "testing")
        registry.register_chain("edit1", "1.0.0", MagicMock(), "editing")
        registry.register_chain("test2", "1.0.0", MagicMock(), "testing")

        testing_chains = registry.list_chains(category="testing")

        assert len(testing_chains) == 2
        assert "test1" in testing_chains
        assert "test2" in testing_chains
        assert "edit1" not in testing_chains

    def test_list_chains_invalid_category(self):
        """list_chains() should return empty list for invalid category."""
        registry = ChainRegistry()

        chains = registry.list_chains(category="invalid")
        assert chains == []

    def test_list_chain_versions(self):
        """list_chain_versions() should return all versions sorted descending."""
        registry = ChainRegistry()

        registry.register_chain("test", "1.0.0", MagicMock(), "testing")
        registry.register_chain("test", "2.0.0", MagicMock(), "testing")
        registry.register_chain("test", "1.5.0", MagicMock(), "testing")

        versions = registry.list_chain_versions("test")

        assert len(versions) == 3
        # Should be sorted descending (latest first)
        assert versions[0] == "2.0.0"
        assert versions[1] == "1.5.0"
        assert versions[2] == "1.0.0"

    def test_list_chain_versions_missing_chain(self):
        """list_chain_versions() should return empty list for missing chain."""
        registry = ChainRegistry()

        versions = registry.list_chain_versions("nonexistent")
        assert versions == []


# =============================================================================
# ChainRegistry Removal Tests
# =============================================================================


class TestChainRegistryRemoval:
    """Tests for ChainRegistry removal methods."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        ChainRegistry._instance = None
        ChainRegistry._initialized = False
        yield
        ChainRegistry._instance = None
        ChainRegistry._initialized = False

    def test_remove_chain_specific_version(self):
        """remove_chain() should remove specific version."""
        registry = ChainRegistry()
        chain_v1 = MagicMock()
        chain_v2 = MagicMock()

        registry.register_chain("test", "1.0.0", chain_v1, "testing")
        registry.register_chain("test", "2.0.0", chain_v2, "testing")

        result = registry.remove_chain("test", version="1.0.0")

        assert result is True
        assert registry.get_chain("test", version="1.0.0") is None
        assert registry.get_chain("test", version="2.0.0") is chain_v2

    def test_remove_chain_all_versions(self):
        """remove_chain() should remove all versions when version is None."""
        registry = ChainRegistry()

        registry.register_chain("test", "1.0.0", MagicMock(), "testing")
        registry.register_chain("test", "2.0.0", MagicMock(), "testing")

        result = registry.remove_chain("test")

        assert result is True
        assert registry.get_chain("test") is None

    def test_remove_chain_missing(self):
        """remove_chain() should return False for missing chain."""
        registry = ChainRegistry()

        result = registry.remove_chain("nonexistent")
        assert result is False

    def test_remove_chain_removes_from_category(self):
        """remove_chain() should remove chain from category."""
        registry = ChainRegistry()

        registry.register_chain("test", "1.0.0", MagicMock(), "testing")

        assert "test" in registry._categories["testing"]

        registry.remove_chain("test")

        assert "test" not in registry._categories["testing"]


# =============================================================================
# ChainRegistry Statistics Tests
# =============================================================================


class TestChainRegistryStatistics:
    """Tests for ChainRegistry statistics methods."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        ChainRegistry._instance = None
        ChainRegistry._initialized = False
        yield
        ChainRegistry._instance = None
        ChainRegistry._initialized = False

    def test_get_registry_stats_empty(self):
        """get_registry_stats() should return stats for empty registry."""
        registry = ChainRegistry()

        stats = registry.get_registry_stats()

        assert stats["total_chains"] == 0
        assert stats["total_versions"] == 0
        assert stats["category_counts"]["exploration"] == 0
        assert stats["category_counts"]["editing"] == 0
        assert stats["category_counts"]["analysis"] == 0
        assert stats["category_counts"]["testing"] == 0
        assert stats["category_counts"]["other"] == 0

    def test_get_registry_stats_with_chains(self):
        """get_registry_stats() should return correct stats."""
        registry = ChainRegistry()

        registry.register_chain("test1", "1.0.0", MagicMock(), "testing")
        registry.register_chain("test2", "1.0.0", MagicMock(), "testing")
        registry.register_chain("edit1", "1.0.0", MagicMock(), "editing")
        registry.register_chain("test1", "2.0.0", MagicMock(), "testing")

        stats = registry.get_registry_stats()

        assert stats["total_chains"] == 3  # test1, test2, edit1
        assert stats["total_versions"] == 4  # 2 versions of test1
        assert stats["category_counts"]["testing"] == 2  # test1, test2
        assert stats["category_counts"]["editing"] == 1  # edit1


# =============================================================================
# ChainRegistry SemVer Sorting Tests
# =============================================================================


class TestChainRegistrySemVerSorting:
    """Tests for ChainRegistry SemVer sorting logic."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        ChainRegistry._instance = None
        ChainRegistry._initialized = False
        yield
        ChainRegistry._instance = None
        ChainRegistry._initialized = False

    def test_semver_key_basic(self):
        """_semver_key() should convert version to sortable tuple."""
        key = ChainRegistry._semver_key("1.2.3")
        assert key == (1, 2, 3)

    def test_semver_key_with_pre_release(self):
        """_semver_key() should strip pre-release for sorting."""
        key1 = ChainRegistry._semver_key("1.2.3-alpha")
        key2 = ChainRegistry._semver_key("1.2.3-beta")
        assert key1 == (1, 2, 3)
        assert key2 == (1, 2, 3)

    def test_semver_key_with_build_metadata(self):
        """_semver_key() should strip build metadata for sorting."""
        key = ChainRegistry._semver_key("1.2.3+build.1")
        assert key == (1, 2, 3)

    def test_semver_key_with_both(self):
        """_semver_key() should strip both pre-release and build."""
        key = ChainRegistry._semver_key("1.2.3-alpha+build.1")
        assert key == (1, 2, 3)

    def test_get_latest_version_basic(self):
        """_get_latest_version() should return highest version."""
        versions = ["1.0.0", "2.0.0", "1.5.0"]
        latest = ChainRegistry._get_latest_version(versions)

        assert latest == "2.0.0"

    def test_get_latest_version_with_pre_release(self):
        """_get_latest_version() should handle pre-release."""
        versions = ["1.0.0", "1.0.0-alpha", "2.0.0"]
        latest = ChainRegistry._get_latest_version(versions)

        assert latest == "2.0.0"

    def test_get_latest_version_empty_list(self):
        """_get_latest_version() should return default for empty list."""
        latest = ChainRegistry._get_latest_version([])

        assert latest == "0.0.0"

    def test_get_latest_version_complex(self):
        """_get_latest_version() should handle complex versions."""
        versions = [
            "1.0.0",
            "2.1.0",
            "1.10.0",
            "2.0.5",
            "1.0.1",
        ]
        latest = ChainRegistry._get_latest_version(versions)

        assert latest == "2.1.0"


# =============================================================================
# ChainRegistry Thread Safety Tests
# =============================================================================


class TestChainRegistryThreadSafety:
    """Tests for ChainRegistry thread safety."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        ChainRegistry._instance = None
        ChainRegistry._initialized = False
        yield
        ChainRegistry._instance = None
        ChainRegistry._initialized = False

    def test_concurrent_registration(self):
        """Registry should handle concurrent registrations safely."""
        registry = ChainRegistry()
        errors = []

        def register_chains(prefix: str, count: int):
            try:
                for i in range(count):
                    registry.register_chain(
                        name=f"{prefix}_chain_{i}",
                        version="1.0.0",
                        chain=MagicMock(),
                        category="testing",
                    )
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
        registry = ChainRegistry()
        errors = []

        # Pre-populate
        registry.register_chain("chain_0", "1.0.0", MagicMock(), "testing")

        def writer():
            try:
                for i in range(1, 50):
                    registry.register_chain(
                        name=f"chain_{i}",
                        version="1.0.0",
                        chain=MagicMock(),
                        category="testing",
                    )
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(50):
                    registry.list_chains()
                    registry.get_chain("chain_0")
                    registry.get_chain_version("chain_0")
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
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        ChainRegistry._instance = None
        ChainRegistry._initialized = False
        yield
        ChainRegistry._instance = None
        ChainRegistry._initialized = False

    def test_get_chain_registry_singleton(self):
        """get_chain_registry() should return singleton instance."""
        registry1 = get_chain_registry()
        registry2 = get_chain_registry()

        assert registry1 is registry2

    def test_get_chain_registry_is_chain_registry(self):
        """get_chain_registry() should return ChainRegistry instance."""
        registry = get_chain_registry()

        assert isinstance(registry, ChainRegistry)


__all__ = [
    "TestChainMetadata",
    "TestChainRegistrySingleton",
    "TestChainRegistryRegistration",
    "TestChainRegistrySemVerValidation",
    "TestChainRegistryCategoryValidation",
    "TestChainRegistryRetrieval",
    "TestChainRegistryListing",
    "TestChainRegistryRemoval",
    "TestChainRegistryStatistics",
    "TestChainRegistrySemVerSorting",
    "TestChainRegistryThreadSafety",
    "TestConvenienceFunctions",
]
