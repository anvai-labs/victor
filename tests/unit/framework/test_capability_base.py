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

"""Tests for BaseCapabilityProvider and CapabilityMetadata.

These tests verify the capability provider framework for verticals.
"""

import pytest
from dataclasses import FrozenInstanceError
from typing import Dict, List

from victor.framework.capabilities import BaseCapabilityProvider, CapabilityMetadata

# =============================================================================
# CapabilityMetadata Tests
# =============================================================================


class TestCapabilityMetadata:
    """Tests for CapabilityMetadata dataclass."""

    def test_create_with_required_fields(self):
        """CapabilityMetadata can be created with only required fields."""
        metadata = CapabilityMetadata(name="test_capability", description="A test capability")
        assert metadata.name == "test_capability"
        assert metadata.description == "A test capability"
        assert metadata.version == "1.0"
        assert metadata.dependencies == []
        assert metadata.tags == []

    def test_create_with_all_fields(self):
        """CapabilityMetadata can be created with all fields."""
        metadata = CapabilityMetadata(
            name="advanced_capability",
            description="An advanced capability with dependencies",
            version="2.5.1",
            dependencies=["base_capability", "helper_capability"],
            tags=["advanced", "experimental", "v2"],
        )
        assert metadata.name == "advanced_capability"
        assert metadata.description == "An advanced capability with dependencies"
        assert metadata.version == "2.5.1"
        assert metadata.dependencies == ["base_capability", "helper_capability"]
        assert metadata.tags == ["advanced", "experimental", "v2"]

    def test_default_version(self):
        """Default version should be '1.0'."""
        metadata = CapabilityMetadata(name="test", description="Test")
        assert metadata.version == "1.0"

    def test_default_dependencies_is_empty_list(self):
        """Default dependencies should be an empty list."""
        metadata = CapabilityMetadata(name="test", description="Test")
        assert metadata.dependencies == []
        assert isinstance(metadata.dependencies, list)

    def test_default_tags_is_empty_list(self):
        """Default tags should be an empty list."""
        metadata = CapabilityMetadata(name="test", description="Test")
        assert metadata.tags == []
        assert isinstance(metadata.tags, list)

    def test_metadata_equality(self):
        """Two CapabilityMetadata with same values should be equal."""
        metadata1 = CapabilityMetadata(
            name="test",
            description="Test",
            version="1.0",
            dependencies=["dep1"],
            tags=["tag1"],
        )
        metadata2 = CapabilityMetadata(
            name="test",
            description="Test",
            version="1.0",
            dependencies=["dep1"],
            tags=["tag1"],
        )
        assert metadata1 == metadata2

    def test_metadata_inequality(self):
        """Two CapabilityMetadata with different values should not be equal."""
        metadata1 = CapabilityMetadata(name="test1", description="Test 1")
        metadata2 = CapabilityMetadata(name="test2", description="Test 2")
        assert metadata1 != metadata2

    def test_metadata_repr(self):
        """CapabilityMetadata should have a readable repr."""
        metadata = CapabilityMetadata(name="test", description="Test capability")
        repr_str = repr(metadata)
        assert "CapabilityMetadata" in repr_str
        assert "test" in repr_str
        assert "Test capability" in repr_str


# =============================================================================
# Mock Capability for Testing
# =============================================================================


class MockCapability:
    """A mock capability for testing."""

    def __init__(self, name: str, value: int = 0):
        self.name = name
        self.value = value

    def execute(self) -> str:
        return f"Executed {self.name} with value {self.value}"


# =============================================================================
# Concrete Implementation for Testing
# =============================================================================


class TestCapabilityProvider(BaseCapabilityProvider[MockCapability]):
    """Concrete implementation of BaseCapabilityProvider for testing."""

    def __init__(self):
        self._capabilities: Dict[str, MockCapability] = {
            "read": MockCapability("read", 1),
            "write": MockCapability("write", 2),
            "execute": MockCapability("execute", 3),
        }
        self._metadata: Dict[str, CapabilityMetadata] = {
            "read": CapabilityMetadata(
                name="read",
                description="Read files and data",
                version="1.0",
                tags=["io", "safe"],
            ),
            "write": CapabilityMetadata(
                name="write",
                description="Write files and data",
                version="1.0",
                dependencies=["read"],
                tags=["io", "destructive"],
            ),
            "execute": CapabilityMetadata(
                name="execute",
                description="Execute commands",
                version="2.0",
                dependencies=["read", "write"],
                tags=["execution", "dangerous"],
            ),
        }

    def get_capabilities(self) -> Dict[str, MockCapability]:
        return self._capabilities

    def get_capability_metadata(self) -> Dict[str, CapabilityMetadata]:
        return self._metadata


class EmptyCapabilityProvider(BaseCapabilityProvider[MockCapability]):
    """A capability provider with no capabilities."""

    def get_capabilities(self) -> Dict[str, MockCapability]:
        return {}

    def get_capability_metadata(self) -> Dict[str, CapabilityMetadata]:
        return {}


# =============================================================================
# BaseCapabilityProvider Tests
# =============================================================================


class TestBaseCapabilityProvider:
    """Tests for BaseCapabilityProvider abstract class."""

    @pytest.fixture
    def provider(self) -> TestCapabilityProvider:
        """Create a test capability provider."""
        return TestCapabilityProvider()

    @pytest.fixture
    def empty_provider(self) -> EmptyCapabilityProvider:
        """Create an empty capability provider."""
        return EmptyCapabilityProvider()

    def test_get_capabilities_returns_all(self, provider: TestCapabilityProvider):
        """get_capabilities should return all registered capabilities."""
        capabilities = provider.get_capabilities()
        assert len(capabilities) == 3
        assert "read" in capabilities
        assert "write" in capabilities
        assert "execute" in capabilities

    def test_get_capabilities_returns_correct_types(self, provider: TestCapabilityProvider):
        """get_capabilities should return MockCapability instances."""
        capabilities = provider.get_capabilities()
        for cap in capabilities.values():
            assert isinstance(cap, MockCapability)

    def test_get_capability_existing(self, provider: TestCapabilityProvider):
        """get_capability should return the capability for existing names."""
        capability = provider.get_capability("read")
        assert capability is not None
        assert isinstance(capability, MockCapability)
        assert capability.name == "read"
        assert capability.value == 1

    def test_get_capability_nonexistent(self, provider: TestCapabilityProvider):
        """get_capability should return None for nonexistent names."""
        capability = provider.get_capability("nonexistent")
        assert capability is None

    def test_get_capability_empty_string(self, provider: TestCapabilityProvider):
        """get_capability should return None for empty string."""
        capability = provider.get_capability("")
        assert capability is None

    def test_list_capabilities(self, provider: TestCapabilityProvider):
        """list_capabilities should return all capability names."""
        names = provider.list_capabilities()
        assert len(names) == 3
        assert "read" in names
        assert "write" in names
        assert "execute" in names

    def test_list_capabilities_returns_list(self, provider: TestCapabilityProvider):
        """list_capabilities should return a list type."""
        names = provider.list_capabilities()
        assert isinstance(names, list)

    def test_list_capabilities_empty_provider(self, empty_provider: EmptyCapabilityProvider):
        """list_capabilities should return empty list for empty provider."""
        names = empty_provider.list_capabilities()
        assert names == []
        assert isinstance(names, list)

    def test_has_capability_existing(self, provider: TestCapabilityProvider):
        """has_capability should return True for existing capabilities."""
        assert provider.has_capability("read") is True
        assert provider.has_capability("write") is True
        assert provider.has_capability("execute") is True

    def test_has_capability_nonexistent(self, provider: TestCapabilityProvider):
        """has_capability should return False for nonexistent capabilities."""
        assert provider.has_capability("nonexistent") is False
        assert provider.has_capability("delete") is False

    def test_has_capability_empty_string(self, provider: TestCapabilityProvider):
        """has_capability should return False for empty string."""
        assert provider.has_capability("") is False

    def test_has_capability_empty_provider(self, empty_provider: EmptyCapabilityProvider):
        """has_capability should return False for all names in empty provider."""
        assert empty_provider.has_capability("read") is False
        assert empty_provider.has_capability("anything") is False

    def test_get_capability_metadata_returns_all(self, provider: TestCapabilityProvider):
        """get_capability_metadata should return metadata for all capabilities."""
        metadata = provider.get_capability_metadata()
        assert len(metadata) == 3
        assert "read" in metadata
        assert "write" in metadata
        assert "execute" in metadata

    def test_get_capability_metadata_correct_types(self, provider: TestCapabilityProvider):
        """get_capability_metadata should return CapabilityMetadata instances."""
        metadata = provider.get_capability_metadata()
        for meta in metadata.values():
            assert isinstance(meta, CapabilityMetadata)

    def test_metadata_matches_capabilities(self, provider: TestCapabilityProvider):
        """Metadata keys should match capability keys."""
        capabilities = provider.get_capabilities()
        metadata = provider.get_capability_metadata()
        assert set(capabilities.keys()) == set(metadata.keys())

    def test_metadata_has_correct_values(self, provider: TestCapabilityProvider):
        """Metadata should have correct values for each capability."""
        metadata = provider.get_capability_metadata()

        read_meta = metadata["read"]
        assert read_meta.name == "read"
        assert read_meta.description == "Read files and data"
        assert read_meta.version == "1.0"
        assert read_meta.tags == ["io", "safe"]

        write_meta = metadata["write"]
        assert write_meta.dependencies == ["read"]

        execute_meta = metadata["execute"]
        assert execute_meta.version == "2.0"
        assert execute_meta.dependencies == ["read", "write"]


class TestCapabilityProviderWithDifferentTypes:
    """Test that BaseCapabilityProvider works with different capability types."""

    def test_string_capability_provider(self):
        """Provider should work with string capabilities."""

        class StringProvider(BaseCapabilityProvider[str]):
            def get_capabilities(self) -> Dict[str, str]:
                return {"greeting": "Hello", "farewell": "Goodbye"}

            def get_capability_metadata(self) -> Dict[str, CapabilityMetadata]:
                return {
                    "greeting": CapabilityMetadata("greeting", "Says hello"),
                    "farewell": CapabilityMetadata("farewell", "Says goodbye"),
                }

        provider = StringProvider()
        assert provider.get_capability("greeting") == "Hello"
        assert provider.has_capability("farewell")
        assert len(provider.list_capabilities()) == 2

    def test_callable_capability_provider(self):
        """Provider should work with callable capabilities."""
        from typing import Callable

        class CallableProvider(BaseCapabilityProvider[Callable[[], int]]):
            def get_capabilities(self) -> Dict[str, Callable[[], int]]:
                return {
                    "get_one": lambda: 1,
                    "get_two": lambda: 2,
                }

            def get_capability_metadata(self) -> Dict[str, CapabilityMetadata]:
                return {
                    "get_one": CapabilityMetadata("get_one", "Returns 1"),
                    "get_two": CapabilityMetadata("get_two", "Returns 2"),
                }

        provider = CallableProvider()
        get_one = provider.get_capability("get_one")
        assert get_one is not None
        assert get_one() == 1

    def test_dict_capability_provider(self):
        """Provider should work with dict capabilities."""
        from typing import Any

        class DictProvider(BaseCapabilityProvider[Dict[str, Any]]):
            def get_capabilities(self) -> Dict[str, Dict[str, Any]]:
                return {
                    "config": {"host": "localhost", "port": 8080},
                    "settings": {"debug": True, "log_level": "INFO"},
                }

            def get_capability_metadata(self) -> Dict[str, CapabilityMetadata]:
                return {
                    "config": CapabilityMetadata("config", "Configuration dict"),
                    "settings": CapabilityMetadata("settings", "Settings dict"),
                }

        provider = DictProvider()
        config = provider.get_capability("config")
        assert config is not None
        assert config["host"] == "localhost"
        assert config["port"] == 8080


class TestCapabilityProviderAbstractMethods:
    """Test that base methods raise NotImplementedError when not overridden."""

    def test_base_class_get_capabilities_raises(self):
        """BaseCapabilityProvider.get_capabilities() raises NotImplementedError."""
        provider = BaseCapabilityProvider()  # type: ignore
        with pytest.raises(NotImplementedError):
            provider.get_capabilities()

    def test_must_implement_get_capabilities(self):
        """Subclass without get_capabilities raises NotImplementedError on call."""

        class IncompleteProvider(BaseCapabilityProvider[str]):
            def get_capability_metadata(self) -> Dict[str, CapabilityMetadata]:
                return {}

        provider = IncompleteProvider()
        with pytest.raises(NotImplementedError):
            provider.get_capabilities()

    def test_must_implement_get_capability_metadata(self):
        """Subclass without get_capability_metadata raises NotImplementedError on call."""

        class IncompleteProvider(BaseCapabilityProvider[str]):
            def get_capabilities(self) -> Dict[str, str]:
                return {}

        provider = IncompleteProvider()
        with pytest.raises(NotImplementedError):
            provider.get_capability_metadata()
