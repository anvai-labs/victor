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

"""Tests for vertical composition framework."""

import pytest

from victor.core.verticals.composition import (
    BaseComposer,
    CapabilityComposer,
    MetadataCapability,
    StagesCapability,
    ExtensionsCapability,
    BaseCapabilityProvider,
)
from victor.core.verticals.base import VerticalBase


class TestBaseCapabilityProvider:
    """Tests for BaseCapabilityProvider."""

    def test_initialization(self):
        """Test provider initialization."""
        provider = BaseCapabilityProvider()
        assert provider.get_config() == {}

    def test_custom_config(self):
        """Test provider with custom config."""

        class CustomProvider(BaseCapabilityProvider):
            def __init__(self):
                super().__init__()
                self._config = {"key": "value"}

        provider = CustomProvider()
        assert provider.get_config() == {"key": "value"}

    def test_validation(self):
        """Test provider validation."""
        provider = BaseCapabilityProvider()
        assert provider.validate() is True


class TestMetadataCapability:
    """Tests for MetadataCapability."""

    def test_initialization(self):
        """Test metadata capability initialization."""
        metadata = MetadataCapability(
            name="test_vertical",
            description="Test description",
            version="1.0.0",
        )

        assert metadata.name == "test_vertical"
        assert metadata.description == "Test description"
        assert metadata.version == "1.0.0"

    def test_with_provider_hints(self):
        """Test metadata with provider hints."""
        provider_hints = {"temperature": 0.7, "max_tokens": 4096}
        metadata = MetadataCapability(
            name="test",
            description="Test",
            provider_hints=provider_hints,
        )

        assert metadata.provider_hints == provider_hints

    def test_get_config(self):
        """Test getting config from metadata."""
        metadata = MetadataCapability("test", "description")
        config = metadata.get_config()

        assert config["name"] == "test"
        assert config["description"] == "description"
        assert config["version"] == "1.0.0"


class TestStagesCapability:
    """Tests for StagesCapability."""

    def test_initialization(self):
        """Test stages capability initialization."""
        from victor.core.vertical_types import StageDefinition

        stages = {
            "planning": StageDefinition(
                name="Planning",
                description="Plan the approach",
                keywords=["plan", "strategy"],
                next_stages={"execution"},
            )
        }

        capability = StagesCapability(stages)
        assert capability.stages == stages

    def test_get_config(self):
        """Test getting config from stages."""
        from victor.core.vertical_types import StageDefinition

        stages = {
            "planning": StageDefinition(
                name="Planning",
                description="Plan",
                keywords=[],
                next_stages=set(),
            )
        }

        capability = StagesCapability(stages)
        config = capability.get_config()

        assert "stages" in config
        assert "planning" in config["stages"]


class TestExtensionsCapability:
    """Tests for ExtensionsCapability."""

    def test_initialization(self):
        """Test extensions capability initialization."""
        extensions = [object(), object()]

        capability = ExtensionsCapability(extensions)
        assert capability.extensions == extensions

    def test_with_extension_types(self):
        """Test extensions with type mapping."""
        ext1 = object()
        ext2 = object()
        extensions = [ext1, ext2]

        extension_types = {
            "object": "test_extension",
        }

        capability = ExtensionsCapability(extensions, extension_types)
        config = capability.get_config()

        assert len(config["extensions"]) == 2


class TestBaseComposer:
    """Tests for BaseComposer."""

    def test_initialization(self):
        """Test composer initialization."""
        composer = BaseComposer(VerticalBase)
        assert composer._vertical == VerticalBase
        assert composer.list_capabilities() == []

    def test_register_capability(self):
        """Test registering a capability."""
        composer = BaseComposer(VerticalBase)
        provider = BaseCapabilityProvider()

        result = composer.register_capability("test", provider)

        assert result is composer  # Method chaining
        assert composer.has_capability("test")

    def test_get_capability(self):
        """Test getting a capability."""
        composer = BaseComposer(VerticalBase)
        provider = BaseCapabilityProvider()

        composer.register_capability("test", provider)
        retrieved = composer.get_capability("test")

        assert retrieved is provider

    def test_get_nonexistent_capability(self):
        """Test getting nonexistent capability returns None."""
        composer = BaseComposer(VerticalBase)
        assert composer.get_capability("nonexistent") is None

    def test_list_capabilities(self):
        """Test listing all capabilities."""
        composer = BaseComposer(VerticalBase)
        provider1 = BaseCapabilityProvider()
        provider2 = BaseCapabilityProvider()

        composer.register_capability("cap1", provider1)
        composer.register_capability("cap2", provider2)

        capabilities = composer.list_capabilities()
        assert set(capabilities) == {"cap1", "cap2"}

    def test_get_all_configs(self):
        """Test getting all capability configs."""
        composer = BaseComposer(VerticalBase)

        class TestProvider(BaseCapabilityProvider):
            def __init__(self, value):
                super().__init__()
                self._config = {"value": value}

        provider1 = TestProvider(1)
        provider2 = TestProvider(2)

        composer.register_capability("cap1", provider1)
        composer.register_capability("cap2", provider2)

        configs = composer.get_all_configs()

        assert configs["cap1"]["value"] == 1
        assert configs["cap2"]["value"] == 2

    def test_validate_all(self):
        """Test validating all capabilities."""
        composer = BaseComposer(VerticalBase)
        provider = BaseCapabilityProvider()

        composer.register_capability("test", provider)
        assert composer.validate_all() is True


class TestCapabilityComposer:
    """Tests for CapabilityComposer."""

    def test_with_metadata(self):
        """Test adding metadata via fluent API."""
        composer = CapabilityComposer(VerticalBase)

        result = composer.with_metadata("test", "description", "1.0.0")

        assert result is composer  # Method chaining
        assert composer.has_capability("metadata")

        metadata = composer.get_capability("metadata")
        assert isinstance(metadata, MetadataCapability)
        assert metadata.name == "test"

    def test_with_stages(self):
        """Test adding stages via fluent API."""
        from victor.core.vertical_types import StageDefinition

        stages = {
            "test": StageDefinition(
                name="Test",
                description="Test stage",
                keywords=[],
                next_stages=set(),
            )
        }

        composer = CapabilityComposer(VerticalBase)
        result = composer.with_stages(stages)

        assert result is composer
        assert composer.has_capability("stages")

    def test_with_extensions(self):
        """Test adding extensions via fluent API."""
        extensions = [object(), object()]

        composer = CapabilityComposer(VerticalBase)
        result = composer.with_extensions(extensions)

        assert result is composer
        assert composer.has_capability("extensions")

    def test_build_creates_vertical(self):
        """Test building a vertical class."""
        composer = CapabilityComposer(VerticalBase)

        # Add required metadata
        composer.with_metadata("test_vertical", "Test description", "1.0.0")

        # Add tools and prompt via helper methods
        composer.with_tools(["read", "write"])
        composer.with_system_prompt("You are a test assistant.")

        # Build the vertical
        Vertical = composer.build()

        # Verify the vertical class
        assert Vertical.name == "test_vertical"
        assert Vertical.description == "Test description"
        assert Vertical.version == "1.0.0"

    def test_build_without_metadata_raises(self):
        """Test that building without metadata raises error."""
        composer = CapabilityComposer(VerticalBase)

        with pytest.raises(ValueError, match="Metadata capability is required"):
            composer.build()

    def test_build_with_invalid_capability_raises(self):
        """Test that building with invalid capability raises error."""

        class InvalidProvider(BaseCapabilityProvider):
            def validate(self):
                return False

        composer = CapabilityComposer(VerticalBase)
        composer.with_metadata("test", "description")
        composer.register_capability("invalid", InvalidProvider())

        with pytest.raises(ValueError, match="Invalid capability"):
            composer.build()


class TestVerticalBaseCompose:
    """Tests for VerticalBase.compose() method."""

    def test_compose_returns_composer(self):
        """Test that compose() returns a CapabilityComposer."""
        composer = VerticalBase.compose()

        assert isinstance(composer, CapabilityComposer)
        assert composer._vertical == VerticalBase

    def test_compose_fluent_api(self):
        """Test using fluent API from compose()."""
        # This should not raise
        composer = VerticalBase.compose().with_metadata("test", "description")

        assert isinstance(composer, CapabilityComposer)

    def test_get_composer_when_none(self):
        """Test get_composer() returns None for inheritance-based verticals."""
        composer = VerticalBase.get_composer()
        assert composer is None

    def test_get_composer_after_build(self):
        """Test get_composer() returns composer after build()."""
        composer = (
            VerticalBase.compose()
            .with_metadata("test", "description")
            .with_tools(["read"])
            .with_system_prompt("Test")
        )

        Vertical = composer.build()
        retrieved_composer = Vertical.get_composer()

        assert retrieved_composer is composer


class TestCompositionIntegration:
    """Integration tests for composition framework."""

    def test_full_composition_workflow(self):
        """Test complete workflow from composition to vertical."""
        from victor.core.vertical_types import StageDefinition

        # Define stages
        stages = {
            "planning": StageDefinition(
                name="Planning",
                description="Plan the approach",
                keywords=["plan", "design"],
                next_stages={"execution"},
            ),
            "execution": StageDefinition(
                name="Execution",
                description="Execute the plan",
                keywords=["run", "execute"],
                next_stages=set(),
            ),
        }

        # Compose the vertical
        TestVertical = (
            VerticalBase.compose()
            .with_metadata("test_vertical", "Test assistant", "2.0.0")
            .with_tools(["read", "write", "search"])
            .with_system_prompt("You are a helpful test assistant.")
            .with_stages(stages)
            .with_extensions([object()])  # Mock extension
            .build()
        )

        # Verify the vertical
        assert TestVertical.name == "test_vertical"
        assert TestVertical.description == "Test assistant"
        assert TestVertical.version == "2.0.0"
        assert TestVertical.get_tools() == ["read", "write", "search"]
        assert "You are a helpful test assistant" in TestVertical.get_system_prompt()

        # Verify stages were injected
        vertical_stages = TestVertical.get_stages()
        assert "planning" in vertical_stages
        assert "execution" in vertical_stages

    def test_multiple_verticals_from_same_base(self):
        """Test creating multiple verticals from same base."""
        Vertical1 = (
            VerticalBase.compose()
            .with_metadata("vertical1", "First vertical")
            .with_tools(["read"])
            .with_system_prompt("You are vertical 1.")
            .build()
        )

        Vertical2 = (
            VerticalBase.compose()
            .with_metadata("vertical2", "Second vertical")
            .with_tools(["write"])
            .with_system_prompt("You are vertical 2.")
            .build()
        )

        # Verticals should be independent
        assert Vertical1.name == "vertical1"
        assert Vertical2.name == "vertical2"
        assert Vertical1.get_tools() == ["read"]
        assert Vertical2.get_tools() == ["write"]
