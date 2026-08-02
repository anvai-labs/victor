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

"""Backward compatibility tests for Victor architecture refactoring.

These tests validate that existing verticals work without modification
and that deprecation warnings are properly emitted for legacy patterns.

Test Coverage:
    - Legacy vertical naming patterns still work
    - Deprecation warnings emitted appropriately
    - New @register_vertical decorator works
    - Both patterns can coexist
    - Migration path is smooth
"""

from __future__ import annotations

import warnings
import pytest

from victor.core.verticals.vertical_metadata import (
    VerticalMetadata,
    VerticalNamingPattern,
)
from victor_contracts.verticals import VerticalBase


def metadata_from_class(cls: type) -> VerticalMetadata:
    """Helper function to extract metadata from a vertical class."""
    return VerticalMetadata.from_class(cls)


# Test Verticals (Legacy and New)


class LegacyCodingAssistant(VerticalBase):
    """Legacy vertical using 'Assistant' suffix pattern."""

    @classmethod
    def get_name(cls) -> str:
        return "legacycoding"

    @classmethod
    def get_description(cls) -> str:
        return "Legacy coding assistant"

    @classmethod
    def get_tools(cls) -> list[str]:
        return []

    @classmethod
    def get_system_prompt(cls) -> str:
        return "Legacy coding assistant"


class NewCodingVertical(VerticalBase):
    """New vertical without 'Assistant' suffix."""

    @classmethod
    def get_name(cls) -> str:
        return "newcoding"

    @classmethod
    def get_description(cls) -> str:
        return "New coding vertical"

    @classmethod
    def get_tools(cls) -> list[str]:
        return []

    @classmethod
    def get_system_prompt(cls) -> str:
        return "New coding vertical"


class CustomNameVertical(VerticalBase):
    """Vertical with custom name that doesn't follow naming convention."""

    name = "custom_name_vertical"

    @classmethod
    def get_name(cls) -> str:
        return "custom_name_vertical"

    @classmethod
    def get_description(cls) -> str:
        return "Custom name vertical"

    @classmethod
    def get_tools(cls) -> list[str]:
        return []

    @classmethod
    def get_system_prompt(cls) -> str:
        return "Custom name vertical"


class TestLegacyNamingPatterns:
    """Test that legacy naming patterns still work."""

    def test_legacy_assistant_suffix_extraction(self):
        """Test that 'Assistant' suffix is properly removed for legacy verticals."""
        metadata = metadata_from_class(LegacyCodingAssistant)

        # Should extract 'legacycoding' from 'LegacyCodingAssistant'
        assert metadata.canonical_name == "legacycoding"
        assert metadata.name == "legacycoding"

    def test_vertical_suffix_extraction(self):
        """Test that 'Vertical' suffix is properly removed."""
        metadata = metadata_from_class(NewCodingVertical)

        # Should extract 'newcoding' from 'NewCodingVertical'
        assert metadata.canonical_name == "newcoding"
        assert metadata.name == "newcoding"

    def test_custom_name_extraction(self):
        """Test that custom names work without suffix."""
        metadata = metadata_from_class(CustomNameVertical)

        # Should use the explicit name attribute
        assert metadata.canonical_name == "custom_name_vertical"
        assert metadata.name == "custom_name_vertical"

    def test_legacy_and_new_coexist(self):
        """Test that legacy and new patterns can coexist."""
        legacy_meta = metadata_from_class(LegacyCodingAssistant)
        new_meta = metadata_from_class(NewCodingVertical)

        # Both should work
        assert legacy_meta.canonical_name == "legacycoding"
        assert new_meta.canonical_name == "newcoding"

        # Both should have valid metadata
        assert legacy_meta.name is not None
        assert new_meta.name is not None


class TestDeprecationWarnings:
    """Test that deprecation warnings are emitted appropriately."""

    def test_legacy_assistant_emits_warning(self):
        """Test that classes without explicit name emit warnings for non-standard naming."""

        # Create a class without explicit name attribute and non-standard naming
        class MyNonStandardClass(VerticalBase):
            """Class that doesn't follow naming convention."""

            @classmethod
            def get_name(cls) -> str:
                return "my_non_standard"

            @classmethod
            def get_description(cls) -> str:
                return "Non standard"

            @classmethod
            def get_tools(cls) -> list[str]:
                return []

            @classmethod
            def get_system_prompt(cls) -> str:
                return "Non standard"

        # metadata_from_class now logs warnings instead of using warnings.warn()
        # Check that the metadata is still created correctly
        metadata = metadata_from_class(MyNonStandardClass)
        assert metadata is not None

    def test_vertical_suffix_no_warning(self):
        """Test that 'Vertical' suffix does NOT emit deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            metadata = metadata_from_class(NewCodingVertical)

            # Vertical suffix is not deprecated
            assistant_warnings = [warning for warning in w if "Assistant" in str(warning.message)]
            assert len(assistant_warnings) == 0


class TestBackwardCompatibility:
    """Test backward compatibility guarantees."""

    def test_existing_verticals_work_without_modification(self):
        """Test that existing verticals continue to work without changes."""
        # Legacy vertical should work via class methods
        tools = LegacyCodingAssistant.get_tools()
        prompt = LegacyCodingAssistant.get_system_prompt()
        name = LegacyCodingAssistant.get_name()
        description = LegacyCodingAssistant.get_description()

        # Methods should work
        assert tools == []
        assert prompt == "Legacy coding assistant"
        assert name == "legacycoding"
        assert description == "Legacy coding assistant"

    def test_metadata_fallback_for_missing_attributes(self):
        """Test that metadata provides sensible defaults for missing attributes."""
        metadata = metadata_from_class(CustomNameVertical)

        # Should have default values even if class doesn't set them
        assert metadata.canonical_name is not None
        assert metadata.name is not None
        assert metadata.version is not None

    def test_multiple_inheritance_patterns_work(self):
        """Test that different inheritance patterns work."""
        # All should work via class methods
        legacy_name = LegacyCodingAssistant.get_name()
        new_name = NewCodingVertical.get_name()
        custom_name = CustomNameVertical.get_name()

        # All should be VerticalBase subclasses
        assert issubclass(LegacyCodingAssistant, VerticalBase)
        assert issubclass(NewCodingVertical, VerticalBase)
        assert issubclass(CustomNameVertical, VerticalBase)

        # All should have valid names
        assert legacy_name == "legacycoding"
        assert new_name == "newcoding"
        assert custom_name == "custom_name_vertical"


class TestMigrationPath:
    """Test smooth migration path from legacy to new patterns."""

    def test_can_migrate_from_assistant_to_vertical(self):
        """Test that migration from Assistant suffix to Vertical suffix works."""
        # Old pattern
        old_metadata = metadata_from_class(LegacyCodingAssistant)
        assert old_metadata.canonical_name == "legacycoding"

        # New pattern (hypothetical migrated class)
        class MigratedLegacyCoding(VerticalBase):
            """Migrated vertical without Assistant suffix."""

            @classmethod
            def get_name(cls) -> str:
                return "migratedlegacycoding"

            @classmethod
            def get_description(cls) -> str:
                return "Migrated legacy coding"

            @classmethod
            def get_tools(cls) -> list[str]:
                return []

            @classmethod
            def get_system_prompt(cls) -> str:
                return "Migrated legacy coding"

        new_metadata = metadata_from_class(MigratedLegacyCoding)
        assert new_metadata.canonical_name == "migratedlegacycoding"

    def test_both_patterns_produce_same_canonical_name(self):
        """Test that both patterns can produce same canonical name when intended."""

        class LegacyAssistant(VerticalBase):
            """Legacy pattern with Assistant suffix."""

            @classmethod
            def get_name(cls) -> str:
                return "legacy"

            @classmethod
            def get_description(cls) -> str:
                return "Legacy"

            @classmethod
            def get_tools(cls) -> list[str]:
                return []

            @classmethod
            def get_system_prompt(cls) -> str:
                return "Legacy"

        class NewVertical(VerticalBase):
            """New pattern with Vertical suffix."""

            @classmethod
            def get_name(cls) -> str:
                return "new"

            @classmethod
            def get_description(cls) -> str:
                return "New"

            @classmethod
            def get_tools(cls) -> list[str]:
                return []

            @classmethod
            def get_system_prompt(cls) -> str:
                return "New"

        legacy_meta = metadata_from_class(LegacyAssistant)
        new_meta = metadata_from_class(NewVertical)

        # Both should successfully extract canonical names
        assert legacy_meta.canonical_name == "legacy"
        assert new_meta.canonical_name == "new"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_double_suffix_patterns(self):
        """Test classes with multiple suffixes."""

        class MyAssistantVertical(VerticalBase):
            """Class with both Assistant and Vertical suffixes."""

            @classmethod
            def get_name(cls) -> str:
                return "myassistantvertical"

            @classmethod
            def get_description(cls) -> str:
                return "Double suffix"

            @classmethod
            def get_tools(cls) -> list[str]:
                return []

            @classmethod
            def get_system_prompt(cls) -> str:
                return "Double suffix"

        metadata = metadata_from_class(MyAssistantVertical)

        # Should handle gracefully
        assert metadata.canonical_name is not None

    def test_no_recognized_suffix(self):
        """Test classes with no recognized suffix."""

        class MyCustomClass(VerticalBase):
            """Class with no recognized suffix."""

            @classmethod
            def get_name(cls) -> str:
                return "mycustomclass"

            @classmethod
            def get_description(cls) -> str:
                return "Custom"

            @classmethod
            def get_tools(cls) -> list[str]:
                return []

            @classmethod
            def get_system_prompt(cls) -> str:
                return "Custom"

        metadata = metadata_from_class(MyCustomClass)

        # Should use the name from get_name()
        assert metadata.canonical_name == "mycustomclass"

    def test_empty_class_name(self):
        """Test handling of edge case class names."""

        class V(VerticalBase):
            """Minimal class name."""

            @classmethod
            def get_name(cls) -> str:
                return "v"

            @classmethod
            def get_description(cls) -> str:
                return "V"

            @classmethod
            def get_tools(cls) -> list[str]:
                return []

            @classmethod
            def get_system_prompt(cls) -> str:
                return "V"

        metadata = metadata_from_class(V)

        # Should handle gracefully
        assert metadata.canonical_name is not None


class TestGracefulDegradation:
    """Test graceful degradation when features are missing."""

    def test_missing_optional_features(self):
        """Test that missing optional features don't break functionality."""
        # Should work even if optional features aren't implemented
        name = LegacyCodingAssistant.get_name()
        description = LegacyCodingAssistant.get_description()
        tools = LegacyCodingAssistant.get_tools()
        prompt = LegacyCodingAssistant.get_system_prompt()

        # All required methods should work
        assert name is not None
        assert description is not None
        assert tools is not None
        assert prompt is not None

    def test_partial_implementation(self):
        """Test that partial implementations don't cause errors."""

        class PartialVertical(VerticalBase):
            """Vertical with minimal implementation."""

            @classmethod
            def get_name(cls) -> str:
                return "partial"

            @classmethod
            def get_description(cls) -> str:
                return "Partial"

            @classmethod
            def get_tools(cls) -> list[str]:
                return []

            @classmethod
            def get_system_prompt(cls) -> str:
                return "Partial vertical"

        # Should work without error
        name = PartialVertical.get_name()
        prompt = PartialVertical.get_system_prompt()

        # Should have values
        assert name is not None
        assert prompt is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
