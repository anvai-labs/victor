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

"""Unit tests for VerticalMetadata module."""

from __future__ import annotations

import pytest

from victor.core.verticals.vertical_metadata import (
    VerticalMetadata,
    VerticalNamingPattern,
)


# Test helper classes
class MockVerticalBase:
    """Mock base class for testing."""

    pass


class CodingAssistant(MockVerticalBase):
    """Vertical with Assistant suffix."""

    name = "coding"
    version = "1.0.0"


class DevOpsVertical(MockVerticalBase):
    """Vertical with Vertical suffix."""

    name = "devops"
    version = "1.2.0"


class ResearchAssistant(MockVerticalBase):
    """Vertical with Assistant suffix but no explicit name."""

    version = "0.5.7"


class CustomNamedVertical(MockVerticalBase):
    """Vertical with custom name not following conventions."""

    name = "custom_security"
    version = "2.0.0"


class TestVerticalMetadata:
    """Test suite for VerticalMetadata class."""

    def test_extract_name_from_classname_assistant_suffix(self):
        """Test name extraction from class names with 'Assistant' suffix."""
        assert VerticalMetadata._extract_name_from_classname("CodingAssistant") == "coding"
        assert VerticalMetadata._extract_name_from_classname("DevOpsAssistant") == "devops"
        assert VerticalMetadata._extract_name_from_classname("ResearchAssistant") == "research"

    def test_extract_name_from_classname_vertical_suffix(self):
        """Test name extraction from class names with 'Vertical' suffix."""
        assert VerticalMetadata._extract_name_from_classname("CodingVertical") == "coding"
        assert VerticalMetadata._extract_name_from_classname("DevOpsVertical") == "devops"
        assert VerticalMetadata._extract_name_from_classname("RAGVertical") == "rag"

    def test_extract_name_from_classname_no_suffix_emits_warning(self, caplog):
        """Test that classes without recognized suffixes log a warning."""
        result = VerticalMetadata._extract_name_from_classname("CustomClass")
        assert result == "customclass"
        assert any("does not follow the recommended" in r.message for r in caplog.records)

    def test_normalize_name(self):
        """Test name normalization."""
        assert VerticalMetadata._normalize_name("Coding") == "coding"
        assert VerticalMetadata._normalize_name("DevOps") == "devops"
        assert VerticalMetadata._normalize_name("AI_ML") == "ai_ml"
        assert VerticalMetadata._normalize_name("data-analysis") == "data_analysis"
        assert VerticalMetadata._normalize_name("machine learning") == "machine_learning"

    def test_make_display_name(self):
        """Test display name generation."""
        assert VerticalMetadata._make_display_name("coding") == "Coding"
        assert VerticalMetadata._make_display_name("devops") == "Devops"
        assert VerticalMetadata._make_display_name("ai_ml") == "Ai Ml"
        assert VerticalMetadata._make_display_name("data_analysis") == "Data Analysis"

    def test_from_class_with_explicit_name(self):
        """Test metadata extraction when vertical has explicit name attribute."""
        metadata = VerticalMetadata.from_class(CodingAssistant)

        assert metadata.name == "coding"
        assert metadata.canonical_name == "coding"
        assert metadata.display_name == "Coding"
        assert metadata.version == "1.0.0"
        assert metadata.api_version == 1
        assert metadata.qualname == "CodingAssistant"

    def test_from_class_without_explicit_name(self):
        """Test metadata extraction when vertical lacks explicit name."""
        metadata = VerticalMetadata.from_class(ResearchAssistant)

        # Should extract name from class name
        assert metadata.name == "research"
        assert metadata.canonical_name == "research"
        assert metadata.display_name == "Research"
        assert metadata.version == "0.5.7"

    def test_from_class_with_vertical_suffix(self):
        """Test metadata extraction for classes with 'Vertical' suffix."""
        metadata = VerticalMetadata.from_class(DevOpsVertical)

        assert metadata.name == "devops"
        assert metadata.canonical_name == "devops"
        assert metadata.display_name == "Devops"
        assert metadata.version == "1.2.0"

    def test_from_class_with_custom_name(self):
        """Test metadata extraction for verticals with custom names."""
        metadata = VerticalMetadata.from_class(CustomNamedVertical)

        assert metadata.name == "custom_security"
        assert metadata.canonical_name == "custom_security"
        assert metadata.display_name == "Custom Security"
        assert metadata.version == "2.0.0"

    def test_from_class_with_manifest(self):
        """Test metadata extraction when class has _victor_manifest."""
        # Create a mock manifest as a dict (what registration.py creates)
        mock_manifest = {
            "name": "test_vertical",
            "version": "1.5.0",
            "api_version": 2,
            "min_framework_version": ">=0.6.0",
        }

        # Attach manifest to class
        class TestVertical(MockVerticalBase):
            pass

        TestVertical._victor_manifest = mock_manifest

        # Since manifest is a dict, from_class should fall back to name attribute
        # or pattern extraction
        TestVertical.name = "test_vertical"
        TestVertical.version = "1.5.0"
        TestVertical.VERTICAL_API_VERSION = 2

        metadata = VerticalMetadata.from_class(TestVertical)

        assert metadata.name == "test_vertical"
        assert metadata.version == "1.5.0"
        assert metadata.api_version == 2

    def test_is_contrib_detection(self):
        """Test detection of contrib verticals."""

        # Create a mock contrib vertical
        class ContribVertical(MockVerticalBase):
            pass

        ContribVertical.__module__ = "victor.verticals.contrib.contrib"

        metadata = VerticalMetadata.from_class(ContribVertical)
        assert metadata.is_contrib is True
        assert metadata.is_external is False

    def test_is_external_detection(self):
        """Test detection of external verticals."""

        # Create a mock external vertical
        class ExternalVertical(MockVerticalBase):
            pass

        ExternalVertical.__module__ = "victor_external.vertical"

        metadata = VerticalMetadata.from_class(ExternalVertical)
        assert metadata.is_contrib is False
        assert metadata.is_external is True

    def test_module_path_tracking(self):
        """Test that module path is correctly tracked."""
        metadata = VerticalMetadata.from_class(CodingAssistant)

        assert metadata.module_path == CodingAssistant.__module__
        assert "coding" in metadata.module_path or "test" in metadata.module_path

    def test_string_representation(self):
        """Test __str__ method."""
        metadata = VerticalMetadata.from_class(CodingAssistant)

        str_repr = str(metadata)
        assert "name='coding'" in str_repr
        assert "version='1.0.0'" in str_repr
        assert "api_version=1" in str_repr

    def test_detailed_representation(self):
        """Test __repr__ method with full details."""
        metadata = VerticalMetadata.from_class(CodingAssistant)

        repr_str = repr(metadata)
        assert "VerticalMetadata(" in repr_str
        assert "name='coding'" in repr_str
        assert "canonical_name='coding'" in repr_str
        assert "display_name='Coding'" in repr_str
        assert "module_path=" in repr_str

    def test_frozen_dataclass(self):
        """Test that VerticalMetadata is frozen (immutable)."""
        metadata = VerticalMetadata.from_class(CodingAssistant)

        with pytest.raises(Exception):  # FrozenInstanceError or similar
            metadata.name = "new_name"

    def test_multiple_suffixes_priority(self):
        """Test that Assistant suffix has priority over Vertical suffix."""

        # This is unlikely in practice but tests the priority logic
        # Assistant suffix is checked first, so "AssistantVertical" -> "assistant"
        class AssistantVertical(MockVerticalBase):
            """Has both suffixes."""

            pass

        # Should detect "Assistant" first (priority order in code)
        result = VerticalMetadata._extract_name_from_classname("AssistantVertical")
        assert result == "assistant"


class TestVerticalNamingPattern:
    """Test suite for VerticalNamingPattern enum."""

    def test_pattern_values(self):
        """Test that all expected patterns are defined."""
        assert VerticalNamingPattern.ASSISTANT_SUFFIX.value == "Assistant"
        assert VerticalNamingPattern.VERTICAL_SUFFIX.value == "Vertical"
        assert VerticalNamingPattern.EXPLICIT_NAME.value == "name"

    def test_pattern_iteration(self):
        """Test that patterns can be iterated."""
        patterns = list(VerticalNamingPattern)
        assert len(patterns) == 3
        assert VerticalNamingPattern.ASSISTANT_SUFFIX in patterns
        assert VerticalNamingPattern.VERTICAL_SUFFIX in patterns
        assert VerticalNamingPattern.EXPLICIT_NAME in patterns


class TestVerticalMetadataIntegration:
    """Integration tests for VerticalMetadata with realistic scenarios."""

    def test_all_built_in_verticals_would_extract(self):
        """Test that all known Victor verticals would extract correctly."""
        test_cases = [
            ("CodingAssistant", "coding"),
            ("DevOpsAssistant", "devops"),
            ("ResearchAssistant", "research"),
            ("RAGAssistant", "rag"),
            ("DataAnalysisAssistant", "dataanalysis"),
            ("CodingVertical", "coding"),
            ("DevOpsVertical", "devops"),
        ]

        for classname, expected_name in test_cases:
            result = VerticalMetadata._extract_name_from_classname(classname)
            assert result == expected_name, f"Failed for {classname}"

    def test_metadata_preserves_all_attributes(self):
        """Test that metadata preserves all important class attributes."""

        class CompleteVertical(MockVerticalBase):
            name = "complete"
            version = "2.3.4"
            VERTICAL_API_VERSION = 2

        metadata = VerticalMetadata.from_class(CompleteVertical)

        assert metadata.name == "complete"
        assert metadata.version == "2.3.4"
        assert metadata.api_version == 2
        # qualname includes test class context
        assert "CompleteVertical" in metadata.qualname

    def test_metadata_with_module_and_qualname(self):
        """Test that metadata correctly captures module and qualname."""

        class OuterClass:
            class InnerVertical(MockVerticalBase):
                name = "inner"

        metadata = VerticalMetadata.from_class(OuterClass.InnerVertical)

        # Python's qualname for nested classes uses dot notation, not <locals>
        assert "OuterClass" in metadata.qualname
        assert "InnerVertical" in metadata.qualname


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
