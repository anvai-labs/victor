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

"""Unit tests for VersionCompatibilityMatrix module."""

from __future__ import annotations

import json
import pytest

from victor.core.verticals.version_matrix import (
    CompatibilityResult,
    CompatibilityRule,
    CompatibilityStatus,
    VersionCompatibilityMatrix,
    check_vertical_compatibility,
    get_compatibility_matrix,
)


class TestCompatibilityRule:
    """Test suite for CompatibilityRule dataclass."""

    def test_create_rule_minimal(self):
        """Test creating a rule with minimal fields."""
        rule = CompatibilityRule(vertical_name="test")

        assert rule.vertical_name == "test"
        assert rule.min_framework_version == ">=0.1.0"
        assert rule.max_framework_version is None
        assert rule.excluded_versions == set()
        assert rule.requires_features == set()
        assert rule.status == CompatibilityStatus.COMPATIBLE

    def test_create_rule_full(self):
        """Test creating a rule with all fields."""
        rule = CompatibilityRule(
            vertical_name="test",
            min_framework_version=">=0.5.0",
            max_framework_version="<1.0.0",
            excluded_versions={"0.5.1", "0.5.2"},
            requires_features={"async_tools", "langchain"},
            status=CompatibilityStatus.INCOMPATIBLE,
            message="Test rule",
        )

        assert rule.vertical_name == "test"
        assert rule.min_framework_version == ">=0.5.0"
        assert rule.max_framework_version == "<1.0.0"
        assert rule.excluded_versions == {"0.5.1", "0.5.2"}
        assert rule.requires_features == {"async_tools", "langchain"}
        assert rule.status == CompatibilityStatus.INCOMPATIBLE
        assert rule.message == "Test rule"

    def test_invalid_version_constraint(self):
        """Test that invalid version constraint raises ValueError."""
        with pytest.raises(ValueError, match="Invalid version constraint"):
            CompatibilityRule(vertical_name="test", min_framework_version=">=invalid")

    def test_invalid_excluded_version(self):
        """Test that invalid excluded version raises ValueError."""
        with pytest.raises(ValueError, match="Invalid version constraint"):
            CompatibilityRule(vertical_name="test", excluded_versions={"invalid"})


class TestCompatibilityResult:
    """Test suite for CompatibilityResult dataclass."""

    def test_create_result(self):
        """Test creating a compatibility result."""
        result = CompatibilityResult(
            vertical_name="test",
            vertical_version="1.0.0",
            framework_version="0.6.0",
            status=CompatibilityStatus.COMPATIBLE,
            message="Compatible",
        )

        assert result.vertical_name == "test"
        assert result.vertical_version == "1.0.0"
        assert result.framework_version == "0.6.0"
        assert result.status == CompatibilityStatus.COMPATIBLE
        assert result.message == "Compatible"
        assert result.required_features == set()

    def test_is_compatible_property(self):
        """Test is_compatible property."""
        compatible_result = CompatibilityResult(
            vertical_name="test",
            vertical_version="1.0.0",
            framework_version="0.6.0",
            status=CompatibilityStatus.COMPATIBLE,
        )
        assert compatible_result.is_compatible is True

        degraded_result = CompatibilityResult(
            vertical_name="test",
            vertical_version="1.0.0",
            framework_version="0.6.0",
            status=CompatibilityStatus.DEGRADED,
        )
        assert degraded_result.is_compatible is True

        incompatible_result = CompatibilityResult(
            vertical_name="test",
            vertical_version="1.0.0",
            framework_version="0.6.0",
            status=CompatibilityStatus.INCOMPATIBLE,
        )
        assert incompatible_result.is_compatible is False

        unknown_result = CompatibilityResult(
            vertical_name="test",
            vertical_version="1.0.0",
            framework_version="0.6.0",
            status=CompatibilityStatus.UNKNOWN,
        )
        assert unknown_result.is_compatible is False

    def test_is_incompatible_property(self):
        """Test is_incompatible property."""
        incompatible_result = CompatibilityResult(
            vertical_name="test",
            vertical_version="1.0.0",
            framework_version="0.6.0",
            status=CompatibilityStatus.INCOMPATIBLE,
        )
        assert incompatible_result.is_incompatible is True

        compatible_result = CompatibilityResult(
            vertical_name="test",
            vertical_version="1.0.0",
            framework_version="0.6.0",
            status=CompatibilityStatus.COMPATIBLE,
        )
        assert compatible_result.is_incompatible is False


class TestVersionCompatibilityMatrix:
    """Test suite for VersionCompatibilityMatrix class."""

    def setup_method(self):
        """Reset matrix before each test."""
        matrix = VersionCompatibilityMatrix.get_instance()
        matrix._rules.clear()
        matrix._loaded = False

    def teardown_method(self):
        """Clean up matrix after each test."""
        matrix = VersionCompatibilityMatrix.get_instance()
        matrix._rules.clear()
        matrix._loaded = False

    def test_singleton_instance(self):
        """Test that matrix returns singleton instance."""
        matrix1 = VersionCompatibilityMatrix.get_instance()
        matrix2 = VersionCompatibilityMatrix.get_instance()

        assert matrix1 is matrix2
        assert isinstance(matrix1, VersionCompatibilityMatrix)

    def test_load_default_rules(self):
        """Test loading default compatibility rules."""
        matrix = VersionCompatibilityMatrix.get_instance()
        matrix.load_default_rules()

        assert matrix.is_loaded()
        assert len(matrix.list_rules()) > 0

        # Check for known verticals
        assert "coding" in matrix.list_rules()
        assert "devops" in matrix.list_rules()

    def test_register_rule(self):
        """Test registering a compatibility rule."""
        matrix = VersionCompatibilityMatrix.get_instance()
        rule = CompatibilityRule(
            vertical_name="test_vertical",
            min_framework_version=">=0.6.0",
        )

        matrix.register_rule(rule)

        assert "test_vertical" in matrix.list_rules()
        retrieved_rule = matrix.get_rule("test_vertical")
        assert retrieved_rule.vertical_name == "test_vertical"
        assert retrieved_rule.min_framework_version == ">=0.6.0"

    def test_unregister_rule(self):
        """Test unregistering a compatibility rule."""
        matrix = VersionCompatibilityMatrix.get_instance()
        rule = CompatibilityRule(vertical_name="test_vertical")

        matrix.register_rule(rule)
        assert "test_vertical" in matrix.list_rules()

        matrix.unregister_rule("test_vertical")
        assert "test_vertical" not in matrix.list_rules()

    def test_check_compatibility_no_rule(self):
        """Test compatibility check with no rule returns UNKNOWN."""
        matrix = VersionCompatibilityMatrix.get_instance()

        result = matrix.check_compatibility("unknown", "1.0.0", "0.6.0")

        assert result.status == CompatibilityStatus.UNKNOWN
        assert "No compatibility rule found" in result.message

    def test_check_compatible(self):
        """Test compatible vertical."""
        matrix = VersionCompatibilityMatrix.get_instance()
        rule = CompatibilityRule(
            vertical_name="test",
            min_framework_version=">=0.5.0",
            status=CompatibilityStatus.COMPATIBLE,
            message="Fully supported",
        )
        matrix.register_rule(rule)

        result = matrix.check_compatibility("test", "1.0.0", "0.6.0")

        assert result.is_compatible
        assert result.status == CompatibilityStatus.COMPATIBLE
        assert result.message == "Fully supported"

    def test_check_incompatible_min_version(self):
        """Test incompatible vertical due to minimum version."""
        matrix = VersionCompatibilityMatrix.get_instance()
        rule = CompatibilityRule(
            vertical_name="test",
            min_framework_version=">=1.0.0",
            status=CompatibilityStatus.COMPATIBLE,
        )
        matrix.register_rule(rule)

        result = matrix.check_compatibility("test", "1.0.0", "0.6.0")

        assert result.is_incompatible
        assert result.status == CompatibilityStatus.INCOMPATIBLE
        assert "requires framework >=1.0.0" in result.message

    def test_check_incompatible_max_version(self):
        """Test incompatible vertical due to maximum version."""
        matrix = VersionCompatibilityMatrix.get_instance()
        rule = CompatibilityRule(
            vertical_name="test",
            min_framework_version=">=0.5.0",
            max_framework_version="<0.6.0",
            status=CompatibilityStatus.COMPATIBLE,
        )
        matrix.register_rule(rule)

        result = matrix.check_compatibility("test", "1.0.0", "0.6.0")

        assert result.is_incompatible
        assert result.status == CompatibilityStatus.INCOMPATIBLE
        assert "requires framework <0.6.0" in result.message

    def test_check_excluded_version(self):
        """Test excluded version is incompatible."""
        matrix = VersionCompatibilityMatrix.get_instance()
        rule = CompatibilityRule(
            vertical_name="test",
            min_framework_version=">=0.5.0",
            excluded_versions={"0.6.0"},
            status=CompatibilityStatus.COMPATIBLE,
        )
        matrix.register_rule(rule)

        result = matrix.check_compatibility("test", "1.0.0", "0.6.0")

        assert result.is_incompatible
        assert result.status == CompatibilityStatus.INCOMPATIBLE
        assert "is excluded" in result.message

    def test_check_degraded_missing_features(self):
        """Test degraded status when features are missing."""
        matrix = VersionCompatibilityMatrix.get_instance()
        rule = CompatibilityRule(
            vertical_name="test",
            min_framework_version=">=0.5.0",
            requires_features={"async_tools", "langchain"},
            status=CompatibilityStatus.COMPATIBLE,
            message="Fully supported",
        )
        matrix.register_rule(rule)

        result = matrix.check_compatibility(
            "test", "1.0.0", "0.6.0", available_features={"async_tools"}
        )

        assert result.is_compatible
        assert result.status == CompatibilityStatus.DEGRADED
        assert "degraded" in result.message.lower()
        assert result.required_features == {"langchain"}

    def test_list_rules(self):
        """Test listing all rules."""
        matrix = VersionCompatibilityMatrix.get_instance()
        matrix.register_rule(CompatibilityRule(vertical_name="test1"))
        matrix.register_rule(CompatibilityRule(vertical_name="test2"))
        matrix.register_rule(CompatibilityRule(vertical_name="test3"))

        rules = matrix.list_rules()

        assert "test1" in rules
        assert "test2" in rules
        assert "test3" in rules
        assert len(rules) == 3

    def test_is_loaded(self):
        """Test is_loaded flag."""
        matrix = VersionCompatibilityMatrix.get_instance()

        assert not matrix.is_loaded()

        matrix.load_default_rules()

        assert matrix.is_loaded()


class TestLoadFromFile:
    """Test suite for loading rules from external JSON file."""

    def setup_method(self):
        """Reset matrix before each test."""
        matrix = VersionCompatibilityMatrix.get_instance()
        matrix._rules.clear()
        matrix._loaded = False

    def teardown_method(self):
        """Clean up matrix after each test."""
        matrix = VersionCompatibilityMatrix.get_instance()
        matrix._rules.clear()
        matrix._loaded = False

    def test_load_from_file_success(self, tmp_path):
        """Test loading rules from JSON file."""
        # Create a temporary JSON file
        rules_data = {
            "rules": [
                {
                    "vertical_name": "test1",
                    "min_framework_version": ">=0.5.0",
                    "status": "compatible",
                    "message": "Test vertical 1",
                },
                {
                    "vertical_name": "test2",
                    "min_framework_version": ">=0.6.0",
                    "excluded_versions": ["0.6.1"],
                    "requires_features": ["async_tools"],
                    "status": "compatible",
                    "message": "Test vertical 2",
                },
            ]
        }

        json_file = tmp_path / "compatibility_matrix.json"
        with open(json_file, "w") as f:
            json.dump(rules_data, f)

        matrix = VersionCompatibilityMatrix.get_instance()
        matrix.load_from_file(json_file)

        assert matrix.is_loaded()
        assert "test1" in matrix.list_rules()
        assert "test2" in matrix.list_rules()

        # Check rule details
        rule1 = matrix.get_rule("test1")
        assert rule1.min_framework_version == ">=0.5.0"
        assert rule1.status == CompatibilityStatus.COMPATIBLE

        rule2 = matrix.get_rule("test2")
        assert rule2.min_framework_version == ">=0.6.0"
        assert "0.6.1" in rule2.excluded_versions
        assert "async_tools" in rule2.requires_features

    def test_load_from_file_not_found(self, tmp_path):
        """Test loading from non-existent file raises FileNotFoundError."""
        matrix = VersionCompatibilityMatrix.get_instance()

        with pytest.raises(FileNotFoundError):
            matrix.load_from_file(tmp_path / "nonexistent.json")

    def test_load_from_file_invalid_json(self, tmp_path):
        """Test loading from invalid JSON raises ValueError."""
        json_file = tmp_path / "invalid.json"
        with open(json_file, "w") as f:
            f.write("{ invalid json }")

        matrix = VersionCompatibilityMatrix.get_instance()

        with pytest.raises(ValueError, match="Invalid JSON"):
            matrix.load_from_file(json_file)

    def test_load_from_file_invalid_rule(self, tmp_path):
        """Test loading file with invalid rule skips the rule."""
        rules_data = {
            "rules": [
                {
                    "vertical_name": "valid",
                    "min_framework_version": ">=0.5.0",
                    "status": "compatible",
                    "message": "Valid rule",
                },
                {
                    "vertical_name": "invalid",
                    "min_framework_version": ">=invalid_version",
                    "status": "compatible",
                },
            ]
        }

        json_file = tmp_path / "compatibility_matrix.json"
        with open(json_file, "w") as f:
            json.dump(rules_data, f)

        matrix = VersionCompatibilityMatrix.get_instance()
        matrix.load_from_file(json_file)

        # Valid rule should be loaded
        assert "valid" in matrix.list_rules()

        # Invalid rule should be skipped
        assert "invalid" not in matrix.list_rules()


class TestConvenienceFunctions:
    """Test suite for convenience functions."""

    def setup_method(self):
        """Reset matrix before each test."""
        matrix = VersionCompatibilityMatrix.get_instance()
        matrix._rules.clear()
        matrix._loaded = False

    def teardown_method(self):
        """Clean up matrix after each test."""
        matrix = VersionCompatibilityMatrix.get_instance()
        matrix._rules.clear()
        matrix._loaded = False

    def test_get_compatibility_matrix(self):
        """Test get_compatibility_matrix convenience function."""
        matrix = get_compatibility_matrix()
        assert isinstance(matrix, VersionCompatibilityMatrix)

    def test_check_vertical_compatibility(self):
        """Test check_vertical_compatibility convenience function."""
        matrix = get_compatibility_matrix()
        matrix.load_default_rules()

        result = check_vertical_compatibility("coding", "2.0.0", "0.6.0")

        assert isinstance(result, CompatibilityResult)
        assert result.vertical_name == "coding"
        assert result.vertical_version == "2.0.0"
        assert result.framework_version == "0.6.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
