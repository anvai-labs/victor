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

"""Tests for victor.core.tool_dependency_schema module."""

import pytest
from pydantic import ValidationError

from victor.core.tool_dependency_schema import (
    ToolCluster,
    ToolDependencyEntry,
    ToolDependencySpec,
    ToolSequence,
    ToolTransition,
)


class TestToolTransition:
    """Tests for ToolTransition model."""

    def test_valid_transition_creation(self):
        """ToolTransition should accept valid tool name and weight."""
        transition = ToolTransition(tool="edit", weight=0.4)
        assert transition.tool == "edit"
        assert transition.weight == 0.4

    def test_transition_weight_at_lower_bound(self):
        """ToolTransition should accept weight of 0.0."""
        transition = ToolTransition(tool="read", weight=0.0)
        assert transition.weight == 0.0

    def test_transition_weight_at_upper_bound(self):
        """ToolTransition should accept weight of 1.0."""
        transition = ToolTransition(tool="read", weight=1.0)
        assert transition.weight == 1.0

    def test_transition_weight_below_lower_bound_raises(self):
        """ToolTransition should reject weight below 0.0."""
        with pytest.raises(ValidationError) as exc_info:
            ToolTransition(tool="edit", weight=-0.1)
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_transition_weight_above_upper_bound_raises(self):
        """ToolTransition should reject weight above 1.0."""
        with pytest.raises(ValidationError) as exc_info:
            ToolTransition(tool="edit", weight=1.1)
        assert "less than or equal to 1" in str(exc_info.value)

    def test_empty_tool_name_raises(self):
        """ToolTransition should reject empty tool name."""
        with pytest.raises(ValidationError) as exc_info:
            ToolTransition(tool="", weight=0.5)
        assert "Tool name cannot be empty" in str(exc_info.value)

    def test_whitespace_only_tool_name_raises(self):
        """ToolTransition should reject whitespace-only tool name."""
        with pytest.raises(ValidationError) as exc_info:
            ToolTransition(tool="   ", weight=0.5)
        assert "Tool name cannot be empty" in str(exc_info.value)

    def test_tool_name_is_stripped(self):
        """ToolTransition should strip whitespace from tool name."""
        transition = ToolTransition(tool="  edit  ", weight=0.4)
        assert transition.tool == "edit"

    def test_transition_missing_tool_raises(self):
        """ToolTransition should require tool field."""
        with pytest.raises(ValidationError) as exc_info:
            ToolTransition(weight=0.5)
        assert "tool" in str(exc_info.value)

    def test_transition_missing_weight_raises(self):
        """ToolTransition should require weight field."""
        with pytest.raises(ValidationError) as exc_info:
            ToolTransition(tool="edit")
        assert "weight" in str(exc_info.value)


class TestToolCluster:
    """Tests for ToolCluster model."""

    def test_valid_cluster_creation(self):
        """ToolCluster should accept valid name and tools list."""
        cluster = ToolCluster(name="file_operations", tools=["read", "write", "edit"])
        assert cluster.name == "file_operations"
        assert cluster.tools == ["read", "write", "edit"]

    def test_cluster_with_single_tool(self):
        """ToolCluster should accept a single tool."""
        cluster = ToolCluster(name="single", tools=["read"])
        assert len(cluster.tools) == 1
        assert cluster.tools[0] == "read"

    def test_empty_tools_list_raises(self):
        """ToolCluster should reject empty tools list."""
        with pytest.raises(ValidationError) as exc_info:
            ToolCluster(name="empty", tools=[])
        assert "Cluster must contain at least one tool" in str(exc_info.value)

    def test_cluster_strips_tool_names(self):
        """ToolCluster should strip whitespace from tool names."""
        cluster = ToolCluster(name="test", tools=["  read  ", " write ", "edit"])
        assert cluster.tools == ["read", "write", "edit"]

    def test_cluster_filters_empty_tool_names(self):
        """ToolCluster should filter out empty tool names."""
        cluster = ToolCluster(name="test", tools=["read", "", "write", "   ", "edit"])
        assert cluster.tools == ["read", "write", "edit"]

    def test_cluster_missing_name_raises(self):
        """ToolCluster should require name field."""
        with pytest.raises(ValidationError) as exc_info:
            ToolCluster(tools=["read", "write"])
        assert "name" in str(exc_info.value)

    def test_cluster_with_only_empty_tool_names_results_in_empty_list(self):
        """ToolCluster with only empty/whitespace tool names results in empty tools list.

        Note: The current validator filters empty names but does not raise an error
        after filtering. This test documents the actual behavior.
        """
        # The validator filters out empty strings, resulting in empty list
        cluster = ToolCluster(name="test", tools=["", "   ", ""])
        assert cluster.tools == []


class TestToolSequence:
    """Tests for ToolSequence model."""

    def test_valid_sequence_creation(self):
        """ToolSequence should accept valid name and tools list."""
        sequence = ToolSequence(name="exploration", tools=["ls", "read", "grep"])
        assert sequence.name == "exploration"
        assert sequence.tools == ["ls", "read", "grep"]

    def test_sequence_with_single_tool(self):
        """ToolSequence should accept a single tool."""
        sequence = ToolSequence(name="single", tools=["read"])
        assert len(sequence.tools) == 1

    def test_empty_sequence_raises(self):
        """ToolSequence should reject empty tools list."""
        with pytest.raises(ValidationError) as exc_info:
            ToolSequence(name="empty", tools=[])
        assert "Sequence must contain at least one tool" in str(exc_info.value)

    def test_sequence_strips_tool_names(self):
        """ToolSequence should strip whitespace from tool names."""
        sequence = ToolSequence(name="test", tools=["  ls  ", " read ", "grep"])
        assert sequence.tools == ["ls", "read", "grep"]

    def test_sequence_filters_empty_tool_names(self):
        """ToolSequence should filter out empty tool names."""
        sequence = ToolSequence(name="test", tools=["ls", "", "read", "   ", "grep"])
        assert sequence.tools == ["ls", "read", "grep"]

    def test_sequence_missing_name_raises(self):
        """ToolSequence should require name field."""
        with pytest.raises(ValidationError) as exc_info:
            ToolSequence(tools=["read", "edit"])
        assert "name" in str(exc_info.value)

    def test_sequence_preserves_order(self):
        """ToolSequence should preserve the order of tools."""
        sequence = ToolSequence(name="ordered", tools=["first", "second", "third"])
        assert sequence.tools == ["first", "second", "third"]


class TestToolDependencyEntry:
    """Tests for ToolDependencyEntry model."""

    def test_valid_entry_creation(self):
        """ToolDependencyEntry should accept valid configuration."""
        entry = ToolDependencyEntry(
            tool="edit",
            depends_on=["read"],
            enables=["test", "git"],
            weight=0.9,
        )
        assert entry.tool == "edit"
        assert entry.depends_on == ["read"]
        assert entry.enables == ["test", "git"]
        assert entry.weight == 0.9

    def test_entry_weight_default_value(self):
        """ToolDependencyEntry weight should default to 1.0."""
        entry = ToolDependencyEntry(tool="edit")
        assert entry.weight == 1.0

    def test_entry_depends_on_default_empty(self):
        """ToolDependencyEntry depends_on should default to empty list."""
        entry = ToolDependencyEntry(tool="edit")
        assert entry.depends_on == []

    def test_entry_enables_default_empty(self):
        """ToolDependencyEntry enables should default to empty list."""
        entry = ToolDependencyEntry(tool="edit")
        assert entry.enables == []

    def test_entry_with_only_depends_on(self):
        """ToolDependencyEntry should accept only depends_on list."""
        entry = ToolDependencyEntry(tool="edit", depends_on=["read", "analyze"])
        assert entry.depends_on == ["read", "analyze"]
        assert entry.enables == []

    def test_entry_with_only_enables(self):
        """ToolDependencyEntry should accept only enables list."""
        entry = ToolDependencyEntry(tool="edit", enables=["test", "commit"])
        assert entry.depends_on == []
        assert entry.enables == ["test", "commit"]

    def test_entry_empty_tool_name_raises(self):
        """ToolDependencyEntry should reject empty tool name."""
        with pytest.raises(ValidationError) as exc_info:
            ToolDependencyEntry(tool="")
        assert "Tool name cannot be empty" in str(exc_info.value)

    def test_entry_whitespace_tool_name_raises(self):
        """ToolDependencyEntry should reject whitespace-only tool name."""
        with pytest.raises(ValidationError) as exc_info:
            ToolDependencyEntry(tool="   ")
        assert "Tool name cannot be empty" in str(exc_info.value)

    def test_entry_tool_name_is_stripped(self):
        """ToolDependencyEntry should strip whitespace from tool name."""
        entry = ToolDependencyEntry(tool="  edit  ")
        assert entry.tool == "edit"

    def test_entry_depends_on_list_is_cleaned(self):
        """ToolDependencyEntry should clean depends_on list."""
        entry = ToolDependencyEntry(tool="edit", depends_on=["  read  ", "", "analyze", "   "])
        assert entry.depends_on == ["read", "analyze"]

    def test_entry_enables_list_is_cleaned(self):
        """ToolDependencyEntry should clean enables list."""
        entry = ToolDependencyEntry(tool="edit", enables=["  test  ", "", "commit", "   "])
        assert entry.enables == ["test", "commit"]

    def test_entry_weight_at_bounds(self):
        """ToolDependencyEntry weight should accept boundary values."""
        entry_low = ToolDependencyEntry(tool="edit", weight=0.0)
        assert entry_low.weight == 0.0

        entry_high = ToolDependencyEntry(tool="edit", weight=1.0)
        assert entry_high.weight == 1.0

    def test_entry_weight_below_bound_raises(self):
        """ToolDependencyEntry should reject weight below 0.0."""
        with pytest.raises(ValidationError) as exc_info:
            ToolDependencyEntry(tool="edit", weight=-0.1)
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_entry_weight_above_bound_raises(self):
        """ToolDependencyEntry should reject weight above 1.0."""
        with pytest.raises(ValidationError) as exc_info:
            ToolDependencyEntry(tool="edit", weight=1.1)
        assert "less than or equal to 1" in str(exc_info.value)


class TestToolDependencySpec:
    """Tests for ToolDependencySpec model."""

    def test_minimal_valid_spec(self):
        """ToolDependencySpec should accept minimal valid configuration."""
        spec = ToolDependencySpec(vertical="coding")
        assert spec.vertical == "coding"
        assert spec.version == "1.0"

    def test_complete_spec_validation(self):
        """ToolDependencySpec should accept complete valid configuration."""
        spec = ToolDependencySpec(
            version="1.0",
            vertical="coding",
            transitions={
                "read": [
                    ToolTransition(tool="edit", weight=0.4),
                    ToolTransition(tool="grep", weight=0.3),
                ]
            },
            clusters={"file_ops": ["read", "write", "edit"]},
            sequences={"exploration": ["ls", "read", "grep"]},
            dependencies=[
                ToolDependencyEntry(tool="edit", depends_on=["read"], enables=["test"], weight=0.9)
            ],
            required_tools=["read", "edit"],
            optional_tools=["grep", "test"],
            default_sequence=["read", "edit", "test"],
            metadata={"author": "test"},
        )
        assert spec.version == "1.0"
        assert spec.vertical == "coding"
        assert len(spec.transitions) == 1
        assert len(spec.clusters) == 1
        assert len(spec.sequences) == 1
        assert len(spec.dependencies) == 1
        assert spec.required_tools == ["read", "edit"]
        assert spec.optional_tools == ["grep", "test"]
        assert spec.default_sequence == ["read", "edit", "test"]
        assert spec.metadata["author"] == "test"

    def test_spec_vertical_required(self):
        """ToolDependencySpec should require vertical field."""
        with pytest.raises(ValidationError) as exc_info:
            ToolDependencySpec()
        assert "vertical" in str(exc_info.value)

    def test_spec_empty_vertical_raises(self):
        """ToolDependencySpec should reject empty vertical."""
        with pytest.raises(ValidationError) as exc_info:
            ToolDependencySpec(vertical="")
        assert "Vertical name cannot be empty" in str(exc_info.value)

    def test_spec_whitespace_vertical_raises(self):
        """ToolDependencySpec should reject whitespace-only vertical."""
        with pytest.raises(ValidationError) as exc_info:
            ToolDependencySpec(vertical="   ")
        assert "Vertical name cannot be empty" in str(exc_info.value)

    def test_spec_vertical_is_lowercased(self):
        """ToolDependencySpec should lowercase vertical name."""
        spec = ToolDependencySpec(vertical="CODING")
        assert spec.vertical == "coding"

    def test_spec_vertical_is_stripped(self):
        """ToolDependencySpec should strip vertical name."""
        spec = ToolDependencySpec(vertical="  coding  ")
        assert spec.vertical == "coding"

    def test_spec_default_values(self):
        """ToolDependencySpec should have sensible default values."""
        spec = ToolDependencySpec(vertical="coding")
        assert spec.version == "1.0"
        assert spec.transitions == {}
        assert spec.clusters == {}
        assert spec.sequences == {}
        assert spec.dependencies == []
        assert spec.required_tools == []
        assert spec.optional_tools == []
        assert spec.default_sequence == ["read", "edit"]
        assert spec.metadata == {} or "all_referenced_tools" in spec.metadata

    def test_spec_clusters_tool_names_cleaned(self):
        """ToolDependencySpec should clean cluster tool names."""
        spec = ToolDependencySpec(
            vertical="coding",
            clusters={"file_ops": ["  read  ", "", "  write  "]},
        )
        assert spec.clusters["file_ops"] == ["read", "write"]

    def test_spec_sequences_tool_names_cleaned(self):
        """ToolDependencySpec should clean sequence tool names."""
        spec = ToolDependencySpec(
            vertical="coding",
            sequences={"exploration": ["  ls  ", "", "  read  "]},
        )
        assert spec.sequences["exploration"] == ["ls", "read"]

    def test_spec_required_tools_cleaned(self):
        """ToolDependencySpec should clean required_tools list."""
        spec = ToolDependencySpec(
            vertical="coding",
            required_tools=["  read  ", "", "  edit  "],
        )
        assert spec.required_tools == ["read", "edit"]

    def test_spec_optional_tools_cleaned(self):
        """ToolDependencySpec should clean optional_tools list."""
        spec = ToolDependencySpec(
            vertical="coding",
            optional_tools=["  grep  ", "", "  test  "],
        )
        assert spec.optional_tools == ["grep", "test"]

    def test_spec_default_sequence_cleaned(self):
        """ToolDependencySpec should clean default_sequence list."""
        spec = ToolDependencySpec(
            vertical="coding",
            default_sequence=["  read  ", "", "  edit  "],
        )
        assert spec.default_sequence == ["read", "edit"]


class TestToolDependencySpecCrossValidation:
    """Tests for ToolDependencySpec cross-validation (validate_tool_references)."""

    def test_validate_tool_references_collects_all_tools(self):
        """validate_tool_references should collect all tools into metadata."""
        spec = ToolDependencySpec(
            vertical="coding",
            transitions={
                "read": [ToolTransition(tool="edit", weight=0.4)],
            },
            clusters={"file_ops": ["write", "delete"]},
            sequences={"explore": ["ls", "grep"]},
            dependencies=[
                ToolDependencyEntry(tool="commit", depends_on=["test"], enables=["deploy"])
            ],
            required_tools=["required1"],
            optional_tools=["optional1"],
            default_sequence=["default1", "default2"],
        )

        # All tools should be collected in metadata
        all_tools = set(spec.metadata.get("all_referenced_tools", []))
        expected_tools = {
            "read",
            "edit",
            "write",
            "delete",
            "ls",
            "grep",
            "commit",
            "test",
            "deploy",
            "required1",
            "optional1",
            "default1",
            "default2",
        }
        assert all_tools == expected_tools

    def test_validate_tool_references_does_not_duplicate_metadata(self):
        """validate_tool_references should not overwrite existing metadata."""
        spec = ToolDependencySpec(
            vertical="coding",
            required_tools=["read"],
            metadata={"custom_key": "custom_value"},
        )
        assert spec.metadata["custom_key"] == "custom_value"
        assert "all_referenced_tools" in spec.metadata


class TestToolDependencySpecGetAllToolNames:
    """Tests for ToolDependencySpec.get_all_tool_names() method."""

    def test_get_all_tool_names_empty_spec(self):
        """get_all_tool_names should return default_sequence tools for minimal spec."""
        spec = ToolDependencySpec(vertical="coding")
        all_tools = spec.get_all_tool_names()
        # Default sequence is ["read", "edit"]
        assert "read" in all_tools
        assert "edit" in all_tools

    def test_get_all_tool_names_from_transitions(self):
        """get_all_tool_names should include tools from transitions."""
        spec = ToolDependencySpec(
            vertical="coding",
            transitions={
                "source_tool": [
                    ToolTransition(tool="target1", weight=0.5),
                    ToolTransition(tool="target2", weight=0.5),
                ]
            },
        )
        all_tools = spec.get_all_tool_names()
        assert "source_tool" in all_tools
        assert "target1" in all_tools
        assert "target2" in all_tools

    def test_get_all_tool_names_from_clusters(self):
        """get_all_tool_names should include tools from clusters."""
        spec = ToolDependencySpec(
            vertical="coding",
            clusters={
                "cluster1": ["tool_a", "tool_b"],
                "cluster2": ["tool_c"],
            },
        )
        all_tools = spec.get_all_tool_names()
        assert "tool_a" in all_tools
        assert "tool_b" in all_tools
        assert "tool_c" in all_tools

    def test_get_all_tool_names_from_sequences(self):
        """get_all_tool_names should include tools from sequences."""
        spec = ToolDependencySpec(
            vertical="coding",
            sequences={
                "seq1": ["tool_x", "tool_y"],
                "seq2": ["tool_z"],
            },
        )
        all_tools = spec.get_all_tool_names()
        assert "tool_x" in all_tools
        assert "tool_y" in all_tools
        assert "tool_z" in all_tools

    def test_get_all_tool_names_from_dependencies(self):
        """get_all_tool_names should include tools from dependencies."""
        spec = ToolDependencySpec(
            vertical="coding",
            dependencies=[
                ToolDependencyEntry(
                    tool="main_tool",
                    depends_on=["dep1", "dep2"],
                    enables=["enabled1", "enabled2"],
                )
            ],
        )
        all_tools = spec.get_all_tool_names()
        assert "main_tool" in all_tools
        assert "dep1" in all_tools
        assert "dep2" in all_tools
        assert "enabled1" in all_tools
        assert "enabled2" in all_tools

    def test_get_all_tool_names_from_required_optional(self):
        """get_all_tool_names should include required and optional tools."""
        spec = ToolDependencySpec(
            vertical="coding",
            required_tools=["req1", "req2"],
            optional_tools=["opt1", "opt2"],
        )
        all_tools = spec.get_all_tool_names()
        assert "req1" in all_tools
        assert "req2" in all_tools
        assert "opt1" in all_tools
        assert "opt2" in all_tools

    def test_get_all_tool_names_deduplicates(self):
        """get_all_tool_names should return unique tool names."""
        spec = ToolDependencySpec(
            vertical="coding",
            transitions={"read": [ToolTransition(tool="edit", weight=0.5)]},
            clusters={"ops": ["read", "edit"]},
            sequences={"seq": ["read", "edit"]},
            required_tools=["read", "edit"],
            default_sequence=["read", "edit"],
        )
        all_tools = spec.get_all_tool_names()
        # Should only contain "read" and "edit" once each
        assert all_tools == {"read", "edit"}

    def test_get_all_tool_names_returns_set(self):
        """get_all_tool_names should return a set."""
        spec = ToolDependencySpec(vertical="coding")
        all_tools = spec.get_all_tool_names()
        assert isinstance(all_tools, set)

    def test_get_all_tool_names_comprehensive(self):
        """get_all_tool_names should work with all configuration sources combined."""
        spec = ToolDependencySpec(
            vertical="coding",
            transitions={"trans_source": [ToolTransition(tool="trans_target", weight=0.5)]},
            clusters={"cluster": ["cluster_tool"]},
            sequences={"sequence": ["sequence_tool"]},
            dependencies=[
                ToolDependencyEntry(tool="dep_tool", depends_on=["dep_dep"], enables=["dep_enable"])
            ],
            required_tools=["required_tool"],
            optional_tools=["optional_tool"],
            default_sequence=["default_tool"],
        )
        all_tools = spec.get_all_tool_names()
        expected = {
            "trans_source",
            "trans_target",
            "cluster_tool",
            "sequence_tool",
            "dep_tool",
            "dep_dep",
            "dep_enable",
            "required_tool",
            "optional_tool",
            "default_tool",
        }
        assert all_tools == expected


class TestToolDependencySpecModelValidation:
    """Tests for ToolDependencySpec model validation using model_validate."""

    def test_model_validate_from_dict(self):
        """ToolDependencySpec should validate from dictionary."""
        data = {
            "version": "1.0",
            "vertical": "coding",
            "transitions": {
                "read": [{"tool": "edit", "weight": 0.4}],
            },
            "clusters": {"file_ops": ["read", "write"]},
            "sequences": {"explore": ["ls", "read"]},
            "dependencies": [{"tool": "edit", "depends_on": ["read"], "enables": ["test"]}],
            "required_tools": ["read"],
            "optional_tools": ["grep"],
            "default_sequence": ["read", "edit"],
        }
        spec = ToolDependencySpec.model_validate(data)
        assert spec.vertical == "coding"
        assert len(spec.transitions["read"]) == 1
        assert spec.transitions["read"][0].tool == "edit"

    def test_model_validate_with_nested_transitions(self):
        """ToolDependencySpec should validate nested transitions correctly."""
        data = {
            "vertical": "devops",
            "transitions": {
                "analyze": [
                    {"tool": "plan", "weight": 0.6},
                    {"tool": "deploy", "weight": 0.4},
                ],
                "plan": [{"tool": "apply", "weight": 0.9}],
            },
        }
        spec = ToolDependencySpec.model_validate(data)
        assert len(spec.transitions["analyze"]) == 2
        assert spec.transitions["plan"][0].weight == 0.9

    def test_model_validate_invalid_transition_weight(self):
        """ToolDependencySpec should reject invalid transition weights in dict."""
        data = {
            "vertical": "coding",
            "transitions": {
                "read": [{"tool": "edit", "weight": 1.5}],  # Invalid
            },
        }
        with pytest.raises(ValidationError) as exc_info:
            ToolDependencySpec.model_validate(data)
        assert "less than or equal to 1" in str(exc_info.value)

    def test_model_validate_missing_vertical(self):
        """ToolDependencySpec should require vertical in dict validation."""
        data = {
            "version": "1.0",
            "transitions": {},
        }
        with pytest.raises(ValidationError) as exc_info:
            ToolDependencySpec.model_validate(data)
        assert "vertical" in str(exc_info.value)


class TestToolDependencySchemaExports:
    """Tests for module exports."""

    def test_all_exports_available(self):
        """All expected classes should be exported."""
        from victor.core.tool_dependency_schema import (
            ToolCluster,
            ToolDependencyEntry,
            ToolDependencySpec,
            ToolSequence,
            ToolTransition,
        )

        assert ToolTransition is not None
        assert ToolCluster is not None
        assert ToolSequence is not None
        assert ToolDependencyEntry is not None
        assert ToolDependencySpec is not None
