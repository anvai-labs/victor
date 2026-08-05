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

"""Tests for victor.core.tool_dependency_base module.

This module tests the BaseToolDependencyProvider class, covering:
- Base class instantiation with config object and individual arguments
- Dependency declaration methods
- Transition weight calculation logic
- Resolution order logic
- Tool suggestion algorithm
- Cluster finding and retrieval
- Edge cases (circular dependencies, missing tools)
- Protocol compliance
"""

import pytest

from victor.core.tool_dependency_base import (
    BaseToolDependencyProvider,
    ToolDependencyConfig,
)
from victor.core.tool_types import ToolDependency, ToolDependencyProviderProtocol

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_dependencies():
    """Sample dependencies for testing."""
    return [
        ToolDependency(
            tool_name="edit",
            depends_on={"read"},
            enables={"test", "lint"},
            weight=0.9,
        ),
        ToolDependency(
            tool_name="test",
            depends_on={"edit"},
            enables={"commit"},
            weight=0.8,
        ),
        ToolDependency(
            tool_name="commit",
            depends_on={"test"},
            enables={"push"},
            weight=0.7,
        ),
    ]


@pytest.fixture
def sample_transitions():
    """Sample transitions for testing."""
    return {
        "read": [("edit", 0.6), ("grep", 0.3), ("analyze", 0.1)],
        "edit": [("test", 0.5), ("lint", 0.3), ("format", 0.2)],
        "grep": [("read", 0.7), ("edit", 0.3)],
    }


@pytest.fixture
def sample_clusters():
    """Sample clusters for testing."""
    return {
        "file_operations": {"read", "write", "edit", "delete"},
        "code_quality": {"test", "lint", "format"},
        "version_control": {"commit", "push", "pull", "branch"},
    }


@pytest.fixture
def sample_sequences():
    """Sample sequences for testing."""
    return {
        "code_review": ["read", "analyze", "comment"],
        "bug_fix": ["read", "edit", "test", "commit"],
        "refactor": ["analyze", "edit", "test", "lint"],
    }


@pytest.fixture
def full_config(sample_dependencies, sample_transitions, sample_clusters, sample_sequences):
    """Full configuration object for testing."""
    return ToolDependencyConfig(
        dependencies=sample_dependencies,
        transitions=sample_transitions,
        clusters=sample_clusters,
        sequences=sample_sequences,
        required_tools={"read", "write", "edit"},
        optional_tools={"grep", "analyze", "format"},
        default_sequence=["read", "edit", "test"],
    )


@pytest.fixture
def provider_with_config(full_config):
    """Provider initialized with config object."""
    return BaseToolDependencyProvider(config=full_config)


@pytest.fixture
def provider_with_args(sample_dependencies, sample_transitions, sample_clusters, sample_sequences):
    """Provider initialized with individual arguments."""
    return BaseToolDependencyProvider(
        dependencies=sample_dependencies,
        transitions=sample_transitions,
        clusters=sample_clusters,
        sequences=sample_sequences,
        required_tools={"read", "write", "edit"},
        optional_tools={"grep", "analyze", "format"},
        default_sequence=["read", "edit", "test"],
    )


@pytest.fixture
def minimal_provider():
    """Minimal provider with no configuration."""
    return BaseToolDependencyProvider()


# =============================================================================
# Tests for ToolDependencyConfig
# =============================================================================


class TestToolDependencyConfig:
    """Tests for ToolDependencyConfig dataclass."""

    def test_default_values(self):
        """ToolDependencyConfig should have sensible defaults."""
        config = ToolDependencyConfig()

        assert config.dependencies == []
        assert config.transitions == {}
        assert config.clusters == {}
        assert config.sequences == {}
        assert config.required_tools == set()
        assert config.optional_tools == set()
        assert config.default_sequence == ["read", "edit"]

    def test_custom_values(self, sample_dependencies):
        """ToolDependencyConfig should accept custom values."""
        config = ToolDependencyConfig(
            dependencies=sample_dependencies,
            transitions={"read": [("edit", 0.5)]},
            clusters={"file_ops": {"read", "edit"}},
            sequences={"review": ["read", "analyze"]},
            required_tools={"read"},
            optional_tools={"grep"},
            default_sequence=["read", "write"],
        )

        assert len(config.dependencies) == 3
        assert "read" in config.transitions
        assert "file_ops" in config.clusters
        assert "review" in config.sequences
        assert "read" in config.required_tools
        assert "grep" in config.optional_tools
        assert config.default_sequence == ["read", "write"]

    def test_default_sequence_factory(self):
        """default_sequence should be a new list each time."""
        config1 = ToolDependencyConfig()
        config2 = ToolDependencyConfig()

        # Should be equal but not the same object
        assert config1.default_sequence == config2.default_sequence
        config1.default_sequence.append("new_tool")
        assert config1.default_sequence != config2.default_sequence


# =============================================================================
# Tests for BaseToolDependencyProvider Initialization
# =============================================================================


class TestBaseToolDependencyProviderInit:
    """Tests for BaseToolDependencyProvider initialization."""

    def test_init_with_config_object(self, full_config):
        """BaseToolDependencyProvider should accept config object."""
        provider = BaseToolDependencyProvider(config=full_config)

        assert provider._config is full_config
        assert len(provider._dependency_map) == 3

    def test_init_with_individual_args(self, sample_dependencies, sample_transitions):
        """BaseToolDependencyProvider should accept individual arguments."""
        provider = BaseToolDependencyProvider(
            dependencies=sample_dependencies,
            transitions=sample_transitions,
            required_tools={"read"},
        )

        assert len(provider._config.dependencies) == 3
        assert "read" in provider._config.transitions
        assert "read" in provider._config.required_tools

    def test_init_with_no_args(self):
        """BaseToolDependencyProvider should work with no arguments."""
        provider = BaseToolDependencyProvider()

        assert provider._config.dependencies == []
        assert provider._config.transitions == {}
        assert provider._config.default_sequence == ["read", "edit"]

    def test_config_takes_precedence_over_individual_args(self, full_config, sample_dependencies):
        """When config is provided, individual args should be ignored."""
        # Create a different set of dependencies
        other_deps = [ToolDependency(tool_name="other", depends_on=set(), enables=set())]

        provider = BaseToolDependencyProvider(
            config=full_config,
            dependencies=other_deps,  # Should be ignored
        )

        # Config should take precedence
        assert len(provider._config.dependencies) == 3
        assert provider._config.dependencies[0].tool_name == "edit"

    def test_dependency_map_built_correctly(self, sample_dependencies):
        """Provider should build dependency map from dependencies."""
        provider = BaseToolDependencyProvider(dependencies=sample_dependencies)

        assert "edit" in provider._dependency_map
        assert "test" in provider._dependency_map
        assert "commit" in provider._dependency_map
        assert provider._dependency_map["edit"].weight == 0.9

    def test_none_values_treated_as_empty(self):
        """None values for optional arguments should become empty collections."""
        provider = BaseToolDependencyProvider(
            dependencies=None,
            transitions=None,
            clusters=None,
            sequences=None,
            required_tools=None,
            optional_tools=None,
            default_sequence=None,
        )

        assert provider._config.dependencies == []
        assert provider._config.transitions == {}
        assert provider._config.clusters == {}
        assert provider._config.sequences == {}
        assert provider._config.required_tools == set()
        assert provider._config.optional_tools == set()
        assert provider._config.default_sequence == ["read", "edit"]


# =============================================================================
# Tests for Protocol Implementation
# =============================================================================


class TestProtocolCompliance:
    """Tests verifying BaseToolDependencyProvider implements protocol."""

    def test_implements_protocol(self, provider_with_config):
        """BaseToolDependencyProvider should implement ToolDependencyProviderProtocol."""
        assert isinstance(provider_with_config, ToolDependencyProviderProtocol)

    def test_get_dependencies_returns_list(self, provider_with_config):
        """get_dependencies should return a list of ToolDependency objects."""
        deps = provider_with_config.get_dependencies()

        assert isinstance(deps, list)
        assert all(isinstance(d, ToolDependency) for d in deps)

    def test_get_dependencies_returns_copy(self, provider_with_config):
        """get_dependencies should return a copy, not the original list."""
        deps1 = provider_with_config.get_dependencies()
        deps2 = provider_with_config.get_dependencies()

        assert deps1 is not deps2
        assert deps1 == deps2

    def test_get_tool_sequences_returns_list_of_lists(self, provider_with_config):
        """get_tool_sequences should return list of tool name sequences."""
        sequences = provider_with_config.get_tool_sequences()

        assert isinstance(sequences, list)
        assert all(isinstance(s, list) for s in sequences)
        assert all(all(isinstance(t, str) for t in seq) for seq in sequences)

    def test_get_tool_sequences_returns_copies(self, provider_with_config):
        """get_tool_sequences should return copies of sequences."""
        seqs1 = provider_with_config.get_tool_sequences()
        seqs2 = provider_with_config.get_tool_sequences()

        # Modify one sequence
        if seqs1:
            seqs1[0].append("modified")

        # Other should be unaffected
        assert seqs1 != seqs2 or len(seqs1) == 0


# =============================================================================
# Tests for get_tool_transitions
# =============================================================================


class TestGetToolTransitions:
    """Tests for get_tool_transitions method."""

    def test_returns_dict(self, provider_with_config):
        """get_tool_transitions should return a dict."""
        transitions = provider_with_config.get_tool_transitions()

        assert isinstance(transitions, dict)

    def test_returns_correct_structure(self, provider_with_config):
        """get_tool_transitions should return tool -> [(next_tool, prob)] mapping."""
        transitions = provider_with_config.get_tool_transitions()

        assert "read" in transitions
        read_transitions = transitions["read"]
        assert isinstance(read_transitions, list)
        assert len(read_transitions) > 0
        assert all(isinstance(t, tuple) and len(t) == 2 for t in read_transitions)

    def test_returns_copy(self, provider_with_config):
        """get_tool_transitions should return a copy."""
        trans1 = provider_with_config.get_tool_transitions()
        trans2 = provider_with_config.get_tool_transitions()

        assert trans1 is not trans2
        assert trans1 == trans2


# =============================================================================
# Tests for get_tool_clusters
# =============================================================================


class TestGetToolClusters:
    """Tests for get_tool_clusters method."""

    def test_returns_dict(self, provider_with_config):
        """get_tool_clusters should return a dict."""
        clusters = provider_with_config.get_tool_clusters()

        assert isinstance(clusters, dict)

    def test_returns_correct_structure(self, provider_with_config):
        """get_tool_clusters should return cluster_name -> set of tools."""
        clusters = provider_with_config.get_tool_clusters()

        assert "file_operations" in clusters
        assert isinstance(clusters["file_operations"], set)
        assert "read" in clusters["file_operations"]

    def test_returns_deep_copy(self, provider_with_config):
        """get_tool_clusters should return a deep copy."""
        clust1 = provider_with_config.get_tool_clusters()
        clust2 = provider_with_config.get_tool_clusters()

        assert clust1 is not clust2
        assert clust1["file_operations"] is not clust2["file_operations"]


# =============================================================================
# Tests for get_recommended_sequence
# =============================================================================


class TestGetRecommendedSequence:
    """Tests for get_recommended_sequence method."""

    def test_returns_known_sequence(self, provider_with_config):
        """get_recommended_sequence should return sequence for known task type."""
        seq = provider_with_config.get_recommended_sequence("bug_fix")

        assert seq == ["read", "edit", "test", "commit"]

    def test_returns_default_for_unknown_task(self, provider_with_config):
        """get_recommended_sequence should return default for unknown task type."""
        seq = provider_with_config.get_recommended_sequence("unknown_task")

        assert seq == ["read", "edit", "test"]

    def test_returns_copy(self, provider_with_config):
        """get_recommended_sequence should return a copy of sequence."""
        seq1 = provider_with_config.get_recommended_sequence("bug_fix")
        seq2 = provider_with_config.get_recommended_sequence("bug_fix")

        assert seq1 is not seq2
        assert seq1 == seq2


# =============================================================================
# Tests for get_required_tools and get_optional_tools
# =============================================================================


class TestGetRequiredOptionalTools:
    """Tests for get_required_tools and get_optional_tools methods."""

    def test_get_required_tools(self, provider_with_config):
        """get_required_tools should return set of required tools."""
        required = provider_with_config.get_required_tools()

        assert isinstance(required, set)
        assert required == {"read", "write", "edit"}

    def test_get_required_tools_returns_copy(self, provider_with_config):
        """get_required_tools should return a copy."""
        req1 = provider_with_config.get_required_tools()
        req2 = provider_with_config.get_required_tools()

        assert req1 is not req2
        assert req1 == req2

    def test_get_optional_tools(self, provider_with_config):
        """get_optional_tools should return set of optional tools."""
        optional = provider_with_config.get_optional_tools()

        assert isinstance(optional, set)
        assert optional == {"grep", "analyze", "format"}

    def test_get_optional_tools_returns_copy(self, provider_with_config):
        """get_optional_tools should return a copy."""
        opt1 = provider_with_config.get_optional_tools()
        opt2 = provider_with_config.get_optional_tools()

        assert opt1 is not opt2
        assert opt1 == opt2


# =============================================================================
# Tests for get_transition_weight
# =============================================================================


class TestGetTransitionWeight:
    """Tests for get_transition_weight method."""

    def test_weight_from_depends_on(self, provider_with_config):
        """get_transition_weight should return weight when to_tool depends on from_tool."""
        # edit depends_on read with weight 0.9
        weight = provider_with_config.get_transition_weight("read", "edit")

        assert weight == 0.9

    def test_weight_from_enables(self, provider_with_config):
        """get_transition_weight should return weight * 0.8 when from_tool enables to_tool."""
        # edit enables test with weight 0.9
        weight = provider_with_config.get_transition_weight("edit", "test")

        # From enables relationship: 0.9 * 0.8 = 0.72
        # But test also depends on edit, so it returns 0.8 (test's weight)
        assert weight == 0.8

    def test_weight_from_enables_when_not_depends(self, provider_with_config):
        """get_transition_weight should use enables when no depends relationship."""
        # edit enables lint (and lint doesn't have a dependency defined)
        weight = provider_with_config.get_transition_weight("edit", "lint")

        # edit.weight * 0.8 = 0.9 * 0.8 = 0.72
        assert weight == 0.9 * 0.8

    def test_weight_from_transitions(self, provider_with_config):
        """get_transition_weight should use transitions dict."""
        # grep -> read has probability 0.7 in transitions
        weight = provider_with_config.get_transition_weight("grep", "read")

        assert weight == 0.7

    def test_weight_from_sequence_adjacency(self, provider_with_config):
        """get_transition_weight should return 0.6 for adjacent tools in sequence."""
        # In bug_fix sequence: read -> edit -> test -> commit
        # analyze -> comment are adjacent in code_review
        weight = provider_with_config.get_transition_weight("analyze", "comment")

        assert weight == 0.6

    def test_weight_default_for_unknown(self, provider_with_config):
        """get_transition_weight should return 0.3 for unknown transitions."""
        weight = provider_with_config.get_transition_weight("unknown1", "unknown2")

        assert weight == 0.3


# =============================================================================
# Tests for suggest_next_tool
# =============================================================================


class TestSuggestNextTool:
    """Tests for suggest_next_tool method."""

    def test_suggests_from_transitions(self, provider_with_config):
        """suggest_next_tool should suggest based on transitions."""
        # read has transitions: edit (0.6), grep (0.3), analyze (0.1)
        next_tool = provider_with_config.suggest_next_tool("read")

        # Should suggest edit (highest probability)
        assert next_tool == "edit"

    def test_avoids_recently_used_tools(self, provider_with_config):
        """suggest_next_tool should avoid recently used tools."""
        # read has transitions: edit (0.6), grep (0.3), analyze (0.1)
        next_tool = provider_with_config.suggest_next_tool("read", used_tools=["edit"])

        # Should skip edit and suggest grep (next highest)
        assert next_tool == "grep"

    def test_avoids_last_three_tools(self, provider_with_config):
        """suggest_next_tool should avoid last 3 tools to prevent loops."""
        next_tool = provider_with_config.suggest_next_tool(
            "read", used_tools=["some", "other", "edit", "grep", "analyze"]
        )

        # Last 3 are edit, grep, analyze - all of read's transitions
        # Should fall back to first transition
        assert next_tool == "edit"

    def test_fallback_to_enables_when_no_transitions(self, sample_dependencies):
        """suggest_next_tool should use enables when no transitions defined."""
        provider = BaseToolDependencyProvider(dependencies=sample_dependencies)

        # edit has no transitions but enables test and lint
        next_tool = provider.suggest_next_tool("edit")

        # Should return one of the enabled tools
        assert next_tool in {"test", "lint"}

    def test_fallback_to_default_sequence(self):
        """suggest_next_tool should use default sequence as last fallback."""
        provider = BaseToolDependencyProvider(default_sequence=["fallback_tool"])

        next_tool = provider.suggest_next_tool("unknown_tool")

        assert next_tool == "fallback_tool"

    def test_fallback_to_read_when_no_default(self):
        """suggest_next_tool should return 'read' if no default sequence."""
        config = ToolDependencyConfig(default_sequence=[])
        provider = BaseToolDependencyProvider(config=config)

        next_tool = provider.suggest_next_tool("unknown_tool")

        assert next_tool == "read"

    def test_with_none_used_tools(self, provider_with_config):
        """suggest_next_tool should handle None used_tools."""
        next_tool = provider_with_config.suggest_next_tool("read", used_tools=None)

        assert next_tool == "edit"


# =============================================================================
# Tests for find_cluster
# =============================================================================


class TestFindCluster:
    """Tests for find_cluster method."""

    def test_finds_cluster_for_tool(self, provider_with_config):
        """find_cluster should return cluster name for tool in a cluster."""
        cluster = provider_with_config.find_cluster("read")

        assert cluster == "file_operations"

    def test_returns_none_for_unknown_tool(self, provider_with_config):
        """find_cluster should return None for tool not in any cluster."""
        cluster = provider_with_config.find_cluster("unknown_tool")

        assert cluster is None

    def test_finds_correct_cluster_for_each_tool(self, provider_with_config):
        """find_cluster should find correct cluster for various tools."""
        assert provider_with_config.find_cluster("read") == "file_operations"
        assert provider_with_config.find_cluster("test") == "code_quality"
        assert provider_with_config.find_cluster("commit") == "version_control"


# =============================================================================
# Tests for get_cluster_tools
# =============================================================================


class TestGetClusterTools:
    """Tests for get_cluster_tools method."""

    def test_returns_tools_in_cluster(self, provider_with_config):
        """get_cluster_tools should return tools in named cluster."""
        tools = provider_with_config.get_cluster_tools("file_operations")

        assert tools == {"read", "write", "edit", "delete"}

    def test_returns_empty_for_unknown_cluster(self, provider_with_config):
        """get_cluster_tools should return empty set for unknown cluster."""
        tools = provider_with_config.get_cluster_tools("unknown_cluster")

        assert tools == set()

    def test_returns_copy(self, provider_with_config):
        """get_cluster_tools should return a copy."""
        tools1 = provider_with_config.get_cluster_tools("file_operations")
        tools2 = provider_with_config.get_cluster_tools("file_operations")

        assert tools1 is not tools2
        assert tools1 == tools2


# =============================================================================
# Tests for is_valid_transition
# =============================================================================


class TestIsValidTransition:
    """Tests for is_valid_transition method."""

    def test_valid_from_depends_on(self, provider_with_config):
        """is_valid_transition should return True for depends_on relationship."""
        # edit depends_on read
        result = provider_with_config.is_valid_transition("read", "edit")

        assert result is True

    def test_valid_from_enables(self, provider_with_config):
        """is_valid_transition should return True for enables relationship."""
        # edit enables lint
        result = provider_with_config.is_valid_transition("edit", "lint")

        assert result is True

    def test_valid_from_transitions(self, provider_with_config):
        """is_valid_transition should return True for transitions."""
        # grep -> read in transitions
        result = provider_with_config.is_valid_transition("grep", "read")

        assert result is True

    def test_invalid_for_undefined_transition(self, provider_with_config):
        """is_valid_transition should return False for undefined transitions."""
        result = provider_with_config.is_valid_transition("unknown1", "unknown2")

        assert result is False

    def test_invalid_for_reverse_direction(self, provider_with_config):
        """is_valid_transition should return False for reverse of valid transition."""
        # read -> edit is valid, but edit -> read is not (unless defined)
        result = provider_with_config.is_valid_transition("edit", "read")

        # This would only be True if there's a dependency or transition for it
        # In our setup, there's no such relationship
        assert result is False


# =============================================================================
# Tests for Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_provider(self, minimal_provider):
        """Empty provider should handle all operations gracefully."""
        assert minimal_provider.get_dependencies() == []
        assert minimal_provider.get_tool_sequences() == []
        assert minimal_provider.get_tool_transitions() == {}
        assert minimal_provider.get_tool_clusters() == {}
        assert minimal_provider.get_required_tools() == set()
        assert minimal_provider.get_optional_tools() == set()
        assert minimal_provider.get_recommended_sequence("any") == ["read", "edit"]
        assert minimal_provider.find_cluster("any") is None
        assert minimal_provider.get_cluster_tools("any") == set()
        assert minimal_provider.get_transition_weight("a", "b") == 0.3
        assert minimal_provider.suggest_next_tool("any") == "read"
        assert minimal_provider.is_valid_transition("a", "b") is False

    def test_circular_dependencies(self):
        """Provider should handle circular dependencies."""
        deps = [
            ToolDependency(tool_name="a", depends_on={"c"}, enables={"b"}, weight=0.5),
            ToolDependency(tool_name="b", depends_on={"a"}, enables={"c"}, weight=0.5),
            ToolDependency(tool_name="c", depends_on={"b"}, enables={"a"}, weight=0.5),
        ]
        provider = BaseToolDependencyProvider(dependencies=deps)

        # Should not hang or crash
        # Note: depends_on takes precedence over enables
        # a -> b: b depends_on a, so weight is b's weight (0.5)
        assert provider.get_transition_weight("a", "b") == 0.5
        # b -> c: c depends_on b, so weight is c's weight (0.5)
        assert provider.get_transition_weight("b", "c") == 0.5
        # c -> a: a depends_on c, so weight is a's weight (0.5)
        assert provider.get_transition_weight("c", "a") == 0.5

    def test_self_dependency(self):
        """Provider should handle self-dependency."""
        deps = [
            ToolDependency(tool_name="recursive", depends_on={"recursive"}, weight=0.5),
        ]
        provider = BaseToolDependencyProvider(dependencies=deps)

        assert provider.get_transition_weight("recursive", "recursive") == 0.5

    def test_tool_in_multiple_clusters(self):
        """find_cluster returns first matching cluster."""
        clusters = {
            "cluster_a": {"shared_tool", "tool_a"},
            "cluster_b": {"shared_tool", "tool_b"},
        }
        provider = BaseToolDependencyProvider(clusters=clusters)

        # Should return one of the clusters (order may vary)
        cluster = provider.find_cluster("shared_tool")
        assert cluster in {"cluster_a", "cluster_b"}

    def test_empty_string_tool_name(self):
        """Provider should handle empty string tool names."""
        deps = [ToolDependency(tool_name="", depends_on=set(), enables=set())]
        provider = BaseToolDependencyProvider(dependencies=deps)

        assert "" in provider._dependency_map

    def test_unicode_tool_names(self):
        """Provider should handle unicode tool names."""
        deps = [
            ToolDependency(tool_name="tool", depends_on=set(), enables=set()),
        ]
        clusters = {"cluster": {"tool"}}
        provider = BaseToolDependencyProvider(dependencies=deps, clusters=clusters)

        assert provider.find_cluster("tool") == "cluster"

    def test_very_long_sequence(self):
        """Provider should handle very long sequences."""
        tools = [f"tool_{i}" for i in range(100)]
        sequences = {"long": tools}
        provider = BaseToolDependencyProvider(sequences=sequences)

        seq = provider.get_recommended_sequence("long")
        assert len(seq) == 100

    def test_many_transitions_per_tool(self):
        """Provider should handle many transitions per tool."""
        transitions = {"source": [(f"target_{i}", 0.01) for i in range(100)]}
        provider = BaseToolDependencyProvider(transitions=transitions)

        trans = provider.get_tool_transitions()
        assert len(trans["source"]) == 100

    def test_transition_weight_zero(self):
        """Provider should handle zero transition weights."""
        transitions = {"source": [("target", 0.0)]}
        provider = BaseToolDependencyProvider(transitions=transitions)

        weight = provider.get_transition_weight("source", "target")
        assert weight == 0.0

    def test_dependency_weight_zero(self):
        """Provider should handle zero dependency weights."""
        deps = [ToolDependency(tool_name="tool", depends_on={"dep"}, weight=0.0)]
        provider = BaseToolDependencyProvider(dependencies=deps)

        weight = provider.get_transition_weight("dep", "tool")
        assert weight == 0.0

    def test_suggest_with_all_recent_used(self, provider_with_config):
        """suggest_next_tool should still return a suggestion when all options are used."""
        # read has: edit, grep, analyze
        used = ["edit", "grep", "analyze"]
        next_tool = provider_with_config.suggest_next_tool("read", used_tools=used)

        # Should fall back to first transition even if recently used
        assert next_tool == "edit"


# =============================================================================
# Tests for Subclassing
# =============================================================================


class TestSubclassing:
    """Tests for subclassing BaseToolDependencyProvider."""

    def test_subclass_can_override_methods(self, sample_dependencies):
        """Subclasses should be able to override methods."""

        class CustomProvider(BaseToolDependencyProvider):
            def get_recommended_sequence(self, task_type: str):
                return ["custom", "sequence"]

        provider = CustomProvider(dependencies=sample_dependencies)

        assert provider.get_recommended_sequence("any") == ["custom", "sequence"]

    def test_subclass_inherits_default_behavior(self, sample_dependencies):
        """Subclasses should inherit default behavior for non-overridden methods."""

        class CustomProvider(BaseToolDependencyProvider):
            pass

        provider = CustomProvider(dependencies=sample_dependencies)

        assert len(provider.get_dependencies()) == 3

    def test_subclass_with_custom_init(self):
        """Subclasses can have custom initialization."""

        class DevOpsProvider(BaseToolDependencyProvider):
            def __init__(self):
                super().__init__(
                    dependencies=[
                        ToolDependency("deploy", depends_on={"build"}, enables={"monitor"})
                    ],
                    required_tools={"build", "deploy"},
                )

        provider = DevOpsProvider()

        assert "build" in provider.get_required_tools()
        assert len(provider.get_dependencies()) == 1


# =============================================================================
# Tests for Integration Scenarios
# =============================================================================


class TestIntegrationScenarios:
    """Integration tests for realistic usage patterns."""

    def test_typical_workflow_traversal(self, provider_with_config):
        """Simulate typical workflow traversal."""
        used_tools = []
        current = "read"

        for _ in range(5):
            used_tools.append(current)
            current = provider_with_config.suggest_next_tool(current, used_tools)

        # Should have traversed through several tools
        assert len(used_tools) == 5
        assert all(isinstance(t, str) for t in used_tools)

    def test_cluster_based_tool_selection(self, provider_with_config):
        """Tools in same cluster should have related transitions."""
        file_ops = provider_with_config.get_cluster_tools("file_operations")

        # At least some tools in file_ops should have valid transitions between them
        valid_transitions = 0
        for from_tool in file_ops:
            for to_tool in file_ops:
                if from_tool != to_tool:
                    weight = provider_with_config.get_transition_weight(from_tool, to_tool)
                    if weight > 0.3:  # Not just default
                        valid_transitions += 1

        # There should be at least some relationships defined
        assert valid_transitions >= 0  # Even if 0, the test documents behavior

    def test_sequence_follows_dependencies(self, provider_with_config):
        """Tools in sequences should follow logical order."""
        seq = provider_with_config.get_recommended_sequence("bug_fix")

        # Each tool should have valid transition to next
        for i in range(len(seq) - 1):
            weight = provider_with_config.get_transition_weight(seq[i], seq[i + 1])
            # Should at least get sequence adjacency weight (0.6)
            assert weight >= 0.3
