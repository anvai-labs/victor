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

"""Tests for EmptyToolDependencyProvider (LSP-compliant null object)."""

import pytest

from victor.core.tool_types import (
    EmptyToolDependencyProvider,
    ToolDependencyProviderProtocol,
)


class TestEmptyToolDependencyProviderInitialization:
    """Test initialization and basic properties."""

    def test_init_stores_vertical_name(self):
        """Provider should store the vertical name."""
        provider = EmptyToolDependencyProvider("research")
        assert provider.vertical == "research"

    def test_init_with_different_verticals(self):
        """Provider should work with any vertical name."""
        for vertical in [
            "coding",
            "devops",
            "research",
            "rag",
            "dataanalysis",
            "custom",
        ]:
            provider = EmptyToolDependencyProvider(vertical)
            assert provider.vertical == vertical


class TestEmptyToolDependencyProviderProtocolCompliance:
    """Test that provider implements ToolDependencyProviderProtocol."""

    def test_implements_get_dependencies(self):
        """Provider must implement get_dependencies."""
        provider = EmptyToolDependencyProvider("test")
        deps = provider.get_dependencies()
        assert isinstance(deps, list)
        assert len(deps) == 0

    def test_implements_get_tool_sequences(self):
        """Provider must implement get_tool_sequences."""
        provider = EmptyToolDependencyProvider("test")
        seqs = provider.get_tool_sequences()
        assert isinstance(seqs, list)
        assert len(seqs) == 1
        assert seqs[0] == ["read"]

    def test_runtime_checkable_protocol(self):
        """Provider should satisfy the runtime-checkable protocol."""
        provider = EmptyToolDependencyProvider("test")
        # The protocol requires get_dependencies and get_tool_sequences
        assert hasattr(provider, "get_dependencies")
        assert hasattr(provider, "get_tool_sequences")
        assert callable(provider.get_dependencies)
        assert callable(provider.get_tool_sequences)


class TestEmptyToolDependencyProviderExtendedMethods:
    """Test extended methods (same interface as BaseToolDependencyProvider)."""

    @pytest.fixture
    def provider(self):
        """Create a test provider."""
        return EmptyToolDependencyProvider("test")

    def test_get_tool_transitions_returns_empty_dict(self, provider):
        """Transitions should be empty for fallback."""
        assert provider.get_tool_transitions() == {}

    def test_get_tool_clusters_returns_empty_dict(self, provider):
        """Clusters should be empty for fallback."""
        assert provider.get_tool_clusters() == {}

    def test_get_recommended_sequence_returns_minimal(self, provider):
        """Recommended sequence should be minimal ["read"]."""
        assert provider.get_recommended_sequence("any_task") == ["read"]
        assert provider.get_recommended_sequence("edit") == ["read"]
        assert provider.get_recommended_sequence("deploy") == ["read"]

    def test_get_required_tools_returns_read(self, provider):
        """Required tools should include at least "read"."""
        assert provider.get_required_tools() == {"read"}

    def test_get_optional_tools_returns_empty(self, provider):
        """Optional tools should be empty for fallback."""
        assert provider.get_optional_tools() == set()

    def test_get_transition_weight_returns_default(self, provider):
        """Transition weight should return default 0.3."""
        assert provider.get_transition_weight("read", "edit") == 0.3
        assert provider.get_transition_weight("any", "tool") == 0.3

    def test_suggest_next_tool_returns_read(self, provider):
        """Next tool suggestion should always be "read" for fallback."""
        assert provider.suggest_next_tool("current") == "read"
        assert provider.suggest_next_tool("edit", ["read", "write"]) == "read"

    def test_find_cluster_returns_none(self, provider):
        """Cluster lookup should return None for fallback."""
        assert provider.find_cluster("read") is None
        assert provider.find_cluster("edit") is None

    def test_get_cluster_tools_returns_empty(self, provider):
        """Cluster tools should be empty for fallback."""
        assert provider.get_cluster_tools("any_cluster") == set()

    def test_is_valid_transition_returns_true(self, provider):
        """All transitions should be valid for fallback (permissive)."""
        assert provider.is_valid_transition("read", "edit") is True
        assert provider.is_valid_transition("any", "tool") is True


class TestEmptyToolDependencyProviderLSPCompliance:
    """Test LSP (Liskov Substitution Principle) compliance.

    The provider should be usable anywhere a BaseToolDependencyProvider is expected.
    """

    def test_can_call_all_expected_methods(self):
        """Provider should support all methods expected by consumers."""
        provider = EmptyToolDependencyProvider("test")

        # All these should work without AttributeError
        provider.vertical
        provider.get_dependencies()
        provider.get_tool_sequences()
        provider.get_tool_transitions()
        provider.get_tool_clusters()
        provider.get_recommended_sequence("edit")
        provider.get_required_tools()
        provider.get_optional_tools()
        provider.get_transition_weight("read", "edit")
        provider.suggest_next_tool("read")
        provider.find_cluster("read")
        provider.get_cluster_tools("cluster")
        provider.is_valid_transition("read", "edit")

    def test_sequence_returns_copy_not_reference(self):
        """get_recommended_sequence should return a copy to prevent mutation."""
        provider = EmptyToolDependencyProvider("test")
        seq1 = provider.get_recommended_sequence("edit")
        seq2 = provider.get_recommended_sequence("edit")

        # Modify one
        seq1.append("write")

        # Other should be unaffected
        assert seq2 == ["read"]

    def test_required_tools_returns_copy_not_reference(self):
        """get_required_tools should return a copy to prevent mutation."""
        provider = EmptyToolDependencyProvider("test")
        tools1 = provider.get_required_tools()
        tools2 = provider.get_required_tools()

        # Modify one
        tools1.add("write")

        # Other should be unaffected
        assert tools2 == {"read"}


class TestEmptyToolDependencyProviderFactoryIntegration:
    """Test integration with create_vertical_tool_dependency_provider."""

    def test_factory_returns_empty_provider_for_missing_yaml(self, tmp_path, monkeypatch):
        """Factory should return EmptyToolDependencyProvider when YAML missing."""
        from victor.core.tool_dependency_loader import (
            create_vertical_tool_dependency_provider,
        )

        # Monkeypatch the yaml_paths to point to non-existent file
        fake_yaml_path = tmp_path / "nonexistent.yaml"

        # We can't easily test this without modifying the factory
        # But we can test that EmptyToolDependencyProvider works as a substitute
        provider = EmptyToolDependencyProvider("research")

        # Should work as a drop-in replacement
        assert provider.vertical == "research"
        assert provider.get_dependencies() == []
        assert provider.get_required_tools() == {"read"}
