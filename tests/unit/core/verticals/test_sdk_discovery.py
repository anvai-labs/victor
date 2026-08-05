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

"""Unit tests for victor-contracts protocol discovery integration.

These tests verify that the framework can discover and use SDK protocols
from external verticals via entry points.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from victor.core.verticals.sdk_discovery import (
    get_sdk_protocol_registry,
    discover_sdk_protocols,
    get_sdk_tool_providers,
    get_sdk_safety_providers,
    get_sdk_workflow_providers,
    get_sdk_prompt_providers,
    get_sdk_capability_providers,
    get_sdk_capability_provider,
    get_sdk_validators,
    get_sdk_validator,
    get_sdk_discovery_stats,
    get_sdk_discovery_summary,
    reload_sdk_discovery,
    reset_sdk_discovery,
    list_sdk_capabilities,
    list_sdk_validators,
    enhance_vertical_with_sdk_protocols,
    ProtocolRegistry,
    DiscoveryStats,
    ProtocolMetadata,
)


class TestSDKDiscoveryModule:
    """Test SDK discovery module structure and imports."""

    def test_module_imports(self):
        """Test that the SDK discovery module can be imported."""
        from victor.core.verticals import sdk_discovery

        assert sdk_discovery is not None

    def test_core_functions_exist(self):
        """Test that core SDK discovery functions exist."""
        assert callable(get_sdk_protocol_registry)
        assert callable(discover_sdk_protocols)
        assert callable(get_sdk_tool_providers)
        assert callable(get_sdk_safety_providers)
        assert callable(get_sdk_workflow_providers)
        assert callable(get_sdk_prompt_providers)
        assert callable(get_sdk_capability_providers)
        assert callable(get_sdk_discovery_stats)
        assert callable(get_sdk_discovery_summary)

    def test_core_verticals_exports(self):
        """Test that SDK discovery functions are exported from victor.core.verticals."""
        from victor.core.verticals import (
            get_sdk_protocol_registry as get_registry,
            discover_sdk_protocols as discover,
            get_sdk_tool_providers as get_tools,
        )

        assert callable(get_registry)
        assert callable(discover)
        assert callable(get_tools)


class TestSDKDiscoveryWithoutSDK:
    """Test SDK discovery behavior when victor-contracts is not installed."""

    def test_get_registry_returns_none_when_sdk_not_available(self):
        """Test that get_sdk_protocol_registry returns None when SDK not available."""
        with patch("victor.core.verticals.sdk_discovery.get_global_registry", None):
            result = get_sdk_protocol_registry()
            assert result is None

    def test_discover_returns_empty_stats_when_sdk_not_available(self):
        """Test that discover_sdk_protocols returns empty stats when SDK not available."""
        with patch("victor.core.verticals.sdk_discovery.get_global_registry", None):
            stats = discover_sdk_protocols()
            assert isinstance(stats, DiscoveryStats)
            assert stats.total_protocols == 0
            assert stats.total_capabilities == 0

    def test_get_tool_providers_returns_empty_list_when_sdk_not_available(self):
        """Test that get_sdk_tool_providers returns empty list when SDK not available."""
        with patch("victor.core.verticals.sdk_discovery.get_global_registry", None):
            providers = get_sdk_tool_providers()
            assert providers == []

    def test_get_capability_providers_returns_empty_dict_when_sdk_not_available(self):
        """Test that get_sdk_capability_providers returns empty dict when SDK not available."""
        with patch("victor.core.verticals.sdk_discovery.get_global_registry", None):
            providers = get_sdk_capability_providers()
            assert providers == {}


class TestSDKDiscoveryWithMockSDK:
    """Test SDK discovery behavior with mocked victor-contracts."""

    @pytest.fixture
    def mock_registry(self):
        """Create a mock ProtocolRegistry."""
        registry = Mock(spec=ProtocolRegistry)

        # Mock discovery stats
        stats = DiscoveryStats(
            total_verticals=5,
            total_protocols=20,
            total_capabilities=10,
            total_validators=3,
            failed_loads=0,
        )
        registry.get_discovery_stats.return_value = stats
        registry.load_from_entry_points.return_value = stats

        # Mock provider lists
        tool_provider = Mock()
        tool_provider.get_tools.return_value = ["read", "write", "edit"]
        tool_provider.__class__.__name__ = "TestToolProvider"

        safety_provider = Mock()
        safety_provider.__class__.__name__ = "TestSafetyProvider"

        workflow_provider = Mock()
        workflow_provider.__class__.__name__ = "TestWorkflowProvider"

        prompt_provider = Mock()
        prompt_provider.__class__.__name__ = "TestPromptProvider"

        registry.get_tool_providers.return_value = [tool_provider]
        registry.get_safety_providers.return_value = [safety_provider]
        registry.get_workflow_providers.return_value = [workflow_provider]
        registry.get_prompt_providers.return_value = [prompt_provider]

        # Mock capability providers
        lsp_capability = Mock()
        git_capability = Mock()

        capability_providers = {
            "coding-lsp": lsp_capability,
            "coding-git": git_capability,
        }
        registry.get_capability_providers.return_value = capability_providers
        registry.get_capability_provider.side_effect = lambda name: capability_providers.get(name)

        # Mock validators
        validators = {
            "test-validator": lambda x: True,
        }
        registry.get_validators.return_value = validators
        registry.get_validator.side_effect = lambda name: validators.get(name)

        # Mock list methods
        registry.list_capability_names.return_value = list(capability_providers.keys())
        registry.list_validator_names.return_value = list(validators.keys())

        return registry

    def test_get_registry_with_mock_sdk(self, mock_registry):
        """Test get_sdk_protocol_registry with mocked SDK."""
        with patch(
            "victor.core.verticals.sdk_discovery.get_global_registry",
            return_value=mock_registry,
        ):
            registry = get_sdk_protocol_registry()
            assert registry is not None
            assert registry == mock_registry

    def test_discover_protocols_with_mock_sdk(self, mock_registry):
        """Test discover_sdk_protocols with mocked SDK."""
        with patch(
            "victor.core.verticals.sdk_discovery.get_global_registry",
            return_value=mock_registry,
        ):
            stats = discover_sdk_protocols()
            assert stats.total_protocols == 20
            assert stats.total_capabilities == 10

    def test_get_tool_providers_with_mock_sdk(self, mock_registry):
        """Test get_sdk_tool_providers with mocked SDK."""
        with patch(
            "victor.core.verticals.sdk_discovery.get_global_registry",
            return_value=mock_registry,
        ):
            providers = get_sdk_tool_providers()
            assert len(providers) == 1
            assert providers[0].get_tools() == ["read", "write", "edit"]

    def test_get_safety_providers_with_mock_sdk(self, mock_registry):
        """Test get_sdk_safety_providers with mocked SDK."""
        with patch(
            "victor.core.verticals.sdk_discovery.get_global_registry",
            return_value=mock_registry,
        ):
            providers = get_sdk_safety_providers()
            assert len(providers) == 1

    def test_get_workflow_providers_with_mock_sdk(self, mock_registry):
        """Test get_sdk_workflow_providers with mocked SDK."""
        with patch(
            "victor.core.verticals.sdk_discovery.get_global_registry",
            return_value=mock_registry,
        ):
            providers = get_sdk_workflow_providers()
            assert len(providers) == 1

    def test_get_prompt_providers_with_mock_sdk(self, mock_registry):
        """Test get_sdk_prompt_providers with mocked SDK."""
        with patch(
            "victor.core.verticals.sdk_discovery.get_global_registry",
            return_value=mock_registry,
        ):
            providers = get_sdk_prompt_providers()
            assert len(providers) == 1

    def test_get_capability_providers_with_mock_sdk(self, mock_registry):
        """Test get_sdk_capability_providers with mocked SDK."""
        with patch(
            "victor.core.verticals.sdk_discovery.get_global_registry",
            return_value=mock_registry,
        ):
            providers = get_sdk_capability_providers()
            assert "coding-lsp" in providers
            assert "coding-git" in providers
            assert len(providers) == 2

    def test_get_specific_capability_provider_with_mock_sdk(self, mock_registry):
        """Test get_sdk_capability_provider with mocked SDK."""
        with patch(
            "victor.core.verticals.sdk_discovery.get_global_registry",
            return_value=mock_registry,
        ):
            lsp = get_sdk_capability_provider("coding-lsp")
            git = get_sdk_capability_provider("coding-git")
            unknown = get_sdk_capability_provider("unknown")

            assert lsp is not None
            assert git is not None
            assert unknown is None

    def test_get_validators_with_mock_sdk(self, mock_registry):
        """Test get_sdk_validators with mocked SDK."""
        with patch(
            "victor.core.verticals.sdk_discovery.get_global_registry",
            return_value=mock_registry,
        ):
            validators = get_sdk_validators()
            assert "test-validator" in validators
            assert validators["test-validator"]("test") is True

    def test_get_specific_validator_with_mock_sdk(self, mock_registry):
        """Test get_sdk_validator with mocked SDK."""
        with patch(
            "victor.core.verticals.sdk_discovery.get_global_registry",
            return_value=mock_registry,
        ):
            validator = get_sdk_validator("test-validator")
            assert validator is not None
            assert validator("test") is True

    def test_list_capabilities_with_mock_sdk(self, mock_registry):
        """Test list_sdk_capabilities with mocked SDK."""
        with patch(
            "victor.core.verticals.sdk_discovery.get_global_registry",
            return_value=mock_registry,
        ):
            capabilities = list_sdk_capabilities()
            assert "coding-lsp" in capabilities
            assert "coding-git" in capabilities

    def test_list_validators_with_mock_sdk(self, mock_registry):
        """Test list_sdk_validators with mocked SDK."""
        with patch(
            "victor.core.verticals.sdk_discovery.get_global_registry",
            return_value=mock_registry,
        ):
            validators = list_sdk_validators()
            assert "test-validator" in validators

    def test_get_discovery_stats_with_mock_sdk(self, mock_registry):
        """Test get_sdk_discovery_stats with mocked SDK."""
        with patch(
            "victor.core.verticals.sdk_discovery.get_global_registry",
            return_value=mock_registry,
        ):
            stats = get_sdk_discovery_stats()
            assert stats.total_verticals == 5
            assert stats.total_protocols == 20
            assert stats.total_capabilities == 10
            assert stats.total_validators == 3
            assert stats.failed_loads == 0

    def test_enhance_vertical_with_sdk_protocols(self, mock_registry):
        """Test enhance_vertical_with_sdk_protocols with mocked SDK."""
        mock_extensions = Mock()

        with patch(
            "victor.core.verticals.sdk_discovery.get_global_registry",
            return_value=mock_registry,
        ):
            enhance_vertical_with_sdk_protocols("coding", mock_extensions)

            # Verify that providers were added to extensions
            if hasattr(mock_extensions, "add_tool_provider"):
                assert mock_extensions.add_tool_provider.called
            if hasattr(mock_extensions, "add_safety_provider"):
                assert mock_extensions.add_safety_provider.called
            if hasattr(mock_extensions, "add_workflow_provider"):
                assert mock_extensions.add_workflow_provider.called


class TestSDKDiscoveryGracefulDegradation:
    """Test that SDK discovery degrades gracefully when SDK is not available."""

    def test_discovery_summary_when_sdk_not_available(self):
        """Test get_sdk_discovery_summary when SDK not available."""
        with patch("victor.core.verticals.sdk_discovery.get_discovery_summary", None):
            summary = get_sdk_discovery_summary()
            assert "victor-contracts not installed" in summary

    def test_reload_when_sdk_not_available(self):
        """Test reload_sdk_discovery when SDK not available."""
        with patch("victor.core.verticals.sdk_discovery.reload_discovery", None):
            stats = reload_sdk_discovery()
            assert isinstance(stats, DiscoveryStats)
            assert stats.total_protocols == 0

    def test_reset_when_sdk_not_available(self):
        """Test reset_sdk_discovery when SDK not available."""
        with patch("victor.core.verticals.sdk_discovery.reset_global_registry", None):
            # Should not raise an exception
            reset_sdk_discovery()


class TestProtocolProviderInterface:
    """Test that protocol providers implement the expected interface."""

    def test_tool_provider_interface(self):
        """Test that ToolProvider has get_tools method."""
        from victor_contracts.verticals.protocols import ToolProvider

        # Check that ToolProvider is a runtime protocol
        assert hasattr(ToolProvider, "_is_protocol")
        assert hasattr(ToolProvider, "_is_runtime_protocol")
        assert ToolProvider._is_protocol
        assert ToolProvider._is_runtime_protocol
        # Check that get_tools is a protocol member
        assert hasattr(ToolProvider, "get_tools")

    def test_safety_provider_interface(self):
        """Test that SafetyProvider has required methods."""
        from victor_contracts.verticals.protocols import SafetyProvider

        # Check that SafetyProvider is a runtime protocol
        assert hasattr(SafetyProvider, "_is_protocol")
        assert hasattr(SafetyProvider, "_is_runtime_protocol")
        assert SafetyProvider._is_protocol
        assert SafetyProvider._is_runtime_protocol
        # SafetyProvider has: get_safety_rules, validate_prompt, validate_tool_call
        assert hasattr(SafetyProvider, "get_safety_rules")

    def test_workflow_provider_interface(self):
        """Test that WorkflowProvider has required methods."""
        from victor_contracts.verticals.protocols import WorkflowProvider

        # Check that WorkflowProvider is a runtime protocol
        assert hasattr(WorkflowProvider, "_is_protocol")
        assert hasattr(WorkflowProvider, "_is_runtime_protocol")
        assert WorkflowProvider._is_protocol
        assert WorkflowProvider._is_runtime_protocol
        # WorkflowProvider has: get_initial_stage, get_stage_definitions, get_workflow_spec
        assert hasattr(WorkflowProvider, "get_workflow_spec")

    def test_prompt_provider_interface(self):
        """Test that PromptProvider has required methods."""
        from victor_contracts.verticals.protocols import PromptProvider

        # Check that PromptProvider is a runtime protocol
        assert hasattr(PromptProvider, "_is_protocol")
        assert hasattr(PromptProvider, "_is_runtime_protocol")
        assert PromptProvider._is_protocol
        assert PromptProvider._is_runtime_protocol
        # PromptProvider has: format_prompt, get_base_prompt, get_prompt_template
        assert hasattr(PromptProvider, "get_base_prompt")
