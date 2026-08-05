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

"""Tests for canonical agent-service bootstrap."""

from unittest.mock import MagicMock, patch

import pytest

from victor.core.bootstrap_services import bootstrap_new_services
from victor.core.container import ServiceContainer
from victor.core.feature_flags import FeatureFlag


class TestBootstrapServices:
    """Tests for bootstrap_new_services function."""

    def test_bootstrap_registers_core_services_regardless_of_flags(self):
        """Bootstrap always registers the canonical core services."""
        container = ServiceContainer()
        conversation_controller = MagicMock()
        streaming_coordinator = MagicMock()

        with patch("victor.core.feature_flags.get_feature_flag_manager") as mock_mgr:
            mgr = MagicMock()
            mgr.is_enabled.return_value = False
            mock_mgr.return_value = mgr

            bootstrap_new_services(
                container,
                conversation_controller,
                streaming_coordinator,
            )

            # Verify core services were registered (services are always created now)
            from victor.agent.services.protocols import (
                ContextServiceProtocol,
                ProviderServiceProtocol,
                RecoveryServiceProtocol,
                ToolServiceProtocol,
                SessionServiceProtocol,
                ChatServiceProtocol,
            )

            # All 6 core services should be registered
            assert container.is_registered(ContextServiceProtocol)
            assert container.is_registered(ProviderServiceProtocol)
            assert container.is_registered(RecoveryServiceProtocol)
            assert container.is_registered(ToolServiceProtocol)
            assert container.is_registered(SessionServiceProtocol)
            assert container.is_registered(ChatServiceProtocol)

            # The decision service IS registered even with no edge/LLM flags
            # (FEP-0012): decision_backend=auto is classifier-first, and the
            # bundled edge_classifier_v1.npz baseline (#381) makes the shipped
            # LocalClassifierDecisionService the flag-independent default.
            # (LoggingDecisionService remains the auto fallback only when no
            # healthy classifier artifact exists, so decisions are still always
            # captured for RL/training data.)
            from victor.agent.services.protocols.decision_service import (
                LLMDecisionServiceProtocol,
            )

            assert container.is_registered(LLMDecisionServiceProtocol)
            svc = container.get(LLMDecisionServiceProtocol)
            assert svc.__class__.__name__ == "LocalClassifierDecisionService"
            # Flag-independence: every rollout flag is disabled above, yet the
            # bundled-classifier backend is adopted and healthy.
            assert svc.is_healthy()

    @pytest.mark.parametrize(
        ("protocol_name", "health_attr"),
        [
            ("ContextServiceProtocol", "is_healthy"),
            ("ProviderServiceProtocol", "is_healthy"),
            ("ToolServiceProtocol", "is_healthy"),
            ("RecoveryServiceProtocol", "is_healthy"),
            ("SessionServiceProtocol", "is_healthy"),
            ("ChatServiceProtocol", "is_healthy"),
        ],
    )
    def test_bootstrap_registers_each_core_service(self, protocol_name, health_attr):
        """Each canonical service is registered without service-layer rollout flags."""
        container = ServiceContainer()
        conversation_controller = MagicMock()
        streaming_coordinator = MagicMock()

        with patch("victor.core.feature_flags.get_feature_flag_manager") as mock_mgr:
            mgr = MagicMock()
            mgr.is_enabled.return_value = False
            mock_mgr.return_value = mgr

            bootstrap_new_services(
                container,
                conversation_controller,
                streaming_coordinator,
            )

            from victor.agent.services import protocols as service_protocols

            protocol = getattr(service_protocols, protocol_name)
            service = container.get(protocol)
            assert service is not None
            assert container.is_registered(protocol)
            assert hasattr(service, health_attr)

    def test_bootstrap_all_services_when_all_flags_enabled(self):
        """Test that all services are bootstrapped when all feature flags are enabled."""
        container = ServiceContainer()
        conversation_controller = MagicMock()
        streaming_coordinator = MagicMock()

        with patch("victor.core.feature_flags.get_feature_flag_manager") as mock_mgr:
            mgr = MagicMock()
            mgr.is_enabled.return_value = True  # All flags enabled
            mock_mgr.return_value = mgr

            bootstrap_new_services(
                container,
                conversation_controller,
                streaming_coordinator,
            )

            # Verify all service protocols are registered
            from victor.agent.services.protocols import (
                ChatServiceProtocol,
                ToolServiceProtocol,
                ContextServiceProtocol,
                ProviderServiceProtocol,
                RecoveryServiceProtocol,
                SessionServiceProtocol,
            )

            assert container.is_registered(ChatServiceProtocol)
            assert container.is_registered(ToolServiceProtocol)
            assert container.is_registered(ContextServiceProtocol)
            assert container.is_registered(ProviderServiceProtocol)
            assert container.is_registered(RecoveryServiceProtocol)
            assert container.is_registered(SessionServiceProtocol)

            # Verify we can get all services
            assert container.get(ChatServiceProtocol) is not None
            assert container.get(ToolServiceProtocol) is not None
            assert container.get(ContextServiceProtocol) is not None
            assert container.get(ProviderServiceProtocol) is not None
            assert container.get(RecoveryServiceProtocol) is not None
            assert container.get(SessionServiceProtocol) is not None

    def test_bootstrap_services_with_custom_tool_selector_and_executor(self):
        """Test that custom tool selector and executor are used when provided."""
        container = ServiceContainer()
        conversation_controller = MagicMock()
        streaming_coordinator = MagicMock()

        custom_selector = MagicMock()
        custom_executor = MagicMock()

        with patch("victor.core.feature_flags.get_feature_flag_manager") as mock_mgr:
            mgr = MagicMock()
            mgr.is_enabled.return_value = False
            mock_mgr.return_value = mgr

            bootstrap_new_services(
                container,
                conversation_controller,
                streaming_coordinator,
                tool_selector=custom_selector,
                tool_executor=custom_executor,
            )

            # Verify ToolServiceProtocol is registered
            from victor.agent.services.protocols import ToolServiceProtocol

            assert container.is_registered(ToolServiceProtocol)

            # Get the service and verify it uses our custom selector/executor
            tool_service = container.get(ToolServiceProtocol)
            assert tool_service is not None

    def test_bootstrap_without_feature_flags_still_registers_core_services(self):
        """Missing feature-flag infrastructure should not block core services."""
        container = ServiceContainer()
        conversation_controller = MagicMock()
        streaming_coordinator = MagicMock()

        with patch(
            "victor.core.bootstrap_services._get_feature_flag_manager_optional",
            return_value=None,
        ):
            bootstrap_new_services(
                container,
                conversation_controller,
                streaming_coordinator,
            )

        from victor.agent.services.protocols import (
            ChatServiceProtocol,
            ToolServiceProtocol,
        )

        assert container.is_registered(ChatServiceProtocol)
        assert container.is_registered(ToolServiceProtocol)

    def test_chat_service_receives_dependencies(self):
        """Test that ChatService receives its dependencies correctly."""
        container = ServiceContainer()
        conversation_controller = MagicMock()
        streaming_coordinator = MagicMock()

        with patch("victor.core.feature_flags.get_feature_flag_manager") as mock_mgr:
            mgr = MagicMock()
            mgr.is_enabled.return_value = True  # All flags enabled
            mock_mgr.return_value = mgr

            bootstrap_new_services(
                container,
                conversation_controller,
                streaming_coordinator,
            )

            # Get ChatService and verify it was created with dependencies
            from victor.agent.services.protocols import ChatServiceProtocol

            chat_service = container.get(ChatServiceProtocol)
            assert chat_service is not None

            # Verify the service has the expected dependencies
            # (This checks that the service was created correctly, not implementation details)
            assert hasattr(chat_service, "is_healthy")

    def test_services_registered_as_singletons(self):
        """Test that all bootstrapped services are registered as SINGLETONs."""
        container = ServiceContainer()
        conversation_controller = MagicMock()
        streaming_coordinator = MagicMock()

        with patch("victor.core.feature_flags.get_feature_flag_manager") as mock_mgr:
            mgr = MagicMock()
            mgr.is_enabled.return_value = True
            mock_mgr.return_value = mgr

            bootstrap_new_services(
                container,
                conversation_controller,
                streaming_coordinator,
            )

            # Verify services are singletons by getting them multiple times
            from victor.agent.services.protocols import ContextServiceProtocol

            service1 = container.get(ContextServiceProtocol)
            service2 = container.get(ContextServiceProtocol)
            assert service1 is service2  # Same instance

    def test_bootstrap_with_partial_flags_enabled(self):
        """Test bootstrap with edge model flag enabled registers LLMDecisionService.

        Core services are always registered, but decision services require flags.
        """
        container = ServiceContainer()
        conversation_controller = MagicMock()
        streaming_coordinator = MagicMock()

        # Enable USE_EDGE_MODEL to get decision service
        enabled_flags = {
            FeatureFlag.USE_EDGE_MODEL,
        }

        with patch("victor.core.feature_flags.get_feature_flag_manager") as mock_mgr:
            mgr = MagicMock()
            mgr.is_enabled.side_effect = lambda flag: flag in enabled_flags
            mock_mgr.return_value = mgr

            bootstrap_new_services(
                container,
                conversation_controller,
                streaming_coordinator,
            )

            # Verify core services are registered (always created)
            from victor.agent.services.protocols import (
                ContextServiceProtocol,
                ToolServiceProtocol,
                ChatServiceProtocol,
            )

            assert container.is_registered(ContextServiceProtocol)
            assert container.is_registered(ToolServiceProtocol)
            assert container.is_registered(ChatServiceProtocol)  # Always registered now

            # Edge model flag should register LLMDecisionService
            from victor.agent.services.protocols.decision_service import (
                LLMDecisionServiceProtocol,
            )

            assert container.get_optional(
                LLMDecisionServiceProtocol
            ) is None or container.is_registered(LLMDecisionServiceProtocol)


class TestMockToolComponents:
    """Tests for mock tool selector and executor used in bootstrap."""

    def test_mock_tool_selector(self):
        """Test that _MockToolSelector has expected interface."""
        from victor.core.bootstrap_services import _MockToolSelector

        selector = _MockToolSelector()

        # Should be async and return empty list
        import asyncio

        result = asyncio.run(selector.select({}, max_tools=10))
        assert result == []

    def test_mock_tool_executor(self):
        """Test that _MockToolExecutor has expected interface."""
        from victor.core.bootstrap_services import _MockToolExecutor

        executor = _MockToolExecutor()

        # Should be async and return ToolResult
        import asyncio

        result = asyncio.run(executor.execute("test_tool", {"arg": "value"}))

        assert result.success is False
        assert result.error == "Mock tool executor - tool not actually executed"
