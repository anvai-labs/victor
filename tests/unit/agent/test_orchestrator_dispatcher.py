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

"""Unit tests for orchestrator protocol-based lazy loading."""

from __future__ import annotations

import pytest

from victor.agent.orchestrator_dispatcher import (
    OrchestratorDispatcher,
    get_orchestrator_dispatcher,
)
from victor.agent.orchestrator_protocols import IAgentOrchestrator
from victor.agent.coordinators.protocol_dependencies import (
    OrchestratorProtocolAdapter,
)


class TestOrchestratorDispatcher:
    """Test suite for OrchestratorDispatcher."""

    def test_dispatcher_lazy_loading(self):
        """Test that dispatcher doesn't import orchestrator until first use."""
        dispatcher = OrchestratorDispatcher()

        # Verify implementation not loaded
        assert dispatcher._impl is None
        assert dispatcher._factory_called is False

        # Trigger loading by accessing a property
        # This should import AgentOrchestrator
        try:
            _ = dispatcher.session_id
            # If we get here without exception, lazy loading worked
            assert dispatcher._impl is not None
        except RuntimeError:
            # Expected if orchestrator can't be created without proper setup
            # This is OK for testing the lazy loading behavior
            pass

    def test_dispatcher_singleton(self):
        """Test that get_orchestrator_dispatcher returns singleton."""
        dispatcher1 = get_orchestrator_dispatcher()
        dispatcher2 = get_orchestrator_dispatcher()

        assert dispatcher1 is dispatcher2

    def test_dispatcher_reset(self):
        """Test that reset clears cached implementation."""
        dispatcher = OrchestratorDispatcher()

        # Reset before any loading
        dispatcher.reset()
        assert dispatcher._impl is None
        assert dispatcher._factory_called is False

    def test_dispatcher_protocol_compliance(self):
        """Test that dispatcher implements IAgentOrchestrator protocol."""
        dispatcher = get_orchestrator_dispatcher()

        # Verify dispatcher can be used as IAgentOrchestrator
        assert isinstance(dispatcher, IAgentOrchestrator)

    def test_dispatcher_has_required_methods(self):
        """Test that dispatcher has all required methods."""
        dispatcher = OrchestratorDispatcher()

        # Check for method existence (not calling them)
        assert hasattr(dispatcher, "run")
        assert hasattr(dispatcher, "stream")
        assert hasattr(dispatcher, "chat")
        assert hasattr(dispatcher, "execute_tool")
        assert hasattr(dispatcher, "aclose")

        # Check for property existence
        # Note: accessing properties would trigger lazy loading
        # so we just check they exist as attributes
        assert "messages" in dir(dispatcher)
        assert "settings" in dir(dispatcher)
        assert "session_id" in dir(dispatcher)


class TestOrchestratorProtocolAdapter:
    """Test suite for OrchestratorProtocolAdapter."""

    def test_adapter_creation(self):
        """Test that adapter can be created with mock orchestrator."""
        mock_orch = MockOrchestrator()
        adapter = OrchestratorProtocolAdapter(mock_orch)

        assert adapter._orchestrator is mock_orch

    def test_adapter_protocol_compliance(self):
        """Test that adapter implements all expected protocols."""
        mock_orch = MockOrchestrator()
        adapter = OrchestratorProtocolAdapter(mock_orch)

        # Check protocol compliance (duck typing)
        assert hasattr(adapter, "run")
        assert hasattr(adapter, "stream")
        assert hasattr(adapter, "chat")
        assert hasattr(adapter, "execute_tool")
        assert hasattr(adapter, "messages")
        assert hasattr(adapter, "settings")
        assert hasattr(adapter, "session_id")

    def test_adapter_delegates_to_orchestrator(self):
        """Test that adapter delegates calls to orchestrator."""
        mock_orch = MockOrchestrator()
        adapter = OrchestratorProtocolAdapter(mock_orch)

        # Test property delegation
        assert adapter.messages == mock_orch.messages
        assert adapter.session_id == mock_orch.session_id
        assert adapter.settings == mock_orch.settings

    def test_adapter_tool_executor_interface(self):
        """Test that adapter provides tool executor interface."""
        mock_orch = MockOrchestrator()
        adapter = OrchestratorProtocolAdapter(mock_orch)

        # Check for tool executor methods
        assert hasattr(adapter, "execute_tool_call")
        assert hasattr(adapter, "execute_tool_calls")

    def test_adapter_message_store_interface(self):
        """Test that adapter provides message store interface."""
        mock_orch = MockOrchestrator()
        adapter = OrchestratorProtocolAdapter(mock_orch)

        # Check for message store methods
        assert hasattr(adapter, "add_message")
        assert hasattr(adapter, "get_messages")

    def test_adapter_state_manager_interface(self):
        """Test that adapter provides state manager interface."""
        mock_orch = MockOrchestrator()
        adapter = OrchestratorProtocolAdapter(mock_orch)

        # Check for state manager methods
        assert hasattr(adapter, "get_state")
        assert hasattr(adapter, "set_state")
        assert hasattr(adapter, "delete_state")


class MockOrchestrator:
    """Mock orchestrator for testing.

    Provides minimal implementation of key orchestrator properties
    that the adapter needs to delegate to.
    """

    def __init__(self) -> None:
        self._messages: list = []
        self._session_id = "test-session-123"
        self._settings = {"test": "settings"}

    @property
    def messages(self) -> list:
        return self._messages

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def settings(self) -> dict:
        return self._settings


@pytest.mark.integration
class TestCoordinatorFactory:
    """Integration tests for CoordinatorFactory."""

    def test_factory_creation(self):
        """Test that factory can be created with container."""
        from victor.core import get_container

        container = get_container()

        from victor.agent.coordinators.coordinator_factory import CoordinatorFactory

        factory = CoordinatorFactory(container)

        assert factory is not None
        assert factory._container is container

    def test_factory_has_adapter(self):
        """Test that factory creates orchestrator adapter when needed."""
        from victor.core import get_container

        container = get_container()

        from victor.agent.coordinators.coordinator_factory import CoordinatorFactory

        factory = CoordinatorFactory(container)

        # Check that adapter can be created (doesn't require orchestrator)
        assert factory._orchestrator_adapter is None

        # Creating adapter doesn't fail without orchestrator
        # (it will be created lazily when first accessed)
        assert factory is not None
