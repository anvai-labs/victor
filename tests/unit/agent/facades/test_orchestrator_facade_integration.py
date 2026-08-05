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

"""Integration tests: orchestrator creates facades and delegates through them."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from victor.agent.coordinators.coordination_state_passed import (
    CoordinationStatePassedCoordinator,
)
from victor.agent.coordinators.exploration_state_passed import (
    ExplorationStatePassedCoordinator,
)
from victor.agent.coordinators.safety_state_passed import SafetyStatePassedCoordinator
from victor.agent.coordinators.system_prompt_state_passed import (
    SystemPromptStatePassedCoordinator,
)
from victor.agent.runtime.provider_runtime import LazyRuntimeProxy
from victor.agent.orchestrator import AgentOrchestrator
from victor.config.settings import Settings


@pytest.fixture
def mock_provider():
    """Create a mock provider."""
    provider = MagicMock()
    provider.name = "mock_provider"
    provider.supports_tools.return_value = True
    provider.get_context_window.return_value = 100000
    provider.chat = AsyncMock(return_value=MagicMock(content="Response", tool_calls=[]))
    return provider


@pytest.fixture
def orchestrator_settings():
    """Create settings for testing."""
    return Settings(
        analytics_enabled=False,
        use_semantic_tool_selection=False,
        use_mcp_tools=False,
    )


@pytest.fixture
def orchestrator(mock_provider, orchestrator_settings):
    """Create an orchestrator for testing."""
    with (
        patch("victor.agent.orchestrator.UsageLogger"),
        patch("victor.core.bootstrap_services.bootstrap_new_services"),
    ):
        return AgentOrchestrator(
            settings=orchestrator_settings,
            provider=mock_provider,
            model="test-model",
        )


class TestFacadeCreation:
    """Tests that orchestrator creates facade instances."""

    def test_orchestration_facade_has_state_passed_handles(self, orchestrator):
        """OrchestrationFacade should expose state-passed coordinator surfaces."""
        assert isinstance(
            orchestrator._orchestration_facade.exploration_state_passed,
            ExplorationStatePassedCoordinator,
        )
        assert isinstance(
            orchestrator._orchestration_facade.system_prompt_state_passed,
            SystemPromptStatePassedCoordinator,
        )
        assert isinstance(
            orchestrator._orchestration_facade.safety_state_passed,
            SafetyStatePassedCoordinator,
        )
        assert isinstance(
            orchestrator._orchestration_facade.coordination_state_passed,
            CoordinationStatePassedCoordinator,
        )

    def test_orchestration_facade_no_longer_exposes_removed_coordinator_shims(self, orchestrator):
        """Removed deprecated coordinator properties should stay absent."""
        facade = orchestrator._orchestration_facade

        for attr in (
            "chat_coordinator",
            "tool_coordinator",
            "session_coordinator",
            "sync_chat_coordinator",
            "streaming_chat_coordinator",
            "unified_chat_coordinator",
        ):
            assert hasattr(facade, attr) is False


class TestBackwardCompatibility:
    """Tests that direct attribute access still works after facade introduction."""

    def test_direct_attribute_access_conversation(self, orchestrator):
        """Direct self.conversation still works."""
        assert orchestrator.conversation is not None

    def test_direct_attribute_access_tools(self, orchestrator):
        """Direct self.tools still works."""
        assert orchestrator.tools is not None

    def test_direct_attribute_access_tool_executor(self, orchestrator):
        """Direct self.tool_executor still works."""
        assert orchestrator.tool_executor is not None

    def test_direct_attribute_access_tool_budget(self, orchestrator):
        """Direct self.tool_budget still works."""
        assert orchestrator.tool_budget > 0

    def test_direct_attribute_access_conversation_state(self, orchestrator):
        """Direct self.conversation_state still works."""
        assert orchestrator.conversation_state is not None

    def test_direct_attribute_access_tool_registrar(self, orchestrator):
        """Direct self.tool_registrar still works."""
        assert orchestrator.tool_registrar is not None

    def test_direct_attribute_access_argument_normalizer(self, orchestrator):
        """Direct self.argument_normalizer still works."""
        assert orchestrator.argument_normalizer is not None

    def test_direct_attribute_access_intent_detector(self, orchestrator):
        """Direct self.intent_detector still works."""
        assert orchestrator.intent_detector is not None
