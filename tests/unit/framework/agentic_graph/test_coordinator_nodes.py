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

"""Tests for coordinator adapter nodes (Phase 2 consolidation)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from victor.framework.agentic_graph.state import (
    create_initial_state,
    AgenticLoopStateModel,
)
from victor.framework.agentic_graph.coordinator_nodes import (
    _create_context_snapshot,
    _apply_transitions_to_state,
    CoordinatorAdapter,
    exploration_node,
    safety_node,
    system_prompt_node,
)


class TestContextSnapshotCreation:
    """Tests for _create_context_snapshot helper."""

    def test_create_snapshot_from_state(self):
        """Test creating ContextSnapshot from AgenticLoopStateModel."""
        state = AgenticLoopStateModel(
            query="Test",
            context={
                "task_type": "code_generation",
                "provider": "test",
                "model": "test-model",
            },
        )

        snapshot = _create_context_snapshot(state)

        assert snapshot is not None
        assert snapshot.provider == "test"
        assert snapshot.model == "test-model"

    def test_create_snapshot_with_orchestrator(self):
        """Test creating snapshot with orchestrator."""
        state = AgenticLoopStateModel(query="Test")
        mock_orchestrator = MagicMock()
        mock_orchestrator.messages = []
        mock_orchestrator.session_id = "test-session"
        mock_orchestrator.conversation_stage = "initial"
        mock_orchestrator.settings = None
        mock_orchestrator.model = "test-model"
        mock_orchestrator.provider_name = "test-provider"
        mock_orchestrator.max_tokens = 4096
        mock_orchestrator.temperature = 0.7
        mock_orchestrator.conversation_state = {}
        mock_orchestrator.session_state = {}
        mock_orchestrator.observed_files = []
        mock_orchestrator._capabilities = {}

        with patch("victor.agent.coordinators.state_context.create_snapshot") as mock_create:
            mock_create.return_value = MagicMock()
            snapshot = _create_context_snapshot(state, mock_orchestrator)
            mock_create.assert_called_once_with(mock_orchestrator)


class TestTransitionApplication:
    """Tests for _apply_transitions_to_state helper."""

    def test_apply_update_state_transition(self):
        """Test applying UPDATE_STATE transition."""
        from victor.agent.coordinators.state_context import TransitionType

        state = AgenticLoopStateModel(query="Test")

        # Create mock transition batch
        mock_batch = MagicMock()
        mock_transition = MagicMock()
        mock_transition.transition_type = TransitionType.UPDATE_STATE
        mock_transition.data = {
            "key": "task_type",
            "value": "debugging",
            "scope": "conversation",
        }
        mock_batch.transitions = [mock_transition]

        result = _apply_transitions_to_state(state, mock_batch)

        assert result.context.get("task_type") == "debugging"

    def test_apply_session_state_transition(self):
        """Test applying session-scoped transition."""
        from victor.agent.coordinators.state_context import TransitionType

        state = AgenticLoopStateModel(query="Test")

        mock_batch = MagicMock()
        mock_transition = MagicMock()
        mock_transition.transition_type = TransitionType.UPDATE_STATE
        mock_transition.data = {
            "key": "session_key",
            "value": "session_value",
            "scope": "session",
        }
        mock_batch.transitions = [mock_transition]

        result = _apply_transitions_to_state(state, mock_batch)

        assert result.context.get("session_state", {}).get("session_key") == "session_value"


class TestCoordinatorAdapter:
    """Tests for CoordinatorAdapter class."""

    @pytest.mark.asyncio
    async def test_adapter_call_success(self):
        """Test successful coordinator call through adapter."""
        # Mock coordinator
        mock_coordinator = AsyncMock()
        mock_result = MagicMock()
        mock_result.transitions = MagicMock()
        mock_result.transitions.transitions = []
        mock_result.metadata = {"test": "data"}
        mock_coordinator.my_method = AsyncMock(return_value=mock_result)

        adapter = CoordinatorAdapter(mock_coordinator)
        state = AgenticLoopStateModel(query="Test")

        result = await adapter.call(state, "my_method", user_message="Test")

        assert isinstance(result, AgenticLoopStateModel)
        mock_coordinator.my_method.assert_called_once()

    @pytest.mark.asyncio
    async def test_adapter_with_transitions(self):
        """Test adapter applies transitions to state."""
        # Mock coordinator with transitions
        mock_coordinator = AsyncMock()

        # Create mock transition
        from victor.agent.coordinators.state_context import (
            TransitionBatch,
            StateTransition,
            TransitionType,
        )

        batch = TransitionBatch()
        batch.update_state("test_key", "test_value", "conversation")
        mock_result = MagicMock()
        mock_result.transitions = batch
        mock_result.metadata = {}
        mock_coordinator.my_method = AsyncMock(return_value=mock_result)

        adapter = CoordinatorAdapter(mock_coordinator)
        state = AgenticLoopStateModel(query="Test")

        result = await adapter.call(state, "my_method")

        assert result.context.get("test_key") == "test_value"


class TestExplorationNode:
    """Tests for exploration_node."""

    @pytest.mark.asyncio
    async def test_exploration_node_without_coordinator(self):
        """Test exploration node without coordinator (graceful skip)."""
        state = create_initial_state(query="Find all Python files")

        with patch(
            "victor.agent.coordinators.exploration_state_passed.ExplorationStatePassedCoordinator",
            side_effect=ImportError,
        ):
            result = await exploration_node(state)
            assert result.query == "Find all Python files"

    @pytest.mark.asyncio
    async def test_exploration_node_with_mock_coordinator(self):
        """Test exploration node with mock coordinator."""
        state = create_initial_state(query="Explore codebase")

        mock_coordinator = AsyncMock()
        mock_result = MagicMock()
        mock_result.transitions = MagicMock()
        mock_result.transitions.transitions = []
        mock_result.metadata = {}
        mock_coordinator.explore = AsyncMock(return_value=mock_result)

        result = await exploration_node(state, exploration_coordinator=mock_coordinator)

        mock_coordinator.explore.assert_called_once()
        assert isinstance(result, AgenticLoopStateModel)

    @pytest.mark.asyncio
    async def test_exploration_node_empty_query(self):
        """Test exploration node skips empty queries."""
        state = AgenticLoopStateModel(query="")

        mock_coordinator = AsyncMock()
        result = await exploration_node(state, exploration_coordinator=mock_coordinator)

        mock_coordinator.explore.assert_not_called()


class TestSafetyNode:
    """Tests for safety_node."""

    @pytest.mark.asyncio
    async def test_safety_node_without_coordinator(self):
        """Test safety node without coordinator (graceful skip)."""
        state = create_initial_state(query="Run tests")
        state = state.model_copy(update={"plan": {"tool_calls": ["pytest"]}})

        with patch(
            "victor.agent.coordinators.safety_state_passed.SafetyStatePassedCoordinator",
            side_effect=ImportError,
        ):
            result = await safety_node(state)
            assert result.query == "Run tests"

    @pytest.mark.asyncio
    async def test_safety_node_with_mock_coordinator(self):
        """Test safety node with mock coordinator."""
        state = create_initial_state(query="Push code")
        state = state.model_copy(update={"plan": {"tool_calls": ["git", "push"]}})

        mock_coordinator = AsyncMock()
        mock_result = MagicMock()
        mock_result.transitions = MagicMock()
        mock_result.transitions.transitions = []
        mock_result.metadata = {"safety_check": {"is_safe": True}}
        mock_coordinator.check = AsyncMock(return_value=mock_result)

        result = await safety_node(state, safety_coordinator=mock_coordinator)

        mock_coordinator.check.assert_called()
        assert isinstance(result, AgenticLoopStateModel)

    @pytest.mark.asyncio
    async def test_safety_node_no_tool_calls(self):
        """Test safety node skips when no tool calls."""
        state = create_initial_state(query="Just chat")
        state = state.model_copy(update={"plan": {}})

        mock_coordinator = AsyncMock()
        result = await safety_node(state, safety_coordinator=mock_coordinator)

        mock_coordinator.check.assert_not_called()

    @pytest.mark.asyncio
    async def test_safety_node_records_error_and_fails_closed(self):
        """Safety node exceptions should not silently fail open."""
        state = create_initial_state(query="Push code")
        state = state.model_copy(update={"plan": {"tool_calls": ["git", "push"]}})

        mock_coordinator = AsyncMock()
        mock_coordinator.check = AsyncMock(side_effect=RuntimeError("boom"))

        result = await safety_node(state, safety_coordinator=mock_coordinator)

        assert result.context is not None
        assert result.context["all_tools_safe"] is False
        assert result.context["safety_error"] == "boom"


class TestSystemPromptNode:
    """Tests for system_prompt_node."""

    @pytest.mark.asyncio
    async def test_system_prompt_node_without_coordinator(self):
        """Test system prompt node without coordinator (graceful skip)."""
        state = create_initial_state(query="Write code")

        with patch(
            "victor.agent.coordinators.system_prompt_state_passed.SystemPromptStatePassedCoordinator",
            side_effect=ImportError,
        ):
            result = await system_prompt_node(state)
            assert result.query == "Write code"

    @pytest.mark.asyncio
    async def test_system_prompt_node_with_mock_coordinator(self):
        """Test system prompt node with mock coordinator."""
        state = create_initial_state(query="Debug code")

        mock_coordinator = AsyncMock()
        mock_result = MagicMock()
        mock_result.transitions = MagicMock()
        mock_result.transitions.transitions = []
        mock_result.metadata = {}
        mock_coordinator.classify = AsyncMock(return_value=mock_result)

        result = await system_prompt_node(state, system_prompt_coordinator=mock_coordinator)

        mock_coordinator.classify.assert_called_once()
        assert isinstance(result, AgenticLoopStateModel)

    @pytest.mark.asyncio
    async def test_system_prompt_node_with_orchestrator(self):
        """Test system prompt node with orchestrator task analyzer."""
        state = create_initial_state(query="Analyze code")

        mock_orchestrator = MagicMock()
        mock_task_analyzer = MagicMock()
        mock_orchestrator.task_analyzer = mock_task_analyzer

        with patch(
            "victor.agent.coordinators.system_prompt_state_passed.SystemPromptStatePassedCoordinator"
        ) as MockCoord:
            mock_coordinator = AsyncMock()
            mock_result = MagicMock()
            mock_result.transitions = MagicMock()
            mock_result.transitions.transitions = []
            mock_result.metadata = {}
            mock_coordinator.classify = AsyncMock(return_value=mock_result)
            MockCoord.return_value = mock_coordinator

            result = await system_prompt_node(state, orchestrator=mock_orchestrator)

            MockCoord.assert_called_once_with(mock_task_analyzer)

    @pytest.mark.asyncio
    async def test_system_prompt_node_empty_query(self):
        """Test system prompt node skips empty queries."""
        state = AgenticLoopStateModel(query="")

        mock_coordinator = AsyncMock()
        result = await system_prompt_node(state, system_prompt_coordinator=mock_coordinator)

        mock_coordinator.classify.assert_not_called()


class TestNodeIntegration:
    """Integration tests for coordinator nodes."""

    @pytest.mark.asyncio
    async def test_exploration_to_safety_flow(self):
        """Test flow from exploration to safety node."""
        state = create_initial_state(query="Explore and check safety")

        # Mock exploration
        mock_exploration = AsyncMock()
        mock_exploration_result = MagicMock()
        mock_exploration_result.transitions = MagicMock()
        mock_exploration_result.transitions.transitions = []
        mock_exploration_result.metadata = {"files_found": 5}
        mock_exploration.explore = AsyncMock(return_value=mock_exploration_result)

        # Run exploration
        state = await exploration_node(state, exploration_coordinator=mock_exploration)

        # Mock safety
        mock_safety = AsyncMock()
        mock_safety_result = MagicMock()
        mock_safety_result.transitions = MagicMock()
        mock_safety_result.transitions.transitions = []
        mock_safety_result.metadata = {}
        mock_safety.check = AsyncMock(return_value=mock_safety_result)

        # Add single tool call to state for safety check
        state = state.model_copy(
            update={"plan": {"tool_calls": [{"name": "git", "arguments": ["status"]}]}}
        )
        state = await safety_node(state, safety_coordinator=mock_safety)

        # Verify both ran
        mock_exploration.explore.assert_called_once()
        mock_safety.check.assert_called_once()

    @pytest.mark.asyncio
    async def test_all_three_nodes_sequence(self):
        """Test running all three coordinator nodes in sequence."""
        state = create_initial_state(query="Full coordinator flow")

        # Mock system prompt
        mock_sysprompt = AsyncMock()
        mock_sysprompt_result = MagicMock()
        mock_sysprompt_result.transitions = MagicMock()
        mock_sysprompt_result.transitions.transitions = []
        mock_sysprompt_result.metadata = {}
        mock_sysprompt.classify = AsyncMock(return_value=mock_sysprompt_result)

        state = await system_prompt_node(state, system_prompt_coordinator=mock_sysprompt)

        # Mock exploration
        mock_exploration = AsyncMock()
        mock_exploration_result = MagicMock()
        mock_exploration_result.transitions = MagicMock()
        mock_exploration_result.transitions.transitions = []
        mock_exploration_result.metadata = {}
        mock_exploration.explore = AsyncMock(return_value=mock_exploration_result)

        state = await exploration_node(state, exploration_coordinator=mock_exploration)

        # Mock safety
        mock_safety = AsyncMock()
        mock_safety_result = MagicMock()
        mock_safety_result.transitions = MagicMock()
        mock_safety_result.transitions.transitions = []
        mock_safety_result.metadata = {"safety_check": {"is_safe": True}}
        mock_safety.check = AsyncMock(return_value=mock_safety_result)

        state = state.model_copy(update={"plan": {"tool_calls": ["read_file"]}})
        state = await safety_node(state, safety_coordinator=mock_safety)

        # Verify all ran
        mock_sysprompt.classify.assert_called_once()
        mock_exploration.explore.assert_called_once()
        mock_safety.check.assert_called_once()
