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

"""Unit tests for state-passed architecture components."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from victor.agent.coordinators.state_context import (
    ContextSnapshot,
    StateTransition,
    TransitionBatch,
    CoordinatorResult,
    TransitionApplier,
    TransitionType,
    create_snapshot,
)


class TestContextSnapshot:
    """Tests for ContextSnapshot immutability and accessors."""

    def test_create_snapshot(self):
        """Test creating a basic snapshot."""
        snapshot = ContextSnapshot(
            messages=(),
            session_id="test-123",
            conversation_stage="initial",
            settings=MagicMock(),
            model="test-model",
            provider="test-provider",
            max_tokens=4096,
            temperature=0.7,
            conversation_state={},
            session_state={},
            observed_files=(),
            capabilities={},
        )

        assert snapshot.session_id == "test-123"
        assert snapshot.conversation_stage == "initial"
        assert snapshot.message_count == 0
        assert snapshot.is_complete is False

    def test_get_state_conversation_priority(self):
        """Test that conversation state takes priority over session state."""
        snapshot = ContextSnapshot(
            messages=(),
            session_id="test",
            conversation_stage="initial",
            settings=MagicMock(),
            model="test",
            provider="test",
            max_tokens=4096,
            temperature=0.7,
            conversation_state={"key": "conv_value"},
            session_state={"key": "sess_value"},
            observed_files=(),
            capabilities={},
        )

        # Conversation state should take priority
        assert snapshot.get_state("key") == "conv_value"

    def test_get_state_default(self):
        """Test get_state with default value."""
        snapshot = ContextSnapshot(
            messages=(),
            session_id="test",
            conversation_stage="initial",
            settings=MagicMock(),
            model="test",
            provider="test",
            max_tokens=4096,
            temperature=0.7,
            conversation_state={},
            session_state={},
            observed_files=(),
            capabilities={},
        )

        assert snapshot.get_state("missing", "default") == "default"

    def test_has_capability(self):
        """Test capability checking."""
        snapshot = ContextSnapshot(
            messages=(),
            session_id="test",
            conversation_stage="initial",
            settings=MagicMock(),
            model="test",
            provider="test",
            max_tokens=4096,
            temperature=0.7,
            conversation_state={},
            session_state={},
            observed_files=(),
            capabilities={"feature_a": True, "feature_b": False},
        )

        assert snapshot.has_capability("feature_a") is True
        assert snapshot.has_capability("feature_b") is False
        assert snapshot.has_capability("feature_c") is False

    def test_get_capability_value(self):
        """Test getting capability values."""
        snapshot = ContextSnapshot(
            messages=(),
            session_id="test",
            conversation_stage="initial",
            settings=MagicMock(),
            model="test",
            provider="test",
            max_tokens=4096,
            temperature=0.7,
            conversation_state={},
            session_state={},
            observed_files=(),
            capabilities={"feature_a": "value_a", "feature_b": 123},
        )

        assert snapshot.get_capability_value("feature_a") == "value_a"
        assert snapshot.get_capability_value("feature_b") == 123
        assert snapshot.get_capability_value("feature_c") is None

    def test_message_count(self):
        """Test message count property."""
        messages = (MagicMock(), MagicMock(), MagicMock())
        snapshot = ContextSnapshot(
            messages=messages,
            session_id="test",
            conversation_stage="initial",
            settings=MagicMock(),
            model="test",
            provider="test",
            max_tokens=4096,
            temperature=0.7,
            conversation_state={},
            session_state={},
            observed_files=(),
            capabilities={},
        )

        assert snapshot.message_count == 3

    def test_is_complete(self):
        """Test is_complete property."""
        snapshot = ContextSnapshot(
            messages=(),
            session_id="test",
            conversation_stage="complete",
            settings=MagicMock(),
            model="test",
            provider="test",
            max_tokens=4096,
            temperature=0.7,
            conversation_state={},
            session_state={},
            observed_files=(),
            capabilities={},
        )

        assert snapshot.is_complete is True


class TestStateTransition:
    """Tests for StateTransition validation."""

    def test_add_message_validation(self):
        """Test ADD_MESSAGE transition requires message data."""
        with pytest.raises(ValueError, match="ADD_MESSAGE.*message"):
            StateTransition(
                transition_type=TransitionType.ADD_MESSAGE,
                data={},  # Missing 'message'
            )

    def test_add_message_valid(self):
        """Test valid ADD_MESSAGE transition."""
        message = MagicMock()
        transition = StateTransition(
            transition_type=TransitionType.ADD_MESSAGE,
            data={"message": message},
        )

        assert transition.transition_type == TransitionType.ADD_MESSAGE
        assert transition.data["message"] == message

    def test_update_state_validation(self):
        """Test UPDATE_STATE transition requires key and value."""
        with pytest.raises(ValueError, match="UPDATE_STATE.*key.*value"):
            StateTransition(
                transition_type=TransitionType.UPDATE_STATE,
                data={"key": "test"},  # Missing 'value'
            )

    def test_update_state_valid(self):
        """Test valid UPDATE_STATE transition."""
        transition = StateTransition(
            transition_type=TransitionType.UPDATE_STATE,
            data={"key": "test_key", "value": "test_value"},
        )

        assert transition.transition_type == TransitionType.UPDATE_STATE
        assert transition.data["key"] == "test_key"
        assert transition.data["value"] == "test_value"

    def test_execute_tool_validation(self):
        """Test EXECUTE_TOOL transition requires tool_name and arguments."""
        with pytest.raises(ValueError, match="EXECUTE_TOOL.*tool_name.*arguments"):
            StateTransition(
                transition_type=TransitionType.EXECUTE_TOOL,
                data={"tool_name": "test"},  # Missing 'arguments'
            )

    def test_execute_tool_valid(self):
        """Test valid EXECUTE_TOOL transition."""
        transition = StateTransition(
            transition_type=TransitionType.EXECUTE_TOOL,
            data={
                "tool_name": "list_files",
                "arguments": {"path": "."},
            },
        )

        assert transition.transition_type == TransitionType.EXECUTE_TOOL
        assert transition.data["tool_name"] == "list_files"


class TestTransitionBatch:
    """Tests for TransitionBatch operations."""

    def test_create_empty_batch(self):
        """Test creating an empty batch."""
        batch = TransitionBatch()
        assert batch.is_empty()
        assert len(batch.transitions) == 0

    def test_add_transition(self):
        """Test adding a transition to batch."""
        batch = TransitionBatch()
        transition = StateTransition(
            transition_type=TransitionType.UPDATE_STATE,
            data={"key": "test", "value": "value"},
        )

        batch.add(transition)

        assert not batch.is_empty()
        assert len(batch.transitions) == 1

    def test_add_message(self):
        """Test add_message helper method."""
        batch = TransitionBatch()
        message = MagicMock()

        result = batch.add_message(message)

        assert result is batch  # Returns self for chaining
        assert len(batch.transitions) == 1
        assert batch.transitions[0].transition_type == TransitionType.ADD_MESSAGE

    def test_update_state(self):
        """Test update_state helper method."""
        batch = TransitionBatch()

        result = batch.update_state("key", "value")

        assert result is batch
        assert len(batch.transitions) == 1
        transition = batch.transitions[0]
        assert transition.data["key"] == "key"
        assert transition.data["value"] == "value"
        assert transition.data["scope"] == "conversation"

    def test_update_state_session_scope(self):
        """Test update_state with session scope."""
        batch = TransitionBatch()

        batch.update_state("key", "value", scope="session")

        transition = batch.transitions[0]
        assert transition.data["scope"] == "session"

    def test_execute_tool(self):
        """Test execute_tool helper method."""
        batch = TransitionBatch()

        result = batch.execute_tool("test_tool", {"arg": "value"})

        assert result is batch
        assert len(batch.transitions) == 1
        transition = batch.transitions[0]
        assert transition.transition_type == TransitionType.EXECUTE_TOOL
        assert transition.data["tool_name"] == "test_tool"

    def test_extend(self):
        """Test extending batches."""
        batch1 = TransitionBatch()
        batch1.update_state("key1", "value1")

        batch2 = TransitionBatch()
        batch2.update_state("key2", "value2")
        batch2.metadata = {"source": "test"}

        batch1.extend(batch2)

        assert len(batch1.transitions) == 2
        assert batch1.metadata == {"source": "test"}

    def test_chaining(self):
        """Test method chaining."""
        batch = TransitionBatch()

        batch.update_state("key1", "value1").update_state("key2", "value2").execute_tool(
            "test_tool", {}
        )

        assert len(batch.transitions) == 3


class TestCoordinatorResult:
    """Tests for CoordinatorResult factory methods."""

    def test_no_op(self):
        """Test creating a no-op result."""
        result = CoordinatorResult.no_op(reasoning="Nothing to do")

        assert result.transitions.is_empty()
        assert result.reasoning == "Nothing to do"
        assert result.should_continue is True

    def test_transitions_only(self):
        """Test creating result with transitions."""
        transition1 = StateTransition(
            transition_type=TransitionType.UPDATE_STATE,
            data={"key": "test", "value": "value"},
        )
        transition2 = StateTransition(
            transition_type=TransitionType.UPDATE_CAPABILITY,
            data={"capability": "test", "value": True},
        )

        result = CoordinatorResult.transitions_only(
            transition1, transition2, reasoning="Multiple updates"
        )

        assert len(result.transitions.transitions) == 2
        assert result.reasoning == "Multiple updates"

    def test_add_message(self):
        """Test add_message on result."""
        result = CoordinatorResult(
            transitions=TransitionBatch(),
        )
        message = MagicMock()

        returned = result.add_message(message)

        assert returned is result
        assert len(result.transitions.transitions) == 1

    def test_update_state(self):
        """Test update_state on result."""
        result = CoordinatorResult(
            transitions=TransitionBatch(),
        )

        result.update_state("key", "value")

        assert len(result.transitions.transitions) == 1


class TestTransitionApplier:
    """Tests for TransitionApplier."""

    @pytest.mark.asyncio
    async def test_apply_add_message(self):
        """Test applying ADD_MESSAGE transition."""
        orchestrator = MagicMock()
        orchestrator.add_message = MagicMock()

        applier = TransitionApplier(orchestrator)
        transition = StateTransition(
            transition_type=TransitionType.ADD_MESSAGE,
            data={"message": "test message"},
        )

        await applier.apply(transition)

        orchestrator.add_message.assert_called_once_with("test message")

    @pytest.mark.asyncio
    async def test_apply_update_state_conversation(self):
        """Test applying UPDATE_STATE for conversation scope."""
        orchestrator = MagicMock()
        orchestrator.conversation_state = {}

        applier = TransitionApplier(orchestrator)
        transition = StateTransition(
            transition_type=TransitionType.UPDATE_STATE,
            data={"key": "test_key", "value": "test_value", "scope": "conversation"},
        )

        await applier.apply(transition)

        assert orchestrator.conversation_state["test_key"] == "test_value"

    @pytest.mark.asyncio
    async def test_apply_update_state_session(self):
        """Test applying UPDATE_STATE for session scope."""
        orchestrator = MagicMock()
        orchestrator.session_state = {}

        applier = TransitionApplier(orchestrator)
        transition = StateTransition(
            transition_type=TransitionType.UPDATE_STATE,
            data={"key": "test_key", "value": "test_value", "scope": "session"},
        )

        await applier.apply(transition)

        assert orchestrator.session_state["test_key"] == "test_value"

    @pytest.mark.asyncio
    async def test_apply_delete_state(self):
        """Test applying DELETE_STATE transition."""
        orchestrator = MagicMock()
        orchestrator.conversation_state = {"test_key": "value"}

        applier = TransitionApplier(orchestrator)
        transition = StateTransition(
            transition_type=TransitionType.DELETE_STATE,
            data={"key": "test_key", "scope": "conversation"},
        )

        await applier.apply(transition)

        assert "test_key" not in orchestrator.conversation_state

    @pytest.mark.asyncio
    async def test_apply_update_capability(self):
        """Test applying UPDATE_CAPABILITY transition."""
        orchestrator = MagicMock()
        orchestrator._capabilities = {}

        applier = TransitionApplier(orchestrator)
        transition = StateTransition(
            transition_type=TransitionType.UPDATE_CAPABILITY,
            data={"capability": "test_cap", "value": True},
        )

        await applier.apply(transition)

        assert orchestrator._capabilities["test_cap"] is True

    @pytest.mark.asyncio
    async def test_apply_update_stage(self):
        """Test applying UPDATE_STAGE transition."""
        orchestrator = MagicMock()
        orchestrator.conversation_stage = "initial"

        applier = TransitionApplier(orchestrator)
        transition = StateTransition(
            transition_type=TransitionType.UPDATE_STAGE,
            data={"stage": "planning"},
        )

        await applier.apply(transition)

        assert orchestrator.conversation_stage == "planning"

    @pytest.mark.asyncio
    async def test_apply_batch(self):
        """Test applying a batch of transitions."""
        orchestrator = MagicMock()
        orchestrator.add_message = MagicMock()
        orchestrator.conversation_state = {}

        batch = TransitionBatch()
        batch.add_message("message1")
        batch.update_state("key", "value")

        applier = TransitionApplier(orchestrator)
        await applier.apply_batch(batch)

        orchestrator.add_message.assert_called_once()
        assert orchestrator.conversation_state["key"] == "value"


class TestCreateSnapshot:
    """Tests for create_snapshot utility."""

    def test_create_snapshot_from_orchestrator(self):
        """Test creating snapshot from orchestrator."""
        orchestrator = MagicMock()
        orchestrator.messages = ["msg1", "msg2"]
        orchestrator.session_id = "test-session"
        orchestrator.conversation_stage = "planning"
        orchestrator.settings = MagicMock()
        orchestrator.model = "test-model"
        orchestrator.provider_name = "test-provider"
        orchestrator.max_tokens = 2048
        orchestrator.temperature = 0.5
        orchestrator.conversation_state = {"key": "value"}
        orchestrator.session_state = {"sess_key": "sess_value"}
        orchestrator.observed_files = ["file1.py", "file2.py"]
        orchestrator._capabilities = {"cap_a": True}

        snapshot = create_snapshot(orchestrator)

        assert snapshot.session_id == "test-session"
        assert snapshot.conversation_stage == "planning"
        assert snapshot.model == "test-model"
        assert snapshot.provider == "test-provider"
        assert snapshot.max_tokens == 2048
        assert snapshot.temperature == 0.5
        assert snapshot.conversation_state == {"key": "value"}
        assert snapshot.session_state == {"sess_key": "sess_value"}
        assert snapshot.observed_files == ("file1.py", "file2.py")
        assert snapshot.capabilities == {"cap_a": True}

    def test_create_snapshot_copies_state(self):
        """Test that snapshot copies state to prevent mutation."""
        orchestrator = MagicMock()
        orchestrator.messages = []
        orchestrator.session_id = "test"
        orchestrator.conversation_stage = "initial"
        orchestrator.settings = MagicMock()
        orchestrator.model = "test"
        orchestrator.provider_name = "test"
        orchestrator.max_tokens = 4096
        orchestrator.temperature = 0.7
        orchestrator.conversation_state = {"key": "original"}
        orchestrator.session_state = {}
        orchestrator.observed_files = []
        orchestrator._capabilities = {}

        snapshot = create_snapshot(orchestrator)

        # Modify snapshot's state (shouldn't affect orchestrator)
        snapshot.conversation_state["key"] = "modified"

        # Orchestrator state should be unchanged
        assert orchestrator.conversation_state["key"] == "original"
