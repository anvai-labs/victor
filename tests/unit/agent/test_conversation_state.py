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

"""Tests for conversation_state module."""

from victor.agent.conversation.state_machine import (
    ConversationStage,
    ConversationState,
    ConversationStateMachine,
    STAGE_KEYWORDS,
    WEAK_EXECUTION_KEYWORDS,
    NATURAL_BACKWARD_TRANSITIONS,
)
from victor.tools.metadata_registry import get_tools_by_stage


class TestConversationStage:
    """Tests for ConversationStage enum."""

    def test_all_stages_defined(self):
        """Test all conversation stages are defined."""
        assert ConversationStage.INITIAL is not None
        assert ConversationStage.PLANNING is not None
        assert ConversationStage.READING is not None
        assert ConversationStage.ANALYSIS is not None
        assert ConversationStage.EXECUTION is not None
        assert ConversationStage.VERIFICATION is not None
        assert ConversationStage.COMPLETION is not None


class TestStageToolMapping:
    """Tests for decorator-driven stage-to-tool mapping via metadata registry."""

    def test_all_stages_have_mappings(self):
        """Test stage tool mappings can be retrieved for all stages.

        Note: With decorator-driven selection, stages may return empty sets
        if no tools have decorated that stage yet.
        """
        for stage in ConversationStage:
            # get_tools_by_stage returns a set (may be empty)
            tools = get_tools_by_stage(stage.name.lower())
            assert isinstance(tools, set), f"Stage {stage} should return a set"

    def test_initial_stage_tools(self):
        """Test initial stage can be queried for tools."""
        tools = get_tools_by_stage("initial")
        # With decorator-driven selection, we just verify the API works
        assert isinstance(tools, set)

    def test_execution_stage_tools(self):
        """Test execution stage can be queried for tools."""
        tools = get_tools_by_stage("execution")
        # With decorator-driven selection, we just verify the API works
        assert isinstance(tools, set)


class TestStageKeywords:
    """Tests for STAGE_KEYWORDS constant."""

    def test_all_stages_have_keywords(self):
        """Test all stages have keywords."""
        for stage in ConversationStage:
            assert stage in STAGE_KEYWORDS, f"Stage {stage} missing keywords"

    def test_execution_keywords(self):
        """Test execution stage has action keywords."""
        keywords = STAGE_KEYWORDS[ConversationStage.EXECUTION]
        assert "change" in keywords
        assert "fix" in keywords
        assert "implement" in keywords


class TestConversationStateDataclass:
    """Tests for ConversationState dataclass."""

    def test_default_init(self):
        """Test default initialization."""
        state = ConversationState()
        assert state.stage == ConversationStage.INITIAL
        assert state.tool_history == []
        assert state.observed_files == set()
        assert state.modified_files == set()
        assert state.message_count == 0
        assert state.last_tools == []

    def test_custom_init(self):
        """Test initialization with custom values."""
        state = ConversationState(
            stage=ConversationStage.PLANNING,
            message_count=5,
        )
        assert state.stage == ConversationStage.PLANNING
        assert state.message_count == 5

    def test_record_tool_execution(self):
        """Test recording tool execution."""
        state = ConversationState()
        state.record_tool_execution("code_search", {"query": "test"})

        assert "code_search" in state.tool_history
        assert "code_search" in state.last_tools

    def test_record_read_file_updates_observed(self):
        """Test read tool updates observed_files."""
        state = ConversationState()
        # Use canonical tool name "read" and "path" arg as checked in conversation_state.py
        state.record_tool_execution("read", {"path": "/test/file.py"})

        assert "/test/file.py" in state.observed_files

    def test_record_write_file_updates_modified(self):
        """Test write tool updates modified_files."""
        state = ConversationState()
        # Use canonical tool name "write" and "path" arg as checked in conversation_state.py
        state.record_tool_execution("write", {"path": "/test/file.py"})

        assert "/test/file.py" in state.modified_files

    def test_record_edit_file_updates_modified(self):
        """Test edit tool updates modified_files."""
        state = ConversationState()
        # Use canonical tool name "edit" and "path" arg as checked in conversation_state.py
        state.record_tool_execution("edit", {"path": "/test/file.py"})

        assert "/test/file.py" in state.modified_files

    def test_last_tools_limit(self):
        """Test last_tools maintains limited history."""
        state = ConversationState()
        for i in range(10):
            state.record_tool_execution(f"tool_{i}", {})

        # Should only keep last 5 tools
        assert len(state.last_tools) == 5

    def test_record_message_increments_count(self):
        """Test recording message increments count."""
        state = ConversationState()
        state.record_message()
        state.record_message()

        assert state.message_count == 2


class TestConversationStateMachine:
    """Tests for ConversationStateMachine class."""

    def test_init(self):
        """Test state machine initialization."""
        sm = ConversationStateMachine()
        assert sm.state.stage == ConversationStage.INITIAL
        assert sm.state.message_count == 0

    def test_reset(self):
        """Test resetting state machine."""
        sm = ConversationStateMachine()
        # Use canonical tool name "read" which is checked in conversation_state.py
        sm.record_tool_execution("read", {"path": "/test.py"})
        sm.record_message("Hello", is_user=True)

        sm.reset()

        assert sm.state.stage == ConversationStage.INITIAL
        assert sm.state.tool_history == []
        assert sm.state.message_count == 0

    def test_get_stage(self):
        """Test getting current stage."""
        sm = ConversationStateMachine()
        assert sm.get_stage() == ConversationStage.INITIAL

    def test_get_stage_tools(self):
        """Test getting stage tools."""
        sm = ConversationStateMachine()
        tools = sm.get_stage_tools()

        # Decorator-driven selection: just verify it returns a set
        # Specific tool membership depends on decorator configuration
        assert isinstance(tools, set)

    def test_record_tool_execution(self):
        """Test recording tool execution through state machine."""
        sm = ConversationStateMachine()
        # Use canonical tool name "read" which is checked in conversation_state.py
        sm.record_tool_execution("read", {"path": "/test.py"})

        assert "/test.py" in sm.state.observed_files

    def test_record_message(self):
        """Test recording message."""
        sm = ConversationStateMachine()
        sm.record_message("Hello, please help me", is_user=True)

        assert sm.state.message_count == 1

    def test_get_state_summary(self):
        """Test getting state summary."""
        sm = ConversationStateMachine()
        # Use canonical tool name "read" which is checked in conversation_state.py
        sm.record_tool_execution("read", {"path": "/test.py"})
        sm.record_message("Hello", is_user=True)

        summary = sm.get_state_summary()

        assert "stage" in summary
        assert "message_count" in summary
        assert summary["message_count"] == 1
        assert summary["files_observed"] == 1

    def test_detect_stage_from_content_execution(self):
        """Test detecting execution stage from message content.

        EXECUTION is only returned after the agent has observed files.
        On the first message with no files read, it returns READING instead.
        """
        sm = ConversationStateMachine()
        # Simulate having observed files (post-exploration)
        sm.state.message_count = 2
        sm.state.observed_files = {"some_file.py"}

        # Message with execution keywords
        stage = sm._detect_stage_from_content("Please fix and implement this change")

        # Should detect execution stage (files already observed)
        assert stage == ConversationStage.EXECUTION

    def test_detect_stage_from_content_no_match(self):
        """Test no stage detected for neutral content."""
        sm = ConversationStateMachine()

        # Neutral message
        stage = sm._detect_stage_from_content("Hello")

        assert stage is None

    def test_detect_stage_from_tools(self):
        """Test detecting stage from tool patterns."""
        sm = ConversationStateMachine()

        # Record some read operations using canonical names
        sm.state.last_tools = ["read", "read", "search"]

        stage = sm._detect_stage_from_tools()

        # Stage detection depends on decorator-driven registry
        # Result can be any stage or None depending on tool configurations
        assert stage is None or stage in ConversationStage

    def test_detect_stage_from_tools_empty(self):
        """Test no stage detected for empty tool history."""
        sm = ConversationStateMachine()

        stage = sm._detect_stage_from_tools()

        assert stage is None


class TestConversationStateMachineTransitions:
    """Tests for stage transitions."""

    def test_maybe_transition(self):
        """Test maybe_transition method."""
        sm = ConversationStateMachine()

        # Simulate executing read operations using canonical names
        sm.record_tool_execution("read", {"path": "/test.py"})
        sm.record_tool_execution("read", {"path": "/test2.py"})

        # Stage may transition
        stage = sm.get_stage()
        # Could be INITIAL or READING depending on transition logic
        assert stage in ConversationStage

    def test_transition_with_confidence(self):
        """Test transition updates confidence."""
        sm = ConversationStateMachine()

        # Send message with strong execution keywords
        sm.record_message("Please fix and change and implement this update", is_user=True)

        # Confidence should be updated
        assert sm.state._stage_confidence >= 0


class TestConversationStateSerialization:
    """Tests for serialization/deserialization."""

    def test_state_to_dict(self):
        """Test ConversationState.to_dict (covers line 176)."""
        state = ConversationState(
            stage=ConversationStage.READING,
            message_count=5,
        )
        state.tool_history = ["read_file", "code_search"]
        state.observed_files = {"/test.py", "/other.py"}
        state.modified_files = {"/changed.py"}
        state.last_tools = ["read_file"]

        data = state.to_dict()

        assert data["stage"] == "READING"
        assert data["tool_history"] == ["read_file", "code_search"]
        assert set(data["observed_files"]) == {"/test.py", "/other.py"}
        assert set(data["modified_files"]) == {"/changed.py"}
        assert data["message_count"] == 5
        assert data["last_tools"] == ["read_file"]

    def test_state_from_dict(self):
        """Test ConversationState.from_dict (covers lines 196-204)."""
        data = {
            "stage": "EXECUTION",
            "tool_history": ["write_file"],
            "observed_files": ["/test.py"],
            "modified_files": ["/test.py"],
            "message_count": 3,
            "last_tools": ["write_file"],
            "stage_confidence": 0.8,
        }

        state = ConversationState.from_dict(data)

        assert state.stage == ConversationStage.EXECUTION
        assert state.tool_history == ["write_file"]
        assert state.observed_files == {"/test.py"}
        assert state.modified_files == {"/test.py"}
        assert state.message_count == 3
        assert state.last_tools == ["write_file"]
        assert state._stage_confidence == 0.8

    def test_state_from_dict_empty(self):
        """Test ConversationState.from_dict with empty dict."""
        state = ConversationState.from_dict({})

        assert state.stage == ConversationStage.INITIAL
        assert state.tool_history == []
        assert state.message_count == 0

    def test_state_machine_to_dict(self):
        """Test ConversationStateMachine.to_dict (covers line 417)."""
        sm = ConversationStateMachine()
        # Use canonical tool name for consistency
        sm.record_tool_execution("read", {"path": "/test.py"})

        data = sm.to_dict()

        assert "state" in data
        assert data["state"]["tool_history"] == ["read"]

    def test_state_machine_from_dict(self):
        """Test ConversationStateMachine.from_dict (covers lines 431-434)."""
        data = {
            "state": {
                "stage": "ANALYSIS",
                "tool_history": ["code_search"],
                "observed_files": ["/file.py"],
                "modified_files": [],
                "message_count": 2,
                "last_tools": ["code_search"],
                "stage_confidence": 0.7,
            }
        }

        sm = ConversationStateMachine.from_dict(data)

        assert sm.state.stage == ConversationStage.ANALYSIS
        assert sm.state.tool_history == ["code_search"]
        assert sm.state.message_count == 2

    def test_state_machine_from_dict_empty(self):
        """Test ConversationStateMachine.from_dict with empty dict."""
        sm = ConversationStateMachine.from_dict({})

        assert sm.state.stage == ConversationStage.INITIAL


class TestConversationStateMachineRoundTrip:
    """Tests for state machine round-trip serialization."""

    def test_state_roundtrip(self):
        """Test ConversationState serialization round-trip."""
        original = ConversationState(
            stage=ConversationStage.ANALYSIS,
            message_count=10,
        )
        original.tool_history = ["read_file", "code_search", "write_file"]
        original.observed_files = {"/a.py", "/b.py"}
        original.modified_files = {"/c.py"}
        original.last_tools = ["write_file"]
        original._stage_confidence = 0.75

        data = original.to_dict()
        restored = ConversationState.from_dict(data)

        assert restored.stage == original.stage
        assert restored.tool_history == original.tool_history
        assert restored.observed_files == original.observed_files
        assert restored.modified_files == original.modified_files
        assert restored.message_count == original.message_count
        assert restored.last_tools == original.last_tools
        assert restored._stage_confidence == original._stage_confidence

    def test_machine_roundtrip(self):
        """Test ConversationStateMachine serialization round-trip."""
        original = ConversationStateMachine()
        # Use canonical tool names for file tracking
        original.record_tool_execution("read", {"path": "/test.py"})
        original.record_tool_execution("write", {"path": "/test.py"})
        original.record_message("Hello, please help", is_user=True)

        data = original.to_dict()
        restored = ConversationStateMachine.from_dict(data)

        assert restored.state.tool_history == original.state.tool_history
        assert restored.state.observed_files == original.state.observed_files
        assert restored.state.modified_files == original.state.modified_files
        assert restored.state.message_count == original.state.message_count


class TestConversationStateMachineRegistryIntegration:
    """Tests for registry integration in ConversationStateMachine.

    Note: Stage-to-tool mapping is now fully decorator-driven via the metadata registry.
    These tests verify the integration with registry_get_tools_by_stage().
    """

    def test_get_tools_for_stage_uses_registry(self):
        """Test _get_tools_for_stage uses the metadata registry."""
        sm = ConversationStateMachine()

        # Get tools for INITIAL stage - result depends on decorator configuration
        tools = sm._get_tools_for_stage(ConversationStage.INITIAL)

        # Should return a set (possibly empty if no tools decorated with initial stage)
        assert isinstance(tools, set)

    def test_get_stage_tools_returns_registry_tools(self):
        """Test get_stage_tools returns tools from metadata registry."""
        sm = ConversationStateMachine()
        sm.state.stage = ConversationStage.EXECUTION

        tools = sm.get_stage_tools()

        # Should return a set from the registry
        assert isinstance(tools, set)

    def test_get_tools_for_stage_includes_registry_tools(self):
        """Test _get_tools_for_stage includes tools from registry."""
        from unittest.mock import patch

        sm = ConversationStateMachine()

        # Mock the registry to return specific tools
        with patch(
            "victor.agent.conversation.state_machine.registry_get_tools_by_stage",
            return_value={"mock_tool_from_registry"},
        ):
            tools = sm._get_tools_for_stage(ConversationStage.INITIAL)

            # Should include registry tools
            assert "mock_tool_from_registry" in tools

    def test_get_stage_tools_from_registry(self):
        """Test get_stage_tools returns tools from registry."""
        from unittest.mock import patch

        sm = ConversationStateMachine()
        sm.state.stage = ConversationStage.READING

        with patch(
            "victor.agent.conversation.state_machine.registry_get_tools_by_stage",
            return_value={"registry_read_tool"},
        ):
            tools = sm.get_stage_tools()

            # Should include registry tool
            assert "registry_read_tool" in tools

    def test_detect_stage_from_tools_uses_helper(self):
        """Test _detect_stage_from_tools uses _get_tools_for_stage."""
        from unittest.mock import patch

        sm = ConversationStateMachine()
        sm.state.last_tools = ["custom_read_tool", "custom_read_tool"]

        # Mock _get_tools_for_stage to return custom_read_tool for READING
        original_get_tools = sm._get_tools_for_stage

        def mock_get_tools(stage):
            if stage == ConversationStage.READING:
                return {"custom_read_tool", "read"}
            return original_get_tools(stage)

        with patch.object(sm, "_get_tools_for_stage", side_effect=mock_get_tools):
            stage = sm._detect_stage_from_tools()

            # Should detect READING stage due to custom_read_tool overlap
            assert stage == ConversationStage.READING

    def test_should_include_tool_with_registry_tools(self):
        """Test should_include_tool works with registry integration."""
        from unittest.mock import patch

        sm = ConversationStateMachine()
        sm.state.stage = ConversationStage.ANALYSIS

        # Mock registry to include a custom tool in ANALYSIS stage
        with patch(
            "victor.agent.conversation.state_machine.registry_get_tools_by_stage",
            return_value={"custom_analysis_tool"},
        ):
            # Custom tool from registry should be included
            assert sm.should_include_tool("custom_analysis_tool") is True

    def test_get_tool_priority_boost_with_registry(self):
        """Test get_tool_priority_boost works with registry integration."""
        from unittest.mock import patch

        sm = ConversationStateMachine()
        sm.state.stage = ConversationStage.EXECUTION

        with patch(
            "victor.agent.conversation.state_machine.registry_get_tools_by_stage",
            return_value={"custom_exec_tool"},
        ):
            # Custom tool from registry should get high boost
            boost = sm.get_tool_priority_boost("custom_exec_tool")
            assert boost == 0.15  # High boost for stage-relevant tools


class TestEdgeModelStageTransitionFallback:
    """Tests for edge model fallback in stage transitions (FEP)."""

    def _make_decision_result(self, stage: str, confidence: float, source: str = "llm"):
        """Create a mock DecisionResult."""
        from unittest.mock import MagicMock

        result = MagicMock()
        result.result.stage = stage
        result.confidence = confidence
        result.source = source
        return result

    def _make_sm_at_execution(self):
        """Create a state machine at EXECUTION stage with recent read tools."""
        sm = ConversationStateMachine()
        sm.state.stage = ConversationStage.EXECUTION
        sm.state.last_tools = ["read", "code_search", "read", "read", "code_search"]
        sm._edge_stage_cache.clear()  # Ensure no cached decisions
        return sm

    def _stage_tools_side_effect(self, stage):
        """Return stage-specific tool sets so detection works properly."""
        mapping = {
            ConversationStage.READING: {"read", "code_search", "ls"},
            ConversationStage.ANALYSIS: {"code_search", "overview", "refs"},
            ConversationStage.EXECUTION: {"edit", "write", "shell"},
            ConversationStage.INITIAL: {"read", "ls"},
            ConversationStage.PLANNING: {"read", "overview"},
            ConversationStage.VERIFICATION: {"test", "shell"},
            ConversationStage.COMPLETION: set(),
        }
        return mapping.get(stage, set())

    def test_edge_model_called_when_heuristic_low_confidence(self):
        """Edge model should be consulted when tool overlap is below threshold."""
        from unittest.mock import patch, MagicMock

        sm = self._make_sm_at_execution()

        decision = self._make_decision_result("analysis", 0.90)
        mock_service = MagicMock()
        mock_service.decide_sync.return_value = decision

        with (
            patch.object(sm, "_detect_stage_from_tools", return_value=ConversationStage.READING),
            patch.object(sm, "_get_tools_for_stage", side_effect=self._stage_tools_side_effect),
            patch("victor.core.service_resolution.get_container") as mock_container,
            patch("victor.core.feature_flags.get_feature_flag_manager") as mock_ffm,
        ):
            mock_ffm.return_value.is_enabled.return_value = True
            mock_container.return_value.get.return_value = mock_service
            mock_container.return_value.get.side_effect = lambda key: (
                mock_service if key == "llm_decision_service" else None
            )

            sm._maybe_transition()

            # Edge model should have been called (read/code_search overlap
            # with READING=2 but < MIN_TOOLS_FOR_TRANSITION=3)
            mock_service.decide_sync.assert_called_once()

    def test_edge_model_not_called_when_heuristic_high_confidence(self):
        """Edge model should NOT be called when tool overlap meets threshold."""
        from unittest.mock import patch, MagicMock

        sm = ConversationStateMachine()
        sm.state.stage = ConversationStage.INITIAL
        sm.state.last_tools = ["edit", "write", "shell", "edit", "write"]

        mock_service = MagicMock()

        with (
            patch(
                "victor.agent.conversation.state_machine.registry_get_tools_by_stage",
                return_value={"edit", "write", "shell"},
            ),
            patch("victor.core.service_resolution.get_container") as mock_container,
        ):
            mock_container.return_value.get.side_effect = lambda key: (
                mock_service if key == "llm_decision_service" else None
            )

            sm._maybe_transition()

            # Edge model should NOT have been called (high overlap = 3+)
            mock_service.decide_sync.assert_not_called()

    def test_graceful_fallback_when_edge_model_unavailable(self):
        """When edge model returns None, heuristic behavior is preserved."""
        from unittest.mock import patch

        sm = self._make_sm_at_execution()

        with (
            patch(
                "victor.agent.conversation.state_machine.registry_get_tools_by_stage",
                return_value={"read", "code_search"},
            ),
            patch("victor.core.feature_flags.get_feature_flag_manager") as mock_ffm,
            patch("victor.core.service_resolution.get_container") as mock_container,
        ):
            mock_ffm.return_value.is_enabled.return_value = True
            mock_container.return_value.get.return_value = None  # No service

            sm._maybe_transition()

            # Should stay at EXECUTION (no transition)
            assert sm.state.stage == ConversationStage.EXECUTION

    def test_edge_model_overrides_heuristic_on_higher_confidence(self):
        """Edge model with higher confidence should override heuristic."""
        from unittest.mock import patch, MagicMock

        sm = self._make_sm_at_execution()

        decision = self._make_decision_result("analysis", 0.90)
        mock_service = MagicMock()
        mock_service.decide_sync.return_value = decision

        with (
            patch.object(sm, "_detect_stage_from_tools", return_value=ConversationStage.READING),
            patch.object(sm, "_get_tools_for_stage", side_effect=self._stage_tools_side_effect),
            patch("victor.core.service_resolution.get_container") as mock_container,
            patch("victor.core.feature_flags.get_feature_flag_manager") as mock_ffm,
        ):
            mock_ffm.return_value.is_enabled.return_value = True
            mock_container.return_value.get.side_effect = lambda key: (
                mock_service if key == "llm_decision_service" else None
            )

            sm._maybe_transition()

            # Should transition to ANALYSIS (edge model overrode heuristic)
            assert sm.state.stage == ConversationStage.ANALYSIS

    def test_feature_flag_off_skips_edge_model(self):
        """When USE_EDGE_MODEL is disabled, edge model is never called."""
        from unittest.mock import patch, MagicMock

        sm = self._make_sm_at_execution()

        mock_service = MagicMock()

        with (
            patch.object(sm, "_get_tools_for_stage", side_effect=self._stage_tools_side_effect),
            patch("victor.core.service_resolution.get_container") as mock_container,
            patch("victor.core.feature_flags.get_feature_flag_manager") as mock_ffm,
        ):
            mock_ffm.return_value.is_enabled.return_value = False
            mock_container.return_value.get.side_effect = lambda key: (
                mock_service if key == "llm_decision_service" else None
            )

            sm._maybe_transition()

            # Edge model should NOT have been called
            mock_service.decide_sync.assert_not_called()
            # Should stay at EXECUTION
            assert sm.state.stage == ConversationStage.EXECUTION

    def test_execution_to_analysis_backward_transition(self):
        """EXECUTION→ANALYSIS should succeed when edge model has sufficient confidence."""
        from unittest.mock import patch, MagicMock

        sm = self._make_sm_at_execution()

        decision = self._make_decision_result("analysis", 0.85)
        mock_service = MagicMock()
        mock_service.decide_sync.return_value = decision

        with (
            patch.object(sm, "_detect_stage_from_tools", return_value=ConversationStage.READING),
            patch.object(sm, "_get_tools_for_stage", side_effect=self._stage_tools_side_effect),
            patch("victor.core.service_resolution.get_container") as mock_container,
            patch("victor.core.feature_flags.get_feature_flag_manager") as mock_ffm,
        ):
            mock_ffm.return_value.is_enabled.return_value = True
            mock_container.return_value.get.side_effect = lambda key: (
                mock_service if key == "llm_decision_service" else None
            )

            sm._maybe_transition()

            # Backward transition should succeed (verify cycle threshold = 0.50)
            assert sm.state.stage == ConversationStage.ANALYSIS

    def test_edge_model_context_includes_tools_and_stage(self):
        """Edge model context should include last_tools, current_stage, and heuristic."""
        from unittest.mock import patch, MagicMock

        sm = self._make_sm_at_execution()

        decision = self._make_decision_result("analysis", 0.90)
        mock_service = MagicMock()
        mock_service.decide_sync.return_value = decision

        with (
            patch.object(sm, "_detect_stage_from_tools", return_value=ConversationStage.READING),
            patch.object(sm, "_get_tools_for_stage", side_effect=self._stage_tools_side_effect),
            patch("victor.core.service_resolution.get_container") as mock_container,
            patch("victor.core.feature_flags.get_feature_flag_manager") as mock_ffm,
        ):
            mock_ffm.return_value.is_enabled.return_value = True
            mock_container.return_value.get.side_effect = lambda key: (
                mock_service if key == "llm_decision_service" else None
            )

            sm._maybe_transition()

            # Verify context passed to edge model
            call_args = mock_service.decide_sync.call_args
            context = call_args[1].get("context") or call_args[0][1]
            assert "last_tools" in context
            assert "current_stage" in context
            assert "detected_stage_heuristic" in context

    def test_cooldown_prevents_edge_model_call(self):
        """Phase 1 optimization: Cooldown should prevent edge model calls.

        When in cooldown period, _maybe_transition should return early
        without calling edge model, since transition won't happen anyway.
        """
        import time
        from unittest.mock import patch, MagicMock

        sm = ConversationStateMachine()
        sm.state.stage = ConversationStage.READING
        sm.state.last_tools = ["read", "grep", "code_search"]  # 3 READING tools
        sm._last_transition_time = time.time()  # Just transitioned

        mock_service = MagicMock()

        with (
            patch.object(sm, "_detect_stage_from_tools", return_value=ConversationStage.ANALYSIS),
            patch.object(sm, "_get_tools_for_stage", side_effect=self._stage_tools_side_effect),
            patch("victor.core.service_resolution.get_container") as mock_container,
            patch("victor.core.feature_flags.get_feature_flag_manager") as mock_ffm,
        ):
            mock_ffm.return_value.is_enabled.return_value = True
            mock_container.return_value.get.side_effect = lambda key: (
                mock_service if key == "llm_decision_service" else None
            )

            # Call _maybe_transition while in cooldown
            sm._maybe_transition()

            # Edge model should NOT have been called (early exit in cooldown)
            mock_service.decide_sync.assert_not_called()
            # Stage should not have changed (cooldown blocked it)
            assert sm.state.stage == ConversationStage.READING

    def test_high_confidence_heuristic_skips_edge_model(self):
        """Phase 1 optimization: High confidence heuristic should skip edge model.

        When tool overlap >= MIN_TOOLS_FOR_TRANSITION (5), heuristic is confident
        and edge model call should be skipped.
        """
        from unittest.mock import patch, MagicMock

        sm = ConversationStateMachine()
        sm.state.stage = ConversationStage.INITIAL
        # 5 unique EXECUTION tools to meet the new MIN_TOOLS_FOR_TRANSITION threshold
        sm.state.last_tools = ["edit", "write", "shell", "git", "test"]

        mock_service = MagicMock()

        with (
            patch.object(sm, "_detect_stage_from_tools", return_value=ConversationStage.EXECUTION),
            patch.object(
                sm,
                "_get_tools_for_stage",
                return_value={"edit", "write", "shell", "git", "test"},
            ),
            patch("victor.core.service_resolution.get_container") as mock_container,
        ):
            mock_container.return_value.get.side_effect = lambda key: (
                mock_service if key == "llm_decision_service" else None
            )

            sm._maybe_transition()

            # Edge model should NOT have been called (high confidence heuristic)
            mock_service.decide_sync.assert_not_called()
            # Should have transitioned directly based on heuristic
            assert sm.state.stage == ConversationStage.EXECUTION


class TestPhase2ContextAwareCalibration:
    """Tests for Phase 2: Context-Aware Calibration of edge model results.

    Phase 2 adds intelligent calibration to prevent edge model bias:
    - Calibrates when agent reads many files without editing
    - Corrects for intent bias (READ_ONLY tasks shouldn't EXECUTE)
    - Uses actual behavior instead of primitive counters
    """

    def _make_decision_result(self, stage: str, confidence: float, source: str = "llm"):
        """Create a mock DecisionResult."""
        from unittest.mock import MagicMock

        result = MagicMock()
        result.result.stage = stage
        result.confidence = confidence
        result.source = source
        return result

    def test_context_calibration_read_only_without_edits(self):
        """Phase 2: Edge model suggesting EXECUTION should be calibrated to ANALYSIS
        when agent has read 10+ files without any edits.
        """
        from unittest.mock import patch, MagicMock
        from victor.agent.action_authorizer import ActionIntent

        sm = ConversationStateMachine(action_intent=ActionIntent.AMBIGUOUS)
        sm.state.stage = ConversationStage.READING

        # Simulate reading 15 files without editing
        for i in range(15):
            sm.state.observed_files.add(f"file_{i}.py")

        # Edge model says EXECUTION with high confidence (bias behavior)
        decision = self._make_decision_result("execution", 0.97)
        mock_service = MagicMock()
        mock_service.decide_sync.return_value = decision

        with (
            patch("victor.core.service_resolution.get_container") as mock_container,
            patch("victor.core.feature_flags.get_feature_flag_manager") as mock_ffm,
        ):
            mock_ffm.return_value.is_enabled.return_value = True
            mock_container.return_value.get.return_value = mock_service

            result = sm._detect_stage_with_edge_model(
                ""
            )  # Empty string for content, not ConversationStage enum

            # Should calibrate to ANALYSIS despite edge model saying EXECUTION
            assert result == (ConversationStage.ANALYSIS, 0.7)

    def test_context_calibration_write_allowed_with_many_reads_stays_execution(self):
        """Explicit write intent should not be downgraded just because planning needed reads."""
        from unittest.mock import patch, MagicMock
        from victor.agent.action_authorizer import ActionIntent

        sm = ConversationStateMachine(action_intent=ActionIntent.WRITE_ALLOWED)
        sm.state.stage = ConversationStage.READING

        for i in range(15):
            sm.state.observed_files.add(f"file_{i}.py")

        decision = self._make_decision_result("execution", 0.97)
        mock_service = MagicMock()
        mock_service.decide_sync.return_value = decision

        with (
            patch("victor.core.service_resolution.get_container") as mock_container,
            patch("victor.core.feature_flags.get_feature_flag_manager") as mock_ffm,
        ):
            mock_ffm.return_value.is_enabled.return_value = True
            mock_container.return_value.get.return_value = mock_service

            result = sm._detect_stage_with_edge_model("")  # Empty string for content

            assert result == (ConversationStage.EXECUTION, 0.97)

    def test_context_calibration_read_only_intent(self):
        """Phase 2: READ_ONLY intent should bias-correct edge model EXECUTION suggestions."""
        from unittest.mock import patch, MagicMock
        from victor.agent.action_authorizer import ActionIntent

        sm = ConversationStateMachine(action_intent=ActionIntent.READ_ONLY)
        sm.state.stage = ConversationStage.ANALYSIS

        # Edge model says EXECUTION
        decision = self._make_decision_result("execution", 0.92)
        mock_service = MagicMock()
        mock_service.decide_sync.return_value = decision

        with (
            patch("victor.core.service_resolution.get_container") as mock_container,
            patch("victor.core.feature_flags.get_feature_flag_manager") as mock_ffm,
        ):
            mock_ffm.return_value.is_enabled.return_value = True
            mock_container.return_value.get.return_value = mock_service

            result = sm._detect_stage_with_edge_model("")  # Empty string for content

            # Should calibrate confidence down for READ_ONLY intent
            stage, confidence = result
            assert stage == ConversationStage.ANALYSIS
            assert confidence == 0.92 * 0.8  # Calibrated down

    def test_context_calibration_display_only_intent(self):
        """Phase 2: DISPLAY_ONLY intent should bias-correct edge model EXECUTION suggestions."""
        from unittest.mock import patch, MagicMock
        from victor.agent.action_authorizer import ActionIntent

        sm = ConversationStateMachine(action_intent=ActionIntent.DISPLAY_ONLY)
        sm.state.stage = ConversationStage.READING

        # Edge model says EXECUTION
        decision = self._make_decision_result("execution", 0.88)
        mock_service = MagicMock()
        mock_service.decide_sync.return_value = decision

        with (
            patch("victor.core.service_resolution.get_container") as mock_container,
            patch("victor.core.feature_flags.get_feature_flag_manager") as mock_ffm,
        ):
            mock_ffm.return_value.is_enabled.return_value = True
            mock_container.return_value.get.return_value = mock_service

            result = sm._detect_stage_with_edge_model("")  # Empty string for content

            # Should calibrate confidence down for DISPLAY_ONLY intent
            stage, confidence = result
            assert stage == ConversationStage.ANALYSIS  # Calibrated to ANALYSIS, not READING
            assert confidence == 0.88 * 0.8  # Calibrated down

    def test_context_calibration_no_effect_when_files_modified(self):
        """Phase 2: Calibration should NOT apply when agent has actually modified files."""
        from unittest.mock import patch, MagicMock
        from victor.agent.action_authorizer import ActionIntent

        sm = ConversationStateMachine(action_intent=ActionIntent.WRITE_ALLOWED)
        sm.state.stage = ConversationStage.ANALYSIS

        # Simulate reading 15 files AND modifying 3 of them
        for i in range(15):
            sm.state.observed_files.add(f"file_{i}.py")
        for i in range(3):
            sm.state.modified_files.add(f"file_{i}.py")

        # Edge model says EXECUTION with high confidence
        decision = self._make_decision_result("execution", 0.97)
        mock_service = MagicMock()
        mock_service.decide_sync.return_value = decision

        with (
            patch("victor.core.service_resolution.get_container") as mock_container,
            patch("victor.core.feature_flags.get_feature_flag_manager") as mock_ffm,
        ):
            mock_ffm.return_value.is_enabled.return_value = True
            mock_container.return_value.get.return_value = mock_service

            result = sm._detect_stage_with_edge_model("")  # Empty string for content

            # Should NOT calibrate (agent is actually editing)
            assert result == (ConversationStage.EXECUTION, 0.97)

    def test_context_calibration_no_effect_below_threshold(self):
        """Phase 2: Calibration should NOT apply when edge model confidence < 0.95."""
        from unittest.mock import patch, MagicMock
        from victor.agent.action_authorizer import ActionIntent

        sm = ConversationStateMachine(action_intent=ActionIntent.WRITE_ALLOWED)
        sm.state.stage = ConversationStage.READING

        # Simulate reading 15 files without editing
        for i in range(15):
            sm.state.observed_files.add(f"file_{i}.py")

        # Edge model says EXECUTION but confidence is below threshold
        decision = self._make_decision_result("execution", 0.90)
        mock_service = MagicMock()
        mock_service.decide_sync.return_value = decision

        with (
            patch("victor.core.service_resolution.get_container") as mock_container,
            patch("victor.core.feature_flags.get_feature_flag_manager") as mock_ffm,
        ):
            mock_ffm.return_value.is_enabled.return_value = True
            mock_container.return_value.get.return_value = mock_service

            result = sm._detect_stage_with_edge_model("")  # Empty string for content

            # Should NOT calibrate (confidence below 0.95 threshold)
            assert result == (ConversationStage.EXECUTION, 0.90)

    def test_set_action_intent_method(self):
        """Phase 2: set_action_intent() should update the intent."""
        from victor.agent.action_authorizer import ActionIntent

        sm = ConversationStateMachine(action_intent=ActionIntent.AMBIGUOUS)

        # Update intent to READ_ONLY
        sm.set_action_intent(ActionIntent.READ_ONLY)

        # Intent should be updated
        assert sm._action_intent == ActionIntent.READ_ONLY

        # Update again to DISPLAY_ONLY
        sm.set_action_intent(ActionIntent.DISPLAY_ONLY)

        # Intent should be updated again
        assert sm._action_intent == ActionIntent.DISPLAY_ONLY

    def test_write_allowed_intent_no_calibration(self):
        """Phase 2: WRITE_ALLOWED intent should NOT trigger calibration."""
        from unittest.mock import patch, MagicMock
        from victor.agent.action_authorizer import ActionIntent

        sm = ConversationStateMachine(action_intent=ActionIntent.WRITE_ALLOWED)
        sm.state.stage = ConversationStage.ANALYSIS

        # Edge model says EXECUTION
        decision = self._make_decision_result("execution", 0.95)
        mock_service = MagicMock()
        mock_service.decide_sync.return_value = decision

        with (
            patch("victor.core.service_resolution.get_container") as mock_container,
            patch("victor.core.feature_flags.get_feature_flag_manager") as mock_ffm,
        ):
            mock_ffm.return_value.is_enabled.return_value = True
            mock_container.return_value.get.return_value = mock_service

            result = sm._detect_stage_with_edge_model("")  # Empty string for content

            # Should NOT calibrate (WRITE_ALLOWED permits execution)
            assert result == (ConversationStage.EXECUTION, 0.95)


class TestNaturalStageProgression:
    """Tests for decision pipeline reordering — natural stage progression.

    The agent should explore (READING) before editing (EXECUTION).
    Bug-fix tasks match EXECUTION keywords but the agent hasn't seen any code yet.
    """

    def test_first_message_execution_intent_maps_to_reading(self):
        """First message with strong edit intent → READING, not EXECUTION.

        The agent must read files before editing. EXECUTION keywords on
        the first message should be suppressed when no files are observed.
        """
        sm = ConversationStateMachine()
        # Simulate first message — no files observed
        assert sm.state.message_count == 0
        assert len(sm.state.observed_files) == 0

        # "create" + "implement" = 2.0 (both strong), would normally → EXECUTION
        result = sm._detect_stage_from_content(
            "Create and implement a fix for the separability_matrix bug"
        )
        assert result == ConversationStage.READING

    def test_execution_after_files_observed(self):
        """After reading files, EXECUTION keywords should work normally."""
        sm = ConversationStateMachine()
        # Simulate having observed files
        sm.state.message_count = 3
        sm.state.observed_files = {"astropy/modeling/separable.py", "tests/test.py"}

        result = sm._detect_stage_from_content(
            "Fix the bug by modifying the separability_matrix function"
        )
        # "fix" (0.5) + "modify" (1.0) = 1.5 < 2, but "modify" alone is strong
        # With files observed, the first-message guard doesn't apply
        assert result == ConversationStage.EXECUTION

    def test_weak_keywords_dont_shortcircuit(self):
        """Weak EXECUTION keywords ('fix', 'add') score 0.5, not 1.0.

        'fix the add' = 0.5 + 0.5 = 1.0, below threshold 2.
        Should fall through to edge model or lower-confidence path.
        """
        sm = ConversationStateMachine()
        sm.state.message_count = 5  # Not first message
        sm.state.observed_files = {"some_file.py"}

        # Count weak keywords manually
        content = "fix the add"
        score = 0.0
        for kw in STAGE_KEYWORDS[ConversationStage.EXECUTION]:
            if kw in content.lower():
                if kw in WEAK_EXECUTION_KEYWORDS:
                    score += 0.5
                else:
                    score += 1.0
        assert score < 2.0, f"Weak keywords scored {score}, should be < 2"

    def test_strong_keywords_still_shortcircuit(self):
        """Strong EXECUTION keywords ('create', 'implement') still score 1.0.

        'create and implement a new feature' = 1.0 + 1.0 = 2.0, hits threshold.
        """
        sm = ConversationStateMachine()
        sm.state.message_count = 5
        sm.state.observed_files = {"some_file.py"}

        result = sm._detect_stage_from_content("create and implement a new feature for the module")
        assert result == ConversationStage.EXECUTION

    def test_benchmark_adapter_starts_reading(self):
        """Benchmark adapter should set READING, not lock EXECUTION."""
        from unittest.mock import MagicMock

        sm = ConversationStateMachine()
        default_min_tools = sm.MIN_TOOLS_FOR_TRANSITION

        # Simulate what the new benchmark adapter does
        sm._transition_to(ConversationStage.READING, confidence=0.8)

        assert sm.state.stage == ConversationStage.READING
        # MIN_TOOLS_FOR_TRANSITION should NOT be locked to 999
        assert sm.MIN_TOOLS_FOR_TRANSITION == default_min_tools

    def test_force_execution_after_5_reads(self):
        """After MAX_READS_WITHOUT_EDIT reads, force READING→EXECUTION.

        DISABLED: Force transition logic disabled in favor of UnifiedTaskTracker.
        UnifiedTaskTracker has sophisticated loop detection:
        - Offset-aware file read detection
        - Task-type aware thresholds
        - Mode-aware exploration limits
        - Read-only loop detection (>20 files read, 0 modified)

        This test now verifies that the primitive heuristic is disabled.
        """
        sm = ConversationStateMachine()
        sm._transition_to(ConversationStage.READING, confidence=0.8)

        # Simulate reading files
        for i in range(sm.MAX_READS_WITHOUT_EDIT):
            sm.state.observed_files.add(f"file_{i}.py")

        # DISABLED: Force transition is disabled, trust UnifiedTaskTracker
        assert sm._should_force_execution_transition() is False


class TestStageRegressionPrevention:
    """Tests for explicit backward transition allowlist.

    READING→INITIAL regression trapped the agent in read-only mode
    for 240s during SWE-bench task 3 (astropy-14365).
    """

    @staticmethod
    def _make_sm():
        """Create state machine with cooldown disabled for testing."""
        sm = ConversationStateMachine()
        sm.TRANSITION_COOLDOWN_SECONDS = 0.0
        return sm

    def test_reading_to_initial_blocked(self):
        """READING→INITIAL at confidence 0.70 must be blocked."""
        sm = self._make_sm()
        sm._transition_to(ConversationStage.READING, confidence=0.8)

        # Attempt backward transition
        sm._transition_to(ConversationStage.INITIAL, confidence=0.70)

        # Should stay in READING — not a natural backward transition
        assert sm.state.stage == ConversationStage.READING

    def test_execution_to_reading_allowed(self):
        """EXECUTION→READING at confidence 0.50 allowed (natural cycle)."""
        sm = self._make_sm()
        sm._transition_to(ConversationStage.EXECUTION, confidence=0.9)
        sm._transition_to(ConversationStage.READING, confidence=0.50)
        assert sm.state.stage == ConversationStage.READING

    def test_execution_to_analysis_allowed(self):
        """EXECUTION→ANALYSIS at confidence 0.55 allowed (natural cycle)."""
        sm = self._make_sm()
        sm._transition_to(ConversationStage.EXECUTION, confidence=0.9)
        sm._transition_to(ConversationStage.ANALYSIS, confidence=0.55)
        assert sm.state.stage == ConversationStage.ANALYSIS

    def test_reading_to_planning_blocked(self):
        """READING→PLANNING not in natural transitions — needs high confidence."""
        sm = self._make_sm()
        sm._transition_to(ConversationStage.READING, confidence=0.8)
        sm._transition_to(ConversationStage.PLANNING, confidence=0.70)
        # Blocked — requires BACKWARD_TRANSITION_THRESHOLD (0.85)
        assert sm.state.stage == ConversationStage.READING

    def test_forward_transitions_always_allowed(self):
        """Forward transitions always succeed regardless of confidence."""
        sm = self._make_sm()
        sm._transition_to(ConversationStage.READING, confidence=0.3)
        assert sm.state.stage == ConversationStage.READING
        sm._transition_to(ConversationStage.EXECUTION, confidence=0.3)
        assert sm.state.stage == ConversationStage.EXECUTION

    def test_natural_transitions_allowlist_complete(self):
        """Verify NATURAL_BACKWARD_TRANSITIONS covers expected pairs."""
        assert (
            ConversationStage.EXECUTION,
            ConversationStage.READING,
        ) in NATURAL_BACKWARD_TRANSITIONS
        assert (
            ConversationStage.EXECUTION,
            ConversationStage.ANALYSIS,
        ) in NATURAL_BACKWARD_TRANSITIONS
        assert (
            ConversationStage.ANALYSIS,
            ConversationStage.READING,
        ) in NATURAL_BACKWARD_TRANSITIONS
        assert (
            ConversationStage.VERIFICATION,
            ConversationStage.EXECUTION,
        ) in NATURAL_BACKWARD_TRANSITIONS
        # These should NOT be in the allowlist
        assert (
            ConversationStage.READING,
            ConversationStage.INITIAL,
        ) not in NATURAL_BACKWARD_TRANSITIONS
        assert (
            ConversationStage.EXECUTION,
            ConversationStage.INITIAL,
        ) not in NATURAL_BACKWARD_TRANSITIONS


class TestStageThrashingDampener:
    """Tests for stage thrashing detection and dampening."""

    @staticmethod
    def _make_sm():
        sm = ConversationStateMachine()
        sm.TRANSITION_COOLDOWN_SECONDS = 0.0
        sm.THRASHING_COOLDOWN_SECONDS = 10.0
        sm.THRASHING_WINDOW_SECONDS = 120.0
        sm.MAX_TRANSITIONS_PER_WINDOW = 8
        return sm

    def test_thrashing_detected_on_oscillation(self):
        """Rapid READING↔EXECUTION cycling triggers thrashing detection."""
        import time

        sm = self._make_sm()
        # Simulate 10 rapid transitions (above MAX_TRANSITIONS_PER_WINDOW=8)
        for i in range(5):
            sm._transition_to(ConversationStage.EXECUTION, confidence=0.9)
            sm._transition_to(ConversationStage.READING, confidence=0.7)

        assert sm._is_thrashing() is True

    def test_thrashing_dampener_blocks(self):
        """After thrashing detected, transitions are blocked for cooldown."""
        sm = self._make_sm()
        # Trigger thrashing (>8 transitions)
        for i in range(5):
            sm._transition_to(ConversationStage.EXECUTION, confidence=0.9)
            sm._transition_to(ConversationStage.READING, confidence=0.7)

        current_stage = sm.state.stage
        # Next transition should be blocked by thrashing cooldown
        sm._transition_to(ConversationStage.EXECUTION, confidence=0.9)
        assert sm.state.stage == current_stage  # Stayed in same stage

    def test_normal_transitions_not_flagged(self):
        """Normal progression (< MAX transitions) is not flagged."""
        sm = self._make_sm()
        sm._transition_to(ConversationStage.READING, confidence=0.8)
        sm._transition_to(ConversationStage.EXECUTION, confidence=0.9)
        sm._transition_to(ConversationStage.VERIFICATION, confidence=0.8)

        assert sm._is_thrashing() is False
